from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMLA_ROOT = PROJECT_ROOT / "semla-flow"
if str(SEMLA_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMLA_ROOT))

from semlaflow.data.qm9_protocol import (
    DELTAE_DEFINITION,
    PROTOCOL_DEFAULT_PREDICTOR_DIR,
    QM9_PROPERTY_SPECS,
    canonicalize_qm9_property_name,
    load_protocol_labels,
    load_protocol_manifest,
    property_values_from_labels,
    property_values_from_y,
    resolve_qm9_root,
    split_indices_tensor,
)

STATS_FILENAME = "qm9_property_stats.json"


def _models_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_checkpoint_path(filename: str) -> Path:
    return _models_dir() / filename


def _legacy_stats_path() -> Path:
    return _models_dir() / STATS_FILENAME


def _default_stats_path(protocol_dir: str | None = None, stats_path: str | None = None) -> Path:
    if stats_path is not None:
        return Path(stats_path)

    if protocol_dir is not None:
        return Path(protocol_dir) / PROTOCOL_DEFAULT_PREDICTOR_DIR / STATS_FILENAME

    return _legacy_stats_path()


def _read_stats_payload(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text())
    if "properties" in payload:
        return payload["properties"]
    return payload


def _compute_legacy_stats(qm9_root: str | None = None) -> Dict[str, Dict[str, float]]:
    dataset = QM9(root=str(resolve_qm9_root(qm9_root, project_root=PROJECT_ROOT)))
    storage = dataset._data if getattr(dataset, "_data", None) is not None else dataset.data
    property_values = property_values_from_y(storage.y[:100000])

    stats = {}
    for name in QM9_PROPERTY_SPECS:
        values = property_values[name].float()
        stats[name] = {
            "mean": float(values.mean()),
            "std": float(values.std().clamp_min(1e-12)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return stats


def _compute_protocol_stats(protocol_dir: str, train_split: str = "da") -> Dict[str, Dict[str, float]]:
    manifest = load_protocol_manifest(protocol_dir)
    labels = load_protocol_labels(protocol_dir)
    split_indices = split_indices_tensor(manifest, train_split)

    stats = {}
    for name in QM9_PROPERTY_SPECS:
        values = property_values_from_labels(labels, name, split_indices).float()
        stats[name] = {
            "mean": float(values.mean()),
            "std": float(values.std().clamp_min(1e-12)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return stats


def load_qm9_property_stats(
    protocol_dir: str | None = None,
    qm9_root: str | None = None,
    stats_path: str | None = None,
    train_split: str = "da",
) -> Dict[str, Dict[str, float]]:
    resolved_stats_path = _default_stats_path(protocol_dir=protocol_dir, stats_path=stats_path)
    if resolved_stats_path.exists():
        stats = _read_stats_payload(resolved_stats_path)
        missing = [name for name in QM9_PROPERTY_SPECS if name not in stats]
        if not missing:
            return stats

        if protocol_dir is None:
            raise KeyError(f"Stats file {resolved_stats_path} is missing properties: {', '.join(missing)}")

        computed = _compute_protocol_stats(protocol_dir, train_split=train_split)
        for name in missing:
            stats[name] = computed[name]

        payload = {
            "deltae_definition": DELTAE_DEFINITION,
            "predictor_train_split": train_split,
            "properties": stats,
        }
        resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_stats_path.write_text(json.dumps(payload, indent=2))
        return stats

    if protocol_dir is not None:
        stats = _compute_protocol_stats(protocol_dir, train_split=train_split)
        payload = {
            "deltae_definition": DELTAE_DEFINITION,
            "predictor_train_split": train_split,
            "properties": stats,
        }
        resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_stats_path.write_text(json.dumps(payload, indent=2))
        return stats

    stats = _compute_legacy_stats(qm9_root=qm9_root)
    resolved_stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def _resolve_checkpoint_path(
    property_name: str,
    model_path: str | None = None,
    protocol_dir: str | None = None,
    model_dir: str | None = None,
) -> Path:
    if model_path is not None:
        return Path(model_path)

    filename = QM9_PROPERTY_SPECS[property_name]["checkpoint"]
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir) / filename)
    if protocol_dir is not None:
        candidates.append(Path(protocol_dir) / PROTOCOL_DEFAULT_PREDICTOR_DIR / filename)
    candidates.append(_default_checkpoint_path(filename))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"QM9 predictor checkpoint not found. Searched: {searched}")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


class QM9PropertyPredictor:
    def __init__(
        self,
        property_name: str,
        model_path: str | None = None,
        qm9_root: str | None = None,
        device: str | torch.device | None = None,
        protocol_dir: str | None = None,
        model_dir: str | None = None,
        stats_path: str | None = None,
        train_split: str = "da",
    ):
        self.property_name = canonicalize_qm9_property_name(property_name)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        checkpoint_path = _resolve_checkpoint_path(
            self.property_name,
            model_path=model_path,
            protocol_dir=protocol_dir,
            model_dir=model_dir,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"QM9 predictor checkpoint not found: {checkpoint_path}")

        self.model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout="add",
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        stats = load_qm9_property_stats(
            protocol_dir=protocol_dir,
            qm9_root=qm9_root,
            stats_path=stats_path,
            train_split=train_split,
        )[self.property_name]
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.tensor(stats["std"], dtype=torch.float32, device=self.device)
        self.min_value = float(stats["min"])
        self.max_value = float(stats["max"])

    def predict(self, data, differentiable: bool = True, return_tensor: bool = False):
        grad_context = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_context:
            if isinstance(data, list):
                raise TypeError("QM9PropertyPredictor.predict does not support list inputs; use PyG Data or Batch.")

            if isinstance(data, Batch):
                z = data.z.to(self.device)
                pos = data.pos.to(self.device)
                batch_vec = data.batch.to(self.device)
            elif isinstance(data, Data):
                z = data.z.to(self.device)
                pos = data.pos.to(self.device)
                if getattr(data, "batch", None) is None:
                    batch_vec = torch.zeros(z.size(0), dtype=torch.long, device=self.device)
                else:
                    batch_vec = data.batch.to(self.device)
            else:
                raise TypeError("Input must be a torch_geometric.data.Data or Batch instance.")

            if z.numel() == 0:
                pred = torch.full((1,), float("nan"), device=self.device)
            else:
                pred = self.model(z, pos, batch_vec)
                if pred.dim() == 2 and pred.size(1) == 1:
                    pred = pred.squeeze(1)
                pred = (pred * self.std) + self.mean

            if differentiable or return_tensor:
                return pred

            pred_cpu = pred.detach().cpu()
            if pred_cpu.numel() == 1:
                return pred_cpu.item()
            return pred_cpu.tolist()
