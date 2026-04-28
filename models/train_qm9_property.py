import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch.utils.data import Subset

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
    property_values_from_batch_targets,
    resolve_qm9_root,
    split_indices_tensor,
)

STATS_FILENAME = "qm9_property_stats.json"


def compute_stats(labels, property_name, indices):
    values = labels["properties"][property_name][indices].float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std().clamp_min(1e-12)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_stats(stats_path: Path, property_name: str, stats: dict, train_split: str, val_split: str):
    payload = {}
    if stats_path.exists():
        payload = json.loads(stats_path.read_text())

    if "properties" not in payload:
        payload = {"properties": payload} if payload else {"properties": {}}

    payload["deltae_definition"] = DELTAE_DEFINITION
    payload["predictor_train_split"] = train_split
    payload["predictor_val_split"] = val_split
    payload["properties"][property_name] = stats

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2))


def main(args):
    property_name = canonicalize_qm9_property_name(args.property_name)
    protocol_dir = Path(args.protocol_dir)
    predictor_dir = protocol_dir / PROTOCOL_DEFAULT_PREDICTOR_DIR
    predictor_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_protocol_manifest(protocol_dir)
    labels = load_protocol_labels(protocol_dir)
    train_indices = split_indices_tensor(manifest, args.train_split)
    val_indices = split_indices_tensor(manifest, args.val_split)

    qm9_root = resolve_qm9_root(args.qm9_root, project_root=PROJECT_ROOT)
    dataset = QM9(root=str(qm9_root))
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stats = compute_stats(labels, property_name, train_indices)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)

    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.z, batch.pos, batch.batch)
            target = property_values_from_batch_targets(batch.y, property_name)
            target = ((target.to(device) - mean) / std).unsqueeze(1)
            loss = F.l1_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.z, batch.pos, batch.batch)
                target = property_values_from_batch_targets(batch.y, property_name)
                target = ((target.to(device) - mean) / std).unsqueeze(1)
                loss = F.l1_loss(pred, target)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = predictor_dir / QM9_PROPERTY_SPECS[property_name]["checkpoint"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "property_name": property_name,
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                    "deltae_definition": DELTAE_DEFINITION,
                },
                checkpoint_path,
            )
            save_stats(predictor_dir / STATS_FILENAME, property_name, stats, args.train_split, args.val_split)
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Saved -> {checkpoint_path}"
            )
        else:
            patience_counter += 1
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--property_name", type=str, required=True)
    parser.add_argument("--protocol_dir", type=str, required=True)
    parser.add_argument("--qm9_root", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="da")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    main(parser.parse_args())
