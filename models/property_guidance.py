from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


DEFAULT_TOKEN_TO_ATOMIC_NUM = {
    "<PAD>": 0,
    "<MASK>": 0,
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def build_vocab_atomic_number_map(vocab) -> torch.Tensor:
    tokens = getattr(vocab, "_tokens", vocab)
    return torch.tensor([DEFAULT_TOKEN_TO_ATOMIC_NUM.get(token, 0) for token in tokens], dtype=torch.long)


def dense_molecular_batch_to_pyg(
    coords: torch.Tensor,
    atomics: torch.Tensor,
    mask: torch.Tensor,
    index_to_atomic_num_map: torch.Tensor,
    coord_scale: float = 1.0,
    coords_are_normalized: bool = True,
    device: str | torch.device | None = None,
) -> Tuple[Batch, Dict[str, torch.Tensor]]:
    if atomics.dim() == 3:
        atom_indices = torch.argmax(atomics, dim=-1)
    else:
        atom_indices = atomics

    atomic_num_map = index_to_atomic_num_map.to(coords.device)
    atomic_nums = atomic_num_map[atom_indices]

    mask_bool = mask.bool()
    batch_size, max_atoms = mask_bool.shape
    batch_vec = torch.arange(batch_size, device=coords.device).unsqueeze(1).expand(-1, max_atoms)

    scaled_coords = coords * coord_scale if coords_are_normalized else coords
    pos_flat = scaled_coords[mask_bool]
    z_flat = atomic_nums[mask_bool].long()
    batch_flat = batch_vec[mask_bool].long()

    target_device = torch.device(device) if device is not None else pos_flat.device
    pos_flat = pos_flat.to(target_device)
    z_flat = z_flat.to(target_device)
    batch_flat = batch_flat.to(target_device)

    pyg_batch = Batch(z=z_flat, pos=pos_flat, batch=batch_flat)
    metadata = {"mask_bool": mask_bool}
    return pyg_batch, metadata


def flat_grad_to_dense(
    flat_grad: torch.Tensor,
    mask_bool: torch.Tensor,
    reference_coords: torch.Tensor,
    coord_scale: float = 1.0,
    grad_is_wrt_raw_coords: bool = True,
) -> torch.Tensor:
    grad = flat_grad * coord_scale if grad_is_wrt_raw_coords else flat_grad
    dense_grad = torch.zeros_like(reference_coords)
    dense_grad[mask_bool] = grad.to(reference_coords.device)
    return dense_grad


@dataclass
class GuidanceObjective:
    name: str
    predictor: object
    weight: float = 1.0
    loss_type: str = "mse"

    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(prediction.device, prediction.dtype)
        if self.loss_type == "mse":
            return (prediction - target) ** 2
        if self.loss_type == "l1":
            return (prediction - target).abs()
        if self.loss_type == "huber":
            return F.huber_loss(prediction, target, reduction="none")

        raise ValueError(f"Unsupported loss type '{self.loss_type}'")


class PropertyGuidanceManager:
    def __init__(
        self,
        objectives: Iterable[GuidanceObjective],
        relation_mode: str = "independent",
        cascade_strength: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        relation_mode = relation_mode.lower()
        if relation_mode not in {"independent", "cascade"}:
            raise ValueError("relation_mode must be either 'independent' or 'cascade'")

        self.objectives: List[GuidanceObjective] = list(objectives)
        if len(self.objectives) == 0:
            raise ValueError("At least one guidance objective is required.")

        self.relation_mode = relation_mode
        self.cascade_strength = cascade_strength
        self.device = torch.device(device)

    @property
    def objective_names(self) -> List[str]:
        return [objective.name for objective in self.objectives]

    def _target_tensor(self, targets: Dict[str, torch.Tensor], name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        if name not in targets:
            raise KeyError(f"Missing target tensor for objective '{name}'")

        target = targets[name]
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)

        target = target.to(device=device, dtype=torch.float32)
        if target.dim() == 0:
            target = target.repeat(batch_size)

        if target.numel() != batch_size:
            raise ValueError(
                f"Target tensor for '{name}' has {target.numel()} values, but batch size is {batch_size}."
            )

        return target

    def predict_properties(self, pyg_batch: Batch, differentiable: bool = False) -> Dict[str, torch.Tensor]:
        predictions = {}
        for objective in self.objectives:
            prediction = objective.predictor.predict(
                pyg_batch,
                differentiable=differentiable,
                return_tensor=not differentiable,
            )
            if not differentiable:
                prediction = prediction.detach()
            predictions[objective.name] = prediction

        return predictions

    def guidance_gradient(
        self,
        pyg_batch: Batch,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.enable_grad():
            if pyg_batch.pos.numel() == 0:
                return torch.zeros_like(pyg_batch.pos), {}

            if self.relation_mode == "independent":
                return self._independent_gradient(pyg_batch, targets)

            return self._cascade_gradient(pyg_batch, targets)

    def _independent_gradient(
        self,
        pyg_batch: Batch,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_grad = torch.zeros_like(pyg_batch.pos)
        predictions = {}
        batch_size = int(pyg_batch.batch.max().item()) + 1 if pyg_batch.batch.numel() > 0 else 0

        for objective in self.objectives:
            pos = pyg_batch.pos.detach().clone().requires_grad_(True)
            batch = Batch(z=pyg_batch.z, pos=pos, batch=pyg_batch.batch)
            prediction = objective.predictor.predict(batch, differentiable=True, return_tensor=True)
            target = self._target_tensor(targets, objective.name, batch_size, pos.device)
            loss = objective.weight * objective.loss(prediction, target).mean()
            grad = torch.autograd.grad(loss, pos, retain_graph=False, create_graph=False)[0].detach()

            total_grad = total_grad + grad
            predictions[objective.name] = prediction.detach()

        return total_grad, predictions

    def _cascade_gradient(
        self,
        pyg_batch: Batch,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cumulative_grad = torch.zeros_like(pyg_batch.pos)
        predictions = {}
        batch_size = int(pyg_batch.batch.max().item()) + 1 if pyg_batch.batch.numel() > 0 else 0

        for objective in self.objectives:
            shifted_pos = (pyg_batch.pos - (self.cascade_strength * cumulative_grad)).detach().clone()
            shifted_pos.requires_grad_(True)
            batch = Batch(z=pyg_batch.z, pos=shifted_pos, batch=pyg_batch.batch)
            prediction = objective.predictor.predict(batch, differentiable=True, return_tensor=True)
            target = self._target_tensor(targets, objective.name, batch_size, shifted_pos.device)
            loss = objective.weight * objective.loss(prediction, target).mean()
            grad = torch.autograd.grad(loss, shifted_pos, retain_graph=False, create_graph=False)[0].detach()

            cumulative_grad = cumulative_grad + grad
            predictions[objective.name] = prediction.detach()

        return cumulative_grad, predictions
