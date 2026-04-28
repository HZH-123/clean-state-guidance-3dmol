from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torch_geometric.data import Batch, Data
from torchmetrics import MetricCollection

import semlaflow.util.functional as smolF
import semlaflow.util.metrics as Metrics
import semlaflow.util.rdkit as smolRD
from semlaflow.models.semla import MolecularGenerator
from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.tokeniser import Vocabulary

_T = torch.Tensor
from typing import Dict  # 确保已导入

_BatchT = Dict[str, _T]


class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        eps=1e-5,
        energy_model: EnergyPredictor = None,
        guidance_scale: float = 0.0,
        vocab: Vocabulary = None
    ):

        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.eps = eps
        self.energy_model = energy_model
        self.guidance_scale = guidance_scale
        self.vocab = vocab
        self.index_to_atomic_num_map = None
        if vocab is not None:
            # 这是一个基于 QM9  的标准映射，您可能需要根据您的 vocab 调整
            token_to_atomic_num = {
                '<MASK>': 0, 'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
                # (如果您的 vocab 更大，请在这里添加)
            }
            tokens = self.vocab._tokens
            self.index_to_atomic_num_map = torch.tensor(
                [token_to_atomic_num.get(token, 0) for token in tokens],
                dtype=torch.long)
        if self.energy_model is not None:
            print(f"[Integrator] 能量引导已启用 | 强度 (Scale): {self.guidance_scale}")

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level
        }

    def step(self, curr: _BatchT, predicted: _BatchT, prior: _BatchT, t: _T, step_size: float) -> _BatchT:
        device = curr["coords"].device
        vocab_size = predicted["atomics"].size(-1)
        n_bonds = predicted["bonds"].size(-1)

        if self.energy_model is not None and self.guidance_scale > 0.0:
            with torch.enable_grad():
                pred_pos_for_grad = predicted["coords"].detach().clone().requires_grad_(True)
                pred_pos_for_grad = pred_pos_for_grad.contiguous()
                curr_z_indices = torch.argmax(predicted["atomics"], dim=-1)
                self.index_to_atomic_num_map = self.index_to_atomic_num_map.to(device)
                curr_z_atomic_nums = self.index_to_atomic_num_map[curr_z_indices]
                curr_mask = curr["mask"]
                batch_size = curr_z_atomic_nums.size(0)

                curr_mask = curr_mask.to(device=pred_pos_for_grad.device)
                mask_bool = curr_mask.bool()
                pos_flat = pred_pos_for_grad[mask_bool]  # [Total_Atoms, 3]

                if pos_flat.shape[0] == 0:
                    energy_gradient_x0 = torch.zeros_like(predicted["coords"])
                else:
                    z_flat = curr_z_atomic_nums[mask_bool]  # [Total_Atoms]
                    batch_vec = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, curr_mask.size(1))
                    batch_vec_flat = batch_vec[mask_bool]  # [Total_Atoms]

                    device =torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
                    z_flat = z_flat.to(device)
                    pos_flat = pos_flat.to(device)
                    batch_vec_flat = batch_vec_flat.to(device)

                    pyg_batch = Batch(z=z_flat, pos=pos_flat, batch=batch_vec_flat)
                    predicted_energy = self.energy_model.predict(pyg_batch)

                    (grad_wrt_pos_flat,) = torch.autograd.grad(
                        outputs=predicted_energy.sum(),
                        inputs=pos_flat,
                        allow_unused=False,
                        retain_graph=False
                    )

                    energy_gradient_x0 = torch.zeros_like(predicted["coords"])
                    energy_gradient_x0[mask_bool] = grad_wrt_pos_flat
                grad_norm = energy_gradient_x0.norm(dim=-1, keepdim=True)
                normalized_grad = energy_gradient_x0 / (grad_norm + 1e-6)
                x0_pred_guided = predicted["coords"] - self.guidance_scale * normalized_grad
                # # =======================================================
                # # === [新增可视化数据捕获逻辑，并直接保存到文件] ===
                # # =======================================================
                # mol_idx = 0  # 仅捕获 batch 中的第一个分子
                # # 仅在 t 接近 0.1 (生成过程接近完成) 且是第一次捕获时执行 (t 从 1 递减到 0)
                # if t[mol_idx].item() < 0.1 and curr["coords"].size(0) > mol_idx:
                #     try:
                #         # 提取有效原子掩码
                #         mask = curr["mask"][mol_idx].cpu().numpy().astype(bool)
                #
                #         # 提取原子类型
                #         z_atomic_nums = curr_z_atomic_nums[mol_idx].cpu().numpy()[mask].tolist()
                #
                #         log_entry = {
                #             "t": t[mol_idx].item(),
                #             "atomic_nums": z_atomic_nums,
                #             # 仅保存有效原子的坐标和梯度，并转换为列表
                #             "coords_pred": predicted["coords"][mol_idx].detach().cpu().numpy()[mask].tolist(),
                #             "coords_guided": x0_pred_guided[mol_idx].detach().cpu().numpy()[mask].tolist(),
                #             "normalized_gradient": energy_gradient_x0[mol_idx].detach().cpu().numpy()[mask].tolist(),
                #         }
                #
                #         # 保存到文件
                #         with open("visualization_data.json", "w") as f:
                #             json.dump(log_entry, f, indent=4)
                #
                #         self._capture_done = True
                #         print(f"--- 成功捕获可视化数据到 visualization_data.json: t = {t[mol_idx].item():.4f} ---")
                #
                #     except Exception as e:
                #         # 确保即使捕获失败也不会中断训练
                #         print(f"可视化数据捕获失败: {e}")
                #         self._capture_done = True  # 即使失败也标记为完成，避免重复尝试
                # # =======================================================
                coord_velocity = (x0_pred_guided - curr["coords"]) / (1 - t.view(-1, 1, 1))

        else:
            coord_velocity = (predicted["coords"] - curr["coords"]) / (1 - t.view(-1, 1, 1))

        coord_velocity += (torch.randn_like(coord_velocity) * self.coord_noise_std)
        coords = curr["coords"] + (step_size * coord_velocity)

        if self.type_strategy == "linear":
            one_hots = torch.eye(vocab_size, device=device).unsqueeze(0).unsqueeze(0)
            type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
            type_velocity = (type_velocity * predicted["atomics"].unsqueeze(-2)).sum(-1)
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Dirichlet refers to sampling from a dirichlet dist, not dirichlet FM
        elif self.type_strategy == "dirichlet":
            type_velocity = torch.distributions.Dirichlet(predicted["atomics"] + self.eps).sample()
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Masking strategy from Discrete Flow Models paper (https://arxiv.org/abs/2402.04997)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                t,
                self.type_mask_index,
                step_size
            )

        # Uniform sampling strategy from Discrete Flow Models paper
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(curr["atomics"], predicted["atomics"], t, step_size)

        # *** Bond update step ***
        # (这部分无需修改)
        if self.type_strategy == "linear":
            one_hots = torch.eye(n_bonds, device=device).view(1, 1, 1, n_bonds, n_bonds)
            bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
            bond_velocity = (bond_velocity * predicted["bonds"].unsqueeze(-2)).sum(-1)
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.type_strategy == "dirichlet":
            bond_velocity = torch.distributions.Dirichlet(predicted["bonds"] + self.eps).sample()
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(curr["bonds"], predicted["bonds"], t, self.bond_mask_index, step_size)

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(curr["bonds"], predicted["bonds"], t, step_size)

        updated = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"]
        }
        return updated

    # TODO test with mask sampling
    def _mask_sampling_step(self, curr_dist, pred_dist, t, mask_index, step_size):
        n_categories = pred_dist.size(-1)

        pred = torch.distributions.Categorical(pred_dist).sample()
        curr = torch.argmax(curr_dist, dim=-1)

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = (step_size * (1 + (self.cat_noise_level * times)) / (1 - times))
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != self.type_mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        curr[unmask] = pred[unmask]
        curr[mask] = mask_index

        return smolF.one_hot_encode_tensor(curr, n_categories)

    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        mult = ((1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times))
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "dirichlet", "mask", "uniform-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(f"{name}_mask_index must be provided if using the mask sampling strategy.")

class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        eps=1e-5,
        energy_model=None,
        guidance_manager=None,
        guidance_scale: float = 0.0,
        guidance_start_t: float = 0.0,
        guidance_end_t: float = 1.0,
        guidance_source: str = "predicted",
        guidance_normalize: bool = True,
        coord_scale: float = 1.0,
        vocab: Vocabulary = None,
    ):
        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.eps = eps
        self.energy_model = energy_model
        self.guidance_manager = guidance_manager
        self.guidance_scale = guidance_scale
        self.guidance_start_t = guidance_start_t
        self.guidance_end_t = guidance_end_t
        if guidance_source not in {"predicted", "curr"}:
            raise ValueError("guidance_source must be either 'predicted' or 'curr'.")
        self.guidance_source = guidance_source
        self.guidance_normalize = guidance_normalize
        self.coord_scale = coord_scale
        self.vocab = vocab
        self.index_to_atomic_num_map = self._build_atomic_number_map(vocab)

        if self.guidance_manager is not None:
            print(
                f"[Integrator] Property guidance enabled | "
                f"scale={self.guidance_scale} | source={self.guidance_source}"
            )
        elif self.energy_model is not None:
            print(f"[Integrator] Energy guidance enabled | scale={self.guidance_scale}")

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
        }

    def step(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        t: _T,
        step_size: float,
        guidance_context: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> _BatchT:
        device = curr["coords"].device
        vocab_size = predicted["atomics"].size(-1)
        n_bonds = predicted["bonds"].size(-1)

        guided_coords = predicted["coords"]
        if self.guidance_scale > 0.0 and self._guidance_is_active(t):
            if self.guidance_manager is not None and guidance_context is not None:
                guidance_state = predicted if self.guidance_source == "predicted" else curr
                guided_coords = self._apply_property_guidance(guidance_state, predicted, curr["mask"], guidance_context)
            elif self.energy_model is not None:
                guided_coords = self._apply_energy_guidance(predicted, curr["mask"])

        coord_velocity = (guided_coords - curr["coords"]) / (1 - t.view(-1, 1, 1))
        coord_velocity += torch.randn_like(coord_velocity) * self.coord_noise_std
        coords = curr["coords"] + (step_size * coord_velocity)

        if self.type_strategy == "linear":
            one_hots = torch.eye(vocab_size, device=device).unsqueeze(0).unsqueeze(0)
            type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
            type_velocity = (type_velocity * predicted["atomics"].unsqueeze(-2)).sum(-1)
            atomics = curr["atomics"] + (step_size * type_velocity)
        elif self.type_strategy == "dirichlet":
            type_velocity = torch.distributions.Dirichlet(predicted["atomics"] + self.eps).sample()
            atomics = curr["atomics"] + (step_size * type_velocity)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                t,
                self.type_mask_index,
                step_size,
            )
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(curr["atomics"], predicted["atomics"], t, step_size)
        else:
            raise ValueError(f"Unknown type strategy '{self.type_strategy}'")

        if self.type_strategy == "linear":
            one_hots = torch.eye(n_bonds, device=device).view(1, 1, 1, n_bonds, n_bonds)
            bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
            bond_velocity = (bond_velocity * predicted["bonds"].unsqueeze(-2)).sum(-1)
            bonds = curr["bonds"] + (step_size * bond_velocity)
        elif self.type_strategy == "dirichlet":
            bond_velocity = torch.distributions.Dirichlet(predicted["bonds"] + self.eps).sample()
            bonds = curr["bonds"] + (step_size * bond_velocity)
        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(curr["bonds"], predicted["bonds"], t, self.bond_mask_index, step_size)
        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(curr["bonds"], predicted["bonds"], t, step_size)
        else:
            raise ValueError(f"Unknown bond strategy '{self.bond_strategy}'")

        return {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"],
        }

    def _build_atomic_number_map(self, vocab):
        if vocab is None:
            return None

        token_to_atomic_num = {
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
        tokens = getattr(vocab, "_tokens", [])
        return torch.tensor([token_to_atomic_num.get(token, 0) for token in tokens], dtype=torch.long)

    def _guidance_is_active(self, t: _T) -> bool:
        time_value = float(t[0].item()) if torch.is_tensor(t) else float(t)
        return self.guidance_start_t <= time_value <= self.guidance_end_t

    def _build_pyg_batch(
        self,
        coords: torch.Tensor,
        atomics: torch.Tensor,
        mask: torch.Tensor,
        target_device: torch.device,
    ):
        if self.index_to_atomic_num_map is None:
            raise RuntimeError("A vocabulary is required to convert atom logits into atomic numbers.")

        atom_indices = torch.argmax(atomics, dim=-1)
        atomic_num_map = self.index_to_atomic_num_map.to(coords.device)
        atomic_nums = atomic_num_map[atom_indices]

        mask_bool = mask.bool()
        batch_size, max_atoms = mask_bool.shape
        batch_vec = torch.arange(batch_size, device=coords.device).unsqueeze(1).expand(-1, max_atoms)

        pos_flat = (coords * self.coord_scale)[mask_bool].to(target_device)
        z_flat = atomic_nums[mask_bool].long().to(target_device)
        batch_flat = batch_vec[mask_bool].long().to(target_device)
        return Batch(z=z_flat, pos=pos_flat, batch=batch_flat), mask_bool

    def _dense_grad_from_flat(self, flat_grad: torch.Tensor, mask_bool: torch.Tensor, reference_coords: torch.Tensor):
        dense_grad = torch.zeros_like(reference_coords)
        dense_grad[mask_bool] = (flat_grad * self.coord_scale).to(reference_coords.device)
        return dense_grad

    def _normalise_dense_grad(self, grad: torch.Tensor, mask: torch.Tensor):
        mask = mask.unsqueeze(-1).to(grad.dtype)
        grad = grad * mask
        mol_norm = grad.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return grad / mol_norm

    def _apply_property_guidance(
        self,
        guidance_state: _BatchT,
        apply_state: _BatchT,
        mask: torch.Tensor,
        guidance_context: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        target_device = getattr(self.guidance_manager, "device", apply_state["coords"].device)
        pyg_batch, mask_bool = self._build_pyg_batch(
            guidance_state["coords"],
            guidance_state["atomics"],
            mask,
            target_device,
        )
        flat_grad, _ = self.guidance_manager.guidance_gradient(pyg_batch, guidance_context["targets"])
        dense_grad = self._dense_grad_from_flat(flat_grad, mask_bool, apply_state["coords"])
        if self.guidance_normalize:
            dense_grad = self._normalise_dense_grad(dense_grad, mask)
        return apply_state["coords"] - (self.guidance_scale * dense_grad)

    def _apply_energy_guidance(self, predicted: _BatchT, mask: torch.Tensor) -> torch.Tensor:
        target_device = predicted["coords"].device
        pyg_batch, mask_bool = self._build_pyg_batch(predicted["coords"], predicted["atomics"], mask, target_device)
        if pyg_batch.pos.numel() == 0:
            return predicted["coords"]

        pyg_batch.pos = pyg_batch.pos.detach().clone().requires_grad_(True)
        energy_batch = Batch(z=pyg_batch.z, pos=pyg_batch.pos, batch=pyg_batch.batch)
        predicted_energy = self.energy_model.predict(energy_batch)
        flat_grad = torch.autograd.grad(
            outputs=predicted_energy.sum(),
            inputs=energy_batch.pos,
            allow_unused=False,
            retain_graph=False,
        )[0].detach()
        dense_grad = self._dense_grad_from_flat(flat_grad, mask_bool, predicted["coords"])
        if self.guidance_normalize:
            dense_grad = self._normalise_dense_grad(dense_grad, mask)
        return predicted["coords"] - (self.guidance_scale * dense_grad)

    def _mask_sampling_step(self, curr_dist, pred_dist, t, mask_index, step_size):
        n_categories = pred_dist.size(-1)

        pred = torch.distributions.Categorical(pred_dist).sample()
        curr = torch.argmax(curr_dist, dim=-1)

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        limit = (step_size * (1 + (self.cat_noise_level * times)) / (1 - times))
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != self.type_mask_index)
        mask[t + step_size >= 1.0] = 0.0

        curr[unmask] = pred[unmask]
        curr[mask] = mask_index
        return smolF.one_hot_encode_tensor(curr, n_categories)

    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        mult = ((1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times))
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        samples = torch.distributions.Categorical(step_probs).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "dirichlet", "mask", "uniform-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(f"{name}_mask_index must be provided if using the mask sampling strategy.")


class MolBuilder:
    def __init__(self, vocab, n_workers=16):
        self.vocab = vocab
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs) for smi in smiles]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(self, coords, atom_dists, mask, bond_dists=None, charge_dists=None, sanitise=True):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # TODO move into from_tensors method of GeometricMolBatch
    def smol_from_tensors(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )

        # mol_dicts = {}
        # for mol_coords, atom_dist, bond_dist, charge_dist in extracted:
        #     mol = {
        #         "coords": mol_coords,
        #         "atomics": atom_dist,
        #         "bonds": bond_dist,
        #         "charges": charge_dist
        #     }
        #     mol_dicts.append(mol)

        self._startup()
        build_fn = partial(self._smol_from_tensors)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        smol_mols = [future.result() for future in futures]
        self.shutdown()

        return smol_mols

    def _mol_from_tensors(self, coords, atom_dists, bond_dists=None, charge_dists=None, sanitise=True):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = self._mol_extract_charges(charge_dists) if charge_dists is not None else None
        return smolRD.mol_from_atoms(coords.numpy(), tokens, bonds=bonds, charges=charges, sanitise=sanitise)

    def _smol_from_tensors(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.size(0)

        charges = torch.tensor(self._mol_extract_charges(charge_dists))
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        bond_types = bond_dists[bond_indices[:, 0], bond_indices[:, 1], :]

        mol = GeometricMol(coords, atom_dists, bond_indices, bond_types, charges)
        return mol

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )
        mol_atom_stabilities = [self.atom_stabilities(*items) for items in extracted]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.shape[0]

        atomics = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists)
        charges = self._mol_extract_charges(charge_dists).tolist()

        # Recreate the adj to ensure it is symmetric
        bond_indices = torch.tensor(bonds[:, :2])
        bond_types = torch.tensor(bonds[:, 2])
        adj = smolF.adj_from_edges(bond_indices, bond_types, n_atoms, symmetric=True)

        adj[adj == 4] = 1.5
        valencies = adj.sum(dim=-1).long()

        stabilities = []
        for i in range(n_atoms):
            atom_type = atomics[i]
            charge = charges[i]
            valence = valencies[i].item()

            if atom_type not in Metrics.ALLOWED_VALENCIES:
                stabilities.append(False)
                continue

            allowed = Metrics.ALLOWED_VALENCIES[atom_type]
            atom_stable = Metrics._is_valid_valence(valence, allowed, charge)
            stabilities.append(atom_stable)

        return stabilities

    # Separate each molecule from the batch
    def _extract_mols(self, coords, atom_dists, mask, bond_dists=None, charge_dists=None):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list, atom_dists_list, bond_dists_list, charge_dists_list)
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = [smolRD.IDX_CHARGE_MAP[idx] for idx in charge_types]
        return np.array(charges)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class MolecularCFM(L.LightningModule):
    def __init__(
        self,
        gen: MolecularGenerator,
        vocab: Vocabulary,
        lr: float,
        integrator: Integrator,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        pairwise_metrics: bool = True,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        sampling_strategy: str = "linear",
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_smiles = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(f"Unsupported type training strategy '{type_strategy}'. "
                             + "Supported are `mse`, `ce` or `mask`.")

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(f"Unsupported bond training strategy '{bond_strategy}'. Supported are `ce` or `mask`.")

        if lr_schedule not in ["constant", "one-cycle"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported. Supported are `constant` or `one-cycle`.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError("total_steps must be provided when using the one-cycle LR scheduler.")

        if distill and (type_strategy == "mask" or bond_strategy == "mask"):
            raise ValueError("Distilled training with masking strategy is not supported.")

        if lr_schedule == "one-cycle" and warm_up_steps is not None:
            print("Note: warm_up_steps is currently ignored if schedule is one-cycle")

        self.gen = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.distill = distill
        self.lr_schedule = lr_schedule
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index

        builder = MolBuilder(vocab)

        if use_ema:
            def ema_avg(avg_param, param, num_averaged):
                return 0.999 * avg_param + (1 - 0.999) * param

            avg_fn = ema_avg
            ema_gen = torch.optim.swa_utils.AveragedModel(gen, avg_fn=avg_fn)

        if compile_model:
            self.gen = self._compile_model(gen)

        self.integrator = integrator
        self.builder = builder
        self.ema_gen = ema_gen if use_ema else None

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "coord_scale": coord_scale,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "distill": distill,
            "lr_schedule": lr_schedule,
            "sampling_strategy": sampling_strategy,
            "use_ema": use_ema,
            "compile_model": compile_model,
            "warm_up_steps": warm_up_steps,
            **gen.hparams,
            **integrator.hparams,
            **kwargs
        }
        self.save_hyperparameters(hparams)

        stability_metrics = {
            "atom-stability": Metrics.AtomStability(),
            "molecule-stability": Metrics.MoleculeStability()
        }
        gen_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "uniqueness": Metrics.Uniqueness(),
            "energy-validity": Metrics.EnergyValidity(),
            "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
            "energy": Metrics.AverageEnergy(),
            "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
            "strain": Metrics.AverageStrainEnergy(),
            "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
            "opt-rmsd": Metrics.AverageOptRmsd()
        }

        if train_smiles is not None:
            print("Creating RDKit mols from training SMILES...")
            train_mols = self.builder.mols_from_smiles(train_smiles, explicit_hs=True)
            train_mols = [mol for mol in train_mols if mol is not None]

            print("Initialising novelty metric...")
            gen_metrics["novelty"] = Metrics.Novelty(train_mols)
            print("Novelty metric complete.")

        self.stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
        self.gen_metrics = MetricCollection(gen_metrics, compute_groups=False)

        if pairwise_metrics:
            pair_metrics = {
                "mol-accuracy": Metrics.MolecularAccuracy(),
                "pair-rmsd": Metrics.MolecularPairRMSD()
            }
            self.pair_metrics = MetricCollection(pair_metrics, compute_groups=False)

        self._init_params()

    def forward(self, batch, t, training=False, cond_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        coords = batch["coords"]
        atom_types = batch["atomics"]
        bonds = batch["bonds"]
        mask = batch["mask"]

        # Prepare invariant atom features
        times = t.view(-1, 1, 1).expand(-1, coords.size(1), -1)
        features = torch.cat((times, atom_types), dim=2)

        # Whether to use the EMA version of the model or not
        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.gen

        if cond_batch is not None:
            out = model(
                coords,
                features,
                edge_feats=bonds,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                atom_mask=mask
            )

        else:
            out = model(coords, features, edge_feats=bonds, atom_mask=mask)

        return out

    def training_step(self, batch, b_idx):
        _, data, interpolated, times = batch

        if self.distill:
            return self._distill_training_step(batch)

        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"])
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        interpolated,
                        times,
                        training=True,
                        cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1)
                    }

        coords, types, bonds, charges = self(
            interpolated,
            times,
            training=True,
            cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges
        }

        losses = self._loss(data, interpolated, predicted)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)

    def validation_step(self, batch, b_idx):
        prior, data, interpolated, times = batch

        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        stabilities = self._generate_stabilities(gen_batch)
        gen_mols = self._generate_mols(gen_batch)

        self.stability_metrics.update(stabilities)
        self.gen_metrics.update(gen_mols)

        # Also measure the model's ability to recreate the original molecule when a bit of prior noise has been added
        if self.pairwise_metrics:
            gen_interp_steps = max(1, int((1 - times[0].item()) * self.integrator.steps))
            gen_interp_batch = self._generate(interpolated, gen_interp_steps)
            gen_interp_mols = self._generate_mols(gen_interp_batch)
            data_mols = self._generate_mols(data)
            self.pair_metrics.update(gen_interp_mols, data_mols)

    def on_validation_epoch_end(self):
        stability_metrics_results = self.stability_metrics.compute()
        gen_metrics_results = self.gen_metrics.compute()
        pair_metrics_results = self.pair_metrics.compute() if self.pairwise_metrics else {}

        metrics = {
            **stability_metrics_results,
            **gen_metrics_results,
            **pair_metrics_results
        }

        for metric, value in metrics.items():
            progbar = True if metric == "validity" else False
            self.log(f"val-{metric}", value, on_epoch=True, logger=True, prog_bar=progbar)

        self.stability_metrics.reset()
        self.gen_metrics.reset()

        if self.pairwise_metrics:
            self.pair_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        prior, _, _, _ = batch
        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=0.0
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3)

        else:
            raise ValueError("Only `constant` or `one-cycle` LR schedules are supported.")

        config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        return config

    def _distill_training_step(self, batch):
        prior, data, interpolated, times = batch

        input_batch = prior
        cond_batch = None
        input_times = torch.zeros_like(times)

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"])
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        input_batch,
                        input_times,
                        training=True,
                        cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1)
                    }

                input_batch = interpolated
                input_times = times

        coords, types, bonds, charges = self(
            input_batch,
            input_times,
            training=True,
            cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges
        }

        losses = self._distill_loss(data, predicted, times)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def _compile_model(self, model):
        return torch.compile(model, dynamic=False, fullgraph=True, mode="reduce-overhead")

    def _loss(self, data, interpolated, predicted):
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"].unsqueeze(2)

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))

        type_loss = self._type_loss(data, interpolated, predicted)
        bond_loss = self._bond_loss(data, interpolated, predicted)
        charge_loss = self._charge_loss(data, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss
        }
        return losses

    def _distill_loss(self, data, predicted, eps=1e-3):
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        mask = data["mask"].unsqueeze(2)

        pred_coords = predicted["coords"]
        pred_atomic_logits = predicted["atomics"]
        pred_bond_logits = predicted["bonds"]

        pred_atomic_dists = F.log_softmax(pred_atomic_logits, dim=-1)
        pred_bond_dists = F.log_softmax(pred_bond_logits, dim=-1)

        # When distilling data should already be given as a dist so use KL div for categoricals
        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        type_loss = F.kl_div(pred_atomic_dists, atomics, reduction="none")
        bond_loss = F.kl_div(pred_bond_dists, bonds, reduction="none")

        adj_matrix = smolF.adj_from_node_mask(mask.squeeze(-1), self_connect=True)
        n_atoms = mask.sum(dim=(1, 2)) + eps
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        coord_loss = (coord_loss * mask).mean(dim=(1, 2))
        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        bond_loss = (bond_loss * adj_matrix.unsqueeze(-1)).sum(dim=(1, 2, 3)) / n_bonds
        charge_loss = self._charge_loss(data, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss
        }
        return losses

    def _type_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["atomics"]
        atomics_dist = data["atomics"]
        mask = data["mask"].unsqueeze(2)
        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            type_loss = F.mse_loss(pred_logits, atomics_dist, reduction="none")
        else:
            atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
            type_loss = F.cross_entropy(pred_logits.flatten(0, 1), atomics, reduction="none")
            type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps

        # If we are training with masking, only compute the loss on masked types
        if self.type_strategy == "mask":
            masked_types = torch.argmax(interpolated["atomics"], dim=-1) == self.type_mask_index
            n_atoms = masked_types.sum(dim=-1) + eps
            type_loss = type_loss * masked_types.float().unsqueeze(-1)

        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        return type_loss

    def _bond_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["bonds"]
        mask = data["mask"]
        bonds = torch.argmax(data["bonds"], dim=-1)
        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(pred_logits.flatten(0, 2), bonds.flatten(0, 2), reduction="none")
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        # Only compute loss on masked bonds if we are training with masking strategy
        if self.bond_strategy == "mask":
            masked_bonds = torch.argmax(interpolated["bonds"], dim=-1) == self.bond_mask_index
            n_bonds = masked_bonds.sum(dim=(1, 2)) + eps
            bond_loss = bond_loss * masked_bonds.float()

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        return bond_loss

    def _charge_loss(self, data, predicted, eps=1e-3):
        pred_logits = predicted["charges"]
        charges = data["charges"]
        mask = data["mask"]
        batch_size, num_atoms, _ = pred_logits.size()

        charges = torch.argmax(charges, dim=-1).flatten(0, 1)
        charge_loss = F.cross_entropy(pred_logits.flatten(0, 1), charges, reduction="none")
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps
        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        return charge_loss

    def _generate(
        self,
        prior,
        steps,
        strategy="linear",
        guidance_context=None,
        trace_callback: Optional[Callable[[int, _T, _BatchT, _BatchT], None]] = None,
    ):
        if self.distill:
            return self._distill_generate(prior)

        if strategy == "linear":
            time_points = np.linspace(0, 1, steps + 1).tolist()

        elif strategy == "log":
            time_points = (1 - np.geomspace(0.01, 1.0, steps + 1)).tolist()
            time_points.reverse()

        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        curr = {k: v.clone() for k, v in prior.items()}

        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"])
        }

        with torch.no_grad():
            for step_idx, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None
                coords, type_logits, bond_logits, charge_logits = self(curr, times, training=False, cond_batch=cond)

                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)

                cond_batch = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs
                }
                predicted = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs,
                    "charges": charge_probs,
                    "mask": curr["mask"]
                }

                if trace_callback is not None:
                    trace_callback(step_idx, times.clone(), curr, predicted)

                curr = self.integrator.step(
                    curr,
                    predicted,
                    prior,
                    times,
                    step_size,
                    guidance_context=guidance_context,
                )
                times = times + step_size

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _distill_generate(self, prior):
        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"])
        }

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        curr = {k: v.clone() for k, v in prior.items()}
        cond = cond_batch if self.self_condition else None

        coords, type_logits, bond_logits, charge_logits = self(curr, times, training=False, cond_batch=cond)

        type_probs = F.softmax(type_logits, dim=-1)
        bond_probs = F.softmax(bond_logits, dim=-1)
        charge_probs = F.softmax(charge_logits, dim=-1)

        predicted = {
            "coords": coords,
            "atomics": type_probs,
            "bonds": bond_probs,
            "charges": charge_probs,
            "mask": curr["mask"]
        }

        if self.self_condition:
            curr = self.integrator.step(curr, predicted, prior, times, 0.5)
            times = times + 0.5
            cond_batch = {
                "coords": coords,
                "atomics": type_probs,
                "bonds": bond_probs
            }
            coords, type_logits, bond_logits, charge_logits = self(
                curr,
                times,
                training=False,
                cond_batch=cond
            )

            type_probs = F.softmax(type_logits, dim=-1)
            bond_probs = F.softmax(bond_logits, dim=-1)
            charge_probs = F.softmax(charge_logits, dim=-1)

            predicted = {
                "coords": coords,
                "atomics": type_probs,
                "bonds": bond_probs,
                "charges": charge_probs,
                "mask": curr["mask"]
            }

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _generate_mols(self, generated, sanitise=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise
        )
        return mols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(coords, atom_dists, masks, bond_dists, charge_dists)
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
