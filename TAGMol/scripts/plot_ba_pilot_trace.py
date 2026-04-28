import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GUIDE_SCORE_KEYS = ('ba', 'qed', 'sa')


def validate_result_schema(result, result_path):
    required_keys = [
        'pred_ligand_pos0_traj',
        'pred_ligand_pos0_guided_traj',
        'guidance_abs_shift',
        'guidance_rel_shift',
        'guide_scores_traj',
    ]
    for key in required_keys:
        if key not in result:
            raise KeyError(f'{result_path} is missing required key: {key}')
    if len(result['pred_ligand_pos0_traj']) != len(result['pred_ligand_pos0_guided_traj']):
        raise ValueError(f'{result_path} has mismatched guided trajectory sample counts.')
    for sample_idx, (pos0, pos0_guided) in enumerate(zip(result['pred_ligand_pos0_traj'], result['pred_ligand_pos0_guided_traj'])):
        if np.asarray(pos0).shape != np.asarray(pos0_guided).shape:
            raise ValueError(f'{result_path} has mismatched guided trajectory shapes for sample {sample_idx}.')
    for key in GUIDE_SCORE_KEYS:
        if key not in result['guide_scores_traj']:
            raise KeyError(f'{result_path} is missing guide score key: {key}')


def mean_atom_norm(traj):
    traj = torch.as_tensor(traj)
    return traj.norm(dim=-1).mean(dim=-1).cpu().numpy()


def mean_relative_shift(pos0_traj, pos0_guided_traj):
    pos0_traj = torch.as_tensor(pos0_traj)
    pos0_guided_traj = torch.as_tensor(pos0_guided_traj)
    diff = (pos0_guided_traj - pos0_traj).norm(dim=-1)
    base = pos0_traj.norm(dim=-1).clamp(min=1e-8)
    return (diff / base).mean(dim=-1).cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', type=str, required=True)
    parser.add_argument('--clean', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    noisy = torch.load(args.noisy, map_location='cpu', weights_only=False)
    clean = torch.load(args.clean, map_location='cpu', weights_only=False)
    validate_result_schema(noisy, args.noisy)
    validate_result_schema(clean, args.clean)

    noisy_pos0 = noisy['pred_ligand_pos0_traj'][args.sample_idx]
    clean_pos0 = clean['pred_ligand_pos0_traj'][args.sample_idx]
    clean_pos0_guided = clean['pred_ligand_pos0_guided_traj'][args.sample_idx]

    steps = np.arange(noisy_pos0.shape[0])
    noisy_curve = mean_atom_norm(noisy_pos0)
    clean_curve = mean_atom_norm(clean_pos0)
    clean_guided_curve = mean_atom_norm(clean_pos0_guided)
    rel_shift_curve = mean_relative_shift(clean_pos0, clean_pos0_guided)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(steps, noisy_curve, label='noisy x0_hat', linewidth=2)
    axes[0].plot(steps, clean_curve, label='clean x0_hat', linewidth=2)
    axes[0].plot(steps, clean_guided_curve, label='clean x0_hat_guided', linewidth=2)
    axes[0].set_ylabel('Mean ||x0_hat||')
    axes[0].legend()
    axes[0].set_title('Single-Pocket Clean-vs-Noisy Trajectory')

    axes[1].plot(steps, rel_shift_curve, color='black', linewidth=2)
    axes[1].set_xlabel('Sampling Step Index')
    axes[1].set_ylabel('Mean Relative Shift')
    axes[1].set_title('Clean Guidance Relative Shift')

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
