#!/usr/bin/env python3
"""
Analyze 3-pocket pilot results for clean guidance scale comparison.
Key metrics:
1. Success rates: recon_success, complete molecule rate
2. BA metrics: Vina Score, Vina Min, Vina Dock
3. Shift magnitudes: guidance_abs_shift, guidance_rel_shift
4. Trajectory smoothness (sample check)
"""

import os
import sys
import json
import glob
import pickle
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("experiments_multi/pilot_3pockets")

CONDITIONS = [
    "ba_noisy",
    "ba_clean_0.25",
    "ba_clean_0.5",
    "ba_clean_1.0",
]

def load_results(condition):
    """Load all results for a condition (supports .pt and .pkl files)."""
    results = []
    cond_dir = BASE_DIR / condition

    if not cond_dir.exists():
        return results

    # Try .pt files first (PyTorch format)
    pt_files = list(cond_dir.glob("result_*.pt"))
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            results.append(data)
        except Exception as e:
            print(f"  Warning: could not load {pt_file}: {e}")

    # Also try .pkl files
    pkl_files = list(cond_dir.glob("*.pkl"))
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                results.append(data)
        except Exception as e:
            print(f"  Warning: could not load {pkl_file}: {e}")

    print(f"  Found {len(results)} result files")
    return results

def extract_metrics(data_list, condition):
    """Extract key metrics from results."""
    metrics = {
        'n_samples': 0,
        'recon_success': [],
        'complete_mol': [],
        'vina_score': [],
        'vina_min': [],
        'vina_dock': [],
        'guidance_abs_shift': [],
        'guidance_rel_shift': [],
    }

    # First try to load eval results
    eval_file = BASE_DIR / condition / 'eval_results' / 'metrics_-1.pt'
    if eval_file.exists():
        try:
            eval_data = torch.load(eval_file, map_location='cpu', weights_only=False)
            if 'all_results' in eval_data and len(eval_data['all_results']) > 0:
                # Extract Vina scores from eval results
                for result in eval_data['all_results']:
                    if 'vina' in result and result['vina']:
                        vina = result['vina']
                        if 'score_only' in vina and vina['score_only']:
                            metrics['vina_score'].append(vina['score_only'][0]['affinity'])
                        if 'minimize' in vina and vina['minimize']:
                            metrics['vina_min'].append(vina['minimize'][0]['affinity'])
                        if 'dock' in vina and vina['dock']:
                            metrics['vina_dock'].append(vina['dock'][0]['affinity'])
                # Get stability info
                if 'stability' in eval_data:
                    stab = eval_data['stability']
                    n_total = 60  # 3 pockets x 20 samples
                    n_success = int(stab.get('eval_success', 0) * n_total)
                    n_complete = int(stab.get('complete', 0) * n_total)
                    metrics['recon_success'] = [True] * n_success + [False] * (n_total - n_success)
                    metrics['complete_mol'] = [True] * n_complete
                    metrics['n_samples'] = n_total
        except Exception as e:
            print(f"  Warning: could not load eval results: {e}")

    # Extract shift metrics from result files
    for data in data_list:
        if isinstance(data, dict):
            abs_shifts = data.get('guidance_abs_shift', [])
            rel_shifts = data.get('guidance_rel_shift', [])

            if isinstance(abs_shifts, list):
                metrics['guidance_abs_shift'].extend(abs_shifts)
            if isinstance(rel_shifts, list):
                metrics['guidance_rel_shift'].extend(rel_shifts)

    if metrics['n_samples'] == 0:
        metrics['n_samples'] = len(metrics['recon_success'])
    return metrics

def compute_stats(values):
    """Compute mean and std for a list of values."""
    if not values:
        return None, None
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))

def print_condition_report(condition, metrics):
    """Print formatted report for a condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    n_total = metrics['n_samples']
    if n_total == 0:
        print("  No data found")
        return

    print(f"\n[1] SUCCESS RATES")
    n_recon = sum(metrics['recon_success'])
    recon_rate = 100 * n_recon / n_total if n_total > 0 else 0
    print(f"  Total samples:      {n_total}")
    print(f"  Recon success:      {n_recon}/{n_total} ({recon_rate:.1f}%)")

    n_complete = len(metrics['complete_mol'])
    complete_rate = 100 * n_complete / n_total if n_total > 0 else 0
    print(f"  Complete molecules: {n_complete}/{n_total} ({complete_rate:.1f}%)")

    print(f"\n[2] BA METRICS (successful recon only)")
    for metric_name in ['vina_score', 'vina_min', 'vina_dock']:
        values = metrics[metric_name]
        mean, std = compute_stats(values)
        if mean is not None:
            print(f"  {metric_name:15s}: {mean:8.2f} ± {std:6.2f}  (n={len(values)})")
        else:
            print(f"  {metric_name:15s}: No data")

    print(f"\n[3] SHIFT MAGNITUDES")
    abs_mean, abs_std = compute_stats(metrics['guidance_abs_shift'])
    rel_mean, rel_std = compute_stats(metrics['guidance_rel_shift'])

    if abs_mean is not None:
        print(f"  guidance_abs_shift: {abs_mean:.4f} ± {abs_std:.4f}")
    else:
        print(f"  guidance_abs_shift: No data")

    if rel_mean is not None:
        print(f"  guidance_rel_shift: {rel_mean:.4f} ± {rel_std:.4f}")
    else:
        print(f"  guidance_rel_shift: No data")

def print_comparison_table(all_metrics):
    """Print comparison table across conditions."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    header = f"{'Metric':<25}"
    for cond in CONDITIONS:
        header += f"{cond:>15}"
    print(header)
    print("-" * 80)

    # Success rates
    for metric_key, metric_label in [
        ('recon_success', 'Recon Success (%)'),
        ('complete_mol', 'Complete Mol (%)'),
    ]:
        row = f"{metric_label:<25}"
        for cond in CONDITIONS:
            m = all_metrics[cond]
            n_total = m['n_samples']
            if metric_key == 'recon_success':
                n = sum(m[metric_key])
            else:
                n = len(m[metric_key])
            pct = 100 * n / n_total if n_total > 0 else 0
            row += f"{pct:>14.1f}%"
        print(row)

    # Vina metrics
    print("-" * 80)
    for metric_key, metric_label in [
        ('vina_score', 'Vina Score (mean)'),
        ('vina_min', 'Vina Min (mean)'),
        ('vina_dock', 'Vina Dock (mean)'),
    ]:
        row = f"{metric_label:<25}"
        for cond in CONDITIONS:
            m = all_metrics[cond]
            mean, _ = compute_stats(m[metric_key])
            if mean is not None:
                row += f"{mean:>15.2f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    # Shift magnitudes
    print("-" * 80)
    for metric_key, metric_label in [
        ('guidance_abs_shift', 'Abs Shift (mean)'),
        ('guidance_rel_shift', 'Rel Shift (mean)'),
    ]:
        row = f"{metric_label:<25}"
        for cond in CONDITIONS:
            m = all_metrics[cond]
            mean, _ = compute_stats(m[metric_key])
            if mean is not None:
                row += f"{mean:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

def check_trajectory_smoothness(condition, pocket_id=0):
    """Check trajectory smoothness for one pocket."""
    print(f"\n[4] TRAJECTORY CHECK: {condition} (pocket {pocket_id})")

    cond_dir = BASE_DIR / condition
    traj_files = list(cond_dir.glob("*_traj.pt")) + list(cond_dir.glob("*_traj.pkl"))

    if not traj_files:
        print(f"  No trajectory files found")
        return

    for traj_file in traj_files[:1]:  # Check first one
        try:
            if str(traj_file).endswith('.pt'):
                traj_data = torch.load(traj_file, map_location='cpu')
            else:
                with open(traj_file, 'rb') as f:
                    traj_data = pickle.load(f)

            # Check if we have position trajectories
            if 'pred_ligand_pos0_traj' in traj_data:
                pos_traj = traj_data['pred_ligand_pos0_traj']
                print(f"  pred_ligand_pos0_traj: {len(pos_traj)} timesteps")

                # Check for jumps (large position changes)
                if len(pos_traj) > 1:
                    diffs = []
                    for i in range(1, len(pos_traj)):
                        diff = np.linalg.norm(pos_traj[i] - pos_traj[i-1], axis=-1).mean()
                        diffs.append(diff)
                    max_diff = max(diffs)
                    mean_diff = np.mean(diffs)
                    print(f"    Position change: mean={mean_diff:.4f}, max={max_diff:.4f}")
                    if max_diff > 1.0:
                        print(f"    WARNING: Large jumps detected!")
                    else:
                        print(f"    OK: Smooth trajectory")

            if 'pred_ligand_pos0_guided_traj' in traj_data:
                guided_traj = traj_data['pred_ligand_pos0_guided_traj']
                print(f"  pred_ligand_pos0_guided_traj: {len(guided_traj)} timesteps")

                if len(guided_traj) > 1:
                    diffs = []
                    for i in range(1, len(guided_traj)):
                        diff = np.linalg.norm(guided_traj[i] - guided_traj[i-1], axis=-1).mean()
                        diffs.append(diff)
                    max_diff = max(diffs)
                    mean_diff = np.mean(diffs)
                    print(f"    Position change: mean={mean_diff:.4f}, max={max_diff:.4f}")
                    if max_diff > 1.0:
                        print(f"    WARNING: Large jumps detected!")
                    else:
                        print(f"    OK: Smooth trajectory")

        except Exception as e:
            print(f"  Error loading trajectory: {e}")

def main():
    print("="*80)
    print("3-POCKET PILOT ANALYSIS")
    print("="*80)
    print(f"Base directory: {BASE_DIR}")

    if not BASE_DIR.exists():
        print(f"\nERROR: Directory not found: {BASE_DIR}")
        print("Run the pilot first: bash scripts/run_pilot_3pockets.sh")
        sys.exit(1)

    # Load and analyze all conditions
    all_metrics = {}
    for condition in CONDITIONS:
        print(f"\nLoading {condition}...")
        data_list = load_results(condition)
        if not data_list:
            print(f"  WARNING: No results found for {condition}")
        metrics = extract_metrics(data_list, condition)
        all_metrics[condition] = metrics
        print_condition_report(condition, metrics)

    # Print comparison table
    print_comparison_table(all_metrics)

    # Check trajectory smoothness for clean_1.0 (most critical)
    print(f"\n{'='*80}")
    print("TRAJECTORY SMOOTHNESS CHECK")
    print(f"{'='*80}")
    for condition in CONDITIONS:
        check_trajectory_smoothness(condition, pocket_id=0)

    # Summary recommendation
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATION")
    print(f"{'='*80}")

    # Compare clean_1.0 vs others
    noisy_recon = 100 * sum(all_metrics['ba_noisy']['recon_success']) / all_metrics['ba_noisy']['n_samples'] if all_metrics['ba_noisy']['n_samples'] > 0 else 0
    clean_1_recon = 100 * sum(all_metrics['ba_clean_1.0']['recon_success']) / all_metrics['ba_clean_1.0']['n_samples'] if all_metrics['ba_clean_1.0']['n_samples'] > 0 else 0

    noisy_vina, _ = compute_stats(all_metrics['ba_noisy']['vina_score'])
    clean_1_vina, _ = compute_stats(all_metrics['ba_clean_1.0']['vina_score'])

    print(f"\nKey comparison (clean_1.0 vs noisy):")
    print(f"  Recon success: {noisy_recon:.1f}% -> {clean_1_recon:.1f}% (Δ {clean_1_recon - noisy_recon:+.1f}%)")
    if noisy_vina and clean_1_vina:
        print(f"  Vina Score:    {noisy_vina:.2f} -> {clean_1_vina:.2f} (Δ {clean_1_vina - noisy_vina:+.2f})")

    print(f"\nRecommendation:")
    if clean_1_recon < noisy_recon - 10:
        print(f"  ⚠️  clean_1.0 drops recon success by >10%. Consider discarding 1.0.")
        print(f"     Full run recommendation: keep 0.25 and 0.5 only.")
    elif clean_1_vina and noisy_vina and clean_1_vina < noisy_vina - 1.0:
        print(f"  ✓  clean_1.0 improves Vina score significantly.")
        print(f"     Worth the stability trade-off if recon drop is acceptable.")
    else:
        print(f"  →  Need more data or review trajectory smoothness to decide.")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
