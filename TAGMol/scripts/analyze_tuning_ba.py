#!/usr/bin/env python3
"""
Analyze BA tuning results (4 BA values x 4 pockets x 50 samples)
Compare QED, SA, Vina metrics to find optimal BA gradient scale.
"""

import os
import sys
import glob
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("experiments_multi/tuning_ba")
BA_VALUES = [0.5, 1.0, 1.5, 2.0]

def load_eval_metrics(ba_val):
    """Load evaluation metrics for a BA value."""
    eval_file = BASE_DIR / f"ba_{ba_val}" / "eval_results" / "metrics_-1.pt"

    if not eval_file.exists():
        print(f"Warning: {eval_file} not found")
        return None

    try:
        data = torch.load(eval_file, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading {eval_file}: {e}")
        return None

def extract_metrics(data):
    """Extract key metrics from evaluation data."""
    if not data or 'all_results' not in data:
        return None

    results = data['all_results']
    if not results:
        return None

    metrics = {
        'n_samples': len(results),
        'qed': [],
        'sa': [],
        'vina_score': [],
        'vina_min': [],
        'vina_dock': [],
    }

    for r in results:
        if 'chem_results' in r:
            metrics['qed'].append(r['chem_results'].get('qed', 0))
            metrics['sa'].append(r['chem_results'].get('sa', 0))

        if 'vina' in r and r['vina']:
            vina = r['vina']
            if 'score_only' in vina and vina['score_only']:
                metrics['vina_score'].append(vina['score_only'][0]['affinity'])
            if 'minimize' in vina and vina['minimize']:
                metrics['vina_min'].append(vina['minimize'][0]['affinity'])
            if 'dock' in vina and vina['dock']:
                metrics['vina_dock'].append(vina['dock'][0]['affinity'])

    return metrics

def compute_hit_rate(qed_list, sa_list, vina_list):
    """Compute hit rate: QED >= 0.4, SA >= 0.5, Vina <= -8.18."""
    if not (qed_list and sa_list and vina_list):
        return 0.0

    n_hits = 0
    for qed, sa, vina in zip(qed_list, sa_list, vina_list):
        if qed >= 0.4 and sa >= 0.5 and vina <= -8.18:
            n_hits += 1

    return 100 * n_hits / len(qed_list)

def print_comparison_table(all_metrics):
    """Print comparison table across BA values."""
    print("\n" + "="*100)
    print("BA TUNING RESULTS COMPARISON")
    print("="*100)
    print(f"{'Metric':<25} {'BA=0.5':<15} {'BA=1.0':<15} {'BA=1.5':<15} {'BA=2.0':<15}")
    print("-"*100)

    # QED
    row = f"{'QED (mean)':<25}"
    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        qed = m.get('qed', [])
        if qed:
            row += f"{np.mean(qed):<15.3f}"
        else:
            row += f"{'N/A':<15}"
    print(row)

    # SA
    row = f"{'SA (mean)':<25}"
    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        sa = m.get('sa', [])
        if sa:
            row += f"{np.mean(sa):<15.3f}"
        else:
            row += f"{'N/A':<15}"
    print(row)

    # Vina Score
    row = f"{'Vina Score (mean)':<25}"
    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        vina = m.get('vina_score', [])
        if vina:
            row += f"{np.mean(vina):<15.3f}"
        else:
            row += f"{'N/A':<15}"
    print(row)

    # Vina Min
    row = f"{'Vina Min (mean)':<25}"
    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        vina = m.get('vina_min', [])
        if vina:
            row += f"{np.mean(vina):<15.3f}"
        else:
            row += f"{'N/A':<15}"
    print(row)

    # Hit Rate
    print("-"*100)
    row = f"{'Hit Rate %':<25}"
    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        qed = m.get('qed', [])
        sa = m.get('sa', [])
        vina = m.get('vina_score', []) or m.get('vina_dock', [])
        if qed and sa and vina:
            hit_rate = compute_hit_rate(qed, sa, vina)
            row += f"{hit_rate:<15.1f}"
        else:
            row += f"{'N/A':<15}"
    print(row)

    # Sample count
    print("-"*100)
    row = f"{'N samples':<25}"
    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        n = m.get('n_samples', 0)
        row += f"{n:<15}"
    print(row)

    print("="*100)

def recommend_ba(all_metrics):
    """Recommend best BA value based on metrics."""
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    best_score = -float('inf')
    best_ba = None

    for ba in BA_VALUES:
        m = all_metrics.get(ba, {})
        qed = m.get('qed', [])
        sa = m.get('sa', [])
        vina = m.get('vina_score', []) or m.get('vina_dock', [])

        if not (qed and sa and vina):
            continue

        # Scoring: balance between properties
        # Normalize: QED (0-1), SA (0-1), Vina (-15 to -5, lower is better)
        qed_mean = np.mean(qed)
        sa_mean = np.mean(sa)
        vina_mean = np.mean(vina)

        # Combined score (higher is better)
        # QED: 0-1, SA: 0-1, Vina: -15 to -5 -> normalize to 0-1 where -10 = 0.5
        vina_norm = (-vina_mean + 5) / 10  # -15->1, -5->0
        vina_norm = max(0, min(1, vina_norm))

        score = qed_mean + sa_mean + vina_norm

        hit_rate = compute_hit_rate(qed, sa, vina)

        print(f"BA={ba}: QED={qed_mean:.3f}, SA={sa_mean:.3f}, Vina={vina_mean:.3f}, HitRate={hit_rate:.1f}%, Score={score:.3f}")

        if score > best_score:
            best_score = score
            best_ba = ba

    if best_ba:
        print(f"\n>>> Recommended BA gradient scale: {best_ba}")
        print(f"    Next step: Fix BA={best_ba}, tune SA: 5, 7.5, 10")
    else:
        print("\n>>> Insufficient data for recommendation")

    print("="*100)

def main():
    print("="*100)
    print("BA TUNING ANALYSIS")
    print("Config: 4 pockets x 50 samples, QED=20, SA=5, clean=0.5")
    print("="*100)

    all_metrics = {}

    for ba in BA_VALUES:
        print(f"\nLoading BA={ba}...")
        data = load_eval_metrics(ba)
        if data:
            metrics = extract_metrics(data)
            if metrics:
                all_metrics[ba] = metrics
                print(f"  Loaded {metrics['n_samples']} samples")
            else:
                print(f"  No metrics extracted")
        else:
            print(f"  No data found")

    if not all_metrics:
        print("\nERROR: No evaluation data found. Run evaluation first:")
        print("  bash scripts/run_tuning_ba_eval.sh")
        sys.exit(1)

    print_comparison_table(all_metrics)
    recommend_ba(all_metrics)

if __name__ == "__main__":
    main()
