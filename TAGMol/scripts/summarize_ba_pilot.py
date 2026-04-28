import argparse
import os
import sys
from glob import glob

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GUIDE_SCORE_KEYS = ('ba', 'qed', 'sa')


def validate_result_schema(result, result_file):
    required_keys = [
        'pred_ligand_pos0_traj',
        'pred_ligand_pos0_guided_traj',
        'guidance_abs_shift',
        'guidance_rel_shift',
        'guide_scores_traj',
    ]
    for key in required_keys:
        if key not in result:
            raise KeyError(f'{result_file} is missing required key: {key}')
    if len(result['pred_ligand_pos0_traj']) != len(result['pred_ligand_pos0_guided_traj']):
        raise ValueError(f'{result_file} has mismatched guided trajectory sample counts.')
    for sample_idx, (pos0, pos0_guided) in enumerate(zip(result['pred_ligand_pos0_traj'], result['pred_ligand_pos0_guided_traj'])):
        if np.asarray(pos0).shape != np.asarray(pos0_guided).shape:
            raise ValueError(f'{result_file} has mismatched guided trajectory shapes for sample {sample_idx}.')
    for key in GUIDE_SCORE_KEYS:
        if key not in result['guide_scores_traj']:
            raise KeyError(f'{result_file} is missing guide score key: {key}')


def load_sample_metrics(sample_path):
    result_files = sorted(
        glob(os.path.join(sample_path, 'result_*.pt')),
        key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]),
    )
    abs_shifts = []
    rel_shifts = []
    for result_file in result_files:
        result = torch.load(result_file, map_location='cpu', weights_only=False)
        validate_result_schema(result, result_file)
        abs_shifts.extend(float(x) for x in result['guidance_abs_shift'])
        rel_shifts.extend(float(x) for x in result['guidance_rel_shift'])
    return {
        'mean_abs_shift': float(np.mean(abs_shifts)) if abs_shifts else 0.0,
        'mean_rel_shift': float(np.mean(rel_shifts)) if rel_shifts else 0.0,
    }


def load_eval_metrics(sample_path):
    metrics_file = os.path.join(sample_path, 'eval_results', 'metrics_-1.pt')
    metrics = torch.load(metrics_file, map_location='cpu', weights_only=False)
    stability = metrics['stability']
    all_results = metrics['all_results']
    out = {
        'complete': float(stability['complete']),
        'eval_success': float(stability['eval_success']),
    }
    if all_results:
        vina_score = [x['vina']['score_only'][0]['affinity'] for x in all_results if x['vina'] is not None]
        vina_min = [x['vina']['minimize'][0]['affinity'] for x in all_results if x['vina'] is not None]
        if vina_score:
            out['vina_score_mean'] = float(np.mean(vina_score))
        if vina_min:
            out['vina_min_mean'] = float(np.mean(vina_min))
    return out


def format_float(value):
    if value is None:
        return 'NA'
    return f'{value:.4f}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--runs', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    assert len(args.runs) == len(args.labels), 'runs and labels must have same length'
    rows = []
    for run_name, label in zip(args.runs, args.labels):
        sample_path = os.path.join(args.root, run_name)
        sample_metrics = load_sample_metrics(sample_path)
        eval_metrics = load_eval_metrics(sample_path)
        row = {
            'label': label,
            **eval_metrics,
            **sample_metrics,
        }
        rows.append(row)

    headers = ['label', 'vina_score_mean', 'vina_min_mean', 'complete', 'eval_success', 'mean_rel_shift']
    table_lines = [
        '| ' + ' | '.join(['Setting', 'Vina Score', 'Vina Min', 'Complete', 'Eval Success', 'Mean Rel Shift']) + ' |',
        '| ' + ' | '.join(['---'] * 6) + ' |',
    ]
    for row in rows:
        table_lines.append(
            '| ' + ' | '.join([
                row['label'],
                format_float(row.get('vina_score_mean')),
                format_float(row.get('vina_min_mean')),
                format_float(row.get('complete')),
                format_float(row.get('eval_success')),
                format_float(row.get('mean_rel_shift')),
            ]) + ' |'
        )

    detail_lines = ['\nDetailed Shift Metrics', 'label\tmean_abs_shift\tmean_rel_shift']
    for row in rows:
        detail_lines.append(
            f"{row['label']}\t{format_float(row.get('mean_abs_shift'))}\t{format_float(row.get('mean_rel_shift'))}"
        )

    output = '\n'.join(table_lines + detail_lines) + '\n'
    print(output)
    if args.out is not None:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(output)
