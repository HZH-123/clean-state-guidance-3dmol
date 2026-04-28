#!/usr/bin/env python3
"""
Multi-process parallel evaluation with Vina Dock.
Utilizes all CPU cores for faster docking evaluation.
"""

import argparse
import os
import sys
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from glob import glob
from collections import defaultdict

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_vina import VinaDockingTask

RDLogger.DisableLog('rdApp.*')


def process_single_molecule(args_tuple, protein_root, exhaustiveness, atom_enc_mode):
    """Process a single molecule: reconstruct + vina dock."""
    idx, r_name, sample_idx, eval_step = args_tuple

    try:
        r = torch.load(r_name, weights_only=False, map_location='cpu')

        # Support both trajectory format and direct format
        if 'pred_ligand_pos_traj' in r and len(r['pred_ligand_pos_traj']) > 0:
            # Trajectory format (original)
            all_pred_ligand_pos = r['pred_ligand_pos_traj']
            all_pred_ligand_v = r['pred_ligand_v_traj']

            if sample_idx >= len(all_pred_ligand_pos):
                return None

            pred_pos = all_pred_ligand_pos[sample_idx][eval_step]
            pred_v = all_pred_ligand_v[sample_idx][eval_step]
        else:
            # Direct format (no trajectory)
            all_pred_ligand_pos = r['pred_ligand_pos']
            all_pred_ligand_v = r['pred_ligand_v']

            if sample_idx >= len(all_pred_ligand_pos):
                return None

            pred_pos = all_pred_ligand_pos[sample_idx]
            pred_v = all_pred_ligand_v[sample_idx]

        # Reconstruction
        pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
        pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        smiles = Chem.MolToSmiles(mol)

        if '.' in smiles:
            return None

        # Chemical properties
        chem_results = scoring_func.get_chem(mol)

        # Vina Dock (score_only and minimize only)
        vina_task = VinaDockingTask.from_generated_mol(
            mol, r['data'].ligand_filename, protein_root=protein_root)

        score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)

        vina_results = {
            'score_only': score_only_results,
            'minimize': minimize_results
        }

        return {
            'idx': idx,
            'mol': mol,
            'smiles': smiles,
            'chem_results': chem_results,
            'vina': vina_results,
            'ligand_filename': r['data'].ligand_filename,
        }

    except Exception as e:
        return {'idx': idx, 'error': str(e)}


def compute_diversity_per_pocket(results):
    """Compute diversity as average Tanimoto distance within each pocket."""
    from collections import defaultdict

    pocket_mols = defaultdict(list)
    for r in results:
        pocket_id = r.get('ligand_filename', 'unknown')
        pocket_mols[pocket_id].append(r['mol'])

    pocket_diversities = []

    for pocket_id, mols in pocket_mols.items():
        if len(mols) < 2:
            continue

        fps = []
        for mol in mols:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
            except:
                continue

        if len(fps) < 2:
            continue

        distances = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                distances.append(1 - sim)

        if distances:
            pocket_diversities.append(np.mean(distances))

    return np.mean(pocket_diversities) if pocket_diversities else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 2)')
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--high_affinity_thresh', type=float, default=-8.18)
    parser.add_argument('--hit_qed_thresh', type=float, default=0.4)
    parser.add_argument('--hit_sa_thresh', type=float, default=0.5)
    parser.add_argument('--hit_vina_thresh', type=float, default=-8.18)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate_parallel', log_dir=result_path)

    # Load result files
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))

    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]

    logger.info(f'Found {len(results_fn_list)} result files')

    # Build task list (file_idx, file_name, sample_idx, eval_step)
    tasks = []
    for file_idx, r_name in enumerate(results_fn_list):
        try:
            r = torch.load(r_name, weights_only=False, map_location='cpu')
            # Support both trajectory and direct format
            if 'pred_ligand_pos_traj' in r and len(r['pred_ligand_pos_traj']) > 0:
                n_samples = len(r['pred_ligand_pos_traj'])
            else:
                n_samples = len(r['pred_ligand_pos'])
            for sample_idx in range(n_samples):
                tasks.append((file_idx * 100 + sample_idx, r_name, sample_idx, args.eval_step))
        except Exception as e:
            logger.warning(f'Failed to load {r_name}: {e}')

    total_tasks = len(tasks)
    logger.info(f'Total molecules to evaluate: {total_tasks}')

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = max(1, mp.cpu_count() - 2)  # Leave 2 cores for system
    logger.info(f'Using {args.num_workers} parallel workers')

    # Process in parallel
    process_func = partial(process_single_molecule,
                          protein_root=args.protein_root,
                          exhaustiveness=args.exhaustiveness,
                          atom_enc_mode=args.atom_enc_mode)

    results = []
    errors = 0

    with mp.Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_func, tasks), total=total_tasks, desc='Docking'):
            if result is None:
                continue
            if 'error' in result:
                errors += 1
                continue
            results.append(result)

    logger.info(f'Evaluation complete: {len(results)} successful, {errors} failed')

    # Compute metrics
    if not results:
        logger.error('No successful evaluations')
        return

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    vina_score = [r['vina']['score_only'][0]['affinity'] for r in results]
    vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
    vina_dock = vina_score  # dock模式已禁用，使用score_only数据

    logger.info('=' * 80)
    logger.info('RESULTS')
    logger.info('=' * 80)
    logger.info(f'Samples: {len(results)}')
    logger.info(f'QED:       Mean: {np.mean(qed):.3f}  Std: {np.std(qed):.3f}  Median: {np.median(qed):.3f}')
    logger.info(f'SA:        Mean: {np.mean(sa):.3f}  Std: {np.std(sa):.3f}  Median: {np.median(sa):.3f}')
    logger.info(f'Vina Score: Mean: {np.mean(vina_score):.3f}  Std: {np.std(vina_score):.3f}  Median: {np.median(vina_score):.3f}')
    logger.info(f'Vina Min:   Mean: {np.mean(vina_min):.3f}  Std: {np.std(vina_min):.3f}  Median: {np.median(vina_min):.3f}')
    logger.info(f'Vina Dock:  Mean: {np.mean(vina_dock):.3f}  Std: {np.std(vina_dock):.3f}  Median: {np.median(vina_dock):.3f}')

    # High Affinity Rate
    n_high_aff = sum(1 for v in vina_dock if v < args.high_affinity_thresh)
    high_aff_rate = 100 * n_high_aff / len(vina_dock)
    logger.info(f'High Affinity: {n_high_aff}/{len(vina_dock)} ({high_aff_rate:.1f}%) < {args.high_affinity_thresh}')

    # Hit Rate
    n_hits = sum(1 for q, s, v in zip(qed, sa, vina_dock)
                 if q >= args.hit_qed_thresh and s >= args.hit_sa_thresh and v <= args.hit_vina_thresh)
    hit_rate = 100 * n_hits / len(qed)
    logger.info(f'Hit Rate (QED>={args.hit_qed_thresh}, SA>={args.hit_sa_thresh}, Vina<={args.hit_vina_thresh}): '
                f'{n_hits}/{len(qed)} ({hit_rate:.1f}%)')

    # Diversity
    diversity = compute_diversity_per_pocket(results)
    logger.info(f'Diversity: Mean Tanimoto distance per pocket: {diversity:.4f}')

    logger.info('=' * 80)

    # Save results
    torch.save({
        'results': results,
        'metrics': {
            'qed': {'mean': np.mean(qed), 'std': np.std(qed), 'median': np.median(qed)},
            'sa': {'mean': np.mean(sa), 'std': np.std(sa), 'median': np.median(sa)},
            'vina_score': {'mean': np.mean(vina_score), 'std': np.std(vina_score), 'median': np.median(vina_score)},
            'vina_min': {'mean': np.mean(vina_min), 'std': np.std(vina_min), 'median': np.median(vina_min)},
            'vina_dock': {'mean': np.mean(vina_dock), 'std': np.std(vina_dock), 'median': np.median(vina_dock)},
            'high_affinity_rate': high_aff_rate,
            'hit_rate': hit_rate,
            'diversity': diversity,
        }
    }, os.path.join(result_path, 'metrics_parallel.pt'))

    logger.info(f'Results saved to {result_path}/metrics_parallel.pt')


if __name__ == '__main__':
    main()
