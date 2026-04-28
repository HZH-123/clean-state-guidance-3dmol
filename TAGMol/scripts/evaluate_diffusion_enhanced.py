#!/usr/bin/env python3
"""
Enhanced evaluation script with additional metrics:
- QED, SA, Vina Score, Vina Min, Vina Dock
- High Affinity % (Vina Score < threshold)
- Diversity (fingerprint-based)
- Hit Rate % (multi-criteria)
"""

import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    if len(all_ring_sizes) == 0:
        logger.info('ring size: NA ratio: None')
        return
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def compute_diversity_per_pocket(results, logger=None):
    """Compute diversity as average Tanimoto distance (1 - similarity) within each pocket.

    Args:
        results: List of result dicts with 'mol' and 'ligand_filename' keys
        logger: Logger instance

    Returns:
        Average diversity across all pockets
    """
    if not results:
        return None

    # Group molecules by pocket
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

        # Compute pairwise distances (1 - similarity)
        distances = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                distances.append(1 - sim)  # Distance = 1 - similarity

        if distances:
            avg_distance = np.mean(distances)
            pocket_diversities.append(avg_distance)

    if not pocket_diversities:
        return None

    overall_diversity = np.mean(pocket_diversities)

    if logger:
        logger.info(f'Diversity:  {len(pocket_diversities)} pockets evaluated')
        logger.info(f'Diversity:  Mean Tanimoto distance per pocket: {overall_diversity:.4f}')

    return overall_diversity


def compute_high_affinity_rate(vina_scores, ref_scores=None, threshold=-8.5, logger=None):
    """Compute percentage of molecules with binding affinity better than reference.

    Args:
        vina_scores: Generated molecule vina scores
        ref_scores: Reference molecule vina scores (per pocket)
        threshold: Alternative threshold if ref_scores not provided
    """
    if not vina_scores:
        return None

    if ref_scores and len(ref_scores) == len(vina_scores):
        # Compare to reference: generated should be better (more negative) than reference
        n_better = sum(1 for gen, ref in zip(vina_scores, ref_scores) if gen < ref)
        rate = 100 * n_better / len(vina_scores)
        if logger:
            logger.info(f'High Affinity: {n_better}/{len(vina_scores)} ({rate:.1f}%) better than reference')
    else:
        # Fallback to threshold
        n_high_affinity = sum(1 for s in vina_scores if s < threshold)
        rate = 100 * n_high_affinity / len(vina_scores)
        if logger:
            logger.info(f'High Affinity: {n_high_affinity}/{len(vina_scores)} ({rate:.1f}%) < {threshold}')

    return rate


def compute_hit_rate(qed_list, sa_list, vina_scores,
                     qed_thresh=0.4, sa_thresh=0.5, vina_thresh=-8.18,
                     logger=None):
    """Compute hit rate based on multi-criteria:
    - QED >= 0.4
    - SA >= 0.5
    - Vina Dock <= -8.18 kcal/mol
    """
    if not (qed_list and sa_list and vina_scores):
        return None

    n_hits = 0
    for qed, sa, vina in zip(qed_list, sa_list, vina_scores):
        if qed >= qed_thresh and sa >= sa_thresh and vina <= vina_thresh:
            n_hits += 1

    rate = 100 * n_hits / len(qed_list)

    if logger:
        logger.info(f'Hit Rate (QED>={qed_thresh}, SA>={sa_thresh}, Vina<={vina_thresh}): '
                   f'{n_hits}/{len(qed_list)} ({rate:.1f}%)')

    return rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    # Default thresholds for hit rate: QED >= 0.4, SA >= 0.5, Vina Dock <= -8.18
    parser.add_argument('--hit_qed_thresh', type=float, default=0.4)
    parser.add_argument('--hit_sa_thresh', type=float, default=0.5)
    parser.add_argument('--hit_vina_thresh', type=float, default=-8.18)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name, weights_only=False)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)

        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                if args.docking_mode == 'qvina':
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ['vina_score', 'vina_dock']:
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                continue

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results,
                'sample_idx': sample_idx
            })

    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    if len(success_atom_types) > 0 and sum(success_atom_types.values()) > 0:
        atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
        logger.info('Atom type JS: %.4f' % atom_type_js)
    else:
        atom_type_js = None
        logger.info('Atom type JS: None')

    if args.save and len(success_pair_dist) > 0:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    # =========================================================================
    # ENHANCED METRICS
    # =========================================================================
    logger.info('=' * 80)
    logger.info('ENHANCED METRICS')
    logger.info('=' * 80)

    if len(results) > 0:
        # Basic metrics
        qed = [r['chem_results']['qed'] for r in results]
        sa = [r['chem_results']['sa'] for r in results]

        logger.info('QED:       Mean: %.3f  Std: %.3f  Median: %.3f' % (
            np.mean(qed), np.std(qed), np.median(qed)))
        logger.info('SA:        Mean: %.3f  Std: %.3f  Median: %.3f' % (
            np.mean(sa), np.std(sa), np.median(sa)))

        # Vina metrics
        if args.docking_mode == 'qvina':
            vina = [r['vina'][0]['affinity'] for r in results]
            logger.info('Vina:      Mean: %.3f  Std: %.3f  Median: %.3f' % (
                np.mean(vina), np.std(vina), np.median(vina)))

            # High affinity rate - compare to reference
            compute_high_affinity_rate(vina, ref_scores=None, threshold=-8.18, logger=logger)

            # Hit rate - QED >= 0.4, SA >= 0.5, Vina <= -8.18
            compute_hit_rate(qed, sa, vina,
                           args.hit_qed_thresh, args.hit_sa_thresh, args.hit_vina_thresh,
                           logger)

        elif args.docking_mode in ['vina_dock', 'vina_score']:
            vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
            vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]

            logger.info('Vina Score: Mean: %.3f  Std: %.3f  Median: %.3f' % (
                np.mean(vina_score_only), np.std(vina_score_only), np.median(vina_score_only)))
            logger.info('Vina Min:   Mean: %.3f  Std: %.3f  Median: %.3f' % (
                np.mean(vina_min), np.std(vina_min), np.median(vina_min)))

            if args.docking_mode == 'vina_dock':
                vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
                logger.info('Vina Dock:  Mean: %.3f  Std: %.3f  Median: %.3f' % (
                    np.mean(vina_dock), np.std(vina_dock), np.median(vina_dock)))

                # High affinity rate (using dock scores) - compare to reference
                compute_high_affinity_rate(vina_dock, ref_scores=None, threshold=-8.18, logger=logger)

                # Hit rate (using dock scores) - QED >= 0.4, SA >= 0.5, Vina Dock <= -8.18
                compute_hit_rate(qed, sa, vina_dock,
                               args.hit_qed_thresh, args.hit_sa_thresh, args.hit_vina_thresh,
                               logger)
            else:
                # High affinity rate (using score_only)
                compute_high_affinity_rate(vina_score_only, ref_scores=None, threshold=-8.18, logger=logger)

                # Hit rate (using score_only)
                compute_hit_rate(qed, sa, vina_score_only,
                               args.hit_qed_thresh, args.hit_sa_thresh, args.hit_vina_thresh,
                               logger)

        # Diversity - per pocket average Tanimoto distance
        diversity = compute_diversity_per_pocket(results, logger)

    else:
        logger.info('No successful evaluations to report.')

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        save_data = {
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results
        }

        # Add enhanced metrics if available
        if len(results) > 0:
            save_data['enhanced_metrics'] = {
                'qed': {'mean': np.mean(qed), 'std': np.std(qed), 'median': np.median(qed)},
                'sa': {'mean': np.mean(sa), 'std': np.std(sa), 'median': np.median(sa)},
            }
            if args.docking_mode in ['vina_dock', 'vina_score']:
                save_data['enhanced_metrics']['vina_score'] = {
                    'mean': np.mean(vina_score_only),
                    'std': np.std(vina_score_only),
                    'median': np.median(vina_score_only)
                }
                save_data['enhanced_metrics']['vina_min'] = {
                    'mean': np.mean(vina_min),
                    'std': np.std(vina_min),
                    'median': np.median(vina_min)
                }
            if args.docking_mode == 'vina_dock':
                save_data['enhanced_metrics']['vina_dock'] = {
                    'mean': np.mean(vina_dock),
                    'std': np.std(vina_dock),
                    'median': np.median(vina_dock)
                }

        torch.save(save_data, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))

    logger.info('=' * 80)
