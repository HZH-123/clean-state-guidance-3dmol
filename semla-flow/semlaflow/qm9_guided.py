import argparse
import json
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import lightning as L
import torch
from rdkit import Chem

THIS_DIR = Path(__file__).resolve().parent
SEMLA_ROOT = THIS_DIR.parent
PROJECT_ROOT = SEMLA_ROOT.parent
MODELS_ROOT = PROJECT_ROOT / "models"

sys.path.insert(0, str(MODELS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SEMLA_ROOT))

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.data.qm9_protocol import DELTAE_DEFINITION, TARGET_SOURCE_QM9_TRUE_TEST, load_protocol_labels
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

from property_guidance import (
    GuidanceObjective,
    PropertyGuidanceManager,
    build_vocab_atomic_number_map,
    dense_molecular_batch_to_pyg,
)
from qm9_predictor import QM9PropertyPredictor, canonicalize_qm9_property_name


DEFAULT_DATASET_SPLIT = "test"
DEFAULT_N_MOLECULES = 10000
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"
DEFAULT_GUIDANCE_SCALE = 0.2
DEFAULT_GUIDANCE_START_T = 0.5
DEFAULT_GUIDANCE_END_T = 1.0
DEFAULT_RELATION_MODE = "independent"

PAPER_UNITS = {
    "mu": ("D", 1.0),
    "alpha": ("Bohr^3", 1.0),
    "ehomo": ("meV", 1000.0),
    "elumo": ("meV", 1000.0),
    "deltae": ("meV", 1000.0),
    "cv": ("cal/mol K", 1.0),
}


def _build_generator(hparams, vocab, args):
    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
        )
        return SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"],
        )

    if hparams["architecture"] == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        return EqgatGenerator(
            hparams["d_model"],
            hparams["n_layers"],
            hparams["n_equi_feats"],
            vocab.size,
            hparams["n_atom_feats"],
            hparams["d_edge"],
            hparams["n_edge_types"],
        )

    if hparams["architecture"] == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        n_layers = args.n_layers if hparams.get("n_layers") is None else hparams["n_layers"]
        if n_layers is None:
            raise ValueError("No hparam for n_layers was saved, use --n_layers to provide one.")

        return VanillaEgnnGenerator(
            hparams["d_model"],
            n_layers,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
        )

    raise ValueError(f"Unknown architecture hyperparameter '{hparams['architecture']}'.")


def load_model(args, vocab, guidance_manager):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    if args.dataset == "qm9":
        coord_scale = util.QM9_COORDS_STD_DEV
    elif args.dataset == "geom-drugs":
        coord_scale = util.GEOM_COORDS_STD_DEV
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    gen = _build_generator(hparams, vocab, args)
    type_mask_index = (
        vocab.indices_from_tokens(["<MASK>"])[0] if hparams["train-type-interpolation"] == "mask" else None
    )

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=None,
        cat_noise_level=args.cat_sampling_noise_level,
        guidance_manager=guidance_manager,
        guidance_scale=args.guidance_scale,
        guidance_start_t=args.guidance_start_t,
        guidance_end_t=args.guidance_end_t,
        guidance_source=args.guidance_source,
        guidance_normalize=not args.disable_grad_normalization,
        coord_scale=coord_scale,
        vocab=vocab,
    )

    model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=None,
        **hparams,
    )
    return model


def build_dm(args, hparams, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS
    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    n_bond_types = 5
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    split_path = Path(args.data_path) / f"{args.dataset_split}.smol"
    dataset = GeometricDataset.load(split_path, transform=transform)
    if args.n_molecules is not None:
        replacement = args.sample_with_replacement or args.n_molecules > len(dataset)
        dataset = dataset.sample(args.n_molecules, replacement=replacement)

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["val-type-interpolation"] == "mask" else None
    bond_mask_index = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_ot=hparams["val-prior-noise-scale-ot"],
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
        equivariant_ot=False,
        batch_ot=False,
    )

    return GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    return build_dm(args, checkpoint["hyper_parameters"], vocab)


def build_guidance_manager(args):
    if len(args.weights) not in [1, len(args.objectives)]:
        raise ValueError("--weights must contain either one value or one per objective.")

    if len(args.losses) not in [1, len(args.objectives)]:
        raise ValueError("--losses must contain either one value or one per objective.")

    if args.target_mode == "constant" and len(args.targets) != len(args.objectives):
        raise ValueError("--targets must contain one value per objective when --target_mode=constant.")

    weights = args.weights if len(args.weights) == len(args.objectives) else args.weights * len(args.objectives)
    losses = args.losses if len(args.losses) == len(args.objectives) else args.losses * len(args.objectives)

    objectives = []
    for name, weight, loss_name in zip(args.objectives, weights, losses):
        canonical_name = canonicalize_qm9_property_name(name)
        predictor = QM9PropertyPredictor(
            canonical_name,
            qm9_root=args.qm9_root,
            device=args.guidance_device,
            protocol_dir=args.protocol_dir,
            model_dir=args.predictor_dir,
            stats_path=args.predictor_stats_path,
            train_split=args.predictor_train_split,
        )
        objectives.append(GuidanceObjective(canonical_name, predictor, weight=weight, loss_type=loss_name))

    return PropertyGuidanceManager(
        objectives,
        relation_mode=args.relation_mode,
        cascade_strength=args.cascade_strength,
        device=args.guidance_device,
    )


def _shuffle_targets_in_place(targets, generator):
    shuffled = {}
    objective_names = list(targets.keys())
    shuffled[objective_names[0]] = targets[objective_names[0]]
    for name in objective_names[1:]:
        permutation = torch.randperm(targets[name].numel(), generator=generator)
        shuffled[name] = targets[name][permutation]
    return shuffled


def build_target_context(args, manager, data_batch, labels, rng):
    if args.target_mode == "constant":
        return {
            "targets": {
                canonicalize_qm9_property_name(name): torch.full(
                    (data_batch["coords"].size(0),),
                    float(target),
                    dtype=torch.float32,
                    device=manager.device,
                )
                for name, target in zip(args.objectives, args.targets)
            }
        }

    if labels is None:
        raise ValueError("Reference target mode requires protocol labels. Pass --protocol_dir with a labels.pt file.")

    if "qm9_dataset_index" not in data_batch:
        raise KeyError("Reference target mode requires qm9_dataset_index metadata in the batch.")

    dataset_indices = data_batch["qm9_dataset_index"].long()
    targets = {
        name: labels["properties"][name][dataset_indices].to(manager.device)
        for name in manager.objective_names
    }

    if args.shuffle_targets_independently and len(targets) > 1:
        targets = _shuffle_targets_in_place(targets, rng)

    return {"targets": targets}


def summarise_errors(records, manager):
    summary = {}
    for objective in manager.objective_names:
        preds = torch.cat([record["predictions"][objective] for record in records]).float()
        targets = torch.cat([record["targets"][objective] for record in records]).float()
        abs_err = (preds - targets).abs()
        paper_unit, paper_scale = PAPER_UNITS.get(objective, ("raw", 1.0))
        summary[objective] = {
            "mae": float(abs_err.mean()),
            "target_mean": float(targets.mean()),
            "prediction_mean": float(preds.mean()),
            "prediction_std": float(preds.std()),
            "paper_unit": paper_unit,
            "paper_scale": paper_scale,
            "paper_mae": float(abs_err.mean() * paper_scale),
            "paper_target_mean": float(targets.mean() * paper_scale),
            "paper_prediction_mean": float(preds.mean() * paper_scale),
        }
    return summary


def _expand_batch_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    view_shape = [mask.size(0)] + ([1] * (target.dim() - 1))
    return mask.view(*view_shape)


def _clone_generated_batch(generated):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in generated.items()}


def _update_selected_batch(best_batch, candidate_batch, better_mask: torch.Tensor):
    for key, value in candidate_batch.items():
        if not torch.is_tensor(value):
            continue
        expanded_mask = _expand_batch_mask(better_mask.to(value.device), value)
        best_batch[key] = torch.where(expanded_mask, value, best_batch[key])
    return best_batch


def _update_selected_predictions(best_predictions, candidate_predictions, better_mask: torch.Tensor):
    updated = {}
    for key, value in candidate_predictions.items():
        updated[key] = torch.where(better_mask.to(value.device), value, best_predictions[key])
    return updated


def _standardized_rerank_score(predictions, targets, manager):
    score = None
    for objective in manager.objectives:
        name = objective.name
        prediction = predictions[name].float()
        target = targets[name].to(prediction.device, prediction.dtype)
        std = float(objective.predictor.std.detach().cpu().item())
        denom = max(std, 1e-6)
        term = (prediction - target).abs() / denom
        score = term if score is None else score + term
    return score


def _predict_properties_for_generated(generated, manager, index_map):
    gen_pyg_batch, _ = dense_molecular_batch_to_pyg(
        generated["coords"],
        generated["atomics"],
        generated["mask"],
        index_map,
        coord_scale=1.0,
        coords_are_normalized=False,
        device=manager.device,
    )
    return manager.predict_properties(gen_pyg_batch, differentiable=False)


def _rerank_generate(args, model, manager, index_map, prior, guidance_context):
    best_generated = None
    best_predictions = None
    best_score = None

    for _ in range(args.rerank_k):
        candidate = model._generate(
            prior,
            args.integration_steps,
            args.ode_sampling_strategy,
            guidance_context=guidance_context,
        )
        candidate_predictions = _predict_properties_for_generated(candidate, manager, index_map)
        candidate_score = _standardized_rerank_score(candidate_predictions, guidance_context["targets"], manager)

        if best_generated is None:
            best_generated = _clone_generated_batch(candidate)
            best_predictions = {name: value.detach().clone() for name, value in candidate_predictions.items()}
            best_score = candidate_score.detach().clone()
            continue

        better_mask = candidate_score < best_score
        if better_mask.any():
            best_generated = _update_selected_batch(best_generated, candidate, better_mask)
            best_predictions = _update_selected_predictions(best_predictions, candidate_predictions, better_mask)
            best_score = torch.where(better_mask, candidate_score, best_score)

    return best_generated, best_predictions


def _init_trace_accumulator(objective_names):
    return {
        "objective_names": list(objective_names),
        "steps": defaultdict(
            lambda: {
                "t": None,
                "count": 0,
                "curr": defaultdict(lambda: {"abs_error_sum": 0.0}),
                "predicted": defaultdict(lambda: {"abs_error_sum": 0.0}),
            }
        ),
    }


def _make_trace_callback(args, model, manager, index_map, targets, trace_accumulator):
    coord_scale = getattr(model.integrator, "coord_scale", 1.0)

    def callback(step_idx, times, curr, predicted):
        if trace_accumulator is None:
            return
        if args.trace_every_n_steps > 1 and step_idx % args.trace_every_n_steps != 0:
            return

        step_entry = trace_accumulator["steps"][int(step_idx)]
        step_entry["t"] = float(times[0].item())

        state_predictions = {}
        for state_name, state_batch in (("curr", curr), ("predicted", predicted)):
            pyg_batch, _ = dense_molecular_batch_to_pyg(
                state_batch["coords"],
                state_batch["atomics"],
                state_batch["mask"],
                index_map,
                coord_scale=coord_scale,
                coords_are_normalized=True,
                device=manager.device,
            )
            state_predictions[state_name] = manager.predict_properties(pyg_batch, differentiable=False)

        batch_size = next(iter(targets.values())).numel()
        step_entry["count"] += int(batch_size)
        for state_name, predictions in state_predictions.items():
            for objective_name, prediction in predictions.items():
                target = targets[objective_name].to(prediction.device, prediction.dtype)
                step_entry[state_name][objective_name]["abs_error_sum"] += float((prediction - target).abs().sum().item())

    return callback


def _summarise_trace(trace_accumulator):
    if trace_accumulator is None:
        return None

    trace_rows = []
    for step_idx in sorted(trace_accumulator["steps"].keys()):
        step_entry = trace_accumulator["steps"][step_idx]
        count = max(int(step_entry["count"]), 1)
        row = {
            "step": int(step_idx),
            "t": float(step_entry["t"]),
            "count": int(step_entry["count"]),
            "curr": {},
            "predicted": {},
        }
        for state_name in ("curr", "predicted"):
            for objective_name in trace_accumulator["objective_names"]:
                abs_error_sum = step_entry[state_name][objective_name]["abs_error_sum"]
                row[state_name][objective_name] = {
                    "mae": abs_error_sum / count,
                    "paper_mae": (abs_error_sum / count) * PAPER_UNITS.get(objective_name, ("raw", 1.0))[1],
                    "paper_unit": PAPER_UNITS.get(objective_name, ("raw", 1.0))[0],
                }
        trace_rows.append(row)

    return {
        "objectives": trace_accumulator["objective_names"],
        "trace_every_n_steps": None,
        "steps": trace_rows,
    }


def save_sdf(filepath, molecules, records, objective_names):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(filepath))
    flat_records = []
    for batch_record in records:
        batch_size = next(iter(batch_record["targets"].values())).numel()
        for idx in range(batch_size):
            item = {"targets": {}, "predictions": {}}
            for name in objective_names:
                item["targets"][name] = float(batch_record["targets"][name][idx].item())
                item["predictions"][name] = float(batch_record["predictions"][name][idx].item())
            flat_records.append(item)

    for molecule, record in zip(molecules, flat_records):
        if molecule is None:
            continue
        for name, value in record["targets"].items():
            molecule.SetProp(f"target_{name}", f"{value:.6f}")
        for name, value in record["predictions"].items():
            molecule.SetProp(f"pred_{name}", f"{value:.6f}")
        writer.write(molecule)

    writer.close()


def run_experiment(args, model, dm, manager, index_map, labels):
    test_dl = dm.test_dataloader()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if args.ablation_mode == "rerank" and args.guidance_scale > 0.0:
        raise ValueError("Rerank mode expects --guidance_scale 0.0 so candidates are generated without guidance.")

    rng = torch.Generator().manual_seed(args.seed)
    objective_names = manager.objective_names

    records = []
    rdkit_molecules = []
    stabilities = []
    trace_accumulator = _init_trace_accumulator(objective_names) if args.save_trace_json is not None else None

    for batch in test_dl:
        prior = {k: v.to(model.device) for k, v in batch[0].items() if torch.is_tensor(v)}
        data = batch[1]
        guidance_context = build_target_context(args, manager, data, labels, rng)
        trace_callback = _make_trace_callback(
            args,
            model,
            manager,
            index_map,
            guidance_context["targets"],
            trace_accumulator,
        ) if trace_accumulator is not None and args.ablation_mode != "rerank" else None

        if args.ablation_mode == "rerank":
            generated, predictions = _rerank_generate(args, model, manager, index_map, prior, guidance_context)
        else:
            generated = model._generate(
                prior,
                args.integration_steps,
                args.ode_sampling_strategy,
                guidance_context=guidance_context,
                trace_callback=trace_callback,
            )
            predictions = _predict_properties_for_generated(generated, manager, index_map)

        predictions = {name: value.detach().cpu() for name, value in predictions.items()}
        targets = {name: value.detach().cpu() for name, value in guidance_context["targets"].items()}

        records.append({"targets": targets, "predictions": predictions})

        if args.compute_quality_metrics or args.save_sdf is not None:
            rdkit_molecules.extend(model._generate_mols(generated))
            if args.compute_quality_metrics:
                stabilities.extend(model._generate_stabilities(generated))

    quality_metrics = None
    if args.compute_quality_metrics:
        metrics, stab_metrics = util.init_metrics(args.data_path, model, novelty_reference_file=args.novelty_reference_file)
        quality_metrics = util.calc_metrics_(rdkit_molecules, metrics, stab_metrics=stab_metrics, mol_stabs=stabilities)
        quality_metrics = {key: float(value) for key, value in quality_metrics.items()}

    if args.save_sdf is not None:
        save_sdf(Path(args.save_sdf), rdkit_molecules, records, objective_names)

    summary = {
        "objectives": objective_names,
        "ablation_mode": args.ablation_mode,
        "protocol_dir": args.protocol_dir,
        "relation_mode": args.relation_mode,
        "target_mode": args.target_mode,
        "target_source": TARGET_SOURCE_QM9_TRUE_TEST if args.target_mode == "reference" else "constant",
        "deltae_definition": DELTAE_DEFINITION,
        "predictor_train_split": args.predictor_train_split,
        "generator_train_split": args.generator_train_split,
        "novelty_reference_file": args.novelty_reference_file,
        "guidance_scale": args.guidance_scale,
        "guidance_start_t": args.guidance_start_t,
        "guidance_end_t": args.guidance_end_t,
        "guidance_source": args.guidance_source,
        "rerank_k": args.rerank_k if args.ablation_mode == "rerank" else None,
        "n_molecules": int(sum(next(iter(record["targets"].values())).numel() for record in records)),
        "metrics": summarise_errors(records, manager),
    }
    if quality_metrics is not None:
        summary["quality_metrics"] = quality_metrics

    trace_summary = _summarise_trace(trace_accumulator)
    if trace_summary is not None:
        trace_summary["trace_every_n_steps"] = args.trace_every_n_steps

    return summary, trace_summary


def print_summary(summary):
    print()
    print("Conditional MAE")
    print("-" * 48)
    for name, values in summary["metrics"].items():
        paper_unit = values.get("paper_unit", "raw")
        print(
            f"{name:<12} MAE={values['mae']:.6f} | "
            f"target_mean={values['target_mean']:.6f} | pred_mean={values['prediction_mean']:.6f}"
        )
        if values.get("paper_scale", 1.0) != 1.0:
            print(
                f"{'':<12} paper_MAE={values['paper_mae']:.3f} {paper_unit} | "
                f"paper_target_mean={values['paper_target_mean']:.3f} {paper_unit} | "
                f"paper_pred_mean={values['paper_prediction_mean']:.3f} {paper_unit}"
            )

    if "quality_metrics" in summary:
        print()
        print("Generation Quality")
        print("-" * 48)
        for name, value in summary["quality_metrics"].items():
            print(f"{name:<24}{value:.6f}")
    print()


def main(args):
    L.seed_everything(args.seed)
    util.disable_lib_stdout()
    util.configure_fs()
    if args.protocol_dir is None:
        args.protocol_dir = args.data_path

    vocab = util.build_vocab()
    index_map = build_vocab_atomic_number_map(vocab)
    manager = build_guidance_manager(args)
    model = load_model(args, vocab, manager)
    dm = dm_from_ckpt(args, vocab)
    labels = load_protocol_labels(args.protocol_dir) if args.target_mode == "reference" else None

    summary, trace_summary = run_experiment(args, model, dm, manager, index_map, labels)
    print_summary(summary)

    if args.save_json is not None:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {output_path}")

    if args.save_trace_json is not None and trace_summary is not None:
        trace_path = Path(args.save_trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace_summary, indent=2))
        print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--protocol_dir", type=str, default=None)
    parser.add_argument("--predictor_dir", type=str, default=None)
    parser.add_argument("--predictor_stats_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--sample_with_replacement", action="store_true")

    parser.add_argument("--objectives", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--losses", nargs="+", type=str, default=["mse"])
    parser.add_argument("--relation_mode", type=str, default=DEFAULT_RELATION_MODE)
    parser.add_argument("--cascade_strength", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--guidance_start_t", type=float, default=DEFAULT_GUIDANCE_START_T)
    parser.add_argument("--guidance_end_t", type=float, default=DEFAULT_GUIDANCE_END_T)
    parser.add_argument("--guidance_source", choices=["predicted", "curr"], default="predicted")
    parser.add_argument("--disable_grad_normalization", action="store_true")
    parser.add_argument("--guidance_device", type=str, default="cpu")
    parser.add_argument("--ablation_mode", choices=["standard", "rerank"], default="standard")
    parser.add_argument("--rerank_k", type=int, default=5)
    parser.add_argument("--trace_every_n_steps", type=int, default=1)

    parser.add_argument("--target_mode", choices=["reference", "constant"], default="reference")
    parser.add_argument("--targets", nargs="*", type=float, default=[])
    parser.add_argument("--shuffle_targets_independently", action="store_true")
    parser.add_argument("--qm9_root", type=str, default=None)
    parser.add_argument("--predictor_train_split", type=str, default="da")
    parser.add_argument("--generator_train_split", type=str, default="db")
    parser.add_argument("--novelty_reference_file", type=str, default="db_train.smol")

    parser.add_argument("--disable_quality_metrics", action="store_false", dest="compute_quality_metrics")
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_trace_json", type=str, default=None)
    parser.add_argument("--save_sdf", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.set_defaults(compute_quality_metrics=True)

    main(parser.parse_args())
