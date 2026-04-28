from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SINGLE_SEEDS = [12345, 23456, 34567]
DEFAULT_MULTI_SEEDS = [12345, 23456, 34567]
DEFAULT_INITIAL_SCALE = 15.0
DEFAULT_EXTRA_SCALES = [5.0, 10.0, 12.0, 20.0, 30.0]
DEFAULT_SINGLE_BASELINE_INTEGRATION_STEPS = 1000
DEFAULT_SINGLE_INTEGRATION_STEPS = 500
DEFAULT_MULTI_BASELINE_INTEGRATION_STEPS = 1000
DEFAULT_MULTI_INTEGRATION_STEPS = 200
DEFAULT_SINGLE_GUIDANCE_START_T = 0.6
DEFAULT_MULTI_GUIDANCE_START_T = 0.5
DEFAULT_SINGLE_GUIDANCE_END_T = 1.0
DEFAULT_MULTI_GUIDANCE_END_T = 1.0


def load_json(path: Path):
    return json.loads(path.read_text())


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_command(cmd, cwd: Path):
    print("Running:", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_guided(
    root: Path,
    output_path: Path,
    ckpt_path: str,
    protocol_dir: str,
    predictor_dir: str,
    objectives: list[str],
    seed: int,
    scale: float,
    guidance_device: str,
    relation_mode: str = "cascade",
    compute_quality: bool = True,
    n_molecules: int = 10000,
    integration_steps: int = 100,
    guidance_start_t: float = 0.5,
    guidance_end_t: float = 1.0,
):
    ensure_parent(output_path)
    if output_path.exists():
        return load_json(output_path)

    cmd = [
        sys.executable,
        "semla-flow/semlaflow/qm9_guided.py",
        "--ckpt_path",
        ckpt_path,
        "--data_path",
        protocol_dir,
        "--protocol_dir",
        protocol_dir,
        "--predictor_dir",
        predictor_dir,
        "--dataset",
        "qm9",
        "--dataset_split",
        "test",
        "--n_molecules",
        str(n_molecules),
        "--integration_steps",
        str(integration_steps),
        "--objectives",
        *objectives,
        "--relation_mode",
        relation_mode,
        "--weights",
        *(["1"] * len(objectives)),
        "--guidance_scale",
        str(scale),
        "--guidance_start_t",
        str(guidance_start_t),
        "--guidance_end_t",
        str(guidance_end_t),
        "--guidance_device",
        guidance_device,
        "--predictor_train_split",
        "da",
        "--generator_train_split",
        "db",
        "--novelty_reference_file",
        "db_train.smol",
        "--seed",
        str(seed),
        "--save_json",
        str(output_path),
    ]
    if not compute_quality:
        cmd.append("--disable_quality_metrics")

    run_command(cmd, root)
    return load_json(output_path)


def single_score(summary: dict, objective: str, mudm_mean: float):
    return summary["metrics"][objective]["paper_mae"] / mudm_mean


def multi_score(summary: dict, objectives: list[str], mudm: list[float]):
    return sum(summary["metrics"][obj]["paper_mae"] / ref for obj, ref in zip(objectives, mudm))


def single_beats(summary: dict, objective: str, mudm_mean: float):
    return summary["metrics"][objective]["paper_mae"] <= mudm_mean


def multi_beats(summary: dict, objectives: list[str], mudm: list[float]):
    return all(summary["metrics"][obj]["paper_mae"] <= ref for obj, ref in zip(objectives, mudm))


def choose_scale(
    tried: dict[float, dict],
    objective_names: list[str],
    mudm_targets,
):
    best_scale = None
    best_score = None
    for scale, summary in tried.items():
        if len(objective_names) == 1:
            score = single_score(summary, objective_names[0], mudm_targets)
        else:
            score = multi_score(summary, objective_names, mudm_targets)

        if best_score is None or score < best_score:
            best_scale = scale
            best_score = score
    return best_scale


def run_single_suite(args, refs, suite_dir: Path):
    results = {}
    candidate_scales = (
        args.single_scales
        if args.single_scales
        else [args.initial_scale] + [scale for scale in args.extra_scales if scale != args.initial_scale]
    )

    for objective, ref in refs["single"].items():
        task_dir = suite_dir / "single" / objective
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Single-objective: {objective} ===")

        baseline_runs = []
        for seed in args.single_seeds:
            output = task_dir / f"baseline_seed{seed}.json"
            run_guided(
                args.root,
                output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                [objective],
                seed,
                0.0,
                args.guidance_device,
                relation_mode="independent",
                compute_quality=True,
                n_molecules=args.n_molecules,
                integration_steps=args.single_baseline_integration_steps,
                guidance_start_t=args.single_guidance_start_t,
                guidance_end_t=args.single_guidance_end_t,
            )
            baseline_runs.append(str(output))

        tuning_seed = args.single_seeds[0]
        tried = {}
        for scale in candidate_scales:
            output = task_dir / "tuning" / f"scale{scale:g}_seed{tuning_seed}.json"
            summary = run_guided(
                args.root,
                output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                [objective],
                tuning_seed,
                scale,
                args.guidance_device,
                relation_mode="independent",
                compute_quality=False,
                n_molecules=args.n_molecules,
                integration_steps=args.single_integration_steps,
                guidance_start_t=args.single_guidance_start_t,
                guidance_end_t=args.single_guidance_end_t,
            )
            tried[scale] = summary
            if scale == args.initial_scale and single_beats(summary, objective, ref["mudm_mean"]):
                break

        best_scale = choose_scale(tried, [objective], ref["mudm_mean"])

        guided_runs = []
        for seed in args.single_seeds:
            output = task_dir / f"guided_scale{best_scale:g}_seed{seed}.json"
            run_guided(
                args.root,
                output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                [objective],
                seed,
                best_scale,
                args.guidance_device,
                relation_mode="independent",
                compute_quality=True,
                n_molecules=args.n_molecules,
                integration_steps=args.single_integration_steps,
                guidance_start_t=args.single_guidance_start_t,
                guidance_end_t=args.single_guidance_end_t,
            )
            guided_runs.append(str(output))

        results[objective] = {
            "baseline": {"final_runs": baseline_runs},
            "guided": {
                "best_scale": best_scale,
                "tuning_runs": {str(scale): str(task_dir / "tuning" / f"scale{scale:g}_seed{tuning_seed}.json") for scale in tried},
                "final_runs": guided_runs,
            },
            "objective": objective,
            "display_name": ref["display_name"],
            "mudm_mean": ref["mudm_mean"],
            "mudm_std": ref["mudm_std"],
        }

    return results


def run_multi_suite(args, refs, suite_dir: Path):
    results = {}
    candidate_scales = (
        args.multi_scales
        if args.multi_scales
        else [args.initial_scale] + [scale for scale in args.extra_scales if scale != args.initial_scale]
    )

    for task in refs["multi"]:
        task_id = task["task_id"]
        objectives = task["objectives"]
        task_dir = suite_dir / "multi" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Multi-objective: {task_id} ===")

        baseline_runs = []
        for seed in args.multi_seeds:
            baseline_output = task_dir / f"baseline_seed{seed}.json"
            run_guided(
                args.root,
                baseline_output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                objectives,
                seed,
                0.0,
                args.guidance_device,
                relation_mode="cascade",
                compute_quality=True,
                n_molecules=args.n_molecules,
                integration_steps=args.multi_baseline_integration_steps,
                guidance_start_t=args.multi_guidance_start_t,
                guidance_end_t=args.multi_guidance_end_t,
            )
            baseline_runs.append(str(baseline_output))

        tuning_seed = args.multi_seeds[0]
        tried = {}
        for scale in candidate_scales:
            output = task_dir / "tuning" / f"scale{scale:g}_seed{tuning_seed}.json"
            summary = run_guided(
                args.root,
                output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                objectives,
                tuning_seed,
                scale,
                args.guidance_device,
                relation_mode="cascade",
                compute_quality=False,
                n_molecules=args.n_molecules,
                integration_steps=args.multi_integration_steps,
                guidance_start_t=args.multi_guidance_start_t,
                guidance_end_t=args.multi_guidance_end_t,
            )
            tried[scale] = summary
            if scale == args.initial_scale and multi_beats(summary, objectives, task["mudm"]):
                break

        best_scale = choose_scale(tried, objectives, task["mudm"])
        guided_runs = []
        for seed in args.multi_seeds:
            guided_output = task_dir / f"guided_scale{best_scale:g}_seed{seed}.json"
            run_guided(
                args.root,
                guided_output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                objectives,
                seed,
                best_scale,
                args.guidance_device,
                relation_mode="cascade",
                compute_quality=True,
                n_molecules=args.n_molecules,
                integration_steps=args.multi_integration_steps,
                guidance_start_t=args.multi_guidance_start_t,
                guidance_end_t=args.multi_guidance_end_t,
            )
            guided_runs.append(str(guided_output))

        results[task_id] = {
            "baseline": {"final_runs": baseline_runs},
            "guided": {
                "best_scale": best_scale,
                "tuning_runs": {str(scale): str(task_dir / "tuning" / f"scale{scale:g}_seed{tuning_seed}.json") for scale in tried},
                "final_runs": guided_runs,
            },
            "objectives": objectives,
            "display_name": task["display_name"],
            "mudm": task["mudm"],
        }

    return results


def main(args):
    root = Path(args.root).resolve()
    refs_path = Path(args.refs).resolve()
    suite_dir = Path(args.output_dir).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    args.root = root
    refs = load_json(refs_path)

    single_results = run_single_suite(args, refs, suite_dir)
    multi_results = run_multi_suite(args, refs, suite_dir)

    suite_manifest = {
        "config": {
            "ckpt_path": args.ckpt_path,
            "protocol_dir": args.protocol_dir,
            "predictor_dir": args.predictor_dir,
            "n_molecules": args.n_molecules,
            "single_baseline_integration_steps": args.single_baseline_integration_steps,
            "single_integration_steps": args.single_integration_steps,
            "multi_baseline_integration_steps": args.multi_baseline_integration_steps,
            "multi_integration_steps": args.multi_integration_steps,
            "single_guidance_start_t": args.single_guidance_start_t,
            "multi_guidance_start_t": args.multi_guidance_start_t,
            "single_guidance_end_t": args.single_guidance_end_t,
            "multi_guidance_end_t": args.multi_guidance_end_t,
            "single_seeds": args.single_seeds,
            "multi_seeds": args.multi_seeds,
            "initial_scale": args.initial_scale,
            "extra_scales": args.extra_scales,
            "single_scales": args.single_scales,
            "multi_scales": args.multi_scales,
            "guidance_device": args.guidance_device,
        },
        "single": single_results,
        "multi": multi_results,
    }
    manifest_path = suite_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(suite_manifest, indent=2), encoding="utf-8")
    print(f"Saved suite manifest to {manifest_path}")

    report_path = suite_dir / "mudm_report.md"
    cmd = [
        sys.executable,
        "experiments/make_mudm_report.py",
        "--refs",
        str(refs_path),
        "--suite_manifest",
        str(manifest_path),
        "--output",
        str(report_path),
    ]
    run_command(cmd, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--refs", type=str, default="experiments/mudm_refs.json")
    parser.add_argument("--output_dir", type=str, default="results/mudm_suite")
    parser.add_argument("--ckpt_path", type=str, default="semla-flow/models/qm9/last.ckpt")
    parser.add_argument("--protocol_dir", type=str, default="results/qm9_mudm_protocol")
    parser.add_argument("--predictor_dir", type=str, default="results/qm9_mudm_protocol/predictors")
    parser.add_argument("--n_molecules", type=int, default=10000)
    parser.add_argument("--single_baseline_integration_steps", type=int, default=DEFAULT_SINGLE_BASELINE_INTEGRATION_STEPS)
    parser.add_argument("--single_integration_steps", type=int, default=DEFAULT_SINGLE_INTEGRATION_STEPS)
    parser.add_argument("--multi_baseline_integration_steps", type=int, default=DEFAULT_MULTI_BASELINE_INTEGRATION_STEPS)
    parser.add_argument("--multi_integration_steps", type=int, default=DEFAULT_MULTI_INTEGRATION_STEPS)
    parser.add_argument("--guidance_device", type=str, default="cpu")
    parser.add_argument("--initial_scale", type=float, default=DEFAULT_INITIAL_SCALE)
    parser.add_argument("--extra_scales", nargs="*", type=float, default=DEFAULT_EXTRA_SCALES)
    parser.add_argument("--single_scales", nargs="*", type=float, default=None)
    parser.add_argument("--multi_scales", nargs="*", type=float, default=None)
    parser.add_argument("--single_guidance_start_t", type=float, default=DEFAULT_SINGLE_GUIDANCE_START_T)
    parser.add_argument("--multi_guidance_start_t", type=float, default=DEFAULT_MULTI_GUIDANCE_START_T)
    parser.add_argument("--single_guidance_end_t", type=float, default=DEFAULT_SINGLE_GUIDANCE_END_T)
    parser.add_argument("--multi_guidance_end_t", type=float, default=DEFAULT_MULTI_GUIDANCE_END_T)
    parser.add_argument("--single_seeds", nargs="*", type=int, default=DEFAULT_SINGLE_SEEDS)
    parser.add_argument("--multi_seeds", nargs="*", type=int, default=DEFAULT_MULTI_SEEDS)
    main(parser.parse_args())
