from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MULTI_SEEDS = [12345, 23456, 34567]
DEFAULT_MULTI_BASELINE_INTEGRATION_STEPS = 100
DEFAULT_MULTI_INTEGRATION_STEPS = 200
DEFAULT_MULTI_GUIDANCE_START_T = 0.5
DEFAULT_MULTI_GUIDANCE_END_T = 1.0
DEFAULT_GUIDED_SCALE = 30.0


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


def select_multi_tasks(refs: dict, requested_task_ids: list[str] | None):
    all_tasks = {task["task_id"]: task for task in refs["multi"]}
    if not requested_task_ids:
        return refs["multi"]

    missing = [task_id for task_id in requested_task_ids if task_id not in all_tasks]
    if missing:
        raise ValueError(f"Unknown multi-objective task(s): {', '.join(missing)}")

    return [all_tasks[task_id] for task_id in requested_task_ids]


def run_multi_suite(args, refs, suite_dir: Path):
    results = {}
    multi_tasks = select_multi_tasks(refs, args.tasks)

    for task in multi_tasks:
        task_id = task["task_id"]
        objectives = task["objectives"]
        task_dir = suite_dir / task_id
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

        guided_runs = []
        for seed in args.multi_seeds:
            guided_output = task_dir / f"guided_scale{args.guided_scale:g}_seed{seed}.json"
            run_guided(
                args.root,
                guided_output,
                args.ckpt_path,
                args.protocol_dir,
                args.predictor_dir,
                objectives,
                seed,
                args.guided_scale,
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
                "best_scale": args.guided_scale,
                "tuning_runs": {},
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

    multi_results = run_multi_suite(args, refs, suite_dir)

    suite_manifest = {
        "config": {
            "ckpt_path": args.ckpt_path,
            "protocol_dir": args.protocol_dir,
            "predictor_dir": args.predictor_dir,
            "n_molecules": args.n_molecules,
            "multi_baseline_integration_steps": args.multi_baseline_integration_steps,
            "multi_integration_steps": args.multi_integration_steps,
            "multi_guidance_start_t": args.multi_guidance_start_t,
            "multi_guidance_end_t": args.multi_guidance_end_t,
            "multi_seeds": args.multi_seeds,
            "guided_scale": args.guided_scale,
            "guidance_device": args.guidance_device,
            "tasks": args.tasks,
        },
        "multi": multi_results,
    }
    manifest_path = suite_dir / "multi_suite_manifest.json"
    manifest_path.write_text(json.dumps(suite_manifest, indent=2), encoding="utf-8")
    print(f"Saved suite manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--refs", type=str, default="experiments/mudm_refs.json")
    parser.add_argument("--output_dir", type=str, default="results/mudm_multi_scale30")
    parser.add_argument("--ckpt_path", type=str, default="semla-flow/models/qm9/last.ckpt")
    parser.add_argument("--protocol_dir", type=str, default="results/qm9_mudm_protocol")
    parser.add_argument("--predictor_dir", type=str, default="results/qm9_mudm_protocol/predictors")
    parser.add_argument("--tasks", nargs="*", type=str, default=None)
    parser.add_argument("--n_molecules", type=int, default=10000)
    parser.add_argument("--multi_baseline_integration_steps", type=int, default=DEFAULT_MULTI_BASELINE_INTEGRATION_STEPS)
    parser.add_argument("--multi_integration_steps", type=int, default=DEFAULT_MULTI_INTEGRATION_STEPS)
    parser.add_argument("--guidance_device", type=str, default="cpu")
    parser.add_argument("--guided_scale", type=float, default=DEFAULT_GUIDED_SCALE)
    parser.add_argument("--multi_guidance_start_t", type=float, default=DEFAULT_MULTI_GUIDANCE_START_T)
    parser.add_argument("--multi_guidance_end_t", type=float, default=DEFAULT_MULTI_GUIDANCE_END_T)
    parser.add_argument("--multi_seeds", nargs="*", type=int, default=DEFAULT_MULTI_SEEDS)
    main(parser.parse_args())
