from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TASK_CONFIGS = {
    "alpha_mu": {"objectives": ["alpha", "mu"], "scale": 15.0},
    "cv_mu": {"objectives": ["cv", "mu"], "scale": 15.0},
    "deltae_mu": {"objectives": ["deltae", "mu"], "scale": 15.0},
    "ehomo_elumo": {"objectives": ["ehomo", "elumo"], "scale": 15.0},
    "elumo_mu": {"objectives": ["elumo", "mu"], "scale": 15.0},
    "elumo_deltae": {"objectives": ["elumo", "deltae"], "scale": 15.0},
    "ehomo_deltae": {"objectives": ["ehomo", "deltae"], "scale": 15.0},
}

DEFAULT_TASKS = [
    "alpha_mu",
    "cv_mu",
    "deltae_mu",
    "elumo_mu",
    "ehomo_deltae",
]


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
    *,
    ckpt_path: str,
    protocol_dir: str,
    predictor_dir: str,
    objectives: list[str],
    n_molecules: int,
    integration_steps: int,
    scale: float,
    guidance_source: str,
    ablation_mode: str,
    guidance_device: str,
    guidance_start_t: float,
    guidance_end_t: float,
    relation_mode: str,
    rerank_k: int,
    seed: int,
    compute_quality: bool = True,
    trace_path: Path | None = None,
):
    ensure_parent(output_path)
    if output_path.exists() and (trace_path is None or trace_path.exists()):
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
        "--guidance_source",
        guidance_source,
        "--guidance_device",
        guidance_device,
        "--ablation_mode",
        ablation_mode,
        "--rerank_k",
        str(rerank_k),
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
    if trace_path is not None:
        ensure_parent(trace_path)
        cmd.extend(["--save_trace_json", str(trace_path)])
    if not compute_quality:
        cmd.append("--disable_quality_metrics")

    run_command(cmd, root)
    return load_json(output_path)


def task_configs_from_args(args):
    task_configs = {}
    for task_id in args.tasks:
        if task_id not in DEFAULT_TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_id}'. Supported tasks: {', '.join(DEFAULT_TASK_CONFIGS.keys())}"
            )
        task_configs[task_id] = dict(DEFAULT_TASK_CONFIGS[task_id])
    return task_configs


def run_task(args, root: Path, suite_dir: Path, task_id: str, task_config: dict):
    objectives = task_config["objectives"]
    scale = float(task_config["scale"])
    task_dir = suite_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Multi-ablation: {task_id} ===")

    results = {}

    base_path = task_dir / "base.json"
    run_guided(
        root,
        base_path,
        ckpt_path=args.ckpt_path,
        protocol_dir=args.protocol_dir,
        predictor_dir=args.predictor_dir,
        objectives=objectives,
        n_molecules=args.n_molecules,
        integration_steps=args.base_integration_steps,
        scale=0.0,
        guidance_source="predicted",
        ablation_mode="standard",
        guidance_device=args.guidance_device,
        guidance_start_t=args.guidance_start_t,
        guidance_end_t=args.guidance_end_t,
        relation_mode=args.relation_mode,
        rerank_k=args.rerank_k,
        seed=args.seed,
        compute_quality=True,
    )
    results["base"] = str(base_path)

    rerank_path = task_dir / "rerank.json"
    run_guided(
        root,
        rerank_path,
        ckpt_path=args.ckpt_path,
        protocol_dir=args.protocol_dir,
        predictor_dir=args.predictor_dir,
        objectives=objectives,
        n_molecules=args.n_molecules,
        integration_steps=args.base_integration_steps,
        scale=0.0,
        guidance_source="predicted",
        ablation_mode="rerank",
        guidance_device=args.guidance_device,
        guidance_start_t=args.guidance_start_t,
        guidance_end_t=args.guidance_end_t,
        relation_mode=args.relation_mode,
        rerank_k=args.rerank_k,
        seed=args.seed,
        compute_quality=True,
    )
    results["rerank"] = str(rerank_path)

    noisy_path = task_dir / "noisy_guidance.json"
    run_guided(
        root,
        noisy_path,
        ckpt_path=args.ckpt_path,
        protocol_dir=args.protocol_dir,
        predictor_dir=args.predictor_dir,
        objectives=objectives,
        n_molecules=args.n_molecules,
        integration_steps=args.guided_integration_steps,
        scale=scale,
        guidance_source="curr",
        ablation_mode="standard",
        guidance_device=args.guidance_device,
        guidance_start_t=args.guidance_start_t,
        guidance_end_t=args.guidance_end_t,
        relation_mode=args.relation_mode,
        rerank_k=args.rerank_k,
        seed=args.seed,
        compute_quality=True,
    )
    results["noisy_guidance"] = str(noisy_path)

    clean_path = task_dir / "clean_guidance.json"
    run_guided(
        root,
        clean_path,
        ckpt_path=args.ckpt_path,
        protocol_dir=args.protocol_dir,
        predictor_dir=args.predictor_dir,
        objectives=objectives,
        n_molecules=args.n_molecules,
        integration_steps=args.guided_integration_steps,
        scale=scale,
        guidance_source="predicted",
        ablation_mode="standard",
        guidance_device=args.guidance_device,
        guidance_start_t=args.guidance_start_t,
        guidance_end_t=args.guidance_end_t,
        relation_mode=args.relation_mode,
        rerank_k=args.rerank_k,
        seed=args.seed,
        compute_quality=True,
    )
    results["clean_guidance"] = str(clean_path)

    if args.trace_n_molecules > 0:
        noisy_trace_summary = task_dir / "noisy_guidance_trace_summary.json"
        noisy_trace_path = task_dir / "noisy_guidance_trace.json"
        run_guided(
            root,
            noisy_trace_summary,
            ckpt_path=args.ckpt_path,
            protocol_dir=args.protocol_dir,
            predictor_dir=args.predictor_dir,
            objectives=objectives,
            n_molecules=args.trace_n_molecules,
            integration_steps=args.guided_integration_steps,
            scale=scale,
            guidance_source="curr",
            ablation_mode="standard",
            guidance_device=args.guidance_device,
            guidance_start_t=args.guidance_start_t,
            guidance_end_t=args.guidance_end_t,
            relation_mode=args.relation_mode,
            rerank_k=args.rerank_k,
            seed=args.seed,
            compute_quality=False,
            trace_path=noisy_trace_path,
        )
        results["noisy_guidance_trace_summary"] = str(noisy_trace_summary)
        results["noisy_guidance_trace"] = str(noisy_trace_path)

        clean_trace_summary = task_dir / "clean_guidance_trace_summary.json"
        clean_trace_path = task_dir / "clean_guidance_trace.json"
        run_guided(
            root,
            clean_trace_summary,
            ckpt_path=args.ckpt_path,
            protocol_dir=args.protocol_dir,
            predictor_dir=args.predictor_dir,
            objectives=objectives,
            n_molecules=args.trace_n_molecules,
            integration_steps=args.guided_integration_steps,
            scale=scale,
            guidance_source="predicted",
            ablation_mode="standard",
            guidance_device=args.guidance_device,
            guidance_start_t=args.guidance_start_t,
            guidance_end_t=args.guidance_end_t,
            relation_mode=args.relation_mode,
            rerank_k=args.rerank_k,
            seed=args.seed,
            compute_quality=False,
            trace_path=clean_trace_path,
        )
        results["clean_guidance_trace_summary"] = str(clean_trace_summary)
        results["clean_guidance_trace"] = str(clean_trace_path)

    return {
        "objectives": objectives,
        "guided_scale": scale,
        "outputs": results,
    }


def main(args):
    root = Path(args.root).resolve()
    suite_dir = Path(args.output_dir).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    task_configs = task_configs_from_args(args)
    task_results = {}
    for task_id, task_config in task_configs.items():
        task_results[task_id] = run_task(args, root, suite_dir, task_id, task_config)

    manifest = {
        "config": {
            "ckpt_path": args.ckpt_path,
            "protocol_dir": args.protocol_dir,
            "predictor_dir": args.predictor_dir,
            "tasks": args.tasks,
            "n_molecules": args.n_molecules,
            "trace_n_molecules": args.trace_n_molecules,
            "base_integration_steps": args.base_integration_steps,
            "guided_integration_steps": args.guided_integration_steps,
            "guidance_start_t": args.guidance_start_t,
            "guidance_end_t": args.guidance_end_t,
            "relation_mode": args.relation_mode,
            "rerank_k": args.rerank_k,
            "guidance_device": args.guidance_device,
            "seed": args.seed,
        },
        "tasks": task_results,
    }
    manifest_path = suite_dir / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--ckpt_path", type=str, default="semla-flow/models/qm9/last.ckpt")
    parser.add_argument("--protocol_dir", type=str, default="results/qm9_mudm_protocol")
    parser.add_argument("--predictor_dir", type=str, default="results/qm9_mudm_protocol/predictors")
    parser.add_argument("--output_dir", type=str, default="results/qm9_multi_ablation")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--n_molecules", type=int, default=10000)
    parser.add_argument("--trace_n_molecules", type=int, default=1024)
    parser.add_argument("--base_integration_steps", type=int, default=100)
    parser.add_argument("--guided_integration_steps", type=int, default=1000)
    parser.add_argument("--guidance_start_t", type=float, default=0.5)
    parser.add_argument("--guidance_end_t", type=float, default=1.0)
    parser.add_argument("--relation_mode", type=str, default="cascade")
    parser.add_argument("--rerank_k", type=int, default=5)
    parser.add_argument("--guidance_device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    main(parser.parse_args())
