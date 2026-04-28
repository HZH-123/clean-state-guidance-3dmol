from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = [
    ("base", "Base"),
    ("rerank", "Rerank"),
    ("noisy_guidance", "Noisy-guidance"),
    ("clean_guidance", "Clean-guidance"),
]


def load_json(path: Path):
    return json.loads(path.read_text())


def resolve_result_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    normalized = path_str.replace("\\", "/")
    marker = "/CEP/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        local = Path.cwd() / Path(suffix)
        if local.exists():
            return local

    for marker in ("results/", "experiments/"):
        idx = normalized.find(marker)
        if idx != -1:
            local = Path.cwd() / Path(normalized[idx:])
            if local.exists():
                return local

    return path


def rows_to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""

    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return "\n".join([header_line, divider, *body])


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_complete_tasks(input_dirs: list[Path]):
    tasks = {}
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for task_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
            required = {name: task_dir / f"{name}.json" for name, _ in METHODS}
            if not all(path.exists() for path in required.values()):
                continue
            tasks[task_dir.name] = required
    return tasks


def format_metric(payload: dict, objective: str):
    metric = payload["metrics"][objective]
    return metric["paper_mae"], metric["paper_unit"]


def build_task_summary(task_id: str, file_map: dict[str, Path]):
    payloads = {name: load_json(path) for name, path in file_map.items()}
    objectives = payloads["base"]["objectives"]
    objective_1, objective_2 = objectives
    unit_1 = payloads["base"]["metrics"][objective_1]["paper_unit"]
    unit_2 = payloads["base"]["metrics"][objective_2]["paper_unit"]

    summary = {
        "task_id": task_id,
        "objective_1": objective_1,
        "objective_2": objective_2,
        "unit_1": unit_1,
        "unit_2": unit_2,
    }

    for method_key, method_label in METHODS:
        payload = payloads[method_key]
        metric_1, _ = format_metric(payload, objective_1)
        metric_2, _ = format_metric(payload, objective_2)
        quality = payload.get("quality_metrics", {})

        summary[f"{method_key}_label"] = method_label
        summary[f"{method_key}_mae_1"] = metric_1
        summary[f"{method_key}_mae_2"] = metric_2
        summary[f"{method_key}_novelty"] = quality.get("novelty")
        summary[f"{method_key}_atom_stability"] = quality.get("atom-stability")
        summary[f"{method_key}_validity"] = quality.get("validity")

    return summary


def summary_to_markdown_row(summary: dict):
    return {
        "Task": f"{summary['objective_1']} + {summary['objective_2']}",
        "Base MAE1": f"{summary['base_mae_1']:.3f}",
        "Base MAE2": f"{summary['base_mae_2']:.3f}",
        "Rerank MAE1": f"{summary['rerank_mae_1']:.3f}",
        "Rerank MAE2": f"{summary['rerank_mae_2']:.3f}",
        "Noisy MAE1": f"{summary['noisy_guidance_mae_1']:.3f}",
        "Noisy MAE2": f"{summary['noisy_guidance_mae_2']:.3f}",
        "Clean MAE1": f"{summary['clean_guidance_mae_1']:.3f}",
        "Clean MAE2": f"{summary['clean_guidance_mae_2']:.3f}",
    }


def summary_to_csv_row(summary: dict):
    return {
        "task_id": summary["task_id"],
        "objective_1": summary["objective_1"],
        "unit_1": summary["unit_1"],
        "objective_2": summary["objective_2"],
        "unit_2": summary["unit_2"],
        "base_mae_1": summary["base_mae_1"],
        "base_mae_2": summary["base_mae_2"],
        "rerank_mae_1": summary["rerank_mae_1"],
        "rerank_mae_2": summary["rerank_mae_2"],
        "noisy_guidance_mae_1": summary["noisy_guidance_mae_1"],
        "noisy_guidance_mae_2": summary["noisy_guidance_mae_2"],
        "clean_guidance_mae_1": summary["clean_guidance_mae_1"],
        "clean_guidance_mae_2": summary["clean_guidance_mae_2"],
        "base_novelty": summary["base_novelty"],
        "rerank_novelty": summary["rerank_novelty"],
        "noisy_guidance_novelty": summary["noisy_guidance_novelty"],
        "clean_guidance_novelty": summary["clean_guidance_novelty"],
        "base_atom_stability": summary["base_atom_stability"],
        "rerank_atom_stability": summary["rerank_atom_stability"],
        "noisy_guidance_atom_stability": summary["noisy_guidance_atom_stability"],
        "clean_guidance_atom_stability": summary["clean_guidance_atom_stability"],
        "base_validity": summary["base_validity"],
        "rerank_validity": summary["rerank_validity"],
        "noisy_guidance_validity": summary["noisy_guidance_validity"],
        "clean_guidance_validity": summary["clean_guidance_validity"],
    }


def quality_rows(summary: dict):
    task_name = f"{summary['objective_1']} + {summary['objective_2']}"
    rows = []
    for method_key, method_label in METHODS:
        rows.append(
            {
                "Task": task_name,
                "Method": method_label,
                "Novelty": f"{summary[f'{method_key}_novelty']:.4f}",
                "Atom Stability": f"{summary[f'{method_key}_atom_stability']:.4f}",
                "Validity": f"{summary[f'{method_key}_validity']:.4f}",
            }
        )
    return rows


def main(args):
    input_dirs = [Path(path) for path in args.input_dirs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    complete_tasks = find_complete_tasks(input_dirs)
    summaries = [build_task_summary(task_id, file_map) for task_id, file_map in sorted(complete_tasks.items())]

    metric_rows = [summary_to_markdown_row(summary) for summary in summaries]
    csv_rows = [summary_to_csv_row(summary) for summary in summaries]
    quality_metric_rows = [row for summary in summaries for row in quality_rows(summary)]

    report = [
        "# Multi-objective Ablation",
        "",
        rows_to_markdown(metric_rows),
        "",
        "# Quality",
        "",
        rows_to_markdown(quality_metric_rows),
        "",
    ]
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report to {output}")

    csv_path = output.with_name(f"{output.stem}.csv")
    quality_csv_path = output.with_name(f"{output.stem}_quality.csv")
    write_csv(csv_path, csv_rows)
    write_csv(quality_csv_path, quality_metric_rows)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved CSV to {quality_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
