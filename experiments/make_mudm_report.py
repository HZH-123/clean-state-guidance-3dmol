from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


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


def format_mean_std(values, digits=3):
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} +/- {pstdev(values):.{digits}f}"


def format_optional(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def value_mean(values):
    return mean(values)


def value_std(values):
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def format_summary(mean_value, std_value, digits=3):
    if std_value == 0:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def resolve_final_runs(task_entry: dict):
    if "final_runs" in task_entry:
        return [resolve_result_path(path) for path in task_entry["final_runs"]]
    if "final_run" in task_entry:
        return [resolve_result_path(task_entry["final_run"])]
    raise KeyError("Expected 'final_runs' or 'final_run' in task entry.")


def collect_single_summary(task_id: str, task_meta: dict, task_runs: dict):
    objective = task_meta.get("objective", task_id)
    baseline_runs = [load_json(path) for path in resolve_final_runs(task_runs["baseline"])]
    guided_runs = [load_json(path) for path in resolve_final_runs(task_runs["guided"])]

    baseline_maes = [run["metrics"][objective]["paper_mae"] for run in baseline_runs]
    guided_maes = [run["metrics"][objective]["paper_mae"] for run in guided_runs]

    baseline_novelty = [run["quality_metrics"]["novelty"] for run in baseline_runs]
    guided_novelty = [run["quality_metrics"]["novelty"] for run in guided_runs]

    baseline_atom = [run["quality_metrics"]["atom-stability"] for run in baseline_runs]
    guided_atom = [run["quality_metrics"]["atom-stability"] for run in guided_runs]

    mudm_mean = task_meta["mudm_mean"]
    mudm_std = task_meta["mudm_std"]
    guided_mean = value_mean(guided_maes)
    delta_vs_mudm = guided_mean - mudm_mean

    return {
        "task_id": task_id,
        "task": task_meta["display_name"],
        "objective": objective,
        "paper_unit": task_meta.get("paper_unit", ""),
        "ubound": task_meta.get("ubound"),
        "num_atoms": task_meta.get("num_atoms"),
        "cond_edm": task_meta.get("cond_edm"),
        "cond_geoldm": task_meta.get("cond_geoldm"),
        "eegsde": task_meta.get("eegsde"),
        "lbound": task_meta.get("lbound"),
        "mudm_mean": mudm_mean,
        "mudm_std": mudm_std,
        "baseline_mae_mean": value_mean(baseline_maes),
        "baseline_mae_std": value_std(baseline_maes),
        "guided_mae_mean": guided_mean,
        "guided_mae_std": value_std(guided_maes),
        "best_scale": task_runs["guided"]["best_scale"],
        "delta_vs_mudm": delta_vs_mudm,
        "baseline_novelty_mean": value_mean(baseline_novelty),
        "baseline_novelty_std": value_std(baseline_novelty),
        "guided_novelty_mean": value_mean(guided_novelty),
        "guided_novelty_std": value_std(guided_novelty),
        "baseline_atom_stability_mean": value_mean(baseline_atom),
        "baseline_atom_stability_std": value_std(baseline_atom),
        "guided_atom_stability_mean": value_mean(guided_atom),
        "guided_atom_stability_std": value_std(guided_atom),
    }


def single_markdown_row(summary: dict):
    return {
        "Task": summary["task"],
        "MUDM": f"{summary['mudm_mean']:.3f} +/- {summary['mudm_std']:.3f}",
        "Baseline MAE": format_summary(summary["baseline_mae_mean"], summary["baseline_mae_std"]),
        "Guided MAE": format_summary(summary["guided_mae_mean"], summary["guided_mae_std"]),
        "Best Scale": f"{summary['best_scale']}",
        "Delta vs MUDM": f"{summary['delta_vs_mudm']:+.3f}",
        "Baseline Novelty": format_summary(summary["baseline_novelty_mean"], summary["baseline_novelty_std"], digits=4),
        "Guided Novelty": format_summary(summary["guided_novelty_mean"], summary["guided_novelty_std"], digits=4),
        "Baseline Atom-Stab": format_summary(summary["baseline_atom_stability_mean"], summary["baseline_atom_stability_std"], digits=4),
        "Guided Atom-Stab": format_summary(summary["guided_atom_stability_mean"], summary["guided_atom_stability_std"], digits=4),
    }


def single_csv_row(summary: dict):
    return {
        "task_id": summary["task_id"],
        "task": summary["task"],
        "objective": summary["objective"],
        "paper_unit": summary["paper_unit"],
        "mudm_mean": f"{summary['mudm_mean']:.6f}",
        "mudm_std": f"{summary['mudm_std']:.6f}",
        "baseline_mae_mean": f"{summary['baseline_mae_mean']:.6f}",
        "baseline_mae_std": f"{summary['baseline_mae_std']:.6f}",
        "guided_mae_mean": f"{summary['guided_mae_mean']:.6f}",
        "guided_mae_std": f"{summary['guided_mae_std']:.6f}",
        "best_scale": f"{summary['best_scale']}",
        "delta_vs_mudm": f"{summary['delta_vs_mudm']:.6f}",
        "baseline_novelty_mean": f"{summary['baseline_novelty_mean']:.6f}",
        "baseline_novelty_std": f"{summary['baseline_novelty_std']:.6f}",
        "guided_novelty_mean": f"{summary['guided_novelty_mean']:.6f}",
        "guided_novelty_std": f"{summary['guided_novelty_std']:.6f}",
        "baseline_atom_stability_mean": f"{summary['baseline_atom_stability_mean']:.6f}",
        "baseline_atom_stability_std": f"{summary['baseline_atom_stability_std']:.6f}",
        "guided_atom_stability_mean": f"{summary['guided_atom_stability_mean']:.6f}",
        "guided_atom_stability_std": f"{summary['guided_atom_stability_std']:.6f}",
    }


def collect_multi_summary(task_meta: dict, task_runs: dict):
    baseline_runs = [load_json(path) for path in resolve_final_runs(task_runs["baseline"])]
    guided_runs = [load_json(path) for path in resolve_final_runs(task_runs["guided"])]
    obj1, obj2 = task_meta["objectives"]
    mudm1, mudm2 = task_meta["mudm"]
    baseline1_values = [run["metrics"][obj1]["paper_mae"] for run in baseline_runs]
    baseline2_values = [run["metrics"][obj2]["paper_mae"] for run in baseline_runs]
    guided1_values = [run["metrics"][obj1]["paper_mae"] for run in guided_runs]
    guided2_values = [run["metrics"][obj2]["paper_mae"] for run in guided_runs]
    baseline_novelty_values = [run["quality_metrics"].get("novelty") for run in baseline_runs if run.get("quality_metrics", {}).get("novelty") is not None]
    guided_novelty_values = [run["quality_metrics"].get("novelty") for run in guided_runs if run.get("quality_metrics", {}).get("novelty") is not None]
    baseline_atom_values = [run["quality_metrics"].get("atom-stability") for run in baseline_runs if run.get("quality_metrics", {}).get("atom-stability") is not None]
    guided_atom_values = [run["quality_metrics"].get("atom-stability") for run in guided_runs if run.get("quality_metrics", {}).get("atom-stability") is not None]

    baseline1_mean = value_mean(baseline1_values)
    baseline2_mean = value_mean(baseline2_values)
    guided1_mean = value_mean(guided1_values)
    guided2_mean = value_mean(guided2_values)
    delta1 = guided1_mean - mudm1
    delta2 = guided2_mean - mudm2

    return {
        "task_id": task_meta["task_id"],
        "task": task_meta["display_name"],
        "objective_1": obj1,
        "objective_2": obj2,
        "correlation": task_meta.get("correlation"),
        "paper_unit_1": task_meta.get("paper_units", ["", ""])[0],
        "paper_unit_2": task_meta.get("paper_units", ["", ""])[1],
        "cond_edm_1": task_meta.get("cond_edm", [None, None])[0],
        "cond_edm_2": task_meta.get("cond_edm", [None, None])[1],
        "eegsde_1": task_meta.get("eegsde", [None, None])[0],
        "eegsde_2": task_meta.get("eegsde", [None, None])[1],
        "mudm_mae_1": mudm1,
        "mudm_mae_2": mudm2,
        "baseline_mae_1": baseline1_mean,
        "baseline_mae_2": baseline2_mean,
        "guided_mae_1": guided1_mean,
        "guided_mae_2": guided2_mean,
        "baseline_mae_1_std": value_std(baseline1_values),
        "baseline_mae_2_std": value_std(baseline2_values),
        "guided_mae_1_std": value_std(guided1_values),
        "guided_mae_2_std": value_std(guided2_values),
        "best_scale": task_runs["guided"]["best_scale"],
        "delta_vs_mudm_1": delta1,
        "delta_vs_mudm_2": delta2,
        "baseline_novelty": value_mean(baseline_novelty_values) if baseline_novelty_values else None,
        "guided_novelty": value_mean(guided_novelty_values) if guided_novelty_values else None,
        "baseline_novelty_std": value_std(baseline_novelty_values) if baseline_novelty_values else 0.0,
        "guided_novelty_std": value_std(guided_novelty_values) if guided_novelty_values else 0.0,
        "baseline_atom_stability": value_mean(baseline_atom_values) if baseline_atom_values else None,
        "guided_atom_stability": value_mean(guided_atom_values) if guided_atom_values else None,
        "baseline_atom_stability_std": value_std(baseline_atom_values) if baseline_atom_values else 0.0,
        "guided_atom_stability_std": value_std(guided_atom_values) if guided_atom_values else 0.0,
        "paper_novelty": task_meta.get("novelty"),
        "paper_atom_stability": task_meta.get("atom_stability"),
    }


def multi_markdown_row(summary: dict):
    return {
        "Task": summary["task"],
        "MUDM": f"{summary['mudm_mae_1']:.3f} / {summary['mudm_mae_2']:.3f}",
        "Baseline MAE": f"{format_summary(summary['baseline_mae_1'], summary['baseline_mae_1_std'])} / {format_summary(summary['baseline_mae_2'], summary['baseline_mae_2_std'])}",
        "Guided MAE": f"{format_summary(summary['guided_mae_1'], summary['guided_mae_1_std'])} / {format_summary(summary['guided_mae_2'], summary['guided_mae_2_std'])}",
        "Best Scale": f"{summary['best_scale']}",
        "Delta vs MUDM": f"{summary['delta_vs_mudm_1']:+.3f} / {summary['delta_vs_mudm_2']:+.3f}",
        "Baseline Novelty": format_optional(summary["baseline_novelty"], digits=4) if summary["baseline_novelty"] is None else format_summary(summary["baseline_novelty"], summary["baseline_novelty_std"], digits=4),
        "Guided Novelty": format_optional(summary["guided_novelty"], digits=4) if summary["guided_novelty"] is None else format_summary(summary["guided_novelty"], summary["guided_novelty_std"], digits=4),
        "Baseline Atom-Stab": format_optional(summary["baseline_atom_stability"], digits=4) if summary["baseline_atom_stability"] is None else format_summary(summary["baseline_atom_stability"], summary["baseline_atom_stability_std"], digits=4),
        "Guided Atom-Stab": format_optional(summary["guided_atom_stability"], digits=4) if summary["guided_atom_stability"] is None else format_summary(summary["guided_atom_stability"], summary["guided_atom_stability_std"], digits=4),
    }


def multi_csv_row(summary: dict):
    return {
        "task_id": summary["task_id"],
        "task": summary["task"],
        "objective_1": summary["objective_1"],
        "objective_2": summary["objective_2"],
        "paper_unit_1": summary["paper_unit_1"],
        "paper_unit_2": summary["paper_unit_2"],
        "mudm_mae_1": f"{summary['mudm_mae_1']:.6f}",
        "mudm_mae_2": f"{summary['mudm_mae_2']:.6f}",
        "baseline_mae_1": f"{summary['baseline_mae_1']:.6f}",
        "baseline_mae_2": f"{summary['baseline_mae_2']:.6f}",
        "baseline_mae_1_std": f"{summary['baseline_mae_1_std']:.6f}",
        "baseline_mae_2_std": f"{summary['baseline_mae_2_std']:.6f}",
        "guided_mae_1": f"{summary['guided_mae_1']:.6f}",
        "guided_mae_2": f"{summary['guided_mae_2']:.6f}",
        "guided_mae_1_std": f"{summary['guided_mae_1_std']:.6f}",
        "guided_mae_2_std": f"{summary['guided_mae_2_std']:.6f}",
        "best_scale": f"{summary['best_scale']}",
        "delta_vs_mudm_1": f"{summary['delta_vs_mudm_1']:.6f}",
        "delta_vs_mudm_2": f"{summary['delta_vs_mudm_2']:.6f}",
        "baseline_novelty": "" if summary["baseline_novelty"] is None else f"{summary['baseline_novelty']:.6f}",
        "baseline_novelty_std": "" if summary["baseline_novelty"] is None else f"{summary['baseline_novelty_std']:.6f}",
        "guided_novelty": "" if summary["guided_novelty"] is None else f"{summary['guided_novelty']:.6f}",
        "guided_novelty_std": "" if summary["guided_novelty"] is None else f"{summary['guided_novelty_std']:.6f}",
        "baseline_atom_stability": "" if summary["baseline_atom_stability"] is None else f"{summary['baseline_atom_stability']:.6f}",
        "baseline_atom_stability_std": "" if summary["baseline_atom_stability"] is None else f"{summary['baseline_atom_stability_std']:.6f}",
        "guided_atom_stability": "" if summary["guided_atom_stability"] is None else f"{summary['guided_atom_stability']:.6f}",
        "guided_atom_stability_std": "" if summary["guided_atom_stability"] is None else f"{summary['guided_atom_stability_std']:.6f}",
    }


def rows_to_markdown(rows):
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def single_paper_csv_row(summary: dict):
    return {
        "Property": summary["task"],
        "Unit": summary["paper_unit"],
        "U-bound": summary["ubound"] or "",
        "#Atoms": summary["num_atoms"] or "",
        "Conditional EDM": summary["cond_edm"] or "",
        "Cond. GeoLDM": summary["cond_geoldm"] or "",
        "EEGSDE": summary["eegsde"] or "",
        "MuDM": f"{summary['mudm_mean']:.3f} +/- {summary['mudm_std']:.3f}",
        "Ours Baseline": f"{summary['baseline_mae_mean']:.3f} +/- {summary['baseline_mae_std']:.3f}",
        "Ours Guided": f"{summary['guided_mae_mean']:.3f} +/- {summary['guided_mae_std']:.3f}",
        "Best Scale": f"{summary['best_scale']}",
        "Delta vs MuDM": f"{summary['delta_vs_mudm']:+.3f}",
        "L-bound": summary["lbound"] or "",
        "Baseline Novelty": f"{summary['baseline_novelty_mean']:.4f} +/- {summary['baseline_novelty_std']:.4f}",
        "Guided Novelty": f"{summary['guided_novelty_mean']:.4f} +/- {summary['guided_novelty_std']:.4f}",
        "Baseline Atom-Stability": f"{summary['baseline_atom_stability_mean']:.4f} +/- {summary['baseline_atom_stability_std']:.4f}",
        "Guided Atom-Stability": f"{summary['guided_atom_stability_mean']:.4f} +/- {summary['guided_atom_stability_std']:.4f}",
    }


def multi_paper_csv_rows(summary: dict):
    common = {
        "Property 1": summary["objective_1"],
        "Property 2": summary["objective_2"],
        "Correlation": "" if summary["correlation"] is None else f"{summary['correlation']:.2f}",
        "Best Scale": f"{summary['best_scale']}",
        "Paper Novelty": "" if summary["paper_novelty"] is None else f"{summary['paper_novelty']:.2f}",
        "Paper Atom-Stability": "" if summary["paper_atom_stability"] is None else f"{summary['paper_atom_stability']:.2f}",
        "Ours Baseline Novelty": "" if summary["baseline_novelty"] is None else f"{summary['baseline_novelty']:.4f}",
        "Ours Guided Novelty": "" if summary["guided_novelty"] is None else f"{summary['guided_novelty']:.4f}",
        "Ours Baseline Atom-Stability": "" if summary["baseline_atom_stability"] is None else f"{summary['baseline_atom_stability']:.4f}",
        "Ours Guided Atom-Stability": "" if summary["guided_atom_stability"] is None else f"{summary['guided_atom_stability']:.4f}",
    }
    row1 = {
        **common,
        "Metric": "MAE 1",
        "Conditional EDM": f"{summary['cond_edm_1']:.3f}",
        "EEGSDE": f"{summary['eegsde_1']:.3f}",
        "MuDM": f"{summary['mudm_mae_1']:.3f}",
        "Ours Baseline": format_summary(summary["baseline_mae_1"], summary["baseline_mae_1_std"]),
        "Ours Guided": format_summary(summary["guided_mae_1"], summary["guided_mae_1_std"]),
        "Delta vs MuDM": f"{summary['delta_vs_mudm_1']:+.3f}",
    }
    row2 = {
        **common,
        "Metric": "MAE 2",
        "Conditional EDM": f"{summary['cond_edm_2']:.3f}",
        "EEGSDE": f"{summary['eegsde_2']:.3f}",
        "MuDM": f"{summary['mudm_mae_2']:.3f}",
        "Ours Baseline": format_summary(summary["baseline_mae_2"], summary["baseline_mae_2_std"]),
        "Ours Guided": format_summary(summary["guided_mae_2"], summary["guided_mae_2_std"]),
        "Delta vs MuDM": f"{summary['delta_vs_mudm_2']:+.3f}",
    }
    return [row1, row2]


def quality_paper_csv_row(summary: dict):
    return {
        "Conditioned properties": f"{summary['objective_1']} + {summary['objective_2']}",
        "MUDM Atom stability": "" if summary["paper_atom_stability"] is None else f"{summary['paper_atom_stability']:.2f}",
        "MUDM Novelty": "" if summary["paper_novelty"] is None else f"{summary['paper_novelty']:.2f}",
        "Ours Baseline Atom stability": "" if summary["baseline_atom_stability"] is None else format_summary(summary["baseline_atom_stability"], summary["baseline_atom_stability_std"], digits=4),
        "Ours Guided Atom stability": "" if summary["guided_atom_stability"] is None else format_summary(summary["guided_atom_stability"], summary["guided_atom_stability_std"], digits=4),
        "Ours Baseline Novelty": "" if summary["baseline_novelty"] is None else format_summary(summary["baseline_novelty"], summary["baseline_novelty_std"], digits=4),
        "Ours Guided Novelty": "" if summary["guided_novelty"] is None else format_summary(summary["guided_novelty"], summary["guided_novelty_std"], digits=4),
    }


def main(args):
    refs = load_json(Path(args.refs))
    suite = load_json(Path(args.suite_manifest))

    single_summaries = []
    for task_id, task_meta in refs["single"].items():
        single_summaries.append(collect_single_summary(task_id, task_meta, suite["single"][task_id]))

    multi_summaries = []
    for task_meta in refs["multi"]:
        multi_summaries.append(collect_multi_summary(task_meta, suite["multi"][task_meta["task_id"]]))

    single_rows = [single_markdown_row(summary) for summary in single_summaries]
    multi_rows = [multi_markdown_row(summary) for summary in multi_summaries]
    single_csv_rows = [single_csv_row(summary) for summary in single_summaries]
    multi_csv_rows = [multi_csv_row(summary) for summary in multi_summaries]
    single_paper_rows = [single_paper_csv_row(summary) for summary in single_summaries]
    multi_paper_rows = [row for summary in multi_summaries for row in multi_paper_csv_rows(summary)]
    quality_paper_rows = [quality_paper_csv_row(summary) for summary in multi_summaries]

    report = [
        "# Single-objective",
        rows_to_markdown(single_rows),
        "",
        "# Multi-objective",
        rows_to_markdown(multi_rows),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report to {output_path}")

    single_csv_path = output_path.with_name(f"{output_path.stem}_single.csv")
    multi_csv_path = output_path.with_name(f"{output_path.stem}_multi.csv")
    single_paper_csv_path = output_path.with_name(f"{output_path.stem}_table1_like.csv")
    multi_paper_csv_path = output_path.with_name(f"{output_path.stem}_table2_like.csv")
    quality_paper_csv_path = output_path.with_name(f"{output_path.stem}_table10_like.csv")
    write_csv(single_csv_path, single_csv_rows)
    write_csv(multi_csv_path, multi_csv_rows)
    write_csv(single_paper_csv_path, single_paper_rows)
    write_csv(multi_paper_csv_path, multi_paper_rows)
    write_csv(quality_paper_csv_path, quality_paper_rows)
    print(f"Saved CSV to {single_csv_path}")
    print(f"Saved CSV to {multi_csv_path}")
    print(f"Saved CSV to {single_paper_csv_path}")
    print(f"Saved CSV to {multi_paper_csv_path}")
    print(f"Saved CSV to {quality_paper_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", type=str, required=True)
    parser.add_argument("--suite_manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    main(parser.parse_args())
