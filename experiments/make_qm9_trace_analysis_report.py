from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRACE_FILES = {
    "noisy": "noisy_guidance_trace.json",
    "clean": "clean_guidance_trace.json",
}
STATE_KEYS = ["curr", "predicted"]


def load_json(path: Path):
    return json.loads(path.read_text())


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
            if all((task_dir / filename).exists() for filename in TRACE_FILES.values()):
                tasks[task_dir.name] = task_dir
    return tasks


def collect_series(trace_payload):
    steps = trace_payload["steps"]
    objectives = trace_payload["objectives"]
    times = [float(step["t"]) for step in steps]

    series = {}
    for state in STATE_KEYS:
        for objective in objectives:
            series[(state, objective)] = [float(step[state][objective]["paper_mae"]) for step in steps]
    return objectives, times, series


def summarise_curve(times, values):
    guided_values = [value for t, value in zip(times, values) if t >= 0.5]
    diffs = [guided_values[i + 1] - guided_values[i] for i in range(len(guided_values) - 1)]

    def _std(items):
        if len(items) <= 1:
            return 0.0
        mean = sum(items) / len(items)
        return math.sqrt(sum((item - mean) ** 2 for item in items) / len(items))

    def _nearest(target_t):
        idx = min(range(len(times)), key=lambda i: abs(times[i] - target_t))
        return values[idx]

    return {
        "start_mev": values[0],
        "t05_mev": _nearest(0.5),
        "t09_mev": _nearest(0.9),
        "final_mev": values[-1],
        "guided_mean_mev": sum(guided_values) / len(guided_values),
        "guided_std_mev": _std(guided_values),
        "guided_diff_std_mev": _std(diffs),
        "guided_total_variation_mev": sum(abs(diff) for diff in diffs),
    }


def plot_task(task_dir: Path, output_dir: Path):
    task_name = task_dir.name
    traces = {}
    objectives = None

    for mode, filename in TRACE_FILES.items():
        payload = load_json(task_dir / filename)
        objectives, times, series = collect_series(payload)
        traces[mode] = {"times": times, "series": series}

    fig, axes = plt.subplots(len(objectives), 1, figsize=(10, 4 * len(objectives)), sharex=True)
    if len(objectives) == 1:
        axes = [axes]

    colors = {
        ("noisy", "curr"): "#1f77b4",
        ("noisy", "predicted"): "#ff7f0e",
        ("clean", "curr"): "#2ca02c",
        ("clean", "predicted"): "#d62728",
    }

    for axis, objective in zip(axes, objectives):
        for mode in ["noisy", "clean"]:
            times = traces[mode]["times"]
            for state in STATE_KEYS:
                values = traces[mode]["series"][(state, objective)]
                axis.plot(times, values, label=f"{mode}/{state}", color=colors[(mode, state)], linewidth=1.8)

        axis.set_title(f"{task_name} - {objective}")
        axis.set_ylabel("Deviation (meV)")
        axis.grid(True, alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Sampling time t")
    fig.tight_layout()
    output_path = output_dir / f"{task_name}_trace_plot.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(args):
    input_dirs = [Path(path) for path in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = find_complete_tasks(input_dirs)

    curve_rows = []
    compare_rows = []
    plot_paths = []

    for task_id, task_dir in sorted(tasks.items()):
        plot_paths.append(plot_task(task_dir, output_dir))

        traces = {mode: load_json(task_dir / filename) for mode, filename in TRACE_FILES.items()}
        objectives = traces["clean"]["objectives"]

        summaries = {}
        for mode in ["noisy", "clean"]:
            _, times, series = collect_series(traces[mode])
            for objective in objectives:
                for state in STATE_KEYS:
                    key = (mode, state, objective)
                    summaries[key] = summarise_curve(times, series[(state, objective)])
                    curve_rows.append(
                        {
                            "task": task_id,
                            "mode": mode,
                            "state": state,
                            "objective": objective,
                            **summaries[key],
                        }
                    )

        for objective in objectives:
            noisy_pred = summaries[("noisy", "predicted", objective)]
            clean_pred = summaries[("clean", "predicted", objective)]
            noisy_curr = summaries[("noisy", "curr", objective)]
            clean_curr = summaries[("clean", "curr", objective)]

            compare_rows.append(
                {
                    "task": task_id,
                    "objective": objective,
                    "noisy_pred_final_mev": round(noisy_pred["final_mev"], 1),
                    "clean_pred_final_mev": round(clean_pred["final_mev"], 1),
                    "final_improvement_mev": round(noisy_pred["final_mev"] - clean_pred["final_mev"], 1),
                    "noisy_pred_diff_std_mev": round(noisy_pred["guided_diff_std_mev"], 2),
                    "clean_pred_diff_std_mev": round(clean_pred["guided_diff_std_mev"], 2),
                    "noisy_curr_final_mev": round(noisy_curr["final_mev"], 1),
                    "clean_curr_final_mev": round(clean_curr["final_mev"], 1),
                }
            )

    markdown_sections = ["# Trace Interpretability Analysis", ""]
    markdown_sections.append("## Predicted-State Comparison")
    markdown_sections.append("")
    markdown_sections.append(rows_to_markdown(compare_rows))
    markdown_sections.append("")
    markdown_sections.append("## Full Curve Summary")
    markdown_sections.append("")
    markdown_sections.append(rows_to_markdown(curve_rows))
    markdown_sections.append("")

    report_path = output_dir / "trace_analysis_report.md"
    report_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    curve_csv = output_dir / "trace_analysis_curves.csv"
    compare_csv = output_dir / "trace_analysis_compare.csv"
    write_csv(curve_csv, curve_rows)
    write_csv(compare_csv, compare_rows)

    manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "tasks": sorted(tasks.keys()),
        "report": str(report_path),
        "curve_csv": str(curve_csv),
        "compare_csv": str(compare_csv),
        "plots": [str(path) for path in plot_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved report to {report_path}")
    print(f"Saved CSV to {curve_csv}")
    print(f"Saved CSV to {compare_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
