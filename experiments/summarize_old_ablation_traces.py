import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINTS = [0.0, 0.5, 0.9, 0.99]
TRACE_FILES = {
    "noisy": "noisy_guidance_trace.json",
    "clean": "clean_guidance_trace.json",
}
STATE_KEYS = ["curr", "predicted"]


def load_json(path: Path):
    return json.loads(path.read_text())


def nearest_step(steps, target_t: float):
    return min(steps, key=lambda step: abs(float(step["t"]) - target_t))


def collect_series(trace_payload):
    steps = trace_payload["steps"]
    objectives = trace_payload["objectives"]
    times = [float(step["t"]) for step in steps]

    series = {}
    for state in STATE_KEYS:
        for objective in objectives:
            key = (state, objective)
            series[key] = [float(step[state][objective]["paper_mae"]) for step in steps]

    return objectives, times, series


def summarise_curve(times, values):
    if not values:
        return {
            "start_mev": math.nan,
            "guidance_start_mev": math.nan,
            "late_mev": math.nan,
            "final_mev": math.nan,
            "guided_mean_mev": math.nan,
            "guided_std_mev": math.nan,
            "guided_diff_std_mev": math.nan,
            "guided_total_variation_mev": math.nan,
        }

    guided_values = [value for t, value in zip(times, values) if t >= 0.5]
    diffs = [guided_values[i + 1] - guided_values[i] for i in range(len(guided_values) - 1)]

    def _std(items):
        if len(items) <= 1:
            return 0.0
        mean = sum(items) / len(items)
        return math.sqrt(sum((item - mean) ** 2 for item in items) / len(items))

    def _nearest_value(target_t):
        idx = min(range(len(times)), key=lambda i: abs(times[i] - target_t))
        return values[idx]

    return {
        "start_mev": values[0],
        "guidance_start_mev": _nearest_value(0.5),
        "late_mev": _nearest_value(0.9),
        "final_mev": values[-1],
        "guided_mean_mev": sum(guided_values) / len(guided_values),
        "guided_std_mev": _std(guided_values),
        "guided_diff_std_mev": _std(diffs),
        "guided_total_variation_mev": sum(abs(diff) for diff in diffs),
    }


def write_summary_csv(rows, output_path: Path):
    fieldnames = [
        "task",
        "mode",
        "state",
        "objective",
        "start_mev",
        "guidance_start_mev",
        "late_mev",
        "final_mev",
        "guided_mean_mev",
        "guided_std_mev",
        "guided_diff_std_mev",
        "guided_total_variation_mev",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoints_csv(rows, output_path: Path):
    fieldnames = ["task", "mode", "state", "objective", "target_t", "actual_t", "paper_mae_mev"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary_rows, output_path: Path):
    grouped = {}
    for row in summary_rows:
        grouped.setdefault((row["task"], row["objective"]), []).append(row)

    lines = ["# Old Trace Summary", ""]
    for (task, objective), rows in sorted(grouped.items()):
        lines.append(f"## {task} - {objective}")
        lines.append("")
        lines.append("| Curve | Start (meV) | t~0.5 (meV) | t~0.9 (meV) | Final (meV) | Guided Mean (meV) | Guided Diff Std (meV) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            curve = f"{row['mode']}/{row['state']}"
            lines.append(
                f"| {curve} | {row['start_mev']:.1f} | {row['guidance_start_mev']:.1f} | "
                f"{row['late_mev']:.1f} | {row['final_mev']:.1f} | {row['guided_mean_mev']:.1f} | "
                f"{row['guided_diff_std_mev']:.1f} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


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
    png_path = output_dir / f"{task_name}_trace_plot.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results/qm9_multi_ablation")
    parser.add_argument("--output_dir", default="results/qm9_multi_ablation_old_trace_summary")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    checkpoint_rows = []
    plot_paths = []

    for task_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if not all((task_dir / filename).exists() for filename in TRACE_FILES.values()):
            continue

        plot_paths.append(plot_task(task_dir, output_dir))

        for mode, filename in TRACE_FILES.items():
            payload = load_json(task_dir / filename)
            steps = payload["steps"]
            objectives, times, series = collect_series(payload)

            for objective in objectives:
                for state in STATE_KEYS:
                    values = series[(state, objective)]
                    summary = summarise_curve(times, values)
                    summary_rows.append(
                        {
                            "task": task_dir.name,
                            "mode": mode,
                            "state": state,
                            "objective": objective,
                            **summary,
                        }
                    )

                    for checkpoint in CHECKPOINTS:
                        step = nearest_step(steps, checkpoint)
                        checkpoint_rows.append(
                            {
                                "task": task_dir.name,
                                "mode": mode,
                                "state": state,
                                "objective": objective,
                                "target_t": checkpoint,
                                "actual_t": float(step["t"]),
                                "paper_mae_mev": float(step[state][objective]["paper_mae"]),
                            }
                        )

    write_summary_csv(summary_rows, output_dir / "trace_stability_summary.csv")
    write_checkpoints_csv(checkpoint_rows, output_dir / "trace_checkpoints.csv")
    write_markdown(summary_rows, output_dir / "trace_summary.md")

    manifest = {
        "input_dir": str(input_dir),
        "plots": [str(path) for path in plot_paths],
        "summary_csv": str(output_dir / "trace_stability_summary.csv"),
        "checkpoints_csv": str(output_dir / "trace_checkpoints.csv"),
        "markdown": str(output_dir / "trace_summary.md"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
