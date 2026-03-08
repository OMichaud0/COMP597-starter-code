#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def build_timeline(rows: List[Dict[str, str]], metrics: List[str]) -> Dict[int, Dict[str, float]]:
    by_sec: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if "elapsed_sec" not in row or row["elapsed_sec"] in ("", None):
            continue
        sec = int(math.floor(float(row["elapsed_sec"])))
        for m in metrics:
            raw = row.get(m, "")
            if raw in ("", None):
                continue
            try:
                by_sec[sec][m].append(float(raw))
            except Exception:
                continue
    out: Dict[int, Dict[str, float]] = {}
    for sec, mvals in by_sec.items():
        out[sec] = {}
        for m, vals in mvals.items():
            out[sec][m] = statistics.mean(vals)
    return out


def aggregate_timelines(per_repeat: List[Dict[int, Dict[str, float]]], metrics: List[str]) -> List[Dict[str, float]]:
    secs = sorted({sec for series in per_repeat for sec in series.keys()})
    aggregated = []
    for sec in secs:
        row: Dict[str, float] = {"elapsed_sec": float(sec)}
        for m in metrics:
            vals = [series[sec][m] for series in per_repeat if sec in series and m in series[sec]]
            avg, std = mean_std(vals)
            row[f"{m}_mean"] = avg
            row[f"{m}_std"] = std
        aggregated.append(row)
    return aggregated


def detect_gpu_idle_windows(timeline_rows: List[Dict[str, float]], threshold: float) -> List[Dict[str, float]]:
    windows: List[Dict[str, float]] = []
    active_start = None
    active_values: List[float] = []
    for row in timeline_rows:
        sec = int(row["elapsed_sec"])
        util = row.get("gpu_util_mean")
        if util is None or math.isnan(util):
            continue
        if util < threshold:
            if active_start is None:
                active_start = sec
                active_values = [util]
            else:
                active_values.append(util)
        else:
            if active_start is not None:
                windows.append(
                    {
                        "start_sec": active_start,
                        "end_sec": sec - 1,
                        "avg_gpu_util": statistics.mean(active_values),
                    }
                )
                active_start = None
                active_values = []
    if active_start is not None:
        end_sec = int(timeline_rows[-1]["elapsed_sec"]) if timeline_rows else active_start
        windows.append(
            {
                "start_sec": active_start,
                "end_sec": end_sec,
                "avg_gpu_util": statistics.mean(active_values) if active_values else float("nan"),
            }
        )
    return windows


def maybe_plot_timeline(path: str, rows: List[Dict[str, float]], metrics: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not rows:
        return
    x = [r["elapsed_sec"] for r in rows]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        y = [r.get(f"{metric}_mean", float("nan")) for r in rows]
        y_std = [r.get(f"{metric}_std", 0.0) for r in rows]
        upper = [a + b for a, b in zip(y, y_std)]
        lower = [a - b for a, b in zip(y, y_std)]
        ax.plot(x, y, linewidth=1.8, label=f"{metric} mean")
        ax.fill_between(x, lower, upper, alpha=0.2, label="std")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("elapsed_sec")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def maybe_plot_phase_bars(path: str, phase_stats: Dict[str, Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = ["forward_ms", "backward_ms", "optimizer_step_ms"]
    means = [phase_stats[k]["mean"] for k in labels]
    stds = [phase_stats[k]["std"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("ms")
    ax.set_title("Average Phase Time (mean +/- std)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate OPT experiment outputs across repeats.")
    parser.add_argument("experiment_dir", type=str, help="Path to one run directory generated by run_opt_experiments.py")
    parser.add_argument("--gpu-util-threshold", type=float, default=99.0, help="GPU util threshold for idle window detection.")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    manifest_path = os.path.join(exp_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json not found under {exp_dir}")

    manifest = read_json(manifest_path)
    reports_dir = os.path.join(exp_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    grouped: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    for run in manifest.get("runs", []):
        if int(run.get("returncode", 1)) == 0:
            grouped[(int(run["batch_size"]), run["profile"])].append(run)

    batch_sizes = sorted({k[0] for k in grouped.keys()}, reverse=True)
    overall_report = {"batches": {}, "overhead": {}}

    for batch_size in batch_sizes:
        batch_dir = os.path.join(reports_dir, f"batch_{batch_size}")
        os.makedirs(batch_dir, exist_ok=True)
        batch_report = {"profiles": {}}

        baseline_key = (batch_size, "A")
        baseline_runs = grouped.get(baseline_key, [])
        baseline_times = [float(r.get("elapsed_sec", 0.0)) for r in baseline_runs]
        baseline_mean, _ = mean_std(baseline_times)

        for profile in ["A", "B", "C"]:
            runs = grouped.get((batch_size, profile), [])
            if not runs:
                continue
            profile_dir = os.path.join(batch_dir, f"profile_{profile}")
            os.makedirs(profile_dir, exist_ok=True)

            durations = [float(r.get("elapsed_sec", 0.0)) for r in runs]
            d_mean, d_std = mean_std(durations)
            p_report: Dict = {
                "duration_sec": {"mean": d_mean, "std": d_std, "n": len(durations)},
            }

            if profile == "C":
                phase_forward = []
                phase_backward = []
                phase_opt = []
                timelines = []
                timeline_metrics = ["gpu_util", "gpu_mem_used", "proc_cpu_percent"]
                for run in runs:
                    run_dir = run["run_dir"]
                    simple_summary_path = os.path.join(run_dir, "simple_summary.json")
                    resource_steps_path = os.path.join(run_dir, "resource_steps.csv")
                    if os.path.isfile(simple_summary_path):
                        simple_summary = read_json(simple_summary_path)
                        avgs = simple_summary.get("averages_ms", {})
                        if "forward_ms" in avgs:
                            phase_forward.append(float(avgs["forward_ms"]))
                        if "backward_ms" in avgs:
                            phase_backward.append(float(avgs["backward_ms"]))
                        if "optimizer_step_ms" in avgs:
                            phase_opt.append(float(avgs["optimizer_step_ms"]))
                    if os.path.isfile(resource_steps_path):
                        rows = read_csv_rows(resource_steps_path)
                        timelines.append(build_timeline(rows, timeline_metrics))

                phase_report = {
                    "forward_ms": {"mean": mean_std(phase_forward)[0], "std": mean_std(phase_forward)[1]},
                    "backward_ms": {"mean": mean_std(phase_backward)[0], "std": mean_std(phase_backward)[1]},
                    "optimizer_step_ms": {"mean": mean_std(phase_opt)[0], "std": mean_std(phase_opt)[1]},
                }
                p_report["phase_bars"] = phase_report
                with open(os.path.join(profile_dir, "phase_bars_summary.json"), "w", encoding="utf-8") as fp:
                    json.dump(phase_report, fp, indent=2)
                maybe_plot_phase_bars(os.path.join(profile_dir, "phase_bars.png"), phase_report)

                if timelines:
                    aggregated = aggregate_timelines(timelines, timeline_metrics)
                    timeline_path = os.path.join(profile_dir, "timeline_aggregate.csv")
                    with open(timeline_path, "w", newline="", encoding="utf-8") as fp:
                        fieldnames = ["elapsed_sec"]
                        for m in timeline_metrics:
                            fieldnames.extend([f"{m}_mean", f"{m}_std"])
                        writer = csv.DictWriter(fp, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in aggregated:
                            writer.writerow(row)
                    maybe_plot_timeline(
                        os.path.join(profile_dir, "timeline_aggregate.png"), aggregated, timeline_metrics
                    )
                    idle_windows = detect_gpu_idle_windows(aggregated, args.gpu_util_threshold)
                    p_report["gpu_idle_windows"] = idle_windows
                    with open(os.path.join(profile_dir, "gpu_idle_windows.json"), "w", encoding="utf-8") as fp:
                        json.dump(idle_windows, fp, indent=2)

            batch_report["profiles"][profile] = p_report

            if profile in ("B", "C") and baseline_mean and not math.isnan(baseline_mean):
                overhead_pct = (d_mean - baseline_mean) / baseline_mean * 100.0
                overall_report["overhead"][f"batch_{batch_size}_profile_{profile}"] = overhead_pct

        overall_report["batches"][f"batch_{batch_size}"] = batch_report
        with open(os.path.join(batch_dir, "summary.json"), "w", encoding="utf-8") as fp:
            json.dump(batch_report, fp, indent=2)

    with open(os.path.join(reports_dir, "overall_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(overall_report, fp, indent=2)

    print(f"Aggregation complete. Reports at: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
