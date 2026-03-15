#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from log_utils import LogLevel, Logger


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.mean(values), statistics.stdev(values)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def safe_float(raw: object) -> Optional[float]:
    if raw is None:
        return None
    try:
        s = str(raw).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_per_second_series(rows: List[Dict[str, str]], metrics: List[str]) -> Dict[int, Dict[str, float]]:
    by_sec: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        elapsed = safe_float(row.get("elapsed_sec"))
        if elapsed is None:
            continue
        sec = int(math.floor(elapsed))
        for metric in metrics:
            value = safe_float(row.get(metric))
            if value is None:
                continue
            by_sec[sec][metric].append(value)

    out: Dict[int, Dict[str, float]] = {}
    for sec, metric_values in by_sec.items():
        out[sec] = {}
        for metric, vals in metric_values.items():
            out[sec][metric] = statistics.mean(vals)
    return out


def aggregate_per_second(
    per_repeat_series: List[Dict[int, Dict[str, float]]], metrics: List[str]
) -> List[Dict[str, float]]:
    secs = sorted({sec for series in per_repeat_series for sec in series.keys()})
    rows: List[Dict[str, float]] = []
    for sec in secs:
        row: Dict[str, float] = {"elapsed_sec": float(sec)}
        for metric in metrics:
            vals = [series[sec][metric] for series in per_repeat_series if sec in series and metric in series[sec]]
            avg, std = mean_std(vals)
            row[f"{metric}_mean"] = avg
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


def detect_gpu_idle_windows(timeline_rows: List[Dict[str, float]], threshold: float) -> List[Dict[str, float]]:
    windows: List[Dict[str, float]] = []
    active_start: Optional[int] = None
    active_vals: List[float] = []

    for row in timeline_rows:
        sec = int(row.get("elapsed_sec", 0.0))
        util = row.get("gpu_util_mean")
        if util is None or math.isnan(util):
            continue
        if util < threshold:
            if active_start is None:
                active_start = sec
                active_vals = [util]
            else:
                active_vals.append(util)
        elif active_start is not None:
            windows.append(
                {
                    "start_sec": active_start,
                    "end_sec": sec - 1,
                    "duration_sec": sec - active_start,
                    "avg_gpu_util": statistics.mean(active_vals),
                }
            )
            active_start = None
            active_vals = []

    if active_start is not None:
        end_sec = int(timeline_rows[-1]["elapsed_sec"]) if timeline_rows else active_start
        windows.append(
            {
                "start_sec": active_start,
                "end_sec": end_sec,
                "duration_sec": max(1, end_sec - active_start + 1),
                "avg_gpu_util": statistics.mean(active_vals) if active_vals else float("nan"),
            }
        )
    return windows


def resolve_manifest_path(input_dir: str) -> str:
    direct = os.path.join(input_dir, "manifest.json")
    if os.path.isfile(direct):
        return direct
    candidates = []
    try:
        for name in os.listdir(input_dir):
            child = os.path.join(input_dir, name)
            if not os.path.isdir(child):
                continue
            candidate = os.path.join(child, "manifest.json")
            if os.path.isfile(candidate):
                candidates.append(candidate)
    except FileNotFoundError:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return ""


def run_identifier(run: Dict[str, object]) -> str:
    return f"batch_{run['batch']}/profile_{run['profile']}/repeat_{run['repeat']}"


def discover_runs(exp_dir: str) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for batch_name in sorted(os.listdir(exp_dir)):
        batch_dir = os.path.join(exp_dir, batch_name)
        if not os.path.isdir(batch_dir) or not batch_name.startswith("batch_"):
            continue
        try:
            batch_size = int(batch_name.split("_", 1)[1])
        except Exception:
            continue
        for profile_name in sorted(os.listdir(batch_dir)):
            profile_dir = os.path.join(batch_dir, profile_name)
            if not os.path.isdir(profile_dir) or not profile_name.startswith("profile_"):
                continue
            profile = profile_name.split("_", 1)[1]
            for repeat_name in sorted(os.listdir(profile_dir)):
                repeat_dir = os.path.join(profile_dir, repeat_name)
                if not os.path.isdir(repeat_dir) or not repeat_name.startswith("repeat_"):
                    continue
                try:
                    repeat = int(repeat_name.split("_", 1)[1])
                except Exception:
                    continue
                run_timing_path = os.path.join(repeat_dir, "run_timing.json")
                if not os.path.isfile(run_timing_path):
                    continue
                timing = read_json(run_timing_path)
                run = {
                    "batch": batch_size,
                    "profile": profile,
                    "repeat": repeat,
                    "run_dir": repeat_dir,
                    "run_timing_path": run_timing_path,
                    "elapsed_sec": float(timing.get("elapsed_sec", float("nan"))),
                    "returncode": int(timing.get("returncode", 1)),
                    "simple_steps_path": os.path.join(repeat_dir, "simple_steps.csv"),
                    "simple_summary_path": os.path.join(repeat_dir, "simple_summary.json"),
                    "resource_steps_path": os.path.join(repeat_dir, "resource_steps.csv"),
                    "resource_summary_path": os.path.join(repeat_dir, "resource_summary.json"),
                }
                runs.append(run)
    runs.sort(key=lambda r: (int(r["batch"]), str(r["profile"]), int(r["repeat"])))
    return runs


def parse_exclusions(raw_values: Sequence[str]) -> set:
    exclusions = set()
    for raw in raw_values:
        parts = raw.split(":")
        if len(parts) != 3:
            continue
        try:
            b = int(parts[0])
            p = parts[1].upper()
            r = int(parts[2])
        except Exception:
            continue
        exclusions.add((b, p, r))
    return exclusions


def ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plot generation. Install dependencies and rerun, "
            "or use --skip-plots for data-only outputs."
        ) from exc


def plot_metric_with_band(ax, x: List[float], y: List[float], y_std: List[float], label: str, color: str) -> None:
    upper = [a + b for a, b in zip(y, y_std)]
    lower = [a - b for a, b in zip(y, y_std)]
    ax.plot(x, y, label=label, color=color, linewidth=1.8)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)


def save_profile_c_resource_timeline(
    plt, out_path: str, rows: List[Dict[str, float]], batch: int, metrics: List[Tuple[str, str, str]]
) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    x = [r["elapsed_sec"] for r in rows]
    for ax, (metric, label, color) in zip(axes, metrics):
        y = [r.get(f"{metric}_mean", float("nan")) for r in rows]
        y_std = [r.get(f"{metric}_std", 0.0) for r in rows]
        plot_metric_with_band(ax, x, y, y_std, "mean ± std", color)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Elapsed Time (sec)")
    fig.suptitle(f"Profile C Resource Timeline (Batch {batch})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_profile_c_phase_timeline(
    plt, out_path: str, rows: List[Dict[str, float]], batch: int, metrics: List[Tuple[str, str, str]]
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = [r["elapsed_sec"] for r in rows]
    for metric, label, color in metrics:
        y = [r.get(f"{metric}_mean", float("nan")) for r in rows]
        y_std = [r.get(f"{metric}_std", 0.0) for r in rows]
        plot_metric_with_band(ax, x, y, y_std, label, color)
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("Phase Time (ms)")
    ax.set_title(f"Profile C Phase Timeline (Batch {batch})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_profile_c_loss_timeline(plt, out_path: str, rows: List[Dict[str, float]], batch: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = [r["elapsed_sec"] for r in rows]
    y = [r.get("loss_mean", float("nan")) for r in rows]
    y_std = [r.get("loss_std", 0.0) for r in rows]
    plot_metric_with_band(ax, x, y, y_std, "Loss", "#8c564b")
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Profile C Loss Timeline (Batch {batch})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_phase_means_vs_batch(
    plt, out_path: str, batch_sizes: List[int], phase_stats: Dict[str, Dict[int, Tuple[float, float]]]
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    style = {
        "forward_ms": ("Forward", "#1f77b4"),
        "backward_ms": ("Backward", "#ff7f0e"),
        "optimizer_step_ms": ("Optimizer", "#2ca02c"),
    }
    for metric in ("forward_ms", "backward_ms", "optimizer_step_ms"):
        label, color = style[metric]
        means = [phase_stats[metric][b][0] for b in batch_sizes]
        stds = [phase_stats[metric][b][1] for b in batch_sizes]
        ax.errorbar(batch_sizes, means, yerr=stds, marker="o", capsize=4, linewidth=1.8, label=label, color=color)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Mean Phase Time (ms)")
    ax.set_title("Profile C Phase Means vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_phase_share_vs_batch(
    plt, out_path: str, batch_sizes: List[int], phase_share_stats: Dict[str, Dict[int, Tuple[float, float]]]
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.24
    x = list(range(len(batch_sizes)))
    phase_config = [
        ("forward_share", "Forward", "#1f77b4"),
        ("backward_share", "Backward", "#ff7f0e"),
        ("optimizer_share", "Optimizer", "#2ca02c"),
    ]
    for idx, (metric, label, color) in enumerate(phase_config):
        means = [phase_share_stats[metric][b][0] * 100.0 for b in batch_sizes]
        stds = [phase_share_stats[metric][b][1] * 100.0 for b in batch_sizes]
        offsets = [pos + (idx - 1) * bar_width for pos in x]
        ax.bar(offsets, means, width=bar_width, color=color, alpha=0.85, label=label, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Share of Step Time (%)")
    ax.set_title("Profile C Phase Share vs Batch Size")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_profile_bc_energy_vs_batch(
    plt, out_path: str, batch_sizes: List[int], energy_stats: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]]
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    profile_style = {"B": ("Profile B", "#9467bd"), "C": ("Profile C", "#d62728")}
    for profile in ("B", "C"):
        label, color = profile_style[profile]
        total_means = [energy_stats[profile][b]["total_energy_kwh"][0] for b in batch_sizes]
        total_stds = [energy_stats[profile][b]["total_energy_kwh"][1] for b in batch_sizes]
        sample_means = [energy_stats[profile][b]["energy_per_sample_mwh"][0] for b in batch_sizes]
        sample_stds = [energy_stats[profile][b]["energy_per_sample_mwh"][1] for b in batch_sizes]
        axes[0].errorbar(batch_sizes, total_means, yerr=total_stds, marker="o", capsize=4, label=label, color=color)
        axes[1].errorbar(batch_sizes, sample_means, yerr=sample_stds, marker="o", capsize=4, label=label, color=color)
    axes[0].set_ylabel("Total GPU Energy (kWh)")
    axes[0].set_title("Profile B/C Energy vs Batch Size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Energy per Sample (mWh)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_profile_bc_gpu_power_timeline(
    plt,
    out_path: str,
    batch: int,
    timeline_b: List[Dict[str, float]],
    timeline_c: List[Dict[str, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for rows, label, color in ((timeline_b, "Profile B", "#9467bd"), (timeline_c, "Profile C", "#d62728")):
        x = [r["elapsed_sec"] for r in rows]
        y = [r.get("gpu_power_w_mean", float("nan")) for r in rows]
        y_std = [r.get("gpu_power_w_std", 0.0) for r in rows]
        plot_metric_with_band(ax, x, y, y_std, label, color)
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("GPU Power (W)")
    ax.set_title(f"Profile B/C GPU Power Timeline (Batch {batch})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_throughput_vs_batch(
    plt, out_path: str, batch_sizes: List[int], throughput_stats: Dict[str, Dict[int, Tuple[float, float]]]
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].errorbar(
        batch_sizes,
        [throughput_stats["steps_per_sec"][b][0] for b in batch_sizes],
        yerr=[throughput_stats["steps_per_sec"][b][1] for b in batch_sizes],
        marker="o",
        capsize=4,
        color="#1f77b4",
        linewidth=1.8,
    )
    axes[1].errorbar(
        batch_sizes,
        [throughput_stats["samples_per_sec"][b][0] for b in batch_sizes],
        yerr=[throughput_stats["samples_per_sec"][b][1] for b in batch_sizes],
        marker="o",
        capsize=4,
        color="#ff7f0e",
        linewidth=1.8,
    )
    axes[0].set_ylabel("Steps / sec")
    axes[0].set_title("Profile C Throughput vs Batch Size")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Samples / sec")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overhead_plot(
    plt, out_path: str, batch_sizes: List[int], overhead_rows: List[Dict[str, object]], title_suffix: str
) -> None:
    rows_by_batch_profile: Dict[Tuple[int, str], Dict[str, object]] = {}
    for row in overhead_rows:
        rows_by_batch_profile[(int(row["batch_size"]), str(row["profile"]))] = row

    fig, ax = plt.subplots(figsize=(10, 5))
    profiles = ["A", "B", "C"]
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    width = 0.24
    x_positions = list(range(len(batch_sizes)))
    for idx, profile in enumerate(profiles):
        vals = []
        errs = []
        offsets = [x + (idx - 1) * width for x in x_positions]
        for batch in batch_sizes:
            row = rows_by_batch_profile.get((batch, profile))
            vals.append(float(row["overhead_pct"]) if row else float("nan"))
            errs.append(float(row["overhead_std_pct"]) if row else 0.0)
        ax.bar(
            offsets,
            vals,
            width=width,
            color=colors[profile],
            alpha=0.85,
            label=f"Profile {profile}",
            yerr=errs,
            capsize=4,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Overhead vs Profile A (%)")
    ax.set_title(f"Instrumentation Overhead by Profile ({title_suffix})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gpu_idle_windows_plot(
    plt, out_path: str, batch_sizes: List[int], idle_windows_by_batch: Dict[int, List[Dict[str, float]]]
) -> None:
    total_idle = []
    avg_idle_util = []
    for batch in batch_sizes:
        windows = idle_windows_by_batch.get(batch, [])
        total_idle.append(sum(float(w.get("duration_sec", 0.0)) for w in windows))
        durations = [float(w.get("duration_sec", 0.0)) for w in windows if float(w.get("duration_sec", 0.0)) > 0]
        if not windows or not durations:
            avg_idle_util.append(float("nan"))
            continue
        weighted_sum = sum(float(w["avg_gpu_util"]) * float(w["duration_sec"]) for w in windows)
        avg_idle_util.append(weighted_sum / sum(durations))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar([str(b) for b in batch_sizes], total_idle, color="#17becf", alpha=0.85)
    axes[0].set_ylabel("Idle Duration (sec)")
    axes[0].set_title("Profile C GPU Idle Window Summary")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar([str(b) for b in batch_sizes], avg_idle_util, color="#bcbd22", alpha=0.85)
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Avg GPU Util During Idle (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def group_successful_runs(runs: List[Dict[str, object]]) -> Dict[Tuple[int, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for run in runs:
        if int(run.get("returncode", 1)) == 0:
            grouped[(int(run["batch"]), str(run["profile"]))].append(run)
    for key in grouped:
        grouped[key].sort(key=lambda r: int(r["repeat"]))
    return grouped


def duration_stats_from_runs(runs: List[Dict[str, object]]) -> Tuple[float, float, int]:
    vals = [float(r["elapsed_sec"]) for r in runs]
    mean, std = mean_std(vals)
    return mean, std, len(vals)


def build_overhead_rows(
    grouped_runs: Dict[Tuple[int, str], List[Dict[str, object]]],
    batch_sizes: List[int],
    excluded_runs: set,
    mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for batch in batch_sizes:
        profile_runs: Dict[str, List[Dict[str, object]]] = {}
        for profile in ("A", "B", "C"):
            selected = list(grouped_runs.get((batch, profile), []))
            if mode == "filtered":
                selected = [
                    r
                    for r in selected
                    if (int(r["batch"]), str(r["profile"]).upper(), int(r["repeat"])) not in excluded_runs
                ]
            profile_runs[profile] = selected

        baseline_mean, baseline_std, baseline_n = duration_stats_from_runs(profile_runs["A"])
        for profile in ("A", "B", "C"):
            mean, std, n = duration_stats_from_runs(profile_runs[profile])
            overhead_pct = float("nan")
            overhead_std_pct = float("nan")
            if baseline_n > 0 and not math.isnan(baseline_mean) and baseline_mean != 0.0 and not math.isnan(mean):
                overhead_pct = (mean - baseline_mean) / baseline_mean * 100.0
                overhead_std_pct = (std / baseline_mean) * 100.0
            rows.append(
                {
                    "mode": mode,
                    "batch_size": batch,
                    "profile": profile,
                    "n": n,
                    "duration_mean_sec": mean,
                    "duration_std_sec": std,
                    "baseline_mean_sec": baseline_mean,
                    "baseline_std_sec": baseline_std,
                    "baseline_n": baseline_n,
                    "overhead_pct": overhead_pct,
                    "overhead_std_pct": overhead_std_pct,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate protocol audit, comparison tables, and plots for an OPT run directory."
    )
    parser.add_argument("experiment_dir", type=str, help="Path to one run directory (contains manifest.json).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots/tables. Defaults to <experiment_dir>/reports/figures.",
    )
    parser.add_argument("--expected-repeats", type=int, default=3, help="Expected number of repeats per batch/profile.")
    parser.add_argument("--target-duration-sec", type=float, default=300.0, help="Target run duration in seconds.")
    parser.add_argument(
        "--duration-tolerance-sec",
        type=float,
        default=15.0,
        help="Tolerance for duration compliance checks in protocol audit.",
    )
    parser.add_argument("--gpu-util-threshold", type=float, default=99.0, help="Threshold for idle window detection.")
    parser.add_argument(
        "--outlier-policy",
        choices=["all", "filtered", "both"],
        default="both",
        help="Which overhead plots to emit.",
    )
    parser.add_argument(
        "--exclude-run",
        action="append",
        default=["16:A:1"],
        help="Run exclusion for filtered overhead mode in the form 'batch:profile:repeat'.",
    )
    parser.add_argument("--validation-tolerance", type=float, default=1e-3, help="Tolerance for report validation checks.")
    parser.add_argument("--skip-plots", action="store_true", help="Generate only JSON/CSV outputs, no PNG files.")
    args = parser.parse_args()

    logger = Logger("PLOT_OPT_REPORT", LogLevel.INFO)
    manifest_path = resolve_manifest_path(os.path.abspath(args.experiment_dir))
    if not manifest_path:
        raise FileNotFoundError(f"manifest.json not found under {args.experiment_dir}")
    exp_dir = os.path.dirname(manifest_path)

    reports_dir = os.path.join(exp_dir, "reports")
    out_dir = args.out_dir.strip() or os.path.join(reports_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Output directory: {out_dir}")

    runs = discover_runs(exp_dir)
    if not runs:
        raise RuntimeError(f"No runs found under {exp_dir}")
    grouped_runs = group_successful_runs(runs)
    batch_sizes = sorted({int(r["batch"]) for r in runs})
    exclusions = parse_exclusions(args.exclude_run)

    protocol_audit: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "experiment_dir": exp_dir,
        "manifest_path": manifest_path,
        "expected_repeats": args.expected_repeats,
        "target_duration_sec": args.target_duration_sec,
        "duration_tolerance_sec": args.duration_tolerance_sec,
        "batch_sizes": batch_sizes,
        "repeat_counts": [],
        "run_durations": [],
        "resource_cadence": [],
        "anomalies": [],
        "validation_against_existing_reports": [],
        "gpu_idle_windows_profileC": {},
    }

    # Protocol audit: repeat counts and per-run duration checks.
    for batch in batch_sizes:
        for profile in ("A", "B", "C"):
            key = (batch, profile)
            runs_for_key = grouped_runs.get(key, [])
            protocol_audit["repeat_counts"].append(
                {
                    "batch_size": batch,
                    "profile": profile,
                    "expected": args.expected_repeats,
                    "actual": len(runs_for_key),
                    "ok": len(runs_for_key) == args.expected_repeats,
                }
            )
            if len(runs_for_key) != args.expected_repeats:
                protocol_audit["anomalies"].append(
                    {
                        "type": "repeat_count_mismatch",
                        "batch_size": batch,
                        "profile": profile,
                        "expected": args.expected_repeats,
                        "actual": len(runs_for_key),
                    }
                )

            durations = []
            for run in runs_for_key:
                elapsed = float(run["elapsed_sec"])
                durations.append(elapsed)
                delta = elapsed - args.target_duration_sec
                within = abs(delta) <= args.duration_tolerance_sec
                protocol_audit["run_durations"].append(
                    {
                        "run_id": run_identifier(run),
                        "batch_size": batch,
                        "profile": profile,
                        "repeat": int(run["repeat"]),
                        "elapsed_sec": elapsed,
                        "delta_from_target_sec": delta,
                        "within_tolerance": within,
                    }
                )
                if not within:
                    protocol_audit["anomalies"].append(
                        {
                            "type": "duration_out_of_tolerance",
                            "run_id": run_identifier(run),
                            "elapsed_sec": elapsed,
                            "target_duration_sec": args.target_duration_sec,
                            "tolerance_sec": args.duration_tolerance_sec,
                        }
                    )

            if durations:
                median_val = statistics.median(durations)
                abs_dev = [abs(v - median_val) for v in durations]
                mad = statistics.median(abs_dev)
                threshold = max(30.0, 6.0 * mad if mad > 0 else 30.0)
                for run in runs_for_key:
                    if abs(float(run["elapsed_sec"]) - median_val) > threshold:
                        protocol_audit["anomalies"].append(
                            {
                                "type": "duration_outlier_within_group",
                                "run_id": run_identifier(run),
                                "group_median_sec": median_val,
                                "group_mad_sec": mad,
                                "threshold_sec": threshold,
                                "elapsed_sec": float(run["elapsed_sec"]),
                            }
                        )

    # Explicit outlier check for filtered overhead default exclusion.
    for run in runs:
        key = (int(run["batch"]), str(run["profile"]).upper(), int(run["repeat"]))
        if key in exclusions and float(run["elapsed_sec"]) > args.target_duration_sec + 120.0:
            protocol_audit["anomalies"].append(
                {
                    "type": "explicit_filtered_overhead_outlier",
                    "run_id": run_identifier(run),
                    "elapsed_sec": float(run["elapsed_sec"]),
                    "reason": "Excluded by configured filtered overhead policy.",
                }
            )

    # Cadence diagnostics (Profiles B and C only).
    for batch in batch_sizes:
        for profile in ("B", "C"):
            for run in grouped_runs.get((batch, profile), []):
                path = str(run["resource_steps_path"])
                if not os.path.isfile(path):
                    protocol_audit["anomalies"].append(
                        {
                            "type": "missing_resource_steps",
                            "run_id": run_identifier(run),
                            "path": path,
                        }
                    )
                    continue
                rows = read_csv_rows(path)
                elapsed = [safe_float(r.get("elapsed_sec")) for r in rows]
                elapsed = [v for v in elapsed if v is not None]
                deltas = [elapsed[i] - elapsed[i - 1] for i in range(1, len(elapsed))]
                record = {
                    "run_id": run_identifier(run),
                    "batch_size": batch,
                    "profile": profile,
                    "samples": len(elapsed),
                    "delta_mean_sec": statistics.mean(deltas) if deltas else float("nan"),
                    "delta_median_sec": statistics.median(deltas) if deltas else float("nan"),
                    "delta_min_sec": min(deltas) if deltas else float("nan"),
                    "delta_p90_sec": percentile(deltas, 0.90) if deltas else float("nan"),
                    "delta_max_sec": max(deltas) if deltas else float("nan"),
                }
                protocol_audit["resource_cadence"].append(record)

    # Build per-batch Profile C aggregated timelines and derived metrics.
    phase_metrics = ["forward_ms", "backward_ms", "optimizer_step_ms", "step_ms", "loss"]
    c_resource_metrics = ["gpu_util", "gpu_mem_used", "proc_cpu_percent", "gpu_power_w"]
    phase_time_stats: Dict[str, Dict[int, Tuple[float, float]]] = {
        "forward_ms": {},
        "backward_ms": {},
        "optimizer_step_ms": {},
    }
    phase_share_stats: Dict[str, Dict[int, Tuple[float, float]]] = {
        "forward_share": {},
        "backward_share": {},
        "optimizer_share": {},
    }
    throughput_stats: Dict[str, Dict[int, Tuple[float, float]]] = {
        "steps_per_sec": {},
        "samples_per_sec": {},
    }
    energy_stats: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = {"B": {}, "C": {}}
    idle_windows_by_batch: Dict[int, List[Dict[str, float]]] = {}
    timeline_profile_c_resource: Dict[int, List[Dict[str, float]]] = {}
    timeline_profile_c_phase: Dict[int, List[Dict[str, float]]] = {}
    timeline_profile_c_loss: Dict[int, List[Dict[str, float]]] = {}
    timeline_profile_b_power: Dict[int, List[Dict[str, float]]] = {}
    timeline_profile_c_power: Dict[int, List[Dict[str, float]]] = {}

    derived_rows: List[Dict[str, object]] = []

    for batch in batch_sizes:
        # Profile C timelines.
        c_runs = grouped_runs.get((batch, "C"), [])
        phase_repeat_series: List[Dict[int, Dict[str, float]]] = []
        resource_repeat_series_c: List[Dict[int, Dict[str, float]]] = []
        resource_repeat_series_b: List[Dict[int, Dict[str, float]]] = []
        metrics_per_repeat: Dict[str, List[float]] = defaultdict(list)
        for run in c_runs:
            simple_steps_path = str(run["simple_steps_path"])
            resource_steps_path = str(run["resource_steps_path"])
            simple_summary_path = str(run["simple_summary_path"])
            resource_summary_path = str(run["resource_summary_path"])
            if not os.path.isfile(simple_steps_path):
                protocol_audit["anomalies"].append(
                    {"type": "missing_simple_steps", "run_id": run_identifier(run), "path": simple_steps_path}
                )
                continue
            if not os.path.isfile(resource_steps_path):
                protocol_audit["anomalies"].append(
                    {"type": "missing_resource_steps", "run_id": run_identifier(run), "path": resource_steps_path}
                )
                continue
            if not os.path.isfile(simple_summary_path):
                protocol_audit["anomalies"].append(
                    {"type": "missing_simple_summary", "run_id": run_identifier(run), "path": simple_summary_path}
                )
                continue
            if not os.path.isfile(resource_summary_path):
                protocol_audit["anomalies"].append(
                    {"type": "missing_resource_summary", "run_id": run_identifier(run), "path": resource_summary_path}
                )
                continue

            simple_steps = read_csv_rows(simple_steps_path)
            resource_steps = read_csv_rows(resource_steps_path)
            phase_repeat_series.append(build_per_second_series(simple_steps, phase_metrics))
            resource_repeat_series_c.append(build_per_second_series(resource_steps, c_resource_metrics))

            simple_summary = read_json(simple_summary_path)
            resource_summary = read_json(resource_summary_path)
            avgs_simple = simple_summary.get("averages_ms", {})
            avgs_resource = resource_summary.get("averages", {})
            iterations = float(simple_summary.get("iterations", 0))
            total_time = float(resource_summary.get("total_training_time_sec", float("nan")))
            total_energy = float(resource_summary.get("cumulative_gpu_energy_kwh", float("nan")))

            step_ms = float(avgs_simple.get("step_ms", float("nan")))
            forward_ms = float(avgs_simple.get("forward_ms", float("nan")))
            backward_ms = float(avgs_simple.get("backward_ms", float("nan")))
            optimizer_ms = float(avgs_simple.get("optimizer_step_ms", float("nan")))

            metrics_per_repeat["step_ms"].append(step_ms)
            metrics_per_repeat["forward_ms"].append(forward_ms)
            metrics_per_repeat["backward_ms"].append(backward_ms)
            metrics_per_repeat["optimizer_step_ms"].append(optimizer_ms)
            metrics_per_repeat["steps_per_sec"].append(iterations / total_time if total_time > 0 else float("nan"))
            metrics_per_repeat["samples_per_sec"].append(
                (iterations * batch) / total_time if total_time > 0 else float("nan")
            )
            metrics_per_repeat["total_energy_kwh"].append(total_energy)
            metrics_per_repeat["energy_per_step_wh"].append(
                (total_energy * 1000.0) / iterations if iterations > 0 else float("nan")
            )
            metrics_per_repeat["energy_per_sample_mwh"].append(
                (total_energy * 1e6) / (iterations * batch) if iterations > 0 and batch > 0 else float("nan")
            )
            metrics_per_repeat["gpu_util_mean"].append(float(avgs_resource.get("gpu_util", float("nan"))))
            metrics_per_repeat["gpu_power_w_mean"].append(float(avgs_resource.get("gpu_power_w", float("nan"))))
            metrics_per_repeat["gpu_mem_used_mib_mean"].append(
                float(avgs_resource.get("gpu_mem_used_mib", float("nan")))
            )
            metrics_per_repeat["proc_cpu_percent_mean"].append(
                float(avgs_resource.get("proc_cpu_percent", float("nan")))
            )
            metrics_per_repeat["forward_share"].append(forward_ms / step_ms if step_ms > 0 else float("nan"))
            metrics_per_repeat["backward_share"].append(backward_ms / step_ms if step_ms > 0 else float("nan"))
            metrics_per_repeat["optimizer_share"].append(optimizer_ms / step_ms if step_ms > 0 else float("nan"))

        if phase_repeat_series:
            phase_agg = aggregate_per_second(
                phase_repeat_series, ["forward_ms", "backward_ms", "optimizer_step_ms", "loss"]
            )
            timeline_profile_c_phase[batch] = phase_agg
            loss_agg = aggregate_per_second(phase_repeat_series, ["loss"])
            timeline_profile_c_loss[batch] = loss_agg
        if resource_repeat_series_c:
            resource_agg = aggregate_per_second(
                resource_repeat_series_c, ["gpu_util", "gpu_mem_used", "proc_cpu_percent", "gpu_power_w"]
            )
            timeline_profile_c_resource[batch] = resource_agg
            idle_windows = detect_gpu_idle_windows(resource_agg, args.gpu_util_threshold)
            idle_windows_by_batch[batch] = idle_windows
            protocol_audit["gpu_idle_windows_profileC"][f"batch_{batch}"] = idle_windows

        # Profile B/C power timeline comparison and energy summary.
        for profile in ("B", "C"):
            profile_runs = grouped_runs.get((batch, profile), [])
            profile_resource_series: List[Dict[int, Dict[str, float]]] = []
            profile_total_energy: List[float] = []
            profile_energy_per_sample: List[float] = []
            for run in profile_runs:
                resource_steps_path = str(run["resource_steps_path"])
                resource_summary_path = str(run["resource_summary_path"])
                if os.path.isfile(resource_steps_path):
                    profile_resource_series.append(
                        build_per_second_series(read_csv_rows(resource_steps_path), ["gpu_power_w"])
                    )
                if os.path.isfile(resource_summary_path):
                    rs = read_json(resource_summary_path)
                    total_energy = float(rs.get("cumulative_gpu_energy_kwh", float("nan")))
                    iterations = float(rs.get("iterations", 0.0))
                    profile_total_energy.append(total_energy)
                    profile_energy_per_sample.append(
                        (total_energy * 1e6) / (iterations * batch) if iterations > 0 else float("nan")
                    )
                else:
                    protocol_audit["anomalies"].append(
                        {
                            "type": "missing_resource_summary",
                            "run_id": run_identifier(run),
                            "path": resource_summary_path,
                        }
                    )
            if profile_resource_series:
                power_agg = aggregate_per_second(profile_resource_series, ["gpu_power_w"])
                if profile == "B":
                    timeline_profile_b_power[batch] = power_agg
                else:
                    timeline_profile_c_power[batch] = power_agg
            energy_stats[profile][batch] = {
                "total_energy_kwh": mean_std(profile_total_energy),
                "energy_per_sample_mwh": mean_std(profile_energy_per_sample),
            }

        # Aggregate derived metrics to single row per batch.
        derived_row = {"batch_size": batch}
        for metric, values in sorted(metrics_per_repeat.items()):
            m, s = mean_std(values)
            derived_row[f"{metric}_mean"] = m
            derived_row[f"{metric}_std"] = s
        derived_rows.append(derived_row)

        # Stats for cross-batch plots.
        phase_time_stats["forward_ms"][batch] = (
            derived_row.get("forward_ms_mean", float("nan")),
            derived_row.get("forward_ms_std", float("nan")),
        )
        phase_time_stats["backward_ms"][batch] = (
            derived_row.get("backward_ms_mean", float("nan")),
            derived_row.get("backward_ms_std", float("nan")),
        )
        phase_time_stats["optimizer_step_ms"][batch] = (
            derived_row.get("optimizer_step_ms_mean", float("nan")),
            derived_row.get("optimizer_step_ms_std", float("nan")),
        )
        phase_share_stats["forward_share"][batch] = (
            derived_row.get("forward_share_mean", float("nan")),
            derived_row.get("forward_share_std", float("nan")),
        )
        phase_share_stats["backward_share"][batch] = (
            derived_row.get("backward_share_mean", float("nan")),
            derived_row.get("backward_share_std", float("nan")),
        )
        phase_share_stats["optimizer_share"][batch] = (
            derived_row.get("optimizer_share_mean", float("nan")),
            derived_row.get("optimizer_share_std", float("nan")),
        )
        throughput_stats["steps_per_sec"][batch] = (
            derived_row.get("steps_per_sec_mean", float("nan")),
            derived_row.get("steps_per_sec_std", float("nan")),
        )
        throughput_stats["samples_per_sec"][batch] = (
            derived_row.get("samples_per_sec_mean", float("nan")),
            derived_row.get("samples_per_sec_std", float("nan")),
        )

    # Build overhead tables.
    overhead_all = build_overhead_rows(grouped_runs, batch_sizes, exclusions, mode="all")
    overhead_filtered = build_overhead_rows(grouped_runs, batch_sizes, exclusions, mode="filtered")
    overhead_rows_combined = overhead_all + overhead_filtered

    # Validate against existing aggregate reports where present.
    for batch in batch_sizes:
        existing_summary_path = os.path.join(reports_dir, f"batch_{batch}", "summary.json")
        if os.path.isfile(existing_summary_path):
            existing_summary = read_json(existing_summary_path)
            profiles = existing_summary.get("profiles", {})
            for profile in ("A", "B", "C"):
                existing_mean = safe_float(
                    profiles.get(profile, {}).get("duration_sec", {}).get("mean")
                )
                runs_for_profile = grouped_runs.get((batch, profile), [])
                computed_mean, _, _ = duration_stats_from_runs(runs_for_profile)
                if existing_mean is not None and not math.isnan(computed_mean):
                    delta = abs(computed_mean - existing_mean)
                    record = {
                        "type": "duration_mean",
                        "batch_size": batch,
                        "profile": profile,
                        "existing": existing_mean,
                        "computed": computed_mean,
                        "abs_delta": delta,
                        "within_tolerance": delta <= args.validation_tolerance,
                    }
                    protocol_audit["validation_against_existing_reports"].append(record)
                    if delta > args.validation_tolerance:
                        protocol_audit["anomalies"].append(
                            {"type": "validation_mismatch", "detail": record}
                        )
        phase_summary_path = os.path.join(
            reports_dir, f"batch_{batch}", "profile_C", "phase_bars_summary.json"
        )
        if os.path.isfile(phase_summary_path):
            existing_phase = read_json(phase_summary_path)
            for metric in ("forward_ms", "backward_ms", "optimizer_step_ms"):
                existing_mean = safe_float(existing_phase.get(metric, {}).get("mean"))
                computed_mean = phase_time_stats[metric].get(batch, (float("nan"), float("nan")))[0]
                if existing_mean is not None and not math.isnan(computed_mean):
                    delta = abs(computed_mean - existing_mean)
                    record = {
                        "type": "phase_mean",
                        "batch_size": batch,
                        "metric": metric,
                        "existing": existing_mean,
                        "computed": computed_mean,
                        "abs_delta": delta,
                        "within_tolerance": delta <= args.validation_tolerance,
                    }
                    protocol_audit["validation_against_existing_reports"].append(record)
                    if delta > args.validation_tolerance:
                        protocol_audit["anomalies"].append(
                            {"type": "validation_mismatch", "detail": record}
                        )

    # Write tables and protocol audit first.
    derived_fields = sorted({k for row in derived_rows for k in row.keys()}, key=lambda x: (x != "batch_size", x))
    write_csv(os.path.join(out_dir, "derived_metrics_summary.csv"), derived_rows, derived_fields)

    overhead_fields = [
        "mode",
        "batch_size",
        "profile",
        "n",
        "duration_mean_sec",
        "duration_std_sec",
        "baseline_mean_sec",
        "baseline_std_sec",
        "baseline_n",
        "overhead_pct",
        "overhead_std_pct",
    ]
    write_csv(os.path.join(out_dir, "overhead_summary_all_vs_filtered.csv"), overhead_rows_combined, overhead_fields)

    with open(os.path.join(out_dir, "protocol_audit.json"), "w", encoding="utf-8") as fp:
        json.dump(protocol_audit, fp, indent=2)

    # Plot outputs.
    if not args.skip_plots:
        plt = ensure_matplotlib()
        for batch in batch_sizes:
            c_resource_rows = timeline_profile_c_resource.get(batch, [])
            if c_resource_rows:
                save_profile_c_resource_timeline(
                    plt,
                    os.path.join(out_dir, f"profileC_timeline_batch_{batch}.png"),
                    c_resource_rows,
                    batch,
                    [
                        ("gpu_util", "GPU Util (%)", "#1f77b4"),
                        ("gpu_mem_used", "GPU Memory Used (MiB)", "#ff7f0e"),
                        ("proc_cpu_percent", "Process CPU (%)", "#2ca02c"),
                    ],
                )
            c_phase_rows = timeline_profile_c_phase.get(batch, [])
            if c_phase_rows:
                save_profile_c_phase_timeline(
                    plt,
                    os.path.join(out_dir, f"profileC_phase_timeline_batch_{batch}.png"),
                    c_phase_rows,
                    batch,
                    [
                        ("forward_ms", "Forward", "#1f77b4"),
                        ("backward_ms", "Backward", "#ff7f0e"),
                        ("optimizer_step_ms", "Optimizer", "#2ca02c"),
                    ],
                )
            c_loss_rows = timeline_profile_c_loss.get(batch, [])
            if c_loss_rows:
                save_profile_c_loss_timeline(
                    plt,
                    os.path.join(out_dir, f"loss_timeline_profileC_batch_{batch}.png"),
                    c_loss_rows,
                    batch,
                )
            b_power_rows = timeline_profile_b_power.get(batch, [])
            c_power_rows = timeline_profile_c_power.get(batch, [])
            if b_power_rows and c_power_rows:
                save_profile_bc_gpu_power_timeline(
                    plt,
                    os.path.join(out_dir, f"profileBC_gpu_power_timeline_batch_{batch}.png"),
                    batch,
                    b_power_rows,
                    c_power_rows,
                )

        save_phase_means_vs_batch(
            plt,
            os.path.join(out_dir, "profileC_phase_means_vs_batch.png"),
            batch_sizes,
            phase_time_stats,
        )
        save_phase_share_vs_batch(
            plt,
            os.path.join(out_dir, "profileC_phase_share_vs_batch.png"),
            batch_sizes,
            phase_share_stats,
        )
        save_profile_bc_energy_vs_batch(
            plt,
            os.path.join(out_dir, "profileBC_energy_vs_batch.png"),
            batch_sizes,
            energy_stats,
        )
        save_throughput_vs_batch(
            plt,
            os.path.join(out_dir, "throughput_vs_batch_profileC.png"),
            batch_sizes,
            throughput_stats,
        )
        save_gpu_idle_windows_plot(
            plt,
            os.path.join(out_dir, "gpu_idle_windows_profileC.png"),
            batch_sizes,
            idle_windows_by_batch,
        )

        if args.outlier_policy in ("all", "both"):
            save_overhead_plot(
                plt,
                os.path.join(out_dir, "overhead_profiles_all_repeats.png"),
                batch_sizes,
                overhead_all,
                "All Repeats",
            )
        if args.outlier_policy in ("filtered", "both"):
            save_overhead_plot(
                plt,
                os.path.join(out_dir, "overhead_profiles_outlier_excluded.png"),
                batch_sizes,
                overhead_filtered,
                "Outlier Excluded",
            )

    logger.info("Completed protocol audit and report generation.")
    logger.info(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
