#!/usr/bin/env python3
"""Plot trainer step metrics over time using fixed-size minute bins.

Example:
  python3 scripts/plot_metrics_by_time.py results/run5_steps.csv --bin-minutes 10
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from statistics import mean


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "metric"


def _try_float(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics with time binned in minute increments.")
    parser.add_argument("csv_path", help="Path to steps CSV (must contain elapsed_sec).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--bin-minutes",
        type=int,
        default=10,
        help="Time bin size in minutes (default: 10).",
    )
    return parser.parse_args()


def read_and_bin(csv_path: str, bin_minutes: int):
    with open(csv_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")
        if "elapsed_sec" not in reader.fieldnames:
            raise ValueError(f"'elapsed_sec' column is required in {csv_path}")

        numeric_fields = [c for c in reader.fieldnames if c != "elapsed_sec"]
        bins = defaultdict(lambda: defaultdict(list))

        for row in reader:
            elapsed_sec = _try_float(row.get("elapsed_sec", ""))
            if elapsed_sec is None:
                continue
            elapsed_min = elapsed_sec / 60.0
            bin_start_min = int(math.floor(elapsed_min / bin_minutes) * bin_minutes)

            for field in numeric_fields:
                val = _try_float(row.get(field, ""))
                if val is not None:
                    bins[bin_start_min][field].append(val)

    return bins, numeric_fields


def aggregate_bins(bins, fields):
    bin_starts = sorted(bins.keys())
    aggregated = []
    for b in bin_starts:
        row = {"time_bin_start_min": b}
        for field in fields:
            values = bins[b].get(field, [])
            row[field] = mean(values) if values else None
        aggregated.append(row)
    return aggregated


def write_binned_csv(path: str, aggregated_rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["time_bin_start_min"] + fields)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow(row)


def plot_metrics(aggregated_rows, fields, plot_dir: str, bin_minutes: int):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to render plots. Install it in your environment."
        ) from exc

    os.makedirs(plot_dir, exist_ok=True)

    x = [row["time_bin_start_min"] for row in aggregated_rows]

    # One PNG per metric.
    for field in fields:
        y = [row[field] for row in aggregated_rows]
        xy = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if len(xy) < 2:
            continue

        px, py = zip(*xy)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(px, py, marker="o", linewidth=1.5, markersize=3)
        ax.set_title(f"{field} vs time ({bin_minutes}-minute bins)")
        ax.set_xlabel("Time since start (minutes)")
        ax.set_ylabel(field)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{_safe_name(field)}_by_time.png"), dpi=150)
        plt.close(fig)

    # Combined overview plot for a useful subset.
    preferred = [
        "step_duration_ms",
        "gpu_util",
        "gpu_power_w",
        "gpu_mem_used",
        "cumulative_gpu_energy_kwh",
        "cumulative_carbon_gco2",
        "sys_cpu_percent",
        "proc_cpu_percent",
        "proc_rss",
        "sys_mem_used",
        "io_write",
        "io_read",
    ]
    selected = [f for f in preferred if f in fields]
    if not selected:
        selected = fields[:12]

    n = len(selected)
    if n == 0:
        return
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx >= n:
                ax.axis("off")
                continue
            field = selected[idx]
            y = [row[field] for row in aggregated_rows]
            xy = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if len(xy) >= 2:
                px, py = zip(*xy)
                ax.plot(px, py, marker="o", linewidth=1.2, markersize=2.8)
            ax.set_title(field, fontsize=10)
            ax.set_xlabel("Minutes", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.35)
            idx += 1

    fig.suptitle(f"Metrics over time ({bin_minutes}-minute bins)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(os.path.join(plot_dir, "overview_metrics_by_time.png"), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if args.bin_minutes <= 0:
        raise ValueError("--bin-minutes must be > 0")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(csv_path)) or "."
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    bins, fields = read_and_bin(csv_path, args.bin_minutes)
    if not bins:
        raise RuntimeError("No valid rows found with numeric elapsed_sec and metric values.")
    aggregated = aggregate_bins(bins, fields)

    binned_csv = os.path.join(output_dir, f"{stem}_binned_{args.bin_minutes}min.csv")
    plot_dir = os.path.join(output_dir, f"{stem}_plots_{args.bin_minutes}min")

    write_binned_csv(binned_csv, aggregated, fields)
    plot_metrics(aggregated, fields, plot_dir, args.bin_minutes)

    print(f"Binned CSV written to: {binned_csv}")
    print(f"Plots written to: {plot_dir}")


if __name__ == "__main__":
    main()

