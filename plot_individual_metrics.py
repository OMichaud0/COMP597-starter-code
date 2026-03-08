import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Get input file from command line argument
if len(sys.argv) < 2:
    print("Usage: python plot_individual_metrics.py <path_to_csv_file>")
    print("Example: python plot_individual_metrics.py results/run-10/run_10_steps.csv")
    sys.exit(1)

csv_file = sys.argv[1]

# Check if file exists
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

# Load the CSV file
df = pd.read_csv(csv_file)
print(f"Loaded: {csv_file}")
print(f"Columns: {df.columns.tolist()}")

# Remove the first 5 rows to eliminate the startup peak
df = df.iloc[5:].reset_index(drop=True)

# Get output directory and create subfolder for plots
base_output_dir = os.path.dirname(csv_file)
csv_basename = os.path.basename(csv_file)
csv_name = os.path.splitext(csv_basename)[0]

# Create 'plots' subfolder within the run directory
output_dir = os.path.join(base_output_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to: {output_dir}\n")

# Dictionary to store plot file paths
plots_created = []

# ============ INDIVIDUAL TIMING PLOTS ============

# 1. Checkpoint Time vs Steps
if 'checkpoint_ms' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['checkpoint_ms'], linewidth=2, color='#d62728')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Checkpoint Time vs Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_01_checkpoint_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 2. Total Step Time vs Steps
if 'step_ms' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['step_ms'], linewidth=2, color='#9467bd')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Total Step Duration vs Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_02_total_step_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 3. Loss vs Steps
if 'loss' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['loss'], linewidth=2, color='#e377c2')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_03_loss.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 4. Forward, Backward, and Optimizer Together (Overlaid Lines)
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    plt.figure(figsize=(14, 7))
    plt.plot(df['step'], df['forward_ms'], linewidth=2.5, label='Forward', color='#1f77b4', alpha=0.85)
    plt.plot(df['step'], df['backward_ms'], linewidth=2.5, label='Backward', color='#ff7f0e', alpha=0.85)
    plt.plot(df['step'], df['optimizer_step_ms'], linewidth=2.5, label='Optimizer', color='#2ca02c', alpha=0.85)
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Forward, Backward, and Optimizer Time vs Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_04_combined_measurements.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# ============ COMPUTED METRICS - INTERESTING ANALYSIS ============

# 5. Backward to Forward Time Ratio
if 'forward_ms' in df.columns and 'backward_ms' in df.columns:
    ratio = df['backward_ms'] / df['forward_ms']
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], ratio, linewidth=2, color='#17becf', label='B/F Ratio')
    plt.axhline(y=ratio.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratio.mean():.2f}')
    plt.fill_between(df['step'], ratio.mean() - ratio.std(), ratio.mean() + ratio.std(), alpha=0.2, color='red', label='±1 Std Dev')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.title('Backward to Forward Time Ratio vs Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_05_backward_forward_ratio.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 6. Time Composition Breakdown vs Steps (Stacked Area)
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    plt.figure(figsize=(14, 7))
    plt.stackplot(df['step'],
                  df['forward_ms'],
                  df['backward_ms'],
                  df['optimizer_step_ms'],
                  labels=['Forward', 'Backward', 'Optimizer'],
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  alpha=0.7)
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Time Composition Breakdown (Stacked Area)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_06_time_composition.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 7. Throughput (Steps per Second)
if 'elapsed_sec' in df.columns and 'step' in df.columns:
    # Calculate throughput as steps per second
    steps_per_second = df['step'] / df['elapsed_sec']
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], steps_per_second, linewidth=2, color='#bcbd22')
    plt.axhline(y=steps_per_second.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {steps_per_second.mean():.2f} steps/sec')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Throughput (steps/sec)', fontsize=12)
    plt.title('Training Throughput vs Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_07_throughput.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 8. Loss Improvement Rate (Loss Gradient)
if 'loss' in df.columns and 'step' in df.columns:
    loss_gradient = df['loss'].diff() / df['step'].diff()
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], loss_gradient, linewidth=2, color='#ff9896', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=loss_gradient.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {loss_gradient.mean():.6f}')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Loss Gradient (per step)', fontsize=12)
    plt.title('Loss Improvement Rate vs Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_08_loss_gradient.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 9. Cumulative Time Analysis
if 'elapsed_sec' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['elapsed_sec'], linewidth=2, color='#8c564b')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Elapsed Time (seconds)', fontsize=12)
    plt.title('Cumulative Elapsed Time vs Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_09_cumulative_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 10. Average Time per Step (Moving Average)
if 'step_ms' in df.columns and 'step' in df.columns:
    window_size = 10
    moving_avg = df['step_ms'].rolling(window=window_size).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['step_ms'], linewidth=1, alpha=0.4, label='Raw Data', color='lightblue')
    plt.plot(df['step'], moving_avg, linewidth=2.5, label=f'{window_size}-Step Moving Average', color='#1f77b4')
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Step Duration with Moving Average', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_10_moving_average.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 11. Component Time Percentage Breakdown
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    forward_pct = (df['forward_ms'] / df['step_ms'] * 100) if 'step_ms' in df.columns else (df['forward_ms'] / (df['forward_ms'] + df['backward_ms'] + df['optimizer_step_ms']) * 100)
    backward_pct = (df['backward_ms'] / df['step_ms'] * 100) if 'step_ms' in df.columns else (df['backward_ms'] / (df['forward_ms'] + df['backward_ms'] + df['optimizer_step_ms']) * 100)
    optimizer_pct = (df['optimizer_step_ms'] / df['step_ms'] * 100) if 'step_ms' in df.columns else (df['optimizer_step_ms'] / (df['forward_ms'] + df['backward_ms'] + df['optimizer_step_ms']) * 100)

    plt.figure(figsize=(14, 7))
    plt.stackplot(df['step'],
                  forward_pct,
                  backward_pct,
                  optimizer_pct,
                  labels=['Forward %', 'Backward %', 'Optimizer %'],
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  alpha=0.7)
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Percentage of Total Step Time (%)', fontsize=12)
    plt.title('Time Percentage Breakdown (Normalized)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_11_percentage_breakdown.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# 12. Elapsed Time vs Step Time (to track efficiency)
if 'elapsed_sec' in df.columns and 'step_ms' in df.columns:
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1 = ax1.plot(df['step'], df['elapsed_sec'], linewidth=2, color='#1f77b4', label='Cumulative Elapsed Time')
    line2 = ax2.plot(df['step'], df['step_ms'], linewidth=2, color='#ff7f0e', label='Step Duration', alpha=0.7)

    ax1.set_xlabel('Step Number', fontsize=12)
    ax1.set_ylabel('Cumulative Elapsed Time (sec)', fontsize=12, color='#1f77b4')
    ax2.set_ylabel('Step Duration (ms)', fontsize=12, color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    plt.title('Cumulative Time vs Step Duration', fontsize=14, fontweight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{csv_name}_12_dual_axis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append(output_file)
    print(f"✓ Created: {output_file}")

# ============ STATISTICS SUMMARY ============
print("\n" + "="*60)
print("STATISTICS SUMMARY")
print("="*60)

print(f"\nDataset Summary:")
print(f"  Total steps: {len(df)}")
print(f"  Step range: {df['step'].min():.0f} to {df['step'].max():.0f}")
print(f"  Time range: {df['elapsed_sec'].min():.2f}s to {df['elapsed_sec'].max():.2f}s")

if 'forward_ms' in df.columns:
    print(f"\nTiming Statistics (mean ± std):")
    print(f"  Forward:     {df['forward_ms'].mean():.2f} ± {df['forward_ms'].std():.2f} ms")
    print(f"  Backward:    {df['backward_ms'].mean():.2f} ± {df['backward_ms'].std():.2f} ms")
    print(f"  Optimizer:   {df['optimizer_step_ms'].mean():.2f} ± {df['optimizer_step_ms'].std():.2f} ms")
    if 'checkpoint_ms' in df.columns:
        print(f"  Checkpoint:  {df['checkpoint_ms'].mean():.2f} ± {df['checkpoint_ms'].std():.2f} ms")

if 'step_ms' in df.columns:
    print(f"  Total Step:  {df['step_ms'].mean():.2f} ± {df['step_ms'].std():.2f} ms")

if 'loss' in df.columns:
    print(f"\nLoss Statistics:")
    print(f"  Initial loss: {df['loss'].iloc[0]:.6f}")
    print(f"  Final loss:   {df['loss'].iloc[-1]:.6f}")
    print(f"  Min loss:     {df['loss'].min():.6f}")
    print(f"  Max loss:     {df['loss'].max():.6f}")
    print(f"  Improvement:  {((df['loss'].iloc[0] - df['loss'].iloc[-1]) / df['loss'].iloc[0] * 100):.2f}%")

if 'elapsed_sec' in df.columns and 'step' in df.columns:
    throughput = len(df) / df['elapsed_sec'].iloc[-1]
    print(f"\nThroughput:")
    print(f"  Average steps/second: {throughput:.2f}")
    print(f"  Average step duration: {(df['elapsed_sec'].iloc[-1] / len(df) * 1000):.2f} ms")

# Time breakdown percentages
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    total_time = df['forward_ms'].sum() + df['backward_ms'].sum() + df['optimizer_step_ms'].sum()
    print(f"\nTime Breakdown (Overall):")
    print(f"  Forward:   {(df['forward_ms'].sum() / total_time * 100):.1f}%")
    print(f"  Backward:  {(df['backward_ms'].sum() / total_time * 100):.1f}%")
    print(f"  Optimizer: {(df['optimizer_step_ms'].sum() / total_time * 100):.1f}%")

print("\n" + "="*60)
print(f"✅ Created {len(plots_created)} individual metric plots")
print("="*60)
