import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Get input file from command line argument
if len(sys.argv) < 2:
    print("Usage: python plot_run10.py <path_to_csv_file>")
    print("Example: python plot_run10.py results/run-10/run_10_steps.csv")
    sys.exit(1)

csv_file = sys.argv[1]

# Check if file exists
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

# Load the CSV file
df = pd.read_csv(csv_file)
print(f"Loaded: {csv_file}")

# Remove the first 5 rows to eliminate the startup peak
df = df.iloc[5:].reset_index(drop=True)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
subplot_count = 0

# 1. Forward, Backward, Optimizer vs Steps
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    subplot_count += 1
    ax1 = plt.subplot(2, 3, subplot_count)
    ax1.plot(df['step'], df['forward_ms'], label='Forward', linewidth=1.5, alpha=0.8)
    ax1.plot(df['step'], df['backward_ms'], label='Backward', linewidth=1.5, alpha=0.8)
    ax1.plot(df['step'], df['optimizer_step_ms'], label='Optimizer', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Forward, Backward, and Optimizer Time vs Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# 2. Forward, Backward, Optimizer vs Elapsed Time
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    subplot_count += 1
    ax2 = plt.subplot(2, 3, subplot_count)
    ax2.plot(df['elapsed_sec'], df['forward_ms'], label='Forward', linewidth=1.5, alpha=0.8)
    ax2.plot(df['elapsed_sec'], df['backward_ms'], label='Backward', linewidth=1.5, alpha=0.8)
    ax2.plot(df['elapsed_sec'], df['optimizer_step_ms'], label='Optimizer', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Elapsed Time (seconds)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Forward, Backward, and Optimizer Time vs Elapsed Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Total step time vs steps
if 'step_ms' in df.columns or 'step_duration_ms' in df.columns:
    subplot_count += 1
    ax3 = plt.subplot(2, 3, subplot_count)
    col_name = 'step_ms' if 'step_ms' in df.columns else 'step_duration_ms'
    ax3.plot(df['step'], df[col_name], label='Total Step Time', linewidth=1.5, color='purple')
    ax3.set_xlabel('Step Number')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Total Step Duration vs Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 4. Loss vs Steps
if 'loss' in df.columns:
    subplot_count += 1
    ax4 = plt.subplot(2, 3, subplot_count)
    ax4.plot(df['step'], df['loss'], label='Loss', linewidth=1.5, color='red')
    ax4.set_xlabel('Step Number')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss vs Steps')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. Stacked area chart - time breakdown vs steps
if 'forward_ms' in df.columns and 'backward_ms' in df.columns and 'optimizer_step_ms' in df.columns:
    subplot_count += 1
    ax5 = plt.subplot(2, 3, subplot_count)
    ax5.stackplot(df['step'],
                  df['forward_ms'],
                  df['backward_ms'],
                  df['optimizer_step_ms'],
                  labels=['Forward', 'Backward', 'Optimizer'],
                  alpha=0.7)
    ax5.set_xlabel('Step Number')
    ax5.set_ylabel('Time (ms)')
    ax5.set_title('Time Composition by Component (Stacked)')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)

# 6. Backward to Forward ratio vs steps
if 'forward_ms' in df.columns and 'backward_ms' in df.columns:
    subplot_count += 1
    ax6 = plt.subplot(2, 3, subplot_count)
    ratio = df['backward_ms'] / df['forward_ms']
    ax6.plot(df['step'], ratio, label='Backward/Forward Ratio', linewidth=1.5, color='green')
    ax6.axhline(y=ratio.mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean: {ratio.mean():.2f}')
    ax6.set_xlabel('Step Number')
    ax6.set_ylabel('Ratio')
    ax6.set_title('Backward to Forward Time Ratio')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

# 7. Energy per step
if 'step_energy_kwh' in df.columns:
    subplot_count += 1
    ax7 = plt.subplot(2, 3, subplot_count)
    ax7.plot(df['step'], df['step_energy_kwh']*1e6, label='Energy per Step', linewidth=1.5, color='orange')
    ax7.set_xlabel('Step Number')
    ax7.set_ylabel('Energy (μWh)')
    ax7.set_title('Energy per Step vs Steps')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

# 8. GPU Power
if 'gpu_power_w' in df.columns:
    subplot_count += 1
    ax8 = plt.subplot(2, 3, subplot_count)
    ax8.plot(df['step'], df['gpu_power_w'], label='GPU Power', linewidth=1.5, color='cyan')
    ax8.set_xlabel('Step Number')
    ax8.set_ylabel('Power (W)')
    ax8.set_title('GPU Power vs Steps')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

# 9. GPU Utilization
if 'gpu_util' in df.columns:
    subplot_count += 1
    ax9 = plt.subplot(2, 3, subplot_count)
    ax9.plot(df['step'], df['gpu_util'], label='GPU Utilization', linewidth=1.5, color='magenta')
    ax9.set_xlabel('Step Number')
    ax9.set_ylabel('GPU Utilization (%)')
    ax9.set_title('GPU Utilization vs Steps')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

# 10. GPU Memory
if 'gpu_mem_used' in df.columns and 'gpu_mem_total' in df.columns:
    subplot_count += 1
    ax10 = plt.subplot(2, 3, subplot_count)
    ax10.plot(df['step'], df['gpu_mem_used'], label='GPU Memory Used', linewidth=1.5, color='blue')
    ax10.set_xlabel('Step Number')
    ax10.set_ylabel('Memory (MiB)')
    ax10.set_title('GPU Memory Usage vs Steps')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

# 11. Cumulative Energy
if 'cumulative_gpu_energy_kwh' in df.columns:
    subplot_count += 1
    ax11 = plt.subplot(2, 3, subplot_count)
    ax11.plot(df['step'], df['cumulative_gpu_energy_kwh'], label='Cumulative Energy', linewidth=1.5, color='darkgreen')
    ax11.set_xlabel('Step Number')
    ax11.set_ylabel('Cumulative Energy (kWh)')
    ax11.set_title('Cumulative Energy vs Steps')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

plt.tight_layout()

# Generate output filename based on input file
output_dir = os.path.dirname(csv_file)
csv_basename = os.path.basename(csv_file)
csv_name = os.path.splitext(csv_basename)[0]
output_file = os.path.join(output_dir, f"{csv_name}_analysis.png")

plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Print some statistics
print("\n=== Statistics ===")
print(f"Total rows: {len(df)} (after removing first 5)")
print(f"Step range: {df['step'].min()} to {df['step'].max()}")
print(f"Time range: {df['elapsed_sec'].min():.2f}s to {df['elapsed_sec'].max():.2f}s")

# Print timing statistics if available
if 'forward_ms' in df.columns:
    print("\n=== Timing Statistics ===")
    print(f"Forward (ms):    mean={df['forward_ms'].mean():.2f}, std={df['forward_ms'].std():.2f}")
    print(f"Backward (ms):   mean={df['backward_ms'].mean():.2f}, std={df['backward_ms'].std():.2f}")
    print(f"Optimizer (ms):  mean={df['optimizer_step_ms'].mean():.2f}, std={df['optimizer_step_ms'].std():.2f}")

    if 'step_ms' in df.columns:
        print(f"Step Total (ms): mean={df['step_ms'].mean():.2f}, std={df['step_ms'].std():.2f}")

    if 'loss' in df.columns:
        print(f"Loss:            min={df['loss'].min():.6f}, max={df['loss'].max():.6f}, final={df['loss'].iloc[-1]:.6f}")

    # Calculate breakdown percentages
    total_component_time = df['forward_ms'] + df['backward_ms'] + df['optimizer_step_ms']
    print(f"\nAverage time breakdown:")
    print(f"  Forward:   {(df['forward_ms'].mean() / total_component_time.mean() * 100):.1f}%")
    print(f"  Backward:  {(df['backward_ms'].mean() / total_component_time.mean() * 100):.1f}%")
    print(f"  Optimizer: {(df['optimizer_step_ms'].mean() / total_component_time.mean() * 100):.1f}%")

# Print energy statistics if available
if 'step_energy_kwh' in df.columns:
    print("\n=== Energy Statistics ===")
    print(f"Step Energy (kWh):       mean={df['step_energy_kwh'].mean():.8f}, std={df['step_energy_kwh'].std():.8f}")
    print(f"Cumulative Energy (kWh): {df['cumulative_gpu_energy_kwh'].iloc[-1]:.4f}")
    print(f"Cumulative Carbon (gCO2e): {df['cumulative_carbon_gco2'].iloc[-1]:.4f}")
    if 'gpu_power_w' in df.columns:
        print(f"GPU Power (W):           mean={df['gpu_power_w'].mean():.2f}, std={df['gpu_power_w'].std():.2f}")
    if 'gpu_util' in df.columns:
        print(f"GPU Util (%):            mean={df['gpu_util'].mean():.1f}, std={df['gpu_util'].std():.1f}")
