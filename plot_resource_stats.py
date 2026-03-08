import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.ndimage import uniform_filter1d

# Get input file from command line argument
if len(sys.argv) < 2:
    print("Usage: python plot_resource_stats.py <path_to_csv_file>")
    print("Example: python plot_resource_stats.py results/run-11/run_11_steps.csv")
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

print("\n=== Statistics ===")
print(f"Total rows: {len(df)} (after removing first 5)")
print(f"Step range: {df['step'].min()} to {df['step'].max()}")
print(f"Time range: {df['elapsed_sec'].min():.2f}s to {df['elapsed_sec'].max():.2f}s")

# Generate output directory and filenames
output_dir = os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.'
csv_basename = os.path.basename(csv_file)
csv_name = os.path.splitext(csv_basename)[0]
combined_output_file = os.path.join(output_dir, f"{csv_name}_analysis.png")

# Create individual plots directory
plots_dir = os.path.join(output_dir, f"{csv_name}_plots")
os.makedirs(plots_dir, exist_ok=True)

# Function to save individual plot
def save_plot(plot_func, filename_suffix):
    """Create and save an individual plot"""
    fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
    plot_func(ax_ind)
    plt.tight_layout()
    filepath = os.path.join(plots_dir, filename_suffix)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig_ind)
    return filepath

# Function to smooth data using a rolling window
def smooth_data(data, window_size=10):
    """Smooth data using uniform filter"""
    return uniform_filter1d(data, size=window_size, mode='nearest')

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 14))
subplot_count = 0

# 1. Energy per step vs Steps
if 'step_energy_kwh' in df.columns:
    subplot_count += 1
    ax1 = plt.subplot(3, 4, subplot_count)
    energy_smooth = smooth_data(df['step_energy_kwh'].values, window_size=10)
    ax1.plot(df['step'], energy_smooth*1e6, label='Energy per Step (Smoothed)', linewidth=1.5, color='orange')
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('Energy (μWh)')
    ax1.set_title('Energy per Step vs Steps')
    # Set y-axis with some margin above the max value
    y_max = (energy_smooth*1e6).max()
    ax1.set_ylim(0, y_max * 1.15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Save individual plot
    save_plot(
        lambda ax: (
            ax.plot(df['step'], energy_smooth*1e6, label='Energy per Step (Smoothed)', linewidth=1.5, color='orange'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('Energy (μWh)'),
            ax.set_title('Energy per Step vs Steps'),
            ax.set_ylim(0, y_max * 1.15),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'energy_per_step_vs_steps.png'
    )

# 2. Energy per step vs Elapsed Time
if 'step_energy_kwh' in df.columns:
    subplot_count += 1
    ax2 = plt.subplot(3, 4, subplot_count)
    energy_smooth = smooth_data(df['step_energy_kwh'].values, window_size=10)
    ax2.plot(df['elapsed_sec'], energy_smooth*1e6, label='Energy per Step (Smoothed)', linewidth=1.5, color='orange')
    ax2.set_xlabel('Elapsed Time (seconds)')
    ax2.set_ylabel('Energy (μWh)')
    ax2.set_title('Energy per Step vs Elapsed Time')
    # Set y-axis with some margin above the max value
    y_max = (energy_smooth*1e6).max()
    ax2.set_ylim(0, y_max * 1.15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['elapsed_sec'], energy_smooth*1e6, label='Energy per Step (Smoothed)', linewidth=1.5, color='orange'),
            ax.set_xlabel('Elapsed Time (seconds)'),
            ax.set_ylabel('Energy (μWh)'),
            ax.set_title('Energy per Step vs Elapsed Time'),
            ax.set_ylim(0, y_max * 1.15),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'energy_per_step_vs_elapsed_time.png'
    )

# 3. Cumulative Energy
if 'cumulative_gpu_energy_kwh' in df.columns:
    subplot_count += 1
    ax3 = plt.subplot(3, 4, subplot_count)
    ax3.plot(df['step'], df['cumulative_gpu_energy_kwh'], label='Cumulative Energy', linewidth=1.5, color='darkgreen')
    ax3.set_xlabel('Step Number')
    ax3.set_ylabel('Cumulative Energy (kWh)')
    ax3.set_title('Cumulative Energy vs Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], df['cumulative_gpu_energy_kwh'], label='Cumulative Energy', linewidth=1.5, color='darkgreen'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('Cumulative Energy (kWh)'),
            ax.set_title('Cumulative Energy vs Steps'),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'cumulative_energy_vs_steps.png'
    )

# 4. GPU Power
if 'gpu_power_w' in df.columns:
    subplot_count += 1
    ax4 = plt.subplot(3, 4, subplot_count)
    gpu_power_smooth = smooth_data(df['gpu_power_w'].values, window_size=15)
    ax4.plot(df['step'], gpu_power_smooth, label='GPU Power (Smoothed)', linewidth=1.5, color='cyan')
    ax4.set_xlabel('Step Number')
    ax4.set_ylabel('Power (W)')
    ax4.set_title('GPU Power vs Steps')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], gpu_power_smooth, label='GPU Power (Smoothed)', linewidth=1.5, color='cyan'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('Power (W)'),
            ax.set_title('GPU Power vs Steps'),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'gpu_power_vs_steps.png'
    )

# 5. GPU Power vs Elapsed Time
if 'gpu_power_w' in df.columns:
    subplot_count += 1
    ax5 = plt.subplot(3, 4, subplot_count)
    gpu_power_smooth = smooth_data(df['gpu_power_w'].values, window_size=15)
    ax5.plot(df['elapsed_sec'], gpu_power_smooth, label='GPU Power (Smoothed)', linewidth=1.5, color='cyan')
    ax5.set_xlabel('Elapsed Time (seconds)')
    ax5.set_ylabel('Power (W)')
    ax5.set_title('GPU Power vs Elapsed Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['elapsed_sec'], gpu_power_smooth, label='GPU Power (Smoothed)', linewidth=1.5, color='cyan'),
            ax.set_xlabel('Elapsed Time (seconds)'),
            ax.set_ylabel('Power (W)'),
            ax.set_title('GPU Power vs Elapsed Time'),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'gpu_power_vs_elapsed_time.png'
    )

# 6. GPU Utilization
if 'gpu_util' in df.columns:
    subplot_count += 1
    ax6 = plt.subplot(3, 4, subplot_count)
    ax6.plot(df['step'], df['gpu_util'], label='GPU Utilization', linewidth=1.5, color='magenta')
    ax6.set_xlabel('Step Number')
    ax6.set_ylabel('GPU Utilization (%)')
    ax6.set_title('GPU Utilization vs Steps')
    ax6.set_ylim(0, 100)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], df['gpu_util'], label='GPU Utilization', linewidth=1.5, color='magenta'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('GPU Utilization (%)'),
            ax.set_title('GPU Utilization vs Steps'),
            ax.set_ylim(0, 100),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'gpu_utilization_vs_steps.png'
    )

# 7. GPU Memory Usage
if 'gpu_mem_used' in df.columns:
    subplot_count += 1
    ax7 = plt.subplot(3, 4, subplot_count)
    ax7.plot(df['step'], df['gpu_mem_used'], label='GPU Memory Used', linewidth=1.5, color='blue')
    ax7.set_xlabel('Step Number')
    ax7.set_ylabel('Memory (MiB)')
    ax7.set_title('GPU Memory Usage vs Steps')
    # Zoom in: set y-axis range to 29000-30000
    ax7.set_ylim(29000, 30000)
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], df['gpu_mem_used'], label='GPU Memory Used', linewidth=1.5, color='blue'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('Memory (MiB)'),
            ax.set_title('GPU Memory Usage vs Steps'),
            ax.set_ylim(29000, 30000),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'gpu_memory_vs_steps.png'
    )

# 8. System Memory Usage
if 'sys_mem_used' in df.columns:
    subplot_count += 1
    ax8 = plt.subplot(3, 4, subplot_count)
    sys_mem_percent = (df['sys_mem_used'] / df['sys_mem_total'].iloc[0]) * 100
    ax8.plot(df['step'], sys_mem_percent, label='System Memory %', linewidth=1.5, color='brown')
    ax8.set_xlabel('Step Number')
    ax8.set_ylabel('System Memory (%)')
    ax8.set_title('System Memory Usage vs Steps')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], sys_mem_percent, label='System Memory %', linewidth=1.5, color='brown'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('System Memory (%)'),
            ax.set_title('System Memory Usage vs Steps'),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'system_memory_vs_steps.png'
    )

# 9. Step Duration
if 'step_duration_ms' in df.columns:
    subplot_count += 1
    ax9 = plt.subplot(3, 4, subplot_count)
    ax9.plot(df['step'], df['step_duration_ms'], label='Step Duration', linewidth=1.5, color='purple')
    ax9.set_xlabel('Step Number')
    ax9.set_ylabel('Duration (ms)')
    ax9.set_title('Step Duration vs Steps')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], df['step_duration_ms'], label='Step Duration', linewidth=1.5, color='purple'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('Duration (ms)'),
            ax.set_title('Step Duration vs Steps'),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'step_duration_vs_steps.png'
    )

# 10. Cumulative Carbon Emissions
if 'cumulative_carbon_gco2' in df.columns:
    subplot_count += 1
    ax10 = plt.subplot(3, 4, subplot_count)
    ax10.plot(df['step'], df['cumulative_carbon_gco2'], label='Cumulative Carbon', linewidth=1.5, color='darkred')
    ax10.set_xlabel('Step Number')
    ax10.set_ylabel('Carbon Emissions (gCO2e)')
    ax10.set_title('Cumulative Carbon Emissions vs Steps')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], df['cumulative_carbon_gco2'], label='Cumulative Carbon', linewidth=1.5, color='darkred'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('Carbon Emissions (gCO2e)'),
            ax.set_title('Cumulative Carbon Emissions vs Steps'),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'cumulative_carbon_vs_steps.png'
    )

# 11. GPU Memory % Utilization
if 'gpu_mem_used' in df.columns and 'gpu_mem_total' in df.columns:
    subplot_count += 1
    ax11 = plt.subplot(3, 4, subplot_count)
    gpu_mem_percent = (df['gpu_mem_used'] / df['gpu_mem_total']) * 100
    ax11.plot(df['step'], gpu_mem_percent, label='GPU Memory %', linewidth=1.5, color='navy')
    ax11.set_xlabel('Step Number')
    ax11.set_ylabel('GPU Memory (%)')
    ax11.set_title('GPU Memory % Utilization vs Steps')
    ax11.margins(y=0.1)  # Add 10% margin on top/bottom for better visibility
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], gpu_mem_percent, label='GPU Memory %', linewidth=1.5, color='navy'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('GPU Memory (%)'),
            ax.set_title('GPU Memory % Utilization vs Steps'),
            ax.margins(y=0.1),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'gpu_memory_percent_vs_steps.png'
    )

# 12. Process CPU Percent
if 'proc_cpu_percent' in df.columns:
    subplot_count += 1
    ax12 = plt.subplot(3, 4, subplot_count)
    ax12.plot(df['step'], df['proc_cpu_percent'], label='Process CPU %', linewidth=1.5, color='lime')
    ax12.set_xlabel('Step Number')
    ax12.set_ylabel('CPU Utilization (%)')
    ax12.set_title('Process CPU Utilization vs Steps')
    ax12.set_ylim(0, 100)
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    save_plot(
        lambda ax: (
            ax.plot(df['step'], df['proc_cpu_percent'], label='Process CPU %', linewidth=1.5, color='lime'),
            ax.set_xlabel('Step Number'),
            ax.set_ylabel('CPU Utilization (%)'),
            ax.set_title('Process CPU Utilization vs Steps'),
            ax.set_ylim(0, 100),
            ax.legend(),
            ax.grid(True, alpha=0.3)
        ),
        'process_cpu_percent_vs_steps.png'
    )

plt.tight_layout()
plt.savefig(combined_output_file, dpi=300, bbox_inches='tight')
print(f"\nCombined plot saved to: {combined_output_file}")
print(f"Individual plots saved to: {plots_dir}/")
plt.close(fig)

# Print energy statistics if available
if 'step_energy_kwh' in df.columns:
    print("\n=== Energy Statistics ===")
    print(f"Step Energy (kWh):       mean={df['step_energy_kwh'].mean():.8f}, std={df['step_energy_kwh'].std():.8f}")
    print(f"Step Energy (μWh):       mean={df['step_energy_kwh'].mean()*1e6:.4f}, std={df['step_energy_kwh'].std()*1e6:.4f}")
    if 'cumulative_gpu_energy_kwh' in df.columns:
        print(f"Cumulative Energy (kWh): {df['cumulative_gpu_energy_kwh'].iloc[-1]:.4f}")
    if 'cumulative_carbon_gco2' in df.columns:
        print(f"Cumulative Carbon (gCO2e): {df['cumulative_carbon_gco2'].iloc[-1]:.4f}")
    if 'gpu_power_w' in df.columns:
        print(f"GPU Power (W):           mean={df['gpu_power_w'].mean():.2f}, std={df['gpu_power_w'].std():.2f}, max={df['gpu_power_w'].max():.2f}")
    if 'gpu_util' in df.columns:
        print(f"GPU Util (%):            mean={df['gpu_util'].mean():.1f}, std={df['gpu_util'].std():.1f}, max={df['gpu_util'].max():.1f}")

if 'step_duration_ms' in df.columns:
    print("\n=== Performance Statistics ===")
    print(f"Step Duration (ms):      mean={df['step_duration_ms'].mean():.2f}, std={df['step_duration_ms'].std():.2f}, min={df['step_duration_ms'].min():.2f}, max={df['step_duration_ms'].max():.2f}")

if 'gpu_mem_used' in df.columns:
    print("\n=== Memory Statistics ===")
    print(f"GPU Memory (MiB):        mean={df['gpu_mem_used'].mean():.1f}, std={df['gpu_mem_used'].std():.1f}, max={df['gpu_mem_used'].max():.1f}")
    if 'gpu_mem_total' in df.columns:
        gpu_mem_percent = (df['gpu_mem_used'] / df['gpu_mem_total']) * 100
        print(f"GPU Memory %:            mean={gpu_mem_percent.mean():.1f}%, max={gpu_mem_percent.max():.1f}%")
    if 'sys_mem_used' in df.columns and 'sys_mem_total' in df.columns:
        sys_mem_percent = (df['sys_mem_used'] / df['sys_mem_total'].iloc[0]) * 100
        print(f"System Memory %:         mean={sys_mem_percent.mean():.1f}%, max={sys_mem_percent.max():.1f}%")

if 'proc_cpu_percent' in df.columns:
    print(f"\n=== CPU Statistics ===")
    print(f"Process CPU %:           mean={df['proc_cpu_percent'].mean():.1f}, std={df['proc_cpu_percent'].std():.1f}, max={df['proc_cpu_percent'].max():.1f}")
    if 'sys_cpu_percent' in df.columns:
        print(f"System CPU %:            mean={df['sys_cpu_percent'].mean():.1f}, std={df['sys_cpu_percent'].std():.1f}, max={df['sys_cpu_percent'].max():.1f}")
