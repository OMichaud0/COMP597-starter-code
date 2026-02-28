import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('results/run-10/run_10_steps.csv')

# Remove the first 5 rows to eliminate the startup peak
df = df.iloc[5:].reset_index(drop=True)

print(f"Data shape after removing initial peak: {df.shape}")
print(f"Step range: {df['step'].min()} to {df['step'].max()}")
print(f"Time range: {df['elapsed_sec'].min():.2f}s to {df['elapsed_sec'].max():.2f}s")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Forward, Backward, Optimizer vs Steps
ax1 = plt.subplot(2, 3, 1)
ax1.plot(df['step'], df['forward_ms'], label='Forward', linewidth=1.5, alpha=0.8)
ax1.plot(df['step'], df['backward_ms'], label='Backward', linewidth=1.5, alpha=0.8)
ax1.plot(df['step'], df['optimizer_step_ms'], label='Optimizer', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Step Number')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Forward, Backward, and Optimizer Time vs Steps')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Forward, Backward, Optimizer vs Elapsed Time
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df['elapsed_sec'], df['forward_ms'], label='Forward', linewidth=1.5, alpha=0.8)
ax2.plot(df['elapsed_sec'], df['backward_ms'], label='Backward', linewidth=1.5, alpha=0.8)
ax2.plot(df['elapsed_sec'], df['optimizer_step_ms'], label='Optimizer', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Elapsed Time (seconds)')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Forward, Backward, and Optimizer Time vs Elapsed Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Total step time vs steps
ax3 = plt.subplot(2, 3, 3)
ax3.plot(df['step'], df['step_ms'], label='Total Step Time', linewidth=1.5, color='purple')
ax3.set_xlabel('Step Number')
ax3.set_ylabel('Time (ms)')
ax3.set_title('Total Step Duration vs Steps')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Loss vs Steps
ax4 = plt.subplot(2, 3, 4)
ax4.plot(df['step'], df['loss'], label='Loss', linewidth=1.5, color='red')
ax4.set_xlabel('Step Number')
ax4.set_ylabel('Loss')
ax4.set_title('Training Loss vs Steps')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Stacked area chart - time breakdown vs steps
ax5 = plt.subplot(2, 3, 5)
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
ax6 = plt.subplot(2, 3, 6)
ratio = df['backward_ms'] / df['forward_ms']
ax6.plot(df['step'], ratio, label='Backward/Forward Ratio', linewidth=1.5, color='green')
ax6.axhline(y=ratio.mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean: {ratio.mean():.2f}')
ax6.set_xlabel('Step Number')
ax6.set_ylabel('Ratio')
ax6.set_title('Backward to Forward Time Ratio')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/run-10/run_10_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to: results/run-10/run_10_analysis.png")

# Print some statistics
print("\n=== Timing Statistics ===")
print(f"Forward (ms):    mean={df['forward_ms'].mean():.2f}, std={df['forward_ms'].std():.2f}")
print(f"Backward (ms):   mean={df['backward_ms'].mean():.2f}, std={df['backward_ms'].std():.2f}")
print(f"Optimizer (ms):  mean={df['optimizer_step_ms'].mean():.2f}, std={df['optimizer_step_ms'].std():.2f}")
print(f"Step Total (ms): mean={df['step_ms'].mean():.2f}, std={df['step_ms'].std():.2f}")
print(f"Loss:            min={df['loss'].min():.6f}, max={df['loss'].max():.6f}, final={df['loss'].iloc[-1]:.6f}")

# Calculate breakdown percentages
total_component_time = df['forward_ms'] + df['backward_ms'] + df['optimizer_step_ms']
print(f"\nAverage time breakdown:")
print(f"  Forward:   {(df['forward_ms'].mean() / total_component_time.mean() * 100):.1f}%")
print(f"  Backward:  {(df['backward_ms'].mean() / total_component_time.mean() * 100):.1f}%")
print(f"  Optimizer: {(df['optimizer_step_ms'].mean() / total_component_time.mean() * 100):.1f}%")
