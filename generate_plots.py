"""
Generate trajectory and error plots for IO-VNBD controlled DR benchmark.
Saves plots to Data/ directory.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# Load results
df = pd.read_csv('Data/IOVNBD_controlled_AI_DR.csv')
with open('Data/IOVNBD_controlled_AI_DR_metrics.json', 'r') as f:
    metrics = json.load(f)

# --- Plot 1: Trajectory ---
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(
    df['reference_east_m'], df['reference_north_m'],
    'b-o', markersize=3, linewidth=2, label='VBOX Reference Trajectory'
)
ax.plot(
    df['estimated_east_m'], df['estimated_north_m'],
    'r--s', markersize=3, linewidth=2, label='IDR Dead Reckoning'
)

# Mark start and end
ax.scatter(df['reference_east_m'].iloc[0], df['reference_north_m'].iloc[0],
           c='green', s=150, zorder=5, label='Start (t=120s)')
ax.scatter(df['reference_east_m'].iloc[-1], df['reference_north_m'].iloc[-1],
           c='blue', s=150, marker='*', zorder=5, label=f'VBOX End (t=150s)')
ax.scatter(df['estimated_east_m'].iloc[-1], df['estimated_north_m'].iloc[-1],
           c='red', s=150, marker='*', zorder=5, label=f'IDR End (t=150s)')

drift_pct = metrics['dr_drift_percent']
final_err = metrics['final_error_m']
ref_path = metrics['reference_path_m']

ax.set_title(
    f'IO-VNBD Controlled DR Benchmark (GNSS Blackout: 120-150s)\n'
    f'DR Drift: {drift_pct:.2f}% | Final Error: {final_err:.2f}m | Path: {ref_path:.1f}m',
    fontsize=13
)
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('Data/IDR_trajectory_integrated.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: Data/IDR_trajectory_integrated.png')

# --- Plot 2: Position Error over time ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax1 = axes[0]
ax1.plot(df['time_s'], df['position_error_m'], 'r-', linewidth=2, label='Position Error (m)')
ax1.axhline(y=metrics['position_mae_m'], color='orange', linestyle='--',
            label=f"MAE={metrics['position_mae_m']:.1f}m")
ax1.fill_between(df['time_s'], 0, df['position_error_m'], alpha=0.2, color='red')
ax1.set_ylabel('Position Error (m)')
ax1.set_title('IDR Dead Reckoning Position Error During GNSS Blackout')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(df['time_s'], df['estimated_speed_mps'] * 3.6, 'g-',
         linewidth=2, label='IDR Estimated Speed (km/h)')
ax2.plot(df['time_s'], df['reference_speed_mps'] * 3.6, 'b--',
         linewidth=2, label='VBOX Reference Speed (km/h)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Speed (km/h)')
ax2.set_title('Speed: IDR vs VBOX Reference')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Data/DR_error_graph.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: Data/DR_error_graph.png')

# --- Plot 3: Summary metrics ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

summary = [
    ['Metric', 'Value', 'Target'],
    ['DR Drift %', f'{drift_pct:.2f}%', '< 10%'],
    ['Final Error', f'{final_err:.2f} m', '< 10% of path'],
    ['Position MAE', f'{metrics["position_mae_m"]:.2f} m', '-'],
    ['Position RMSE', f'{metrics["position_rmse_m"]:.2f} m', '-'],
    ['Max Error', f'{metrics["maximum_error_m"]:.2f} m', '-'],
    ['Reference Path', f'{ref_path:.2f} m', '-'],
    ['Reference Straight', f'{metrics["reference_straight_m"]:.2f} m', '-'],
    ['Outage Duration', '30 s', '-'],
    ['Initial Speed', '56.8 km/h', '-'],
    ['AI Model', 'Loaded (accel)', '-'],
]

colors = []
for row in summary:
    if row[0] == 'Metric':
        colors.append(['#2c3e50'] * 3)
    elif row[0] == 'DR Drift %':
        c = '#27ae60' if drift_pct < 10 else '#e74c3c'
        colors.append([c, c, c])
    else:
        colors.append(['#ecf0f1', '#ecf0f1', '#ecf0f1'])

table = ax.table(
    cellText=summary,
    cellLoc='center',
    loc='center',
    cellColours=colors,
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 1.8)

# Style header
for j in range(3):
    cell = table[(0, j)]
    cell.set_text_props(color='white', fontweight='bold')

status = 'PASS' if drift_pct < 10 else 'NEEDS IMPROVEMENT'
ax.set_title(
    f'IO-VNBD DR Benchmark Results — Status: {status}',
    fontsize=14, fontweight='bold', pad=20
)

plt.tight_layout()
plt.savefig('Data/IDR_metrics.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: Data/IDR_metrics.png')

print(f'\nDR Drift: {drift_pct:.2f}% (target < 10%) — {"PASS" if drift_pct < 10 else "FAIL"}')
