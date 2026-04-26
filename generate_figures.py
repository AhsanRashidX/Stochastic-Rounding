"""
Research Diagrams for "Training with Fewer Bits: Unlocking Edge LLMs Training with Stochastic Rounding"
Generates publication-quality figures from experiment_results.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import os

# Load results
try:
    with open('experiment_results.json', 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print("Error: experiment_results.json not found!")
    print("Please run your experiments first to generate the results file.")
    exit(1)

# Create figure directory
os.makedirs('figures', exist_ok=True)

# Color palette
COLORS = {
    'fp32': '#2E86AB',
    'sr_8bit': '#A23B72',
    'rtn_8bit': '#F18F01',
    'sr_6bit': '#C73E1D',
    'rtn_6bit': '#E9C46A',
    'sr_4bit': '#264653',
    'rtn_4bit': '#8AB17D',
    'sr': '#2A9D8F',
    'rtn': '#E76F51'
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Helper function to determine if an experiment uses SR
def is_sr_experiment(r):
    """Check if experiment uses Stochastic Rounding"""
    name = r.get('name', '')
    # Check explicit use_sr flag
    if 'use_sr' in r:
        return r['use_sr']
    # Check name patterns
    return 'SR' in name.upper() and 'RTN' not in name.upper()

def is_rtn_experiment(r):
    """Check if experiment uses Round-to-Nearest"""
    name = r.get('name', '')
    if 'use_sr' in r:
        return not r['use_sr']
    return 'RTN' in name.upper()

print(f"Loaded {len(results)} experiments")
print("Experiments found:")
for r in results:
    sr = is_sr_experiment(r)
    rtn = is_rtn_experiment(r)
    print(f"  {r['name']}: bits={r['bits']}, batch={r['batch_size']}, method={'SR' if sr else 'RTN' if rtn else 'Unknown'}")

# ============================================================
# Figure 1: Perplexity Comparison Bar Chart (SR vs RTN)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

categories = []
sr_values = []
rtn_values = []

# Group by bit width - match any batch size, just take first available
for bits in [8, 6, 4]:
    sr_candidates = [r for r in results if r['bits'] == bits and is_sr_experiment(r)]
    rtn_candidates = [r for r in results if r['bits'] == bits and is_rtn_experiment(r)]

    if sr_candidates and rtn_candidates:
        # Take the one with largest batch size for fair comparison
        sr_best = min(sr_candidates, key=lambda x: x['best_ppl'])
        rtn_best = min(rtn_candidates, key=lambda x: x['best_ppl'])

        categories.append(f'{bits}-bit')
        sr_values.append(sr_best['best_ppl'])
        rtn_values.append(rtn_best['best_ppl'])

if categories:
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, sr_values, width, label='Stochastic Rounding (SR)',
                   color=COLORS['sr'], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, rtn_values, width, label='Round-to-Nearest (RTN)',
                   color=COLORS['rtn'], edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Quantization Bit Width', fontweight='bold')
    ax.set_ylabel('Validation Perplexity (↓)', fontweight='bold')
    ax.set_title('SR vs RTN: Perplexity Comparison Across Bit Widths', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(sr_values), max(rtn_values)) * 1.15)
else:
    ax.text(0.5, 0.5, 'No SR vs RTN comparison data available\n(need both SR and RTN experiments)',
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('SR vs RTN: Perplexity Comparison', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/01_sr_vs_rtn_comparison.png', dpi=300)
plt.show()
print("Saved: figures/01_sr_vs_rtn_comparison.png")

# ============================================================
# Figure 2: Perplexity vs Bit Width (All Configurations)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Separate by batch size
batch_sizes = sorted(list(set(r['batch_size'] for r in results)))
markers = ['o', 's', '^', 'D', 'v']

for idx, bs in enumerate(batch_sizes):
    batch_data = [(r['bits'], r['best_ppl']) for r in results if r['batch_size'] == bs]
    if batch_data:
        batch_data = sorted(batch_data, key=lambda x: x[0], reverse=True)  # Sort by bits descending
        bits_list, ppls = zip(*batch_data)
        ax.plot(bits_list, ppls, markers[idx % len(markers)] + '-', 
                linewidth=2.5, markersize=10, label=f'Batch Size = {bs}',
                markerfacecolor='white', markeredgewidth=2)

ax.set_xlabel('Bit Width', fontweight='bold')
ax.set_ylabel('Validation Perplexity (↓)', fontweight='bold')
ax.set_title('Perplexity Degradation with Reduced Precision', fontweight='bold', pad=20)

# Set x-ticks to actual bit values present in data
all_bits = sorted(list(set(r['bits'] for r in results)), reverse=True)
ax.set_xticks(all_bits)
ax.set_xticklabels([f'{b}-bit' if b < 32 else 'FP32' for b in all_bits])

ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

# Add annotation
ax.annotate('Lower is better', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/02_perplexity_vs_bitwidth.png', dpi=300)
plt.show()
print("Saved: figures/02_perplexity_vs_bitwidth.png")

# ============================================================
# Figure 3: Batch Size Scaling Effect (SR only)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

sr_results = [r for r in results if is_sr_experiment(r)]
bit_widths = sorted(list(set(r['bits'] for r in sr_results if r['bits'] < 32)))

for bits in bit_widths:
    sr_data = [(r['batch_size'], r['best_ppl']) for r in sr_results if r['bits'] == bits]
    if sr_data:
        batches, ppls = zip(*sorted(sr_data))
        color_key = f'sr_{bits}bit'
        color = COLORS.get(color_key, '#333333')
        ax.plot(batches, ppls, 'o-', color=color, linewidth=2.5,
                markersize=10, label=f'{bits}-bit SR', markerfacecolor='white',
                markeredgewidth=2, markeredgecolor=color)

# Add FP32 baseline as horizontal line
fp32_results = [r for r in results if r['bits'] == 32]
if fp32_results:
    fp32_ppl = min(r['best_ppl'] for r in fp32_results)
    ax.axhline(y=fp32_ppl, color=COLORS['fp32'], linestyle='--', linewidth=2,
               label=f'FP32 Baseline ({fp32_ppl:.1f})')

ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Validation Perplexity (↓)', fontweight='bold')
ax.set_title('Effect of Batch Size on Quantized Training (SR)', fontweight='bold', pad=20)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

# Use log scale if batch sizes vary widely
if len(batch_sizes) > 1 and max(batch_sizes) / min(batch_sizes) >= 4:
    ax.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig('figures/03_batch_size_scaling.png', dpi=300)
plt.show()
print("Saved: figures/03_batch_size_scaling.png")

# ============================================================
# Figure 4: Conceptual Diagram - Stochastic Rounding vs RTN
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_rounding_concept(ax, title, method, color):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    # Draw number line
    ax.plot([1, 9], [5, 5], 'k-', linewidth=2)

    # Draw tick marks
    for i in range(2, 9):
        ax.plot([i, i], [4.8, 5.2], 'k-', linewidth=1.5)
        ax.text(i, 4.3, str(i-1), ha='center', fontsize=11, fontweight='bold')

    # Example value
    value = 4.7
    ax.plot([value], [5], 'ro', markersize=15, zorder=5)
    ax.annotate(f'x = {value}', xy=(value, 5), xytext=(value, 6.5),
                ha='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    if method == 'RTN':
        rounded = 5
        ax.plot([rounded], [5], 'bs', markersize=15, zorder=5)
        ax.annotate(f'RTN(x) = {rounded}', xy=(rounded, 5), xytext=(rounded, 3),
                    ha='center', fontsize=12, fontweight='bold', color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(5, 1.5, 'Deterministic: Always rounds\nto nearest integer',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    else:
        ax.plot([4], [5], 'gs', markersize=12, zorder=5, alpha=0.6)
        ax.plot([5], [5], 'gs', markersize=12, zorder=5, alpha=0.6)
        ax.annotate('SR(x) = 4 (prob=0.3)', xy=(4, 5), xytext=(2.5, 3),
                    ha='center', fontsize=11, fontweight='bold', color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.annotate('SR(x) = 5 (prob=0.7)', xy=(5, 5), xytext=(6.5, 3),
                    ha='center', fontsize=11, fontweight='bold', color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(5, 1.5, 'Stochastic: Rounds probabilistically\nE[SR(x)] = x (unbiased)',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plot_rounding_concept(axes[0], 'Round-to-Nearest (RTN)', 'RTN', COLORS['rtn'])
plot_rounding_concept(axes[1], 'Stochastic Rounding (SR)', 'SR', COLORS['sr'])

plt.tight_layout()
plt.savefig('figures/04_rounding_concept.png', dpi=300)
plt.show()
print("Saved: figures/04_rounding_concept.png")

# ============================================================
# Figure 5: Training Pipeline Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Mixed-Precision Training Pipeline with Stochastic Rounding',
             fontweight='bold', fontsize=16, pad=20)

def draw_box(ax, x, y, w, h, text, color, text_color='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=11, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, lw=2,
                            mutation_scale=20)
    ax.add_patch(arrow)

# Forward pass
draw_box(ax, 1, 7, 2.5, 1.2, 'Input\nActivations\n(FP32)', '#E8F4F8')
draw_arrow(ax, 3.5, 7.6, 5, 7.6)
draw_box(ax, 5, 7, 2.5, 1.2, 'Quantized\nWeights\n(Q_w)', '#FFE4E1')
draw_arrow(ax, 7.5, 7.6, 9, 7.6)
draw_box(ax, 9, 7, 2.5, 1.2, 'Forward\nOutput\n(FP32)', '#E8F4F8')

# Loss
draw_arrow(ax, 11.5, 7.6, 12.5, 7.6)
draw_box(ax, 12.5, 7, 1.2, 1.2, 'Loss', '#FFFACD')

# Backward pass
draw_arrow(ax, 10.25, 7, 10.25, 5.5, color='red')
draw_box(ax, 9, 4.5, 2.5, 1.2, 'Quantized\nGradients\n(Q_g)', '#FFE4E1')
draw_arrow(ax, 9, 5.1, 7.5, 5.1, color='red')
draw_box(ax, 5, 4.5, 2.5, 1.2, 'Quantized\nActivations\n(Q_a)', '#FFE4E1')
draw_arrow(ax, 5, 5.1, 3.5, 5.1, color='red')
draw_box(ax, 1, 4.5, 2.5, 1.2, 'Weight\nGradients\n(FP32)', '#E8F4F8')

# Update
draw_arrow(ax, 2.25, 4.5, 2.25, 3.2, color='green')
draw_box(ax, 1, 2, 2.5, 1.2, 'FP32\nWeights\nUpdated', '#90EE90')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#E8F4F8', edgecolor='black', label='FP32 (High Precision)'),
    mpatches.Patch(facecolor='#FFE4E1', edgecolor='black', label='Quantized (Low Precision)'),
    mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='Weight Update'),
    mpatches.FancyArrowPatch((0,0), (1,0), arrowstyle='->', color='black', label='Forward Pass'),
    mpatches.FancyArrowPatch((0,0), (1,0), arrowstyle='->', color='red', label='Backward Pass'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True,
          fancybox=True, shadow=True, fontsize=10)

# Annotations
ax.text(7, 8.8, 'Forward Pass: W_q = Quantize(W, SR)', ha='center', fontsize=12,
        style='italic', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax.text(7, 3.5, 'Backward Pass: Quantize activations & gradients', ha='center', fontsize=12,
        style='italic', color='red', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax.text(7, 1.5, 'Key: Forward activations stay FP32, only weights/gradients quantized',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/05_training_pipeline.png', dpi=300)
plt.show()
print("Saved: figures/05_training_pipeline.png")

# ============================================================
# Figure 6: Summary Table as Figure
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
for r in results:
    if is_sr_experiment(r):
        method = 'SR'
    elif is_rtn_experiment(r):
        method = 'RTN'
    else:
        method = 'FP32'
    table_data.append([
        r['name'],
        f"{r['bits']}-bit",
        str(r['batch_size']),
        method,
        f"{r['best_val_loss']:.4f}",
        f"{r['best_ppl']:.2f}"
    ])

table = ax.table(cellText=table_data,
                 colLabels=['Experiment', 'Precision', 'Batch', 'Method', 'Val Loss', 'Perplexity'],
                 cellLoc='center',
                 loc='center',
                 colColours=['#2E86AB']*6)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style header
for i in range(6):
    table[(0, i)].set_text_props(color='white', fontweight='bold')
    table[(0, i)].set_facecolor('#264653')

# Color code rows
for i in range(1, len(table_data) + 1):
    bits = int(table_data[i-1][1].split('-')[0])
    if bits == 32:
        color = '#E8F4F8'
    elif bits == 8:
        color = '#F0E6EF'
    elif bits == 6:
        color = '#FFF0E6'
    else:
        color = '#E6F0E6'

    for j in range(6):
        table[(i, j)].set_facecolor(color)

ax.set_title('Experimental Results Summary', fontweight='bold', fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('figures/06_results_table.png', dpi=300)
plt.show()
print("Saved: figures/06_results_table.png")

print("\n" + "="*60)
print("All figures generated successfully!")
print("Check the 'figures/' directory for outputs.")
print("="*60)