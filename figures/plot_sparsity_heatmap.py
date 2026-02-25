#!/usr/bin/env python3
"""
Sparsity heatmap: blue = observed, white = missing.
Shows the model × benchmark matrix fill pattern.
"""
import numpy as np
import sys, os, warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES,
)

# Sort models by number of observed scores (most filled at top)
obs_per_model = OBSERVED.sum(axis=1)
model_order = np.argsort(-obs_per_model)

# Sort benchmarks by number of observed scores (most filled at left)
obs_per_bench = OBSERVED.sum(axis=0)
bench_order = np.argsort(-obs_per_bench)

# Reorder the observation matrix
obs_sorted = OBSERVED[model_order][:, bench_order].astype(float)

# Model and benchmark labels in sorted order
model_labels = [MODEL_NAMES.get(MODEL_IDS[i], MODEL_IDS[i]) for i in model_order]
bench_labels = [BENCH_NAMES.get(BENCH_IDS[j], BENCH_IDS[j]) for j in bench_order]

# ── Plot ──
fig, ax = plt.subplots(figsize=(16, 14))

# Custom colormap: white=0 (missing), blue=1 (observed)
cmap = mcolors.ListedColormap(['white', '#3274A1'])

ax.imshow(obs_sorted, aspect='auto', cmap=cmap, interpolation='nearest')

# Grid lines to separate cells
ax.set_xticks(np.arange(-0.5, N_BENCH, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N_MODELS, 1), minor=True)
ax.grid(which='minor', color='#E0E0E0', linewidth=0.3)
ax.tick_params(which='minor', size=0)

# Labels
ax.set_xticks(range(N_BENCH))
ax.set_xticklabels(bench_labels, rotation=90, fontsize=7, ha='center')
ax.set_yticks(range(N_MODELS))
ax.set_yticklabels(model_labels, fontsize=6)

ax.set_xlabel('Benchmarks (sorted by coverage)', fontsize=14)
ax.set_ylabel('Models (sorted by coverage)', fontsize=14)

fill_pct = OBSERVED.sum() / (N_MODELS * N_BENCH) * 100
ax.set_title(f'{N_MODELS} models × {N_BENCH} benchmarks — {int(OBSERVED.sum())} scores ({fill_pct:.0f}% filled)',
             fontsize=16)

# Add fill-rate annotations on right side
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(N_MODELS))
ax2.set_yticklabels([f'{int(obs_per_model[i])}' for i in model_order], fontsize=5, color='gray')
ax2.set_ylabel('# benchmarks', fontsize=10, color='gray')
ax2.tick_params(axis='y', length=0)

plt.tight_layout()
out_path = os.path.join(REPO_ROOT, 'figures', 'sparsity_heatmap.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.close()
