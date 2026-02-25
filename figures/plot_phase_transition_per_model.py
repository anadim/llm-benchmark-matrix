#!/usr/bin/env python3
"""
Per-model phase transition: prediction error vs number of revealed benchmarks.

For each selected model, hide ALL its scores, then reveal them 1-at-a-time
in random order. After each reveal, run LogitSVD Blend and measure median
absolute error on the remaining hidden scores. Average over multiple random
orderings. k=0 means "no known scores" = column-mean baseline.
"""
import numpy as np
import sys, os, warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES,
)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))
from all_methods import predict_logit_svd_blend

# ── Configuration ──
TARGET_MODELS = ['gpt-5.2', 'gemini-3-pro', 'claude-opus-4.6', 'deepseek-r1']
N_SEEDS = 10  # random orderings to average over
MAX_REVEAL = 20  # max benchmarks to reveal (plot up to this)

# Nice display names
DISPLAY_NAMES = {
    'gpt-5.2': 'GPT-5.2 (37 benchmarks)',
    'gemini-3-pro': 'Gemini 3 Pro (36)',
    'claude-opus-4.6': 'Claude Opus 4.6 (29)',
    'deepseek-r1': 'DeepSeek-R1 (25)',
}

COLORS = {
    'gpt-5.2': '#2CA02C',
    'gemini-3-pro': '#1F77B4',
    'claude-opus-4.6': '#7B3FF2',
    'deepseek-r1': '#E74C3C',
}


def compute_baseline_error(model_id):
    """k=0: no known scores. Predict every benchmark with its column mean."""
    mi = MODEL_IDS.index(model_id)
    obs_j = np.where(OBSERVED[mi])[0]
    col_means = np.nanmean(M_FULL, axis=0)

    abs_errors = []
    for j in obs_j:
        actual = M_FULL[mi, j]
        pred = col_means[j]
        if np.isfinite(pred) and np.isfinite(actual):
            abs_errors.append(abs(pred - actual))
    return np.median(abs_errors) if abs_errors else np.nan


def run_phase_transition(model_id):
    """For one model: reveal 0..k scores, predict the rest, measure median absolute error."""
    mi = MODEL_IDS.index(model_id)
    obs_j = np.where(OBSERVED[mi])[0]
    n_obs = len(obs_j)
    max_k = min(MAX_REVEAL, n_obs - 2)  # need at least 2 hidden to measure error

    print(f"\n{DISPLAY_NAMES.get(model_id, model_id)}: {n_obs} observed benchmarks, testing k=0..{max_k}")

    # k=0 baseline
    baseline_err = compute_baseline_error(model_id)
    print(f"  k=0 (column mean): {baseline_err:.2f}")

    # For each k (number of revealed scores): collect median absolute errors across seeds
    results = {k: [] for k in range(1, max_k + 1)}

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(42 + seed)
        order = obs_j.copy()
        rng.shuffle(order)

        for k in range(1, max_k + 1):
            # Build training matrix: hide ALL of this model's scores, then reveal k
            M_train = M_FULL.copy()
            revealed = set(order[:k])
            hidden = [j for j in obs_j if j not in revealed]

            # Hide all of this model's scores
            for j in obs_j:
                M_train[mi, j] = np.nan
            # Reveal k scores
            for j in revealed:
                M_train[mi, j] = M_FULL[mi, j]

            # Predict
            M_pred = predict_logit_svd_blend(M_train)

            # Measure absolute error on hidden scores
            abs_errors = []
            for j in hidden:
                actual = M_FULL[mi, j]
                pred = M_pred[mi, j]
                if np.isfinite(pred) and np.isfinite(actual):
                    abs_errors.append(abs(pred - actual))

            if len(abs_errors) > 0:
                results[k].append(np.median(abs_errors))

        print(f"  Seed {seed+1}/{N_SEEDS} done")

    # Build full series starting at k=0
    ks = [0] + sorted(results.keys())
    mean_errors = [baseline_err] + [np.mean(results[k]) if results[k] else np.nan for k in sorted(results.keys())]
    std_errors = [0.0] + [np.std(results[k]) if results[k] else np.nan for k in sorted(results.keys())]

    return ks, mean_errors, std_errors


# ── Run for all target models ──
all_results = {}
for mid in TARGET_MODELS:
    if mid not in MODEL_IDS:
        print(f"WARNING: {mid} not found in matrix, skipping")
        continue
    ks, means, stds = run_phase_transition(mid)
    all_results[mid] = (ks, means, stds)

# ── Figure 1: 2×2 grid ──
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
axes_flat = axes.flatten()

for idx, mid in enumerate(TARGET_MODELS):
    if mid not in all_results:
        continue
    ax = axes_flat[idx]
    ks, means, stds = all_results[mid]
    color = COLORS.get(mid, '#333333')
    name = DISPLAY_NAMES.get(mid, mid)

    ax.plot(ks, means, '-o', color=color, linewidth=2.5, markersize=5, zorder=3)
    ax.fill_between(ks,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color=color)

    # Highlight k=0 baseline point
    ax.plot(0, means[0], 's', color='gray', markersize=10, zorder=4)

    ax.set_title(name, fontsize=16, fontweight='bold', color=color)
    ax.set_xlabel('Known scores', fontsize=14)
    ax.set_ylabel('Median absolute error (points)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-0.5, MAX_REVEAL + 0.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(bottom=0, top=16)
    ax.grid(True, alpha=0.3)

    # 5-point reference line
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # Annotate k=0 baseline
    ax.annotate('benchmark\naverage', (0, means[0]), textcoords='offset points',
                xytext=(12, -5), fontsize=10, color='gray', va='top')

fig.suptitle('How quickly can we predict a model\'s scores?', fontsize=19, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = os.path.join(REPO_ROOT, 'figures', 'phase_transition_per_model.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved 2x2 grid to {out_path}")
plt.close()

# ── Figure 2: all together ──
fig, ax = plt.subplots(figsize=(10, 6))

for mid in TARGET_MODELS:
    if mid not in all_results:
        continue
    ks, means, stds = all_results[mid]
    color = COLORS.get(mid, '#333333')
    label = DISPLAY_NAMES.get(mid, mid)

    ax.plot(ks, means, '-o', color=color, label=label, linewidth=2.5, markersize=5, zorder=3)
    ax.fill_between(ks,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color=color)

    # k=0 baseline marker
    ax.plot(0, means[0], 's', color=color, markersize=9, zorder=4)

ax.set_xlabel('Number of known benchmark scores', fontsize=16)
ax.set_ylabel('Median absolute error (points)', fontsize=16)
ax.set_title('How quickly can we predict a model\'s scores?', fontsize=18)
ax.legend(fontsize=13, loc='upper right')
ax.tick_params(labelsize=14)
ax.set_xlim(-0.5, MAX_REVEAL + 0.5)
ax.set_xticks(range(0, MAX_REVEAL + 1, 2))
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# Reference lines
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.text(MAX_REVEAL - 0.3, 5.4, '5-point error', ha='right', fontsize=11, color='gray')

# Annotate the k=0 region
ax.annotate('benchmark\naverage only', (0.3, 12), fontsize=11, color='gray',
            ha='left', va='bottom', style='italic')

plt.tight_layout()
out_path2 = os.path.join(REPO_ROOT, 'figures', 'phase_transition_all_models.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved overlay to {out_path2}")
plt.close()
