#!/usr/bin/env python3
"""
Phase transition comparison: with vs without ARC-AGI benchmarks.

Runs the same per-model phase transition experiment twice:
1. Full matrix (all benchmarks)
2. ARC-AGI-1 and ARC-AGI-2 excluded from the error measurement

This tests whether the bimodal ARC-AGI benchmarks are driving the error floor.
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
N_SEEDS = 10
MAX_REVEAL = 20

EXCLUDE_BENCHMARKS = {'arc_agi_1', 'arc_agi_2'}
EXCLUDE_IDX = {BENCH_IDS.index(b) for b in EXCLUDE_BENCHMARKS if b in BENCH_IDS}
print(f"Excluding benchmarks: {EXCLUDE_BENCHMARKS} (indices: {EXCLUDE_IDX})")

DISPLAY_NAMES = {
    'gpt-5.2': 'GPT-5.2',
    'gemini-3-pro': 'Gemini 3 Pro',
    'claude-opus-4.6': 'Claude Opus 4.6',
    'deepseek-r1': 'DeepSeek-R1',
}

COLORS = {
    'gpt-5.2': '#2CA02C',
    'gemini-3-pro': '#1F77B4',
    'claude-opus-4.6': '#7B3FF2',
    'deepseek-r1': '#E74C3C',
}


def run_phase_transition(model_id, exclude_from_error=None):
    """
    For one model: reveal 0..k scores, predict the rest, measure median absolute error.
    exclude_from_error: set of bench indices to exclude from error computation (but keep in training).
    """
    if exclude_from_error is None:
        exclude_from_error = set()

    mi = MODEL_IDS.index(model_id)
    obs_j = np.where(OBSERVED[mi])[0]
    n_obs = len(obs_j)
    max_k = min(MAX_REVEAL, n_obs - 2)

    # k=0 baseline: column mean
    col_means = np.nanmean(M_FULL, axis=0)
    baseline_errors = []
    for j in obs_j:
        if j in exclude_from_error:
            continue
        actual = M_FULL[mi, j]
        pred = col_means[j]
        if np.isfinite(pred) and np.isfinite(actual):
            baseline_errors.append(abs(pred - actual))
    baseline_err = np.median(baseline_errors) if baseline_errors else np.nan

    results = {k: [] for k in range(1, max_k + 1)}

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(42 + seed)
        order = obs_j.copy()
        rng.shuffle(order)

        for k in range(1, max_k + 1):
            M_train = M_FULL.copy()
            revealed = set(order[:k])
            hidden = [j for j in obs_j if j not in revealed]

            for j in obs_j:
                M_train[mi, j] = np.nan
            for j in revealed:
                M_train[mi, j] = M_FULL[mi, j]

            M_pred = predict_logit_svd_blend(M_train)

            abs_errors = []
            for j in hidden:
                if j in exclude_from_error:
                    continue
                actual = M_FULL[mi, j]
                pred = M_pred[mi, j]
                if np.isfinite(pred) and np.isfinite(actual):
                    abs_errors.append(abs(pred - actual))

            if len(abs_errors) > 0:
                results[k].append(np.median(abs_errors))

    ks = [0] + sorted(results.keys())
    mean_errors = [baseline_err] + [np.mean(results[k]) if results[k] else np.nan for k in sorted(results.keys())]
    std_errors = [0.0] + [np.std(results[k]) if results[k] else np.nan for k in sorted(results.keys())]

    return ks, mean_errors, std_errors


# ── Run both conditions ──
results_full = {}
results_no_arc = {}

for mid in TARGET_MODELS:
    if mid not in MODEL_IDS:
        print(f"WARNING: {mid} not found, skipping")
        continue
    print(f"\n{'='*50}")
    print(f"  {DISPLAY_NAMES[mid]}")
    print(f"{'='*50}")

    print("  [1/2] Full matrix...")
    ks, means, stds = run_phase_transition(mid, exclude_from_error=set())
    results_full[mid] = (ks, means, stds)
    print(f"    k=0: {means[0]:.1f}, k=5: {means[5]:.1f}, k=10: {means[10]:.1f}")

    print("  [2/2] Excluding ARC-AGI from error...")
    ks2, means2, stds2 = run_phase_transition(mid, exclude_from_error=EXCLUDE_IDX)
    results_no_arc[mid] = (ks2, means2, stds2)
    print(f"    k=0: {means2[0]:.1f}, k=5: {means2[5]:.1f}, k=10: {means2[10]:.1f}")


# ── Plot: overlay with solid (full) vs dashed (no ARC-AGI) ──
fig, ax = plt.subplots(figsize=(10, 6))

for mid in TARGET_MODELS:
    if mid not in results_full:
        continue
    color = COLORS[mid]
    name = DISPLAY_NAMES[mid]

    ks, means, stds = results_full[mid]
    ax.plot(ks, means, '-o', color=color, linewidth=2.5, markersize=5,
            label=f'{name} (all)', zorder=3)

    ks2, means2, stds2 = results_no_arc[mid]
    ax.plot(ks2, means2, '--s', color=color, linewidth=2, markersize=4,
            label=f'{name} (no ARC-AGI)', alpha=0.7, zorder=3)

ax.set_xlabel('Number of known benchmark scores', fontsize=16)
ax.set_ylabel('Median absolute error (points)', fontsize=16)
ax.set_title('Effect of removing ARC-AGI benchmarks from error', fontsize=18)
ax.legend(fontsize=10, loc='upper right', ncol=2)
ax.tick_params(labelsize=14)
ax.set_xlim(-0.5, MAX_REVEAL + 0.5)
ax.set_xticks(range(0, MAX_REVEAL + 1, 2))
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.4, linewidth=1)

plt.tight_layout()
out_path = os.path.join(REPO_ROOT, 'figures', 'phase_transition_no_arcagi.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to {out_path}")
plt.close()

# ── Print summary ──
print(f"\n{'='*60}")
print("SUMMARY: Median error at k=5 and k=15")
print(f"{'='*60}")
print(f"{'Model':<25s} {'Full k=5':>10s} {'NoARC k=5':>10s} {'Full k=15':>10s} {'NoARC k=15':>10s}")
for mid in TARGET_MODELS:
    if mid not in results_full:
        continue
    _, mf, _ = results_full[mid]
    _, mn, _ = results_no_arc[mid]
    print(f"{DISPLAY_NAMES[mid]:<25s} {mf[5]:>10.2f} {mn[5]:>10.2f} {mf[15]:>10.2f} {mn[15]:>10.2f}")
