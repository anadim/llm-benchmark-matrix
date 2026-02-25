#!/usr/bin/env python3
"""Analyze principal components: top benchmarks, top models, variance explained."""
import numpy as np
import sys, os, warnings

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES,
)

# ── Build matrix with z-score normalization ──
# Column-mean impute, then z-score each benchmark column
M = M_FULL.copy()
col_means = np.nanmean(M, axis=0)
col_stds = np.nanstd(M, axis=0)
for j in range(N_BENCH):
    mask = np.isnan(M[:, j])
    M[mask, j] = col_means[j]

# Z-score normalize each column (puts all benchmarks on same scale)
M_z = np.zeros_like(M)
for j in range(N_BENCH):
    if col_stds[j] > 1e-6:
        M_z[:, j] = (M[:, j] - col_means[j]) / col_stds[j]
    else:
        M_z[:, j] = 0.0

M_centered = M_z  # already centered by z-scoring

# SVD
U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

# ── Variance explained ──
total_var = np.sum(S**2)
var_explained = S**2 / total_var * 100
cum_var = np.cumsum(var_explained)

print("=" * 60)
print("  SINGULAR VALUE SPECTRUM")
print("=" * 60)
for i in range(min(10, len(S))):
    print(f"  PC{i+1}: {var_explained[i]:6.2f}%  (cumulative: {cum_var[i]:6.2f}%)")

print(f"\n  PC1 + PC2 = {cum_var[1]:.1f}% of total variance")

# ── Top benchmarks per component ──
# Vt[k, :] = loadings of component k on benchmarks
print(f"\n{'='*60}")
print("  TOP BENCHMARKS BY COMPONENT")
print(f"{'='*60}")

for pc in range(2):
    loadings = Vt[pc, :]
    # Sort by absolute value
    order = np.argsort(-np.abs(loadings))
    print(f"\n  PC{pc+1} ({var_explained[pc]:.1f}% variance):")
    print(f"  {'Rank':<6s} {'Benchmark':<35s} {'Loading':>8s}")
    print(f"  {'-'*52}")
    for rank, j in enumerate(order[:10]):
        bname = BENCH_NAMES.get(BENCH_IDS[j], BENCH_IDS[j])
        print(f"  {rank+1:<6d} {bname:<35s} {loadings[j]:>8.3f}")

# ── Top models per component ──
# U[:, k] * S[k] = model scores on component k
print(f"\n{'='*60}")
print("  TOP MODELS BY COMPONENT")
print(f"{'='*60}")

for pc in range(2):
    scores = U[:, pc] * S[pc]
    # Sort: most negative first (convention: negative = high capability for PC1)
    order_neg = np.argsort(scores)  # most negative first
    order_pos = np.argsort(-scores)  # most positive first

    print(f"\n  PC{pc+1} ({var_explained[pc]:.1f}% variance):")

    # Check sign convention: if most benchmarks have negative loadings on PC1,
    # then negative model scores = high capability
    mean_loading = np.mean(Vt[pc, :])
    if mean_loading < 0:
        # Negative loadings dominate → negative model scores = strong
        print(f"  (negative = stronger, convention: most benchmarks load negatively)")
        print(f"\n  STRONGEST (most negative):")
        print(f"  {'Rank':<6s} {'Model':<35s} {'Score':>8s}")
        print(f"  {'-'*52}")
        for rank, i in enumerate(order_neg[:10]):
            mname = MODEL_NAMES.get(MODEL_IDS[i], MODEL_IDS[i])
            print(f"  {rank+1:<6d} {mname:<35s} {scores[i]:>8.1f}")
        print(f"\n  WEAKEST (most positive):")
        for rank, i in enumerate(order_pos[:5]):
            mname = MODEL_NAMES.get(MODEL_IDS[i], MODEL_IDS[i])
            print(f"  {rank+1:<6d} {mname:<35s} {scores[i]:>8.1f}")
    else:
        # Positive loadings dominate → positive model scores = strong
        print(f"  (positive = stronger)")
        print(f"\n  STRONGEST (most positive):")
        print(f"  {'Rank':<6s} {'Model':<35s} {'Score':>8s}")
        print(f"  {'-'*52}")
        for rank, i in enumerate(order_pos[:10]):
            mname = MODEL_NAMES.get(MODEL_IDS[i], MODEL_IDS[i])
            print(f"  {rank+1:<6d} {mname:<35s} {scores[i]:>8.1f}")
        print(f"\n  WEAKEST (most negative):")
        for rank, i in enumerate(order_neg[:5]):
            mname = MODEL_NAMES.get(MODEL_IDS[i], MODEL_IDS[i])
            print(f"  {rank+1:<6d} {mname:<35s} {scores[i]:>8.1f}")
