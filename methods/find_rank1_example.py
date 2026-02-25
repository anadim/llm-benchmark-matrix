#!/usr/bin/env python3
"""Find the most striking near-rank-1 sub-matrix for the report."""
import numpy as np
import sys, os, warnings
from itertools import combinations

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES,
)

# Find all sub-matrices: 3 models × 2 benchmarks, fully observed, near rank-1
# Rank-1 means: a_1/a_2 ≈ b_1/b_2 ≈ c_1/c_2 (ratios are constant)
# Or equivalently: the 3×2 sub-matrix has very small second singular value

print("Searching for striking near-rank-1 sub-matrices (3 models × 2 benchmarks)...")
print("=" * 100)

best = []

# Only consider benchmarks with reasonable spread (not saturated)
bench_candidates = []
for j in range(N_BENCH):
    vals = M_FULL[:, j][OBSERVED[:, j]]
    if len(vals) >= 10 and np.std(vals) > 5:
        bench_candidates.append(j)
print(f"Benchmarks with spread > 5: {len(bench_candidates)}")

# For each pair of benchmarks
for j1, j2 in combinations(bench_candidates, 2):
    # Find models observed on both
    both = np.where(OBSERVED[:, j1] & OBSERVED[:, j2])[0]
    if len(both) < 3:
        continue

    # For each triple of models
    for trio in combinations(both, 3):
        trio = list(trio)
        sub = M_FULL[np.ix_(trio, [j1, j2])]

        # Check for zeros (can't compute ratios)
        if np.any(np.abs(sub) < 0.5):
            continue

        # Compute rank-1 approximation error
        U, S, Vt = np.linalg.svd(sub - sub.mean(axis=0), full_matrices=False)
        if S[0] < 1e-6:
            continue
        ratio = S[1] / S[0]  # smaller = more rank-1

        # BOTH benchmarks must vary meaningfully within the sub-matrix
        range_j1 = sub[:, 0].max() - sub[:, 0].min()
        range_j2 = sub[:, 1].max() - sub[:, 1].min()
        if range_j1 < 10 or range_j2 < 10:
            continue

        spread = sub.max() - sub.min()
        if spread < 15:
            continue

        # Check that the pattern is visually compelling: consistent ordering
        order_j1 = np.argsort(sub[:, 0])
        order_j2 = np.argsort(sub[:, 1])
        if not np.array_equal(order_j1, order_j2):
            continue  # Not monotonically ordered — less visually striking

        best.append({
            'models': trio,
            'benches': [j1, j2],
            'sub': sub,
            'ratio': ratio,
            'spread': spread,
            'range_j1': range_j1,
            'range_j2': range_j2,
        })

# Sort by ratio (smaller = more rank-1)
best.sort(key=lambda x: x['ratio'])

print(f"\nFound {len(best)} candidates. Top 20:\n")

for rank, b in enumerate(best[:20]):
    j1, j2 = b['benches']
    print(f"#{rank+1} (σ2/σ1 = {b['ratio']:.4f}, spread = {b['spread']:.0f}, ranges: {b['range_j1']:.0f}/{b['range_j2']:.0f})")
    print(f"  Benchmarks: {BENCH_NAMES[BENCH_IDS[j1]]} | {BENCH_NAMES[BENCH_IDS[j2]]}")

    # Print as table
    print(f"  {'Model':<35s} {BENCH_NAMES[BENCH_IDS[j1]]:<22s} {BENCH_NAMES[BENCH_IDS[j2]]:<22s}")
    print(f"  {'─'*80}")
    for k, i in enumerate(b['models']):
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s} {b['sub'][k, 0]:>8.1f}           {b['sub'][k, 1]:>8.1f}")
    # Show ratio to help verify rank-1 structure
    ratios = b['sub'][:, 0] / b['sub'][:, 1]
    print(f"  Ratios: {', '.join(f'{r:.3f}' for r in ratios)}")
    print()

# Also search 4 models × 2 benchmarks
print("\n\nSearching 4 models × 2 benchmarks...")
best4 = []

for j1, j2 in combinations(bench_candidates, 2):
    both = np.where(OBSERVED[:, j1] & OBSERVED[:, j2])[0]
    if len(both) < 4:
        continue

    for quad in combinations(both, 4):
        quad = list(quad)
        sub = M_FULL[np.ix_(quad, [j1, j2])]

        if np.any(np.abs(sub) < 0.5):
            continue

        U, S, Vt = np.linalg.svd(sub - sub.mean(axis=0), full_matrices=False)
        if S[0] < 1e-6:
            continue
        ratio = S[1] / S[0]

        # BOTH benchmarks must vary meaningfully
        range_j1 = sub[:, 0].max() - sub[:, 0].min()
        range_j2 = sub[:, 1].max() - sub[:, 1].min()
        if range_j1 < 10 or range_j2 < 10:
            continue

        spread = sub.max() - sub.min()
        if spread < 20:
            continue

        order_j1 = np.argsort(sub[:, 0])
        order_j2 = np.argsort(sub[:, 1])
        if not np.array_equal(order_j1, order_j2):
            continue

        best4.append({
            'models': quad,
            'benches': [j1, j2],
            'sub': sub,
            'ratio': ratio,
            'spread': spread,
        })

best4.sort(key=lambda x: x['ratio'])

print(f"\nFound {len(best4)} candidates. Top 15:\n")

for rank, b in enumerate(best4[:15]):
    j1, j2 = b['benches']
    print(f"#{rank+1} (σ2/σ1 = {b['ratio']:.4f}, spread = {b['spread']:.0f})")
    print(f"  Benchmarks: {BENCH_NAMES[BENCH_IDS[j1]]} | {BENCH_NAMES[BENCH_IDS[j2]]}")

    print(f"  {'Model':<35s} {BENCH_NAMES[BENCH_IDS[j1]]:<22s} {BENCH_NAMES[BENCH_IDS[j2]]:<22s}")
    print(f"  {'─'*80}")
    for k, i in enumerate(b['models']):
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s} {b['sub'][k, 0]:>8.1f}           {b['sub'][k, 1]:>8.1f}")
    print()

# Also try 3 models × 3 benchmarks for a nice square-ish table
print("\n\nSearching 3 models × 3 benchmarks...")
best33 = []

for j1, j2, j3 in combinations(bench_candidates, 3):
    all3 = np.where(OBSERVED[:, j1] & OBSERVED[:, j2] & OBSERVED[:, j3])[0]
    if len(all3) < 3:
        continue

    for trio in combinations(all3, 3):
        trio = list(trio)
        sub = M_FULL[np.ix_(trio, [j1, j2, j3])]

        if np.any(np.abs(sub) < 0.5):
            continue

        U, S, Vt = np.linalg.svd(sub - sub.mean(axis=0), full_matrices=False)
        if S[0] < 1e-6:
            continue
        ratio = S[1] / S[0]

        # ALL 3 benchmarks must vary meaningfully
        ranges = [sub[:, c].max() - sub[:, c].min() for c in range(3)]
        if any(r < 8 for r in ranges):
            continue

        spread = sub.max() - sub.min()
        if spread < 20:
            continue

        # Check consistent ordering across all 3 benchmarks
        orders = [np.argsort(sub[:, c]).tolist() for c in range(3)]
        if not (orders[0] == orders[1] == orders[2]):
            continue

        best33.append({
            'models': trio,
            'benches': [j1, j2, j3],
            'sub': sub,
            'ratio': ratio,
            'spread': spread,
        })

best33.sort(key=lambda x: x['ratio'])

print(f"\nFound {len(best33)} candidates. Top 10:\n")

for rank, b in enumerate(best33[:10]):
    j1, j2, j3 = b['benches']
    print(f"#{rank+1} (σ2/σ1 = {b['ratio']:.4f}, spread = {b['spread']:.0f})")
    bnames = [BENCH_NAMES[BENCH_IDS[j]] for j in b['benches']]
    print(f"  {'Model':<35s} {bnames[0]:<20s} {bnames[1]:<20s} {bnames[2]:<20s}")
    print(f"  {'─'*95}")
    for k, i in enumerate(b['models']):
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s} {b['sub'][k, 0]:>8.1f}       {b['sub'][k, 1]:>8.1f}       {b['sub'][k, 2]:>8.1f}")
    print()
