#!/usr/bin/env python3
"""
Question 13 - 5-Benchmark Minimum Eval Set Predictions Spot-Check
=================================================================
Uses {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA} as the
only input features. For each of the other 44 benchmarks, trains a ridge
regression on models that have both the 5 selected benchmarks AND the target
benchmark observed. Predicts all other observed entries and computes APE.
"""

import numpy as np
import sys, warnings, csv, os
from collections import defaultdict
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, BENCH_CATS,
)

# ── The 5-benchmark minimum eval set ────────────────────────────────────────
FIVE_BENCH_IDS = ['hle', 'aime_2025', 'livecodebench', 'swe_bench_verified', 'simpleqa']
five_bench_idx = [BENCH_IDS.index(b) for b in FIVE_BENCH_IDS]

print("=" * 90)
print("  5-BENCHMARK MINIMUM EVAL SET: SPOT-CHECK PREDICTIONS")
print("=" * 90)
print(f"\n  Selected benchmarks: {[BENCH_NAMES[b] for b in FIVE_BENCH_IDS]}")
print(f"  Benchmark indices:   {five_bench_idx}")

# Check coverage: how many models have all 5 benchmarks?
has_all_five = np.ones(N_MODELS, dtype=bool)
for j in five_bench_idx:
    has_all_five &= OBSERVED[:, j]
n_has_all = has_all_five.sum()
print(f"\n  Models with all 5 benchmarks observed: {n_has_all}/{N_MODELS}")
print(f"  Models: {[MODEL_NAMES[MODEL_IDS[i]] for i in range(N_MODELS) if has_all_five[i]]}")

# ── For each target benchmark (not in the 5), train ridge regression ────────
all_predictions = []  # (model_name, bench_name, actual, predicted, ape_pct, bench_cat)

target_bench_stats = {}  # bench_name -> {n_train, n_pred, medape, ...}

for j_target in range(N_BENCH):
    if j_target in five_bench_idx:
        continue

    bench_name = BENCH_NAMES[BENCH_IDS[j_target]]
    bench_cat = BENCH_CATS[j_target]

    # Find models that have all 5 selected benchmarks AND the target benchmark
    has_target = OBSERVED[:, j_target]
    train_mask = has_all_five & has_target

    n_train = train_mask.sum()
    if n_train < 3:
        # Not enough training data; skip or use fallback
        target_bench_stats[bench_name] = {
            'n_train': n_train, 'n_pred': 0, 'medape': np.nan,
            'cat': bench_cat, 'reason': 'too few training models'
        }
        continue

    # Build training data
    X_train = M_FULL[train_mask][:, five_bench_idx]
    y_train = M_FULL[train_mask, j_target]

    # Train ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Leave-one-out style: for each model in the training set, predict its
    # target value using a model trained WITHOUT it (proper LOO)
    # BUT the question says "predict all other observed entries (the ones we can verify)"
    # So we do a simpler approach: train on ALL eligible models, predict all eligible
    # models' target values. This is in-sample but for spot-checking predictions.
    #
    # Actually, to be fair, let's do PROPER leave-one-out for models in training set,
    # and direct prediction for models that have all 5 but NOT the target (out-of-sample).
    # But actually: the point is to predict entries we CAN verify (i.e., observed entries).
    # The training set IS the set of models with both features and target.
    # Let's do LOO cross-validation to get honest error estimates.

    apes_this_bench = []

    for i in range(N_MODELS):
        if not has_all_five[i]:
            continue
        if not OBSERVED[i, j_target]:
            continue

        # Leave this model out of training
        loo_mask = train_mask.copy()
        loo_mask[i] = False

        if loo_mask.sum() < 3:
            continue

        X_loo = M_FULL[loo_mask][:, five_bench_idx]
        y_loo = M_FULL[loo_mask, j_target]

        ridge_loo = Ridge(alpha=1.0)
        ridge_loo.fit(X_loo, y_loo)

        x_i = M_FULL[i, five_bench_idx].reshape(1, -1)
        pred = ridge_loo.predict(x_i)[0]
        actual = M_FULL[i, j_target]

        if abs(actual) > 1e-6:
            ape = abs(pred - actual) / abs(actual) * 100
        else:
            ape = np.nan

        all_predictions.append({
            'model': MODEL_NAMES[MODEL_IDS[i]],
            'benchmark': bench_name,
            'category': bench_cat,
            'actual': actual,
            'predicted': round(pred, 2),
            'ape_pct': round(ape, 2) if not np.isnan(ape) else np.nan,
        })
        if not np.isnan(ape):
            apes_this_bench.append(ape)

    if apes_this_bench:
        target_bench_stats[bench_name] = {
            'n_train': n_train, 'n_pred': len(apes_this_bench),
            'medape': np.median(apes_this_bench),
            'meanape': np.mean(apes_this_bench),
            'cat': bench_cat,
        }
    else:
        target_bench_stats[bench_name] = {
            'n_train': n_train, 'n_pred': 0, 'medape': np.nan,
            'cat': bench_cat, 'reason': 'no valid predictions'
        }

# ── Summary Stats ────────────────────────────────────────────────────────────
valid_preds = [p for p in all_predictions if not np.isnan(p['ape_pct'])]
all_apes = [p['ape_pct'] for p in valid_preds]

print(f"\n{'='*90}")
print(f"  SUMMARY STATISTICS")
print(f"{'='*90}")
print(f"  Total predictions made:    {len(valid_preds)}")
print(f"  Median APE:                {np.median(all_apes):.2f}%")
print(f"  Mean APE:                  {np.mean(all_apes):.2f}%")
print(f"  Pct within 5%:             {np.mean([a < 5 for a in all_apes])*100:.1f}%")
print(f"  Pct within 10%:            {np.mean([a < 10 for a in all_apes])*100:.1f}%")
print(f"  Pct within 20%:            {np.mean([a < 20 for a in all_apes])*100:.1f}%")

# ── Top 20 BEST predictions (lowest APE) ────────────────────────────────────
valid_sorted_best = sorted(valid_preds, key=lambda x: x['ape_pct'])

print(f"\n{'='*90}")
print(f"  TOP 20 BEST PREDICTIONS (lowest APE)")
print(f"{'='*90}")
print(f"  {'#':>3s}  {'Model':<35s}  {'Benchmark':<28s}  {'Actual':>8s}  {'Predicted':>9s}  {'APE%':>7s}")
print(f"  {'─'*3}  {'─'*35}  {'─'*28}  {'─'*8}  {'─'*9}  {'─'*7}")
for rank, p in enumerate(valid_sorted_best[:20]):
    print(f"  {rank+1:>3d}  {p['model']:<35s}  {p['benchmark']:<28s}  "
          f"{p['actual']:>8.1f}  {p['predicted']:>9.1f}  {p['ape_pct']:>6.2f}%")

# ── Top 20 WORST predictions (highest APE) ──────────────────────────────────
valid_sorted_worst = sorted(valid_preds, key=lambda x: -x['ape_pct'])

print(f"\n{'='*90}")
print(f"  TOP 20 WORST PREDICTIONS (highest APE)")
print(f"{'='*90}")
print(f"  {'#':>3s}  {'Model':<35s}  {'Benchmark':<28s}  {'Actual':>8s}  {'Predicted':>9s}  {'APE%':>7s}")
print(f"  {'─'*3}  {'─'*35}  {'─'*28}  {'─'*8}  {'─'*9}  {'─'*7}")
for rank, p in enumerate(valid_sorted_worst[:20]):
    print(f"  {rank+1:>3d}  {p['model']:<35s}  {p['benchmark']:<28s}  "
          f"{p['actual']:>8.1f}  {p['predicted']:>9.1f}  {p['ape_pct']:>6.2f}%")

# ── Per-benchmark breakdown ─────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  PER-BENCHMARK PREDICTION QUALITY (sorted by MedAPE)")
print(f"{'='*90}")
print(f"  {'Benchmark':<30s}  {'Category':<20s}  {'N_train':>7s}  {'N_pred':>6s}  {'MedAPE':>8s}  {'MeanAPE':>8s}")
print(f"  {'─'*30}  {'─'*20}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}")

bench_rows = [(name, stats) for name, stats in target_bench_stats.items() if stats['n_pred'] > 0]
bench_rows.sort(key=lambda x: x[1]['medape'])

for name, stats in bench_rows:
    print(f"  {name:<30s}  {stats['cat']:<20s}  {stats['n_train']:>7d}  {stats['n_pred']:>6d}  "
          f"{stats['medape']:>7.1f}%  {stats.get('meanape', np.nan):>7.1f}%")

# Benchmarks with too few training data
no_pred = [(name, stats) for name, stats in target_bench_stats.items() if stats['n_pred'] == 0]
if no_pred:
    print(f"\n  Benchmarks with NO predictions (insufficient overlap):")
    for name, stats in no_pred:
        print(f"    {name:<30s}  N_train={stats['n_train']}  reason={stats.get('reason', 'unknown')}")

# ── Breakdown by benchmark CATEGORY ─────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  BREAKDOWN BY BENCHMARK CATEGORY")
print(f"{'='*90}")

cat_preds = defaultdict(list)
for p in valid_preds:
    cat_preds[p['category']].append(p['ape_pct'])

print(f"  {'Category':<25s}  {'N':>5s}  {'MedAPE':>8s}  {'MeanAPE':>8s}  {'<5%':>6s}  {'<10%':>6s}  {'<20%':>6s}")
print(f"  {'─'*25}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}")

cat_rows = []
for cat, apes in sorted(cat_preds.items()):
    n = len(apes)
    med = np.median(apes)
    mean = np.mean(apes)
    p5 = np.mean([a < 5 for a in apes]) * 100
    p10 = np.mean([a < 10 for a in apes]) * 100
    p20 = np.mean([a < 20 for a in apes]) * 100
    cat_rows.append((cat, n, med, mean, p5, p10, p20))

# Sort by MedAPE
cat_rows.sort(key=lambda x: x[2])
for cat, n, med, mean, p5, p10, p20 in cat_rows:
    print(f"  {cat:<25s}  {n:>5d}  {med:>7.1f}%  {mean:>7.1f}%  {p5:>5.1f}%  {p10:>5.1f}%  {p20:>5.1f}%")

# ── Which of the 5 selected benchmarks cover which categories? ──────────────
print(f"\n{'='*90}")
print(f"  CATEGORY COVERAGE OF THE 5 SELECTED BENCHMARKS")
print(f"{'='*90}")
for bid in FIVE_BENCH_IDS:
    j = BENCH_IDS.index(bid)
    print(f"  {BENCH_NAMES[bid]:<35s}  Category: {BENCH_CATS[j]}")
print(f"\n  Note: The 5 benchmarks cover {len(set(BENCH_CATS[BENCH_IDS.index(b)] for b in FIVE_BENCH_IDS))} "
      f"of {len(set(BENCH_CATS))} categories directly.")
print(f"  Categories represented: {sorted(set(BENCH_CATS[BENCH_IDS.index(b)] for b in FIVE_BENCH_IDS))}")
print(f"  Categories NOT represented: "
      f"{sorted(set(BENCH_CATS) - set(BENCH_CATS[BENCH_IDS.index(b)] for b in FIVE_BENCH_IDS))}")

# ── Save ALL predictions to CSV ─────────────────────────────────────────────
outpath = os.path.join(REPO_ROOT, 'results', 'five_bench_predictions.csv')
with open(outpath, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model', 'benchmark', 'category', 'actual', 'predicted', 'ape_pct'])
    writer.writeheader()
    for p in sorted(all_predictions, key=lambda x: x['ape_pct'] if not np.isnan(x['ape_pct']) else 9999):
        writer.writerow(p)

print(f"\n  All {len(all_predictions)} predictions saved to {outpath}")
print(f"\n{'='*90}")
print(f"  DONE")
print(f"{'='*90}")
