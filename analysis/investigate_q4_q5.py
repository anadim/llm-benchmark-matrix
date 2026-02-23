#!/usr/bin/env python3
"""
Investigation script for Questions 4 and 5 about the 5-benchmark minimum eval set.

Q4: Why does restricting the POOL of benchmarks to 5 reduce overfitting
    compared to BenchReg which already uses k=5 per prediction?

Q5: What happens if greedy forward selection starts with GPQA Diamond or MMLU
    instead of HLE?
"""

import numpy as np
import sys, warnings
from sklearn.linear_model import Ridge
from collections import defaultdict

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, BENCH_CATS,
    compute_metrics,
)
from methods.all_methods import predict_benchreg

# ============================================================================
#  UTILITIES
# ============================================================================

# Map benchmark names to indices
BENCH_NAME_TO_IDX = {BENCH_IDS[j]: j for j in range(N_BENCH)}

# The 5-benchmark minimum eval set from the greedy selection (HLE-first)
FIVE_SET_IDS = ['hle', 'aime_2025', 'livecodebench', 'swe_bench_verified', 'simpleqa']
FIVE_SET_IDX = [BENCH_NAME_TO_IDX[b] for b in FIVE_SET_IDS]
FIVE_SET_NAMES = [BENCH_NAMES[b] for b in FIVE_SET_IDS]


def predict_from_subset(selected_j):
    """Predict all benchmarks from a subset using per-target ridge regression.
    Returns M_pred and also per-target training sample counts."""
    M_pred = np.full_like(M_FULL, np.nan)
    train_counts = {}

    for jj in selected_j:
        M_pred[:, jj] = M_FULL[:, jj]

    for j_target in range(N_BENCH):
        if j_target in selected_j:
            train_counts[j_target] = -1  # it's in the selected set
            continue
        # Find models that have both all selected benchmarks AND target
        has_selected = np.ones(N_MODELS, dtype=bool)
        for jj in selected_j:
            has_selected &= OBSERVED[:, jj]
        has_target = OBSERVED[:, j_target]
        train_mask = has_selected & has_target

        train_counts[j_target] = train_mask.sum()

        if train_mask.sum() < 3:
            M_pred[:, j_target] = np.nanmean(M_FULL[:, j_target])
            continue

        X_train = M_FULL[train_mask][:, selected_j]
        y_train = M_FULL[train_mask, j_target]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        for i in range(N_MODELS):
            if has_selected[i]:
                x = M_FULL[i, selected_j].reshape(1, -1)
                M_pred[i, j_target] = ridge.predict(x)[0]
            else:
                M_pred[i, j_target] = np.nanmean(M_FULL[:, j_target])

    return M_pred, train_counts


def compute_subset_medape(selected_j):
    """Compute MedAPE for predicting all non-selected benchmarks from selected set."""
    M_pred, train_counts = predict_from_subset(selected_j)
    actuals, preds = [], []
    for i in range(N_MODELS):
        for j2 in range(N_BENCH):
            if j2 in selected_j:
                continue
            if OBSERVED[i, j2] and not np.isnan(M_pred[i, j2]):
                if abs(M_FULL[i, j2]) > 1e-6:
                    actuals.append(M_FULL[i, j2])
                    preds.append(M_pred[i, j2])
    if actuals:
        ape = np.abs((np.array(preds) - np.array(actuals)) / np.array(actuals))
        return np.median(ape) * 100, train_counts
    return np.inf, train_counts


def greedy_forward_selection(forced_first=None, max_steps=10):
    """Greedy forward selection. Optionally force the first benchmark."""
    selected = []
    remaining = list(range(N_BENCH))

    if forced_first is not None:
        selected.append(forced_first)
        remaining.remove(forced_first)
        medape, _ = compute_subset_medape(selected)
        print(f"    Step 1 (forced): +{BENCH_NAMES[BENCH_IDS[forced_first]]:<35s}  MedAPE={medape:.2f}%")

    start_step = len(selected)
    for step in range(start_step, max_steps):
        best_j = None
        best_err = np.inf

        for j in remaining:
            candidate = selected + [j]
            err, _ = compute_subset_medape(candidate)
            if err < best_err:
                best_err = err
                best_j = j

        if best_j is not None:
            selected.append(best_j)
            remaining.remove(best_j)
            names = [BENCH_NAMES[BENCH_IDS[j]] for j in selected]
            print(f"    Step {step+1}: +{BENCH_NAMES[BENCH_IDS[best_j]]:<35s}  MedAPE={best_err:.2f}%  Set: {', '.join(names)}")

    return selected


# ============================================================================
#  QUESTION 4: Why does restricting the pool reduce overfitting?
# ============================================================================

print("=" * 100)
print("  QUESTION 4: Why does the 5-benchmark ridge beat BenchReg's locally-selected k=5?")
print("=" * 100)

# --- Part A: Training sample counts for 5-benchmark ridge ---
print("\n--- A) 5-benchmark ridge: training samples per target benchmark ---")
_, train_counts_5bench = predict_from_subset(FIVE_SET_IDX)

# How many models have ALL 5 benchmarks?
has_all_5 = np.ones(N_MODELS, dtype=bool)
for jj in FIVE_SET_IDX:
    has_all_5 &= OBSERVED[:, jj]
n_have_all_5 = has_all_5.sum()
print(f"\n  Models that have ALL 5 selected benchmarks: {n_have_all_5} / {N_MODELS}")
print(f"  Selected benchmarks and their individual coverages:")
for jj in FIVE_SET_IDX:
    cov = OBSERVED[:, jj].sum()
    print(f"    {BENCH_NAMES[BENCH_IDS[jj]]:<35s}  coverage={cov}")

print(f"\n  Training samples (models with all 5 + target) per target benchmark:")
print(f"  {'Target Benchmark':<40s}  {'N_train':>8s}  {'Cov':>5s}")
counts_list = []
for j in range(N_BENCH):
    if j in FIVE_SET_IDX:
        continue
    n_train = train_counts_5bench[j]
    cov = OBSERVED[:, j].sum()
    counts_list.append((j, n_train, cov))

counts_list.sort(key=lambda x: -x[1])
for j, n_train, cov in counts_list:
    print(f"  {BENCH_NAMES[BENCH_IDS[j]]:<40s}  {n_train:>8d}  {cov:>5d}")

median_5bench_train = np.median([c[1] for c in counts_list])
mean_5bench_train = np.mean([c[1] for c in counts_list])
print(f"\n  Median training samples: {median_5bench_train:.0f}")
print(f"  Mean training samples:   {mean_5bench_train:.1f}")
print(f"  Min training samples:    {min(c[1] for c in counts_list)}")
print(f"  Max training samples:    {max(c[1] for c in counts_list)}")


# --- Part B: BenchReg's locally-selected benchmarks and training samples ---
print("\n\n--- B) BenchReg: which 5 benchmarks does it select per target, and how many training samples? ---")

obs = ~np.isnan(M_FULL)
benchreg_stats = []

# For each target benchmark, find BenchReg's top-5 and compute shared coverage
print(f"\n  {'Target':<35s}  {'Top-5 selected benchmarks':<80s}  {'Shared':>7s}")
print(f"  {'─'*35}  {'─'*80}  {'─'*7}")

for j in range(N_BENCH):
    targets_obs = np.where(obs[:, j])[0]
    if len(targets_obs) < 5:
        continue
    correlations = []
    for j2 in range(N_BENCH):
        if j2 == j:
            continue
        shared = obs[:, j] & obs[:, j2]
        if shared.sum() < 5:
            correlations.append((j2, -1, shared.sum()))
            continue
        x, y = M_FULL[shared, j2], M_FULL[shared, j]
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot < 1e-10:
            correlations.append((j2, -1, shared.sum()))
            continue
        var_x = np.sum((x - x.mean())**2)
        if var_x < 1e-10:
            correlations.append((j2, -1, shared.sum()))
            continue
        slope = np.sum((x-x.mean())*(y-y.mean())) / var_x
        intercept = y.mean() - slope * x.mean()
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat)**2)
        r2 = 1 - ss_res / ss_tot
        correlations.append((j2, r2, shared.sum()))

    correlations.sort(key=lambda x: -x[1])
    best5 = [(j2, r2, sh) for j2, r2, sh in correlations[:5] if r2 >= 0.2]

    if not best5:
        continue

    # Compute: how many models have ALL of target + best5 benchmarks?
    selected_js = [j2 for j2, _, _ in best5]
    has_all = obs[:, j].copy()
    for j2 in selected_js:
        has_all &= obs[:, j2]
    n_shared = has_all.sum()

    bench_names_str = ", ".join(f"{BENCH_NAMES[BENCH_IDS[j2]]}(r²={r2:.2f},sh={sh})"
                                 for j2, r2, sh in best5)

    benchreg_stats.append((j, BENCH_NAMES[BENCH_IDS[j]], n_shared, best5,
                           OBSERVED[:, j].sum()))

# Sort by shared count
benchreg_stats.sort(key=lambda x: x[2])

# Print all
for j, name, n_shared, best5, target_cov in benchreg_stats:
    bench_str = ", ".join(f"{BENCH_NAMES[BENCH_IDS[j2]]}(r²={r2:.2f})" for j2, r2, _ in best5)
    print(f"  {name:<35s}  {bench_str:<80s}  {n_shared:>7d}")

# Summary statistics
shared_counts = [x[2] for x in benchreg_stats]
print(f"\n  BenchReg's shared observation counts (target + its 5 selected benchmarks):")
print(f"    Median: {np.median(shared_counts):.0f}")
print(f"    Mean:   {np.mean(shared_counts):.1f}")
print(f"    Min:    {min(shared_counts)}")
print(f"    Max:    {max(shared_counts)}")
print(f"    # targets with shared < 10: {sum(1 for s in shared_counts if s < 10)}")
print(f"    # targets with shared < 20: {sum(1 for s in shared_counts if s < 20)}")
print(f"    # targets with shared < 30: {sum(1 for s in shared_counts if s < 30)}")


# --- Part C: Direct comparison ---
print("\n\n--- C) Head-to-head comparison: training data per target ---")
print(f"\n  {'Target':<35s}  {'5-bench ridge N':>16s}  {'BenchReg shared':>16s}  {'Diff':>8s}")
print(f"  {'─'*35}  {'─'*16}  {'─'*16}  {'─'*8}")

benchreg_dict = {x[0]: x[2] for x in benchreg_stats}
comparison_data = []
for j, n_5bench, cov in counts_list:
    n_benchreg = benchreg_dict.get(j, -1)
    if n_benchreg < 0:
        continue
    diff = n_5bench - n_benchreg
    comparison_data.append((j, n_5bench, n_benchreg, diff))
    print(f"  {BENCH_NAMES[BENCH_IDS[j]]:<35s}  {n_5bench:>16d}  {n_benchreg:>16d}  {diff:>+8d}")

diffs = [c[3] for c in comparison_data]
print(f"\n  5-bench ridge has MORE training data in {sum(1 for d in diffs if d > 0)}/{len(diffs)} targets")
print(f"  5-bench ridge has FEWER training data in {sum(1 for d in diffs if d < 0)}/{len(diffs)} targets")
print(f"  Mean difference (5bench - BenchReg): {np.mean(diffs):+.1f}")
print(f"  Median difference: {np.median(diffs):+.1f}")


# --- Part D: The key insight ---
print("\n\n--- D) KEY HYPOTHESIS TESTED ---")
print(f"""
  The 5 globally-selected benchmarks {FIVE_SET_NAMES} have:
  - {n_have_all_5} models with ALL 5 (= effective training pool for ridge)
  - Median training samples per target: {median_5bench_train:.0f}

  BenchReg's locally-correlated 5 benchmarks have:
  - Median shared coverage with target: {np.median(shared_counts):.0f}
  - Many targets with very few shared observations (< 10): {sum(1 for s in shared_counts if s < 10)}

  But BenchReg doesn't require ALL 5 predictors to be present. It uses each
  predictor independently (weighted average of individual regressions).
  Let's check: how many models does each individual BenchReg predictor regression use?
""")

# For BenchReg, the ACTUAL training data per individual regression is just
# the pairwise shared count between predictor and target
print("  BenchReg actually fits per-predictor simple regressions (not joint ridge).")
print("  Each predictor j2 for target j uses: models having BOTH j and j2")
print("  Then averages predictions weighted by r².\n")

# Show pairwise counts for a few example targets
example_targets = ['hle', 'aime_2025', 'arc_agi_2', 'imo_2025', 'usamo_2025',
                   'gpqa_diamond', 'mmlu', 'swe_bench_verified']
for tid in example_targets:
    if tid not in BENCH_NAME_TO_IDX:
        continue
    j = BENCH_NAME_TO_IDX[tid]
    # Find this target in benchreg_stats
    match = [x for x in benchreg_stats if x[0] == j]
    if not match:
        continue
    _, name, n_shared, best5, target_cov = match[0]
    print(f"  Target: {name} (coverage={target_cov})")
    for j2, r2, sh in best5:
        cov_j2 = OBSERVED[:, j2].sum()
        print(f"    Predictor: {BENCH_NAMES[BENCH_IDS[j2]]:<30s}  r²={r2:.2f}  "
              f"pairwise_shared={sh}  predictor_cov={cov_j2}")
    print()

# --- Part E: The REAL mechanism ---
print("\n--- E) THE REAL MECHANISM: Joint ridge vs independent regressions ---")
print("""
  5-benchmark ridge fits a JOINT model: y = w1*x1 + w2*x2 + ... + w5*x5 + b
  - Needs models that have ALL 5 features simultaneously
  - More parameters to fit (5 weights) but gets multivariate signal
  - Ridge regularization (alpha=1.0) prevents overfitting

  BenchReg fits INDEPENDENT simple regressions: y = a*x_k + b for each k
  - Each regression uses different (possibly larger) training sets
  - But predictions are a HEURISTIC average, not an optimal multivariate fit
  - No regularization on the averaging step
  - Selected benchmarks vary per target (non-stationary predictor set)

  Let's verify via proper holdout: hide 20% of observed entries, predict them
  using (a) 5-bench ridge, (b) BenchReg restricted to same 5, (c) BenchReg full.
""")

# BenchReg restricted to the 5-benchmark pool
def predict_benchreg_restricted(M_train, allowed_js):
    """BenchReg but only allowed to use benchmarks in allowed_js as predictors."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    for j in range(N_BENCH):
        targets_obs = np.where(obs[:, j])[0]
        if len(targets_obs) < 5:
            continue

        preds_all = []

        for j2 in allowed_js:
            if j2 == j:
                continue
            shared = obs[:, j] & obs[:, j2]
            if shared.sum() < 5:
                continue
            x, y = M_train[shared, j2], M_train[shared, j]
            ss_tot = np.sum((y - y.mean())**2)
            if ss_tot < 1e-10:
                continue
            var_x = np.sum((x - x.mean())**2)
            if var_x < 1e-10:
                continue
            slope = np.sum((x-x.mean())*(y-y.mean())) / var_x
            intercept = y.mean() - slope * x.mean()
            y_hat = slope * x + intercept
            ss_res = np.sum((y - y_hat)**2)
            r2 = 1 - ss_res / ss_tot
            if r2 < 0.2:
                continue
            preds_all.append((j2, slope, intercept, r2))

        for i in range(N_MODELS):
            if not np.isnan(M_train[i, j]):
                continue
            preds, weights = [], []
            for j2, slope, intercept, r2 in preds_all:
                if np.isnan(M_train[i, j2]):
                    continue
                preds.append(slope * M_train[i, j2] + intercept)
                weights.append(r2)
            if preds:
                M_pred[i, j] = np.average(preds, weights=weights)
    return M_pred


def predict_5bench_ridge(M_train, selected_j):
    """Predict all benchmarks from a fixed subset using joint ridge regression."""
    M_pred = M_train.copy()
    for j_target in range(N_BENCH):
        if j_target in selected_j:
            continue
        has_selected = np.ones(N_MODELS, dtype=bool)
        for jj in selected_j:
            has_selected &= ~np.isnan(M_train[:, jj])
        has_target = ~np.isnan(M_train[:, j_target])
        train_mask = has_selected & has_target

        if train_mask.sum() < 3:
            M_pred[:, j_target] = np.where(np.isnan(M_train[:, j_target]),
                                            np.nanmean(M_train[:, j_target]),
                                            M_train[:, j_target])
            continue

        X_train = M_train[train_mask][:, selected_j]
        y_train = M_train[train_mask, j_target]
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        for i in range(N_MODELS):
            if np.isnan(M_train[i, j_target]) and has_selected[i]:
                x = M_train[i, selected_j].reshape(1, -1)
                M_pred[i, j_target] = ridge.predict(x)[0]
    return M_pred


# Proper holdout evaluation: random 20% cell holdout, 3 folds
from evaluation_harness import holdout_random_cells, evaluate_method

folds = holdout_random_cells(frac=0.2, n_folds=3, seed=42)

print("  Computing MedAPE via proper holdout (20% cells, 3-fold)...\n")

# Method 1: 5-bench ridge (joint)
overall_ridge, _ = evaluate_method(lambda M: predict_5bench_ridge(M, FIVE_SET_IDX), folds)
medape_5ridge_ho = overall_ridge['medape']
print(f"  5-benchmark Ridge (joint):              MedAPE = {medape_5ridge_ho:.2f}%")

# Method 2: BenchReg restricted to same 5
overall_restr, _ = evaluate_method(lambda M: predict_benchreg_restricted(M, set(FIVE_SET_IDX)), folds)
medape_restricted_ho = overall_restr['medape']
print(f"  BenchReg restricted to same 5 (indep):  MedAPE = {medape_restricted_ho:.2f}%")

# Method 3: BenchReg full
overall_full, _ = evaluate_method(predict_benchreg, folds)
medape_full_ho = overall_full['medape']
print(f"  BenchReg full 49-pool (indep, k=5):     MedAPE = {medape_full_ho:.2f}%")

# Also: non-holdout MedAPE for 5-bench ridge (what we report as 4.8%)
medape_5ridge, _ = compute_subset_medape(FIVE_SET_IDX)
print(f"\n  (For reference, 5-bench ridge on FULL data: MedAPE = {medape_5ridge:.2f}%)")
print(f"  (This is the 'oracle' number since ridge sees all targets during training)")

print(f"""
  INTERPRETATION:
  - 5-bench Ridge (holdout):          {medape_5ridge_ho:.2f}%
  - BenchReg restricted to same 5:    {medape_restricted_ho:.2f}%
  - BenchReg full 49-pool:            {medape_full_ho:.2f}%

  If ridge beats BenchReg-restricted: benefit is partly from JOINT regression
  If BenchReg-restricted beats full: benefit is from BENCHMARK CHOICE (fewer noisier predictors)
""")


# ============================================================================
#  QUESTION 5: Greedy selection starting with different first benchmarks
# ============================================================================

print("\n" + "=" * 100)
print("  QUESTION 5: Greedy forward selection with different starting benchmarks")
print("=" * 100)

# --- 5a: GPQA Diamond alone ---
gpqa_idx = BENCH_NAME_TO_IDX['gpqa_diamond']
hle_idx = BENCH_NAME_TO_IDX['hle']
mmlu_idx = BENCH_NAME_TO_IDX['mmlu']

print(f"\n--- A) Single-benchmark MedAPE ---")
for name, idx in [('HLE', hle_idx), ('GPQA Diamond', gpqa_idx), ('MMLU', mmlu_idx)]:
    medape_single, _ = compute_subset_medape([idx])
    cov = OBSERVED[:, idx].sum()
    print(f"  {name:<20s}  MedAPE={medape_single:.2f}%  (coverage={cov})")

# --- 5b: Greedy starting with GPQA Diamond ---
print(f"\n--- B) Greedy selection starting with GPQA Diamond ---")
selected_gpqa = greedy_forward_selection(forced_first=gpqa_idx, max_steps=10)

gpqa_5set = selected_gpqa[:5]
medape_gpqa5, _ = compute_subset_medape(gpqa_5set)
print(f"\n  GPQA-first 5-set: {[BENCH_NAMES[BENCH_IDS[j]] for j in gpqa_5set]}")
print(f"  MedAPE: {medape_gpqa5:.2f}%")

# --- 5c: Greedy starting with MMLU ---
print(f"\n--- C) Greedy selection starting with MMLU ---")
selected_mmlu = greedy_forward_selection(forced_first=mmlu_idx, max_steps=10)

mmlu_5set = selected_mmlu[:5]
medape_mmlu5, _ = compute_subset_medape(mmlu_5set)
print(f"\n  MMLU-first 5-set: {[BENCH_NAMES[BENCH_IDS[j]] for j in mmlu_5set]}")
print(f"  MedAPE: {medape_mmlu5:.2f}%")

# --- 5d: Original HLE-first for reference ---
print(f"\n--- D) Original HLE-first greedy selection (for reference) ---")
selected_hle = greedy_forward_selection(forced_first=hle_idx, max_steps=10)

hle_5set = selected_hle[:5]
medape_hle5, _ = compute_subset_medape(hle_5set)
print(f"\n  HLE-first 5-set: {[BENCH_NAMES[BENCH_IDS[j]] for j in hle_5set]}")
print(f"  MedAPE: {medape_hle5:.2f}%")

# --- 5e: Also try fully unconstrained greedy (no forced first) ---
print(f"\n--- E) Fully unconstrained greedy (no forced first) ---")
selected_free = greedy_forward_selection(forced_first=None, max_steps=10)

free_5set = selected_free[:5]
medape_free5, _ = compute_subset_medape(free_5set)
print(f"\n  Unconstrained 5-set: {[BENCH_NAMES[BENCH_IDS[j]] for j in free_5set]}")
print(f"  MedAPE: {medape_free5:.2f}%")


# --- Summary ---
print("\n\n" + "=" * 100)
print("  SUMMARY")
print("=" * 100)

print(f"""
  QUESTION 4 ANSWER:
  ==================
  5-benchmark ridge training samples per target:
    - Median: {median_5bench_train:.0f}, Mean: {mean_5bench_train:.1f}
    - Models with all 5: {n_have_all_5}

  BenchReg (49-pool) shared observations per target:
    - Median: {np.median(shared_counts):.0f}, Mean: {np.mean(shared_counts):.1f}
    - Targets with < 10 shared: {sum(1 for s in shared_counts if s < 10)}
    - Targets with < 20 shared: {sum(1 for s in shared_counts if s < 20)}

  5-bench ridge has MORE training data in {sum(1 for d in diffs if d > 0)}/{len(diffs)} targets
  (mean diff: {np.mean(diffs):+.1f})

  Key results (proper holdout):
    5-benchmark Ridge (joint):             {medape_5ridge_ho:.2f}%
    BenchReg restricted to same 5 (indep): {medape_restricted_ho:.2f}%
    BenchReg full 49-pool (k=5):           {medape_full_ho:.2f}%

  5-bench ridge on full data (no holdout): {medape_5ridge:.2f}%

  QUESTION 5 ANSWER:
  ==================
  Single-benchmark MedAPE:
    HLE alone:          see above
    GPQA Diamond alone: see above
    MMLU alone:         see above

  5-benchmark set MedAPE:
    HLE-first:           {medape_hle5:.2f}%  {[BENCH_NAMES[BENCH_IDS[j]] for j in hle_5set]}
    GPQA-first:          {medape_gpqa5:.2f}%  {[BENCH_NAMES[BENCH_IDS[j]] for j in gpqa_5set]}
    MMLU-first:          {medape_mmlu5:.2f}%  {[BENCH_NAMES[BENCH_IDS[j]] for j in mmlu_5set]}
    Unconstrained:       {medape_free5:.2f}%  {[BENCH_NAMES[BENCH_IDS[j]] for j in free_5set]}
""")
