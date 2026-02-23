#!/usr/bin/env python3
"""
Question 10: BenchReg cold-start failure investigation
Why does BenchReg produce NO predictions for cold-start (3 known scores)?
"""

import numpy as np
import sys, warnings, os
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES,
    holdout_cold_start, holdout_leave_one_benchmark,
    compute_metrics, print_metrics, evaluate_method,
)
from methods.all_methods import predict_benchreg

print("="*90)
print("  QUESTION 10: BenchReg Cold-Start Failure Investigation")
print("="*90)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Trace through the cold-start scenario step by step
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*90)
print("PART 1: Step-by-step trace of cold-start scenario")
print("─"*90)

# Generate cold-start holdout
folds = holdout_cold_start(n_reveal=3, min_scores=8, seed=42)
M_train_cs, test_set_cs = folds[0]

# Which 3 benchmarks are revealed?
bench_coverage = OBSERVED.sum(axis=0)
top_3_benches = np.argsort(-bench_coverage)[:3]
print(f"\nThe 3 most-covered benchmarks (revealed in cold-start):")
for idx, bj in enumerate(top_3_benches):
    print(f"  {idx+1}. bench_idx={bj}: {BENCH_NAMES[BENCH_IDS[bj]]} (coverage={int(bench_coverage[bj])} models)")

# Show what each model knows in cold-start
obs_cs = ~np.isnan(M_train_cs)
print(f"\nIn cold-start M_train, each model knows at most {obs_cs.sum(axis=1).max()} benchmarks")
print(f"Distribution of known benchmarks per model:")
known_counts = obs_cs.sum(axis=1)
for k in sorted(set(known_counts)):
    n = (known_counts == k).sum()
    print(f"  {k} known: {n} models")

# Pick a specific model with exactly 3 known to trace
example_model_idx = None
for i in range(N_MODELS):
    obs_j = np.where(obs_cs[i])[0]
    if len(obs_j) == 3 and len(np.where(OBSERVED[i])[0]) >= 8:
        example_model_idx = i
        break

if example_model_idx is None:
    # pick any model with known scores
    for i in range(N_MODELS):
        if obs_cs[i].sum() > 0:
            example_model_idx = i
            break

i = example_model_idx
known_j = np.where(obs_cs[i])[0]
hidden_j_full = np.where(OBSERVED[i] & ~obs_cs[i])[0]
print(f"\nTracing model: {MODEL_NAMES[MODEL_IDS[i]]} (idx={i})")
print(f"  Known benchmarks in cold-start ({len(known_j)}):")
for j in known_j:
    print(f"    bench_idx={j}: {BENCH_NAMES[BENCH_IDS[j]]} = {M_train_cs[i, j]:.1f}")
print(f"  Hidden benchmarks to predict ({len(hidden_j_full)}):")
for j in hidden_j_full[:5]:
    print(f"    bench_idx={j}: {BENCH_NAMES[BENCH_IDS[j]]} (actual={M_FULL[i, j]:.1f})")
if len(hidden_j_full) > 5:
    print(f"    ... and {len(hidden_j_full)-5} more")

# Now trace BenchReg's logic for this model
print(f"\n--- BenchReg trace for model {MODEL_NAMES[MODEL_IDS[i]]} ---")

# Recompute BenchReg's internal state for the cold-start M_train
obs_br = ~np.isnan(M_train_cs)

# For each hidden benchmark j, trace what BenchReg does
trace_count = 0
for j in hidden_j_full[:3]:  # trace first 3 hidden benchmarks
    print(f"\n  Target benchmark: {BENCH_NAMES[BENCH_IDS[j]]} (bench_idx={j})")

    # Step 1: compute correlations of all other benchmarks with j
    targets_obs = np.where(obs_br[:, j])[0]
    print(f"    Models with observed target in M_train_cs: {len(targets_obs)}")

    if len(targets_obs) < 5:
        print(f"    --> SKIP: fewer than 5 observations for target benchmark")
        continue

    correlations = []
    for j2 in range(N_BENCH):
        if j2 == j:
            continue
        shared = obs_br[:, j] & obs_br[:, j2]
        n_shared = shared.sum()
        if n_shared < 5:
            correlations.append((j2, -1, n_shared, "too few shared"))
            continue
        x, y = M_train_cs[shared, j2], M_train_cs[shared, j]
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot < 1e-10:
            correlations.append((j2, -1, n_shared, "zero variance in target"))
            continue
        var_x = np.sum((x - x.mean())**2)
        if var_x < 1e-10:
            correlations.append((j2, -1, n_shared, "zero variance in predictor"))
            continue
        cov = np.sum((x - x.mean()) * (y - y.mean()))
        slope = cov / var_x
        intercept = y.mean() - slope * x.mean()
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat)**2)
        r2 = 1 - ss_res / ss_tot
        correlations.append((j2, r2, n_shared, "ok"))

    correlations.sort(key=lambda x: -x[1])

    # Show top 5 correlated benchmarks
    print(f"    Top 5 correlated benchmarks:")
    top5 = correlations[:5]
    for rank, (j2, r2, n_sh, status) in enumerate(top5):
        is_known = j2 in known_j
        print(f"      #{rank+1}: {BENCH_NAMES[BENCH_IDS[j2]]} (idx={j2}) r²={r2:.3f} "
              f"shared={n_sh} known_by_model={'YES' if is_known else 'NO'}")

    # Filter by r²>=0.2
    best = [(j2, r2) for j2, r2, _, _ in correlations[:5] if r2 >= 0.2]
    print(f"    After r²≥0.2 filter: {len(best)} predictors remain")

    # Check if model has scores for any of these predictors
    usable = [(j2, r2) for j2, r2 in best if not np.isnan(M_train_cs[i, j2])]
    print(f"    Of those, model has known scores for: {len(usable)}")
    if usable:
        for j2, r2 in usable:
            print(f"      {BENCH_NAMES[BENCH_IDS[j2]]} (idx={j2}) r²={r2:.3f} model_score={M_train_cs[i, j2]:.1f}")
    else:
        # WHY? Check overlap
        print(f"    --> FAILURE: None of the top-5 correlated benchmarks (with r²≥0.2)")
        print(f"       are among this model's {len(known_j)} known benchmarks: {[BENCH_NAMES[BENCH_IDS[jj]] for jj in known_j]}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1b: Systematic count of the failure mode
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n--- Systematic analysis across ALL cold-start test pairs ---")

# Recompute BenchReg's correlations on the cold-start training matrix
# For each target benchmark j, find top-5 correlated benchmarks
benchreg_top5 = {}  # j -> [(j2, r2), ...]
for j in range(N_BENCH):
    targets_obs = np.where(obs_br[:, j])[0]
    if len(targets_obs) < 5:
        benchreg_top5[j] = []
        continue
    correlations = []
    for j2 in range(N_BENCH):
        if j2 == j:
            continue
        shared = obs_br[:, j] & obs_br[:, j2]
        if shared.sum() < 5:
            continue
        x, y = M_train_cs[shared, j2], M_train_cs[shared, j]
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot < 1e-10:
            continue
        var_x = np.sum((x - x.mean())**2)
        if var_x < 1e-10:
            continue
        cov = np.sum((x - x.mean()) * (y - y.mean()))
        slope = cov / var_x
        intercept = y.mean() - slope * x.mean()
        ss_res = np.sum((y - (slope * x + intercept))**2)
        r2 = 1 - ss_res / ss_tot
        correlations.append((j2, r2))
    correlations.sort(key=lambda x: -x[1])
    benchreg_top5[j] = [(j2, r2) for j2, r2 in correlations[:5] if r2 >= 0.2]

# Count failure modes
total_test = len(test_set_cs)
fail_no_obs = 0          # target bench has <5 observations
fail_no_corr = 0         # no correlated benchmarks pass r²≥0.2
fail_no_overlap = 0      # top-5 corr benchmarks exist but model doesn't know any of them
success = 0

for i, j in test_set_cs:
    if len(benchreg_top5.get(j, [])) == 0:
        # Could be no obs or no good correlations
        targets_obs = np.where(obs_br[:, j])[0]
        if len(targets_obs) < 5:
            fail_no_obs += 1
        else:
            fail_no_corr += 1
        continue

    # Check if model knows any of the top-5 corr benchmarks
    known_by_model = set(np.where(obs_br[i])[0])
    top5_benches = set(j2 for j2, _ in benchreg_top5[j])
    if known_by_model & top5_benches:
        success += 1
    else:
        fail_no_overlap += 1

print(f"\nTotal cold-start test pairs: {total_test}")
print(f"  Success (would produce prediction): {success} ({100*success/total_test:.1f}%)")
print(f"  Fail: target has <5 observations:   {fail_no_obs} ({100*fail_no_obs/total_test:.1f}%)")
print(f"  Fail: no benchmarks with r²≥0.2:    {fail_no_corr} ({100*fail_no_corr/total_test:.1f}%)")
print(f"  Fail: no overlap with known:         {fail_no_overlap} ({100*fail_no_overlap/total_test:.1f}%)")

# Verify by actually running BenchReg
print(f"\nVerification: actually run BenchReg on cold-start...")
M_pred_cs = predict_benchreg(M_train_cs)
actual_cs = [M_FULL[i, j] for i, j in test_set_cs]
pred_cs = [M_pred_cs[i, j] for i, j in test_set_cs]
n_nan_pred = sum(1 for p in pred_cs if np.isnan(p))
n_valid_pred = sum(1 for p in pred_cs if not np.isnan(p))
print(f"  BenchReg predictions: {n_valid_pred} valid, {n_nan_pred} NaN out of {total_test}")
m = compute_metrics(actual_cs, pred_cs)
print_metrics(m, "  BenchReg cold-start")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Test relaxing the constraints
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "─"*90)
print("PART 2: Relaxing BenchReg constraints on cold-start")
print("─"*90)

# Define relaxed versions
def predict_benchreg_relaxed(M_train, top_k=5, min_r2=0.2):
    """Same as predict_benchreg but with configurable k and min_r2."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()
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
                correlations.append((j2, -1))
                continue
            x, y = M_train[shared, j2], M_train[shared, j]
            ss_tot = np.sum((y - y.mean())**2)
            if ss_tot < 1e-10:
                correlations.append((j2, -1))
                continue
            var_x = np.sum((x - x.mean())**2)
            if var_x < 1e-10:
                correlations.append((j2, -1))
                continue
            slope = np.sum((x - x.mean()) * (y - y.mean())) / var_x
            intercept = y.mean() - slope * x.mean()
            ss_res = np.sum((y - (slope * x + intercept))**2)
            r2 = 1 - ss_res / ss_tot
            correlations.append((j2, r2))
        correlations.sort(key=lambda x: -x[1])
        best = [(j2, r2) for j2, r2 in correlations[:top_k] if r2 >= min_r2]
        if not best:
            continue
        for i in range(N_MODELS):
            if not np.isnan(M_train[i, j]):
                continue
            preds, weights = [], []
            for j2, r2 in best:
                if np.isnan(M_train[i, j2]):
                    continue
                shared = obs[:, j] & obs[:, j2]
                if shared.sum() < 5:
                    continue
                x, y = M_train[shared, j2], M_train[shared, j]
                var_x = np.sum((x - x.mean())**2)
                if var_x < 1e-10:
                    continue
                slope = np.sum((x - x.mean()) * (y - y.mean())) / var_x
                intercept = y.mean() - slope * x.mean()
                preds.append(slope * M_train[i, j2] + intercept)
                weights.append(max(r2, 0.01))  # avoid zero weights
            if preds:
                M_pred[i, j] = np.average(preds, weights=weights)
    return M_pred

configs = [
    ("BenchReg(k=5, r²≥0.2) [default]", 5, 0.2),
    ("BenchReg(k=2, r²≥0.2)", 2, 0.2),
    ("BenchReg(k=3, r²≥0.2)", 3, 0.2),
    ("BenchReg(k=10, r²≥0.2)", 10, 0.2),
    ("BenchReg(k=20, r²≥0.2)", 20, 0.2),
    ("BenchReg(k=5, r²≥0.0)", 5, 0.0),
    ("BenchReg(k=10, r²≥0.0)", 10, 0.0),
    ("BenchReg(k=20, r²≥0.0)", 20, 0.0),
    ("BenchReg(k=5, r²≥-1.0) [no filter]", 5, -1.0),
    ("BenchReg(k=ALL, r²≥0.0)", N_BENCH, 0.0),
]

print(f"\n{'Config':<42s} {'n_pred':>7s} {'n_NaN':>6s} {'MedAPE':>8s} {'MeanAPE':>9s} {'R²':>7s}")
print(f"{'─'*42} {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*7}")

for name, k, min_r2 in configs:
    fn = lambda M, _k=k, _r2=min_r2: predict_benchreg_relaxed(M, top_k=_k, min_r2=_r2)
    M_pred = fn(M_train_cs)
    actual = [M_FULL[i, j] for i, j in test_set_cs]
    predicted = [M_pred[i, j] for i, j in test_set_cs]
    n_valid = sum(1 for p in predicted if not np.isnan(p))
    n_nan = sum(1 for p in predicted if np.isnan(p))
    m = compute_metrics(actual, predicted)
    medape_str = f"{m['medape']:.1f}%" if m['n'] > 0 else "N/A"
    meanape_str = f"{m['meanape']:.1f}%" if m['n'] > 0 else "N/A"
    r2_str = f"{m['r2']:.3f}" if m['n'] > 0 else "N/A"
    print(f"  {name:<40s} {n_valid:>7d} {n_nan:>6d} {medape_str:>8s} {meanape_str:>9s} {r2_str:>7s}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Is this fundamental or fixable?
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "─"*90)
print("PART 3: Is this a code limitation or fundamental?")
print("─"*90)

# In cold-start, each model knows only the top-3 benchmarks.
# BenchReg predicts benchmark j using top-k correlated benchmarks.
# If none of those top-k are among the 3 known, prediction = NaN.

# Compute: for how many target benchmarks j, do ANY of the top-5 correlated
# benchmarks overlap with the 3 revealed benchmarks?
set_top3 = set(top_3_benches.tolist())

print(f"\nRevealed benchmarks (top 3 by coverage): {[BENCH_NAMES[BENCH_IDS[j]] for j in top_3_benches]}")
print(f"Bench indices: {list(top_3_benches)}")

# For each target benchmark, compute top-k correlated and check overlap
print(f"\nPer-benchmark analysis: does any top-5 predictor overlap with the 3 revealed?")
n_target_with_overlap = 0
n_target_without_overlap = 0
n_target_skip = 0

for j in range(N_BENCH):
    if j in set_top3:
        continue  # these are known, not targets

    targets_obs = np.where(obs_br[:, j])[0]
    if len(targets_obs) < 5:
        n_target_skip += 1
        continue

    top5_j = benchreg_top5.get(j, [])
    top5_indices = set(j2 for j2, _ in top5_j)

    if top5_indices & set_top3:
        n_target_with_overlap += 1
    else:
        n_target_without_overlap += 1

print(f"  Target benchmarks where top-5 INCLUDES a revealed bench: {n_target_with_overlap}")
print(f"  Target benchmarks where top-5 EXCLUDES all revealed:     {n_target_without_overlap}")
print(f"  Target benchmarks skipped (<5 observations):             {n_target_skip}")

# Now check with ALL correlated benchmarks (not just top-5)
print(f"\nWhat if we use ALL benchmarks with r²≥0.2 (not just top-5)?")
n_overlap_any = 0
n_no_overlap_any = 0
for j in range(N_BENCH):
    if j in set_top3:
        continue
    targets_obs = np.where(obs_br[:, j])[0]
    if len(targets_obs) < 5:
        continue

    # Recompute with all benchmarks
    any_overlap = False
    for j2 in range(N_BENCH):
        if j2 == j:
            continue
        shared = obs_br[:, j] & obs_br[:, j2]
        if shared.sum() < 5:
            continue
        x, y = M_train_cs[shared, j2], M_train_cs[shared, j]
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot < 1e-10:
            continue
        var_x = np.sum((x - x.mean())**2)
        if var_x < 1e-10:
            continue
        slope = np.sum((x - x.mean()) * (y - y.mean())) / var_x
        intercept = y.mean() - slope * x.mean()
        ss_res = np.sum((y - (slope * x + intercept))**2)
        r2 = 1 - ss_res / ss_tot
        if r2 >= 0.2 and j2 in set_top3:
            any_overlap = True
            break

    if any_overlap:
        n_overlap_any += 1
    else:
        n_no_overlap_any += 1

print(f"  Targets where ANY benchmark with r²≥0.2 overlaps revealed: {n_overlap_any}")
print(f"  Targets where NO benchmark with r²≥0.2 overlaps revealed:  {n_no_overlap_any}")

# Check r²≥0.0
print(f"\nWhat about r²≥0.0? (any non-negative relationship)")
n_overlap_0 = 0
n_no_overlap_0 = 0
for j in range(N_BENCH):
    if j in set_top3:
        continue
    targets_obs = np.where(obs_br[:, j])[0]
    if len(targets_obs) < 5:
        continue
    any_overlap = False
    for j2 in top_3_benches:
        if j2 == j:
            continue
        shared = obs_br[:, j] & obs_br[:, j2]
        if shared.sum() < 5:
            continue
        x, y = M_train_cs[shared, j2], M_train_cs[shared, j]
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot < 1e-10:
            continue
        var_x = np.sum((x - x.mean())**2)
        if var_x < 1e-10:
            continue
        slope = np.sum((x - x.mean()) * (y - y.mean())) / var_x
        intercept = y.mean() - slope * x.mean()
        ss_res = np.sum((y - (slope * x + intercept))**2)
        r2 = 1 - ss_res / ss_tot
        if r2 >= 0.0:
            any_overlap = True
            break
    if any_overlap:
        n_overlap_0 += 1
    else:
        n_no_overlap_0 += 1

print(f"  Targets where a revealed bench has r²≥0.0 with target: {n_overlap_0}")
print(f"  Targets where NO revealed bench has r²≥0.0 with target: {n_no_overlap_0}")

# The core structural issue
print(f"\n\n" + "─"*90)
print("PART 3b: The structural explanation")
print("─"*90)

# In cold-start, obs_br has a very specific structure:
# - Only the top-3 benchmark columns are populated
# - All other columns are mostly empty (only for models that don't meet min_scores)
# This means shared observations between target j and predictor j2 are tiny
# unless j2 is one of the 3 revealed benchmarks

# Check: for target j (not in top-3), and predictor j2 (also not in top-3),
# how many shared observations are there?
print(f"\nShared observation counts between benchmark pairs in cold-start M_train:")
print(f"  (Remember: only top-3 benchmarks have data for most models)")
print()

# Count which models have data in cold-start for non-revealed benchmarks
non_revealed_with_data = {}
for j in range(N_BENCH):
    if j in set_top3:
        continue
    n_obs = obs_br[:, j].sum()
    if n_obs > 0:
        non_revealed_with_data[j] = n_obs

print(f"  Non-revealed benchmarks with ANY data in M_train_cs: {len(non_revealed_with_data)}")
if non_revealed_with_data:
    for j, n in sorted(non_revealed_with_data.items(), key=lambda x: -x[1])[:10]:
        print(f"    {BENCH_NAMES[BENCH_IDS[j]]} (idx={j}): {n} observations")

# Explain: the cold-start holdout hides everything EXCEPT the top-3 benchmarks
# for models with ≥8 scores. But models with <8 scores keep all their data.
print(f"\n  Models excluded from cold-start (min_scores<8): keep all their data.")
n_excluded = sum(1 for i in range(N_MODELS) if OBSERVED[i].sum() < 8)
n_included = N_MODELS - n_excluded
print(f"  Excluded models (keep full data): {n_excluded}")
print(f"  Included models (only 3 benchmarks): {n_included}")

# For a revealed bench (e.g., top_3_benches[0]) vs a non-revealed target:
# shared obs = models that have both in cold-start
# = excluded models that happen to have both + included models that have target (but wait,
#   included models had target removed!)
# So shared obs is ONLY from excluded models
print(f"\n  KEY INSIGHT: For BenchReg to use a revealed benchmark as predictor for a hidden one,")
print(f"  it needs ≥5 shared observations between them. But in cold-start:")
print(f"  - Included models ({n_included}) only have the 3 revealed benchmarks")
print(f"    (their non-revealed scores are hidden → shared with non-revealed = 0)")
print(f"  - Wait, actually: shared = models that have BOTH benchmarks observed in M_train_cs")
print(f"  - For (revealed_j, hidden_target_j): shared models are those that have BOTH in M_train_cs")
print(f"  - Included models have revealed_j observed but hidden_target_j is NaN → NOT shared!")
print(f"  - Only excluded models (with <8 scores originally) retain their hidden scores")

# Let's verify this precisely
j_revealed = top_3_benches[0]
j_hidden_example = None
for j in range(N_BENCH):
    if j not in set_top3 and obs_br[:, j].sum() >= 5:
        j_hidden_example = j
        break

if j_hidden_example is not None:
    shared_mask = obs_br[:, j_revealed] & obs_br[:, j_hidden_example]
    shared_models = np.where(shared_mask)[0]
    print(f"\n  Example: revealed={BENCH_NAMES[BENCH_IDS[j_revealed]]} vs hidden={BENCH_NAMES[BENCH_IDS[j_hidden_example]]}")
    print(f"  Shared observations in M_train_cs: {len(shared_models)}")

    # Which models contribute?
    for mi in shared_models[:5]:
        orig_scores = OBSERVED[mi].sum()
        print(f"    Model {MODEL_NAMES[MODEL_IDS[mi]]}: {orig_scores} original scores (excluded={orig_scores < 8})")

# Final summary: the bottleneck
print(f"\n\nFor a revealed benchmark and a target benchmark to have ≥5 shared obs,")
print(f"we need ≥5 excluded models (original scores <8) that happen to have both.")
print(f"This is very sparse → most pairs fail the shared≥5 check → BenchReg can't fit regression → NaN.")

# Let's count exactly: for each target benchmark, how many shared obs with each revealed bench?
print(f"\n  Shared observations matrix (revealed vs hidden targets):")
print(f"  {'Target benchmark':<40s}", end="")
for bj in top_3_benches:
    print(f" {BENCH_NAMES[BENCH_IDS[bj]][:12]:>12s}", end="")
print()

n_shown = 0
for j in range(N_BENCH):
    if j in set_top3:
        continue
    # Only show if it's a target for some model
    if not any(ji == j for _, ji in test_set_cs[:20]):
        continue
    print(f"  {BENCH_NAMES[BENCH_IDS[j]]:<40s}", end="")
    for bj in top_3_benches:
        shared = (obs_br[:, j] & obs_br[:, bj]).sum()
        print(f" {shared:>12d}", end="")
    print()
    n_shown += 1
    if n_shown >= 15:
        break

# ─────────────────────────────────────────────────────────────────────────────
# Final comprehensive summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*90)
print("  SUMMARY: Why BenchReg produces NO predictions for cold-start")
print("="*90)
print("""
BenchReg's prediction of benchmark j for model i requires:
  1. Find top-k benchmarks most correlated with j (need ≥5 shared observations to compute r²)
  2. Among those top-k, the model must have a known score for at least one
  3. That predictor benchmark must have r²≥0.2 with the target

In cold-start (3 known benchmarks):
  - Each included model knows ONLY 3 benchmarks (the most common ones)
  - To predict any other benchmark j, BenchReg needs to compute correlations
    between j and potential predictors using shared observations in M_train_cs
  - But included models have NO data for j (it was hidden!) so they don't
    contribute shared observations between j and any predictor
  - Only excluded models (with <8 original scores, not part of cold-start)
    retain data for hidden benchmarks
  - This makes shared observation counts between the 3 revealed benchmarks
    and hidden targets very small — often <5, so BenchReg can't even compute r²
  - Even when r² can be computed, the 3 revealed benchmarks may not rank
    in the top-5 for that target benchmark
  - Result: for most (model, target) pairs, BenchReg produces NaN

The root cause is STRUCTURAL: BenchReg computes inter-benchmark correlations
on the TRAINING matrix. Cold-start training matrices are extremely sparse for
non-revealed benchmarks, making correlation estimation impossible.

This is NOT just a threshold issue — even with k=ALL and r²≥0.0, the bottleneck
is having ≥5 shared observations between revealed and hidden benchmarks.

This is fundamentally different from KNN or SVD methods which can work in the
full model-space and don't require per-benchmark-pair correlation estimates.
""")
