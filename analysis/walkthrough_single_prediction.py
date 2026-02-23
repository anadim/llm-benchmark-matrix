#!/usr/bin/env python3
"""
End-to-end walkthrough of a single BenchReg+KNN(alpha=0.6) prediction.
Target: Claude Sonnet 4 on AIME 2025 (actual = 76.3).
"""

import numpy as np
import sys, warnings, os
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))

from build_benchmark_matrix import MODELS, BENCHMARKS, DATA
from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, col_normalize, col_stats,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0: Identify the target cell
# ═══════════════════════════════════════════════════════════════════════════════
model_id = "claude-sonnet-4"
bench_id = "aime_2025"

i_target = MODEL_IDS.index(model_id)
j_target = BENCH_IDS.index(bench_id)
actual_value = M_FULL[i_target, j_target]

print("=" * 90)
print("  END-TO-END WALKTHROUGH: BenchReg + KNN(alpha=0.6)")
print("=" * 90)
print(f"\n  Target: {MODEL_NAMES[model_id]} on {BENCH_NAMES[bench_id]}")
print(f"  Matrix indices: model i={i_target}, benchmark j={j_target}")
print(f"  Actual value: {actual_value}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Create modified matrix (hide the target cell)
# ═══════════════════════════════════════════════════════════════════════════════
M_train = M_FULL.copy()
M_train[i_target, j_target] = np.nan

print(f"\n  Hid entry M[{i_target},{j_target}]. Now it's NaN in M_train.")
print(f"  Claude Sonnet 4 has {(~np.isnan(M_train[i_target])).sum()} observed benchmarks remaining.")

# Show what benchmarks Claude Sonnet 4 still has
obs_j = np.where(~np.isnan(M_train[i_target]))[0]
print(f"\n  Claude Sonnet 4's remaining observed benchmarks:")
for j in obs_j:
    print(f"    {BENCH_NAMES[BENCH_IDS[j]]:<35s}  = {M_train[i_target, j]:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: BenchReg — find top-5 predictor benchmarks for AIME 2025
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STEP 2: BenchReg — Predicting AIME 2025 from correlated benchmarks")
print("=" * 90)

obs = ~np.isnan(M_train)
top_k = 5
min_r2 = 0.2

# Compute R² of each benchmark with AIME 2025 (pairwise complete observations)
correlations = []
for j2 in range(N_BENCH):
    if j2 == j_target:
        continue
    shared = obs[:, j_target] & obs[:, j2]
    n_shared = shared.sum()
    if n_shared < 5:
        correlations.append((j2, -1, 0, n_shared, 0, 0, 0))
        continue
    x = M_train[shared, j2]
    y = M_train[shared, j_target]
    ss_tot = np.sum((y - y.mean())**2)
    if ss_tot < 1e-10:
        correlations.append((j2, -1, 0, n_shared, 0, 0, 0))
        continue
    cov = np.sum((x - x.mean()) * (y - y.mean()))
    var_x = np.sum((x - x.mean())**2)
    if var_x < 1e-10:
        correlations.append((j2, -1, 0, n_shared, 0, 0, 0))
        continue
    slope = cov / var_x
    intercept = y.mean() - slope * x.mean()
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat)**2)
    r2 = 1 - ss_res / ss_tot

    # Also compute Pearson correlation for display
    pearson_r = np.corrcoef(x, y)[0, 1]

    correlations.append((j2, r2, pearson_r, n_shared, slope, intercept, var_x))

correlations.sort(key=lambda x: -x[1])

print(f"\n  How correlations are computed:")
print(f"  - For each other benchmark j2, find models observed on BOTH j2 AND AIME 2025")
print(f"  - Compute R² from simple linear regression y=slope*x+intercept")
print(f"  - This is Pearson on PAIRWISE COMPLETE OBSERVATIONS (no imputation)")
print(f"  - Minimum 5 shared observations required")

print(f"\n  All benchmarks ranked by R² with AIME 2025:")
print(f"  {'Rank':>4s}  {'Benchmark':<35s}  {'R²':>6s}  {'Pearson r':>10s}  {'n_shared':>8s}  {'slope':>8s}  {'intercept':>10s}")
print(f"  {'─'*4}  {'─'*35}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
for rank, (j2, r2, pr, ns, sl, ic, vx) in enumerate(correlations[:20]):
    marker = " <── SELECTED" if rank < top_k and r2 >= min_r2 else ""
    print(f"  {rank+1:>4d}  {BENCH_NAMES[BENCH_IDS[j2]]:<35s}  {r2:>6.3f}  {pr:>+10.3f}  {ns:>8d}  {sl:>8.3f}  {ic:>10.2f}{marker}")

# Select top-k with r2 >= min_r2
best = [(j2, r2, pr, ns, sl, ic) for j2, r2, pr, ns, sl, ic, vx in correlations[:top_k] if r2 >= min_r2]
print(f"\n  Selected {len(best)} predictor benchmarks (top-{top_k} with R² >= {min_r2}):")
for j2, r2, pr, ns, sl, ic in best:
    print(f"    {BENCH_NAMES[BENCH_IDS[j2]]:<35s}  R²={r2:.3f}  Pearson r={pr:+.3f}  n_shared={ns}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: BenchReg prediction for Claude Sonnet 4
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  Now predict Claude Sonnet 4 on AIME 2025 using these predictors:")
print(f"  For each predictor benchmark j2:")
print(f"    1. Re-fit slope/intercept on models with BOTH j2 and AIME 2025 observed")
print(f"       (same as before, but now Claude Sonnet 4's AIME 2025 is hidden)")
print(f"    2. pred_j2 = slope * claude_sonnet4_score_on_j2 + intercept")
print(f"    3. Weight = R²")
print(f"    4. Final = weighted average of per-predictor predictions")

preds_benchreg = []
weights_benchreg = []
for j2, r2, pr, ns, sl, ic in best:
    if np.isnan(M_train[i_target, j2]):
        print(f"\n    SKIP {BENCH_NAMES[BENCH_IDS[j2]]}: Claude Sonnet 4 has no score here")
        continue
    # Recompute regression on shared observations
    shared = obs[:, j_target] & obs[:, j2]
    x = M_train[shared, j2]
    y = M_train[shared, j_target]
    cov = np.sum((x - x.mean()) * (y - y.mean()))
    var_x = np.sum((x - x.mean())**2)
    slope = cov / var_x
    intercept = y.mean() - slope * x.mean()

    cs4_score = M_train[i_target, j2]
    pred = slope * cs4_score + intercept

    print(f"\n    Predictor: {BENCH_NAMES[BENCH_IDS[j2]]}")
    print(f"      Regression: AIME_2025 = {slope:.4f} * {BENCH_NAMES[BENCH_IDS[j2]]} + {intercept:.2f}")
    print(f"      (fitted on {shared.sum()} shared models)")
    print(f"      Claude Sonnet 4 score on {BENCH_NAMES[BENCH_IDS[j2]]} = {cs4_score:.1f}")
    print(f"      Prediction: {slope:.4f} * {cs4_score:.1f} + {intercept:.2f} = {pred:.2f}")
    print(f"      Weight (R²) = {r2:.3f}")

    preds_benchreg.append(pred)
    weights_benchreg.append(r2)

if preds_benchreg:
    benchreg_pred = np.average(preds_benchreg, weights=weights_benchreg)
    print(f"\n  BenchReg final prediction:")
    print(f"    Weighted average = sum(w_i * pred_i) / sum(w_i)")
    numerator = sum(w * p for w, p in zip(weights_benchreg, preds_benchreg))
    denominator = sum(weights_benchreg)
    print(f"    = ({' + '.join(f'{w:.3f}*{p:.2f}' for w, p in zip(weights_benchreg, preds_benchreg))}) / {denominator:.3f}")
    print(f"    = {numerator:.2f} / {denominator:.3f}")
    print(f"    = {benchreg_pred:.2f}")
else:
    benchreg_pred = np.nan
    print(f"\n  BenchReg: NO prediction possible (no shared predictors)")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: KNN — find 5 most similar models
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STEP 4: KNN (B2) — Finding 5 most similar models to Claude Sonnet 4")
print("=" * 90)

M_norm, cm, cs = col_normalize(M_train)
obs_norm = ~np.isnan(M_norm)
k = 5

# Claude Sonnet 4's observed benchmarks in normalized space
shared_all = obs_norm[i_target]
print(f"\n  Working in z-score normalized space (per-column z-scores)")
print(f"  Claude Sonnet 4 has {shared_all.sum()} observed benchmarks for similarity computation")

sims = np.full(N_MODELS, -999.0)
sim_details = []
for k2 in range(N_MODELS):
    if k2 == i_target:
        continue
    shared = shared_all & obs_norm[k2]
    n_shared = shared.sum()
    if n_shared < 3:
        sim_details.append((k2, -999, n_shared))
        continue
    a = M_norm[i_target, shared]
    b = M_norm[k2, shared]
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    cos_sim = np.dot(a, b) / denom
    sims[k2] = cos_sim
    sim_details.append((k2, cos_sim, n_shared))

# Sort by similarity
sim_details.sort(key=lambda x: -x[1])
top_k_indices = [k2 for k2, s, ns in sim_details[:k] if s > -999]

print(f"\n  Cosine similarity computed on z-scored shared benchmarks")
print(f"\n  All models ranked by cosine similarity with Claude Sonnet 4:")
print(f"  {'Rank':>4s}  {'Model':<40s}  {'CosSim':>8s}  {'n_shared':>8s}")
print(f"  {'─'*4}  {'─'*40}  {'─'*8}  {'─'*8}")
for rank, (k2, s, ns) in enumerate(sim_details[:15]):
    marker = " <── TOP-5" if rank < k and s > -999 else ""
    if s > -999:
        print(f"  {rank+1:>4d}  {MODEL_NAMES[MODEL_IDS[k2]]:<40s}  {s:>8.4f}  {ns:>8d}{marker}")

# Now compute KNN prediction
print(f"\n  Computing KNN prediction for AIME 2025:")
weights_knn_raw = np.maximum(np.array([sims[k2] for k2 in top_k_indices]), 0.01)
weights_knn = weights_knn_raw / weights_knn_raw.sum()

print(f"\n  {'Model':<40s}  {'CosSim':>8s}  {'Weight':>8s}  {'AIME2025 z':>10s}  {'AIME2025 raw':>12s}")
print(f"  {'─'*40}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}")
vals_knn = []
ws_knn = []
for idx, ki in enumerate(top_k_indices):
    cos_sim_val = sims[ki]
    weight = weights_knn[idx]
    if not np.isnan(M_norm[ki, j_target]):
        z_val = M_norm[ki, j_target]
        raw_val = M_train[ki, j_target]
        vals_knn.append(z_val)
        ws_knn.append(weight)
        print(f"  {MODEL_NAMES[MODEL_IDS[ki]]:<40s}  {cos_sim_val:>8.4f}  {weight:>8.4f}  {z_val:>+10.4f}  {raw_val:>12.1f}")
    else:
        print(f"  {MODEL_NAMES[MODEL_IDS[ki]]:<40s}  {cos_sim_val:>8.4f}  {weight:>8.4f}  {'NaN':>10s}  {'NaN':>12s}  (no AIME 2025)")

if vals_knn:
    pred_z = np.average(vals_knn, weights=ws_knn)
    knn_pred = pred_z * cs[j_target] + cm[j_target]
    print(f"\n  KNN prediction (in z-score space):")
    print(f"    Weighted average z = {pred_z:+.4f}")
    print(f"    De-normalize: {pred_z:.4f} * {cs[j_target]:.4f} + {cm[j_target]:.4f} = {knn_pred:.2f}")
else:
    knn_pred = cm[j_target]
    print(f"\n  KNN: No neighbors have AIME 2025, falling back to column mean = {knn_pred:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Blend with alpha=0.6
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STEP 5: BLEND — alpha=0.6 * BenchReg + (1-alpha) * KNN")
print("=" * 90)

alpha = 0.6
final_pred = alpha * benchreg_pred + (1 - alpha) * knn_pred
error = final_pred - actual_value
ape = abs(error) / abs(actual_value) * 100

print(f"\n  BenchReg prediction:  {benchreg_pred:.2f}")
print(f"  KNN prediction:       {knn_pred:.2f}")
print(f"  alpha = {alpha}")
print(f"\n  final = {alpha} * {benchreg_pred:.2f} + {1-alpha} * {knn_pred:.2f}")
print(f"        = {alpha * benchreg_pred:.2f} + {(1-alpha) * knn_pred:.2f}")
print(f"        = {final_pred:.2f}")
print(f"\n  Actual value:         {actual_value}")
print(f"  Error:                {error:+.2f}")
print(f"  APE:                  {ape:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Verify against the actual predict_blend function
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STEP 6: VERIFICATION — Compare with actual predict_blend function")
print("=" * 90)

sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))
from all_methods import predict_blend, predict_benchreg, predict_B2

M_breg = predict_benchreg(M_train)
M_knn = predict_B2(M_train, k=5)
M_blend = predict_blend(M_train, alpha=0.6)

print(f"\n  predict_benchreg:  M_breg[{i_target},{j_target}]  = {M_breg[i_target, j_target]:.2f}")
print(f"  predict_B2(k=5):   M_knn[{i_target},{j_target}]   = {M_knn[i_target, j_target]:.2f}")
print(f"  predict_blend(0.6): M_blend[{i_target},{j_target}] = {M_blend[i_target, j_target]:.2f}")
print(f"  Our manual calc:                          = {final_pred:.2f}")
print(f"  Match: {abs(final_pred - M_blend[i_target, j_target]) < 0.01}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7: How was alpha=0.6 chosen?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STEP 7: HOW WAS alpha=0.6 CHOSEN?")
print("=" * 90)

print("""
  Looking at the codebase:

  1. In matrix_completion_v8.py (the development file), THREE alpha
     values were tested as separate methods in the evaluation sweep:
       - BReg+KNN(alpha=0.6)
       - BReg+KNN(alpha=0.65)
       - BReg+KNN(alpha=0.7)
     Plus log variants:
       - LogBReg+KNN(alpha=0.6)
       - LogBReg+KNN(alpha=0.65)

  2. These were evaluated via cv_evaluate() on random holdout folds and
     per-model holdout folds, comparing MedAPE, MAPE, and %within-10%.

  3. The final evaluation_harness/all_methods.py hardcodes alpha=0.6 as the
     default for predict_blend(). The LogBlend uses alpha=0.65.

  4. There is NO automated grid search or scipy.optimize call. The alpha
     was chosen by manual comparison of 3 candidate values {0.6, 0.65, 0.7}
     on cross-validation metrics. This is essentially a coarse manual sweep.

  ANSWER: alpha=0.6 was selected by manually comparing 3 values (0.6, 0.65, 0.7)
  on cross-validation holdout performance in matrix_completion_v8.py. It is NOT
  the result of fine-grained optimization — just the best of 3 manual candidates.
""")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8: How are BenchReg correlations computed?
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  STEP 8: HOW ARE BENCHREG CORRELATIONS COMPUTED?")
print("=" * 90)

print("""
  The BenchReg method computes R² (not raw Pearson r) between benchmark pairs.
  Specifically, in predict_benchreg():

  1. For each target benchmark j:
     For each candidate predictor j2:
       - Find the set of models observed on BOTH j and j2 (pairwise complete)
       - Require at least 5 shared observations
       - Fit simple linear regression: y = slope * x + intercept
       - Compute R² = 1 - SS_res / SS_tot

  2. This is PAIRWISE COMPLETE OBSERVATIONS — NOT imputed data.
     Different benchmark pairs may use different subsets of models.
     For example:
       - AIME 2025 vs MATH 500 might share 30 models
       - AIME 2025 vs OSWorld might share only 8 models

  3. The R² is numerically equal to Pearson r² (since it's simple linear
     regression with one predictor), but the code computes it via the
     regression formulation.

  4. For the final prediction, each predictor benchmark's regression is
     re-fitted on the pairwise-complete set, and predictions are combined
     via R²-weighted average.

  ANSWER: Correlations are computed as R² on PAIRWISE COMPLETE observations.
  No imputation is used. Each benchmark pair uses its own subset of shared models.
""")

print("=" * 90)
print("  SUMMARY")
print("=" * 90)
print(f"""
  Target:     {MODEL_NAMES[model_id]} on {BENCH_NAMES[bench_id]}
  Actual:     {actual_value}
  BenchReg:   {benchreg_pred:.2f}  (from {len(preds_benchreg)} predictor benchmarks)
  KNN:        {knn_pred:.2f}  (from {len(top_k_indices)} nearest models)
  Blend:      {final_pred:.2f}  (alpha=0.6)
  Error:      {error:+.2f} ({ape:.1f}% APE)
""")
