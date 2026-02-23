#!/usr/bin/env python3
"""
Creative Methods Round 3: Final optimization
=============================================
Focus on:
  1. SVD in logit space (currently only z-score)
  2. LOO weight tuning for the ensemble
  3. Fair coverage-corrected comparison
  4. Simplified best methods for clarity
"""

import numpy as np
import sys, warnings, os, time
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_PROVIDERS, MODEL_REASONING,
    MODEL_OPEN, MODEL_PARAMS, MODEL_ACTIVE, BENCH_CATS,
    col_normalize, col_denormalize, col_stats,
    holdout_random_cells, holdout_per_model,
)
from all_methods import predict_benchreg, predict_B2, predict_blend, predict_svd, predict_B0
from creative_methods import (
    build_model_features, extended_metrics, evaluate_extended,
    BIMODAL_BENCH_IDX, BIMODAL_THRESHOLD,
    predict_logit_benchreg, predict_gbt,
)
from creative_methods_r2 import (
    predict_logit_cat_benchreg, predict_logit_cat_blend,
    predict_gbt_logit, predict_kitchen_sink, predict_confidence_blend,
    _is_pct_bench, _to_logit, _from_logit,
)


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 1: SVD in Logit Space
# ══════════════════════════════════════════════════════════════════════════════

def predict_svd_logit(M_train, rank=2, max_iter=100, tol=1e-4):
    """Soft-Impute SVD but in logit space for percentage benchmarks.
    Non-percentage benchmarks use z-score as before."""
    obs = ~np.isnan(M_train)
    is_pct = np.array([_is_pct_bench(j, M_train) for j in range(N_BENCH)])

    # Transform to working space
    M_work = M_train.copy()
    for j in range(N_BENCH):
        if is_pct[j]:
            valid = obs[:, j]
            M_work[valid, j] = _to_logit(M_train[valid, j])

    # Z-score normalize the working space
    cm = np.nanmean(M_work, axis=0)
    cs = np.nanstd(M_work, axis=0)
    cs[cs < 1e-8] = 1.0

    M_norm = (M_work - cm) / cs
    M_norm[np.isnan(M_work)] = np.nan

    # Initialize missing with 0
    M_imp = M_norm.copy()
    M_imp[np.isnan(M_imp)] = 0

    for it in range(max_iter):
        M_old = M_imp.copy()
        try:
            U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        U_r, s_r, Vt_r = U[:, :rank], s[:rank], Vt[:rank, :]
        M_approx = U_r @ np.diag(s_r) @ Vt_r
        M_imp = np.where(obs, M_norm, M_approx)
        M_imp[np.isnan(M_imp)] = 0
        diff = np.sqrt(np.mean((M_imp - M_old)**2))
        rel_diff = diff / (np.sqrt(np.mean(M_old**2)) + 1e-12)
        if rel_diff < tol:
            break

    # Denormalize
    M_pred_work = M_imp * cs + cm

    # Inverse transform
    M_pred = np.full_like(M_train, np.nan)
    for j in range(N_BENCH):
        if is_pct[j]:
            M_pred[:, j] = _from_logit(M_pred_work[:, j])
        else:
            M_pred[:, j] = M_pred_work[:, j]

    M_pred[obs] = M_train[obs]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 2: Logit SVD + LogitCatBenchReg Blend
# ══════════════════════════════════════════════════════════════════════════════

def predict_logit_svd_breg(M_train, alpha_breg=0.5, alpha_svd=0.3, alpha_knn=0.2):
    """Blend of LogitCatBenchReg, Logit-SVD, and KNN with NaN fallback.
    All three methods share the logit space philosophy."""
    M_lcb = predict_logit_cat_benchreg(M_train)
    M_svd = predict_svd_logit(M_train, rank=2)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            preds, weights = [], []
            b = M_lcb[i, j]
            if np.isfinite(b) and not np.isnan(b):
                preds.append(b); weights.append(alpha_breg)
            s = M_svd[i, j]
            if np.isfinite(s):
                preds.append(s); weights.append(alpha_svd)
            k = M_knn[i, j]
            if np.isfinite(k):
                preds.append(k); weights.append(alpha_knn)

            if preds:
                weights = np.array(weights)
                weights /= weights.sum()
                M_pred[i, j] = np.average(preds, weights=weights)
            else:
                M_pred[i, j] = col_mean[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 3: Kitchen Sink V2 (with Logit-SVD replacing normal SVD)
# ══════════════════════════════════════════════════════════════════════════════

def predict_kitchen_sink_v2(M_train):
    """Like KitchenSink but uses Logit-SVD instead of normal SVD."""
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_lcb = predict_logit_cat_benchreg(M_train)
    M_gbt = predict_gbt_logit(M_train)
    M_knn = predict_B2(M_train, k=5)
    M_svd = predict_svd_logit(M_train, rank=2)

    n_obs_per_model = obs.sum(axis=1)
    n_obs_per_bench = obs.sum(axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue

            candidates = []

            if np.isfinite(M_lcb[i, j]) and not np.isnan(M_lcb[i, j]):
                w = 0.45 if n_obs_per_model[i] >= 10 else 0.35
                candidates.append((M_lcb[i, j], w))

            if np.isfinite(M_gbt[i, j]):
                w = 0.25 if n_obs_per_bench[j] >= 15 else 0.15
                candidates.append((M_gbt[i, j], w))

            if np.isfinite(M_knn[i, j]):
                candidates.append((M_knn[i, j], 0.15))

            if np.isfinite(M_svd[i, j]):
                candidates.append((M_svd[i, j], 0.15))

            if candidates:
                vals, wts = zip(*candidates)
                wts = np.array(wts)
                wts /= wts.sum()
                M_pred[i, j] = np.average(vals, weights=wts)
            else:
                M_pred[i, j] = col_mean[j]

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 4: Simple Trio (LogitCatBenchReg + LogitSVD + KNN, equal weight)
# ══════════════════════════════════════════════════════════════════════════════

def predict_simple_trio(M_train):
    """Simple equal-weight average of 3 complementary methods.
    No fuss, no tuning — just diversity."""
    M_lcb = predict_logit_cat_benchreg(M_train)
    M_svd = predict_svd_logit(M_train, rank=2)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            preds = []
            if np.isfinite(M_lcb[i, j]) and not np.isnan(M_lcb[i, j]):
                preds.append(M_lcb[i, j])
            if np.isfinite(M_svd[i, j]):
                preds.append(M_svd[i, j])
            if np.isfinite(M_knn[i, j]):
                preds.append(M_knn[i, j])
            if preds:
                M_pred[i, j] = np.mean(preds)
            else:
                M_pred[i, j] = col_mean[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 5: Logit Blend with SVD fallback (no KNN)
# ══════════════════════════════════════════════════════════════════════════════

def predict_logit_svd_only(M_train, alpha=0.6):
    """LogitCatBenchReg primary, LogitSVD fallback. No KNN at all.
    Tests whether KNN is actually adding value."""
    M_lcb = predict_logit_cat_benchreg(M_train)
    M_svd = predict_svd_logit(M_train, rank=2)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            b, s = M_lcb[i, j], M_svd[i, j]
            b_ok = np.isfinite(b) and not np.isnan(b)
            s_ok = np.isfinite(s)
            if b_ok and s_ok:
                M_pred[i, j] = alpha * b + (1 - alpha) * s
            elif b_ok:
                M_pred[i, j] = b
            elif s_ok:
                M_pred[i, j] = s
            else:
                M_pred[i, j] = col_mean[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 6: GBT-Logit with model features + LogitCatBenchReg as feature
# ══════════════════════════════════════════════════════════════════════════════

def predict_gbt_stacked(M_train, max_depth=3, n_estimators=50, min_samples_leaf=5):
    """GBT where LogitCatBenchReg predictions are an additional feature.
    This 'stacks' the two best methods."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    M_lcb = predict_logit_cat_benchreg(M_train)
    model_feats = build_model_features(M_train)
    col_mean = np.nanmean(M_train, axis=0)
    is_pct = np.array([_is_pct_bench(j, M_train) for j in range(N_BENCH)])

    for j in range(N_BENCH):
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 10:
            M_pred[np.isnan(M_train[:, j]), j] = col_mean[j] if not np.isnan(col_mean[j]) else 50.0
            continue

        # Features: other benchmarks (imputed) + model metadata + LCB prediction
        M_imp = M_train.copy()
        for jj in range(N_BENCH):
            nan_mask = np.isnan(M_imp[:, jj])
            M_imp[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0

        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        lcb_col = np.where(np.isfinite(M_lcb[:, j]) & ~np.isnan(M_lcb[:, j]),
                           M_lcb[:, j], col_mean[j]).reshape(-1, 1)
        X_all = np.column_stack([M_imp[:, other_cols], model_feats, lcb_col])

        y_train = M_train[train_idx, j].copy()
        if is_pct[j]:
            y_train = _to_logit(y_train)

        gbt = GradientBoostingRegressor(
            max_depth=max_depth, n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf, learning_rate=0.1,
            random_state=42
        )
        gbt.fit(X_all[train_idx], y_train)

        missing_idx = np.where(np.isnan(M_train[:, j]))[0]
        if len(missing_idx) > 0:
            preds = gbt.predict(X_all[missing_idx])
            if is_pct[j]:
                preds = _from_logit(preds)
            M_pred[missing_idx, j] = preds

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 7: Weighted Ensemble with LOO-tuned weights per benchmark category
# ══════════════════════════════════════════════════════════════════════════════

def predict_cat_weighted_ensemble(M_train):
    """Learn optimal weights for (LogitCatBenchReg, SVD-logit, KNN) per benchmark category
    using LOO on observed entries within each category."""
    obs = ~np.isnan(M_train)

    M_lcb = predict_logit_cat_benchreg(M_train)
    M_svd = predict_svd_logit(M_train, rank=2)
    M_knn = predict_B2(M_train, k=5)

    bases = [M_lcb, M_svd, M_knn]
    n_bases = len(bases)
    col_mean = np.nanmean(M_train, axis=0)

    cats = sorted(set(BENCH_CATS))

    # Learn weights per category
    cat_weights = {}
    for cat in cats:
        cat_benchmarks = [j for j in range(N_BENCH) if BENCH_CATS[j] == cat]
        # Collect errors for each base method on observed entries in this category
        base_preds_list = [[] for _ in range(n_bases)]
        actuals = []

        for j in cat_benchmarks:
            for i in range(N_MODELS):
                if obs[i, j]:
                    actual = M_FULL[i, j]
                    actuals.append(actual)
                    for bi, base in enumerate(bases):
                        val = base[i, j]
                        if np.isfinite(val) and not np.isnan(val):
                            base_preds_list[bi].append(val)
                        else:
                            base_preds_list[bi].append(col_mean[j])

        if len(actuals) < 10:
            cat_weights[cat] = np.array([0.5, 0.3, 0.2])  # default
            continue

        # Find weights that minimize MAE on in-sample (this is a very simple optimizer)
        actuals = np.array(actuals)
        base_preds_arr = np.array(base_preds_list)  # (n_bases, n_samples)

        best_mae = np.inf
        best_w = np.array([0.5, 0.3, 0.2])

        # Grid search over simplex
        for w0 in np.arange(0.1, 0.9, 0.1):
            for w1 in np.arange(0.05, 0.9 - w0, 0.1):
                w2 = 1.0 - w0 - w1
                if w2 < 0.05:
                    continue
                w = np.array([w0, w1, w2])
                pred = w @ base_preds_arr
                mae = np.median(np.abs(pred - actuals))
                if mae < best_mae:
                    best_mae = mae
                    best_w = w.copy()

        cat_weights[cat] = best_w

    # Apply per-category weights
    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            cat = BENCH_CATS[j]
            w = cat_weights.get(cat, np.array([0.5, 0.3, 0.2]))

            preds, weights = [], []
            for bi, base in enumerate(bases):
                val = base[i, j]
                if np.isfinite(val) and not np.isnan(val):
                    preds.append(val)
                    weights.append(w[bi])

            if preds:
                weights = np.array(weights)
                weights /= weights.sum()
                M_pred[i, j] = np.average(preds, weights=weights)
            else:
                M_pred[i, j] = col_mean[j]

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 8: Residual Correction — predict error of best method, subtract it
# ══════════════════════════════════════════════════════════════════════════════

def predict_residual_correction(M_train):
    """1. Get LogitCatBenchReg predictions
    2. On observed entries, compute residuals
    3. Predict residuals using model features + benchmark features
    4. Subtract predicted residuals from LogitCatBenchReg"""
    obs = ~np.isnan(M_train)
    M_lcb = predict_logit_cat_benchreg(M_train)
    M_svd = predict_svd_logit(M_train, rank=2)
    M_knn = predict_B2(M_train, k=5)
    col_mean = np.nanmean(M_train, axis=0)

    # First pass: use LCB where available, SVD/KNN elsewhere
    M_base = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            b = M_lcb[i, j]
            if np.isfinite(b) and not np.isnan(b):
                M_base[i, j] = b
            elif np.isfinite(M_svd[i, j]):
                M_base[i, j] = 0.6 * M_svd[i, j] + 0.4 * (M_knn[i, j] if np.isfinite(M_knn[i, j]) else col_mean[j])
            elif np.isfinite(M_knn[i, j]):
                M_base[i, j] = M_knn[i, j]
            else:
                M_base[i, j] = col_mean[j]

    # Compute residuals on observed entries
    model_feats = build_model_features(M_train)
    cm_w = col_mean.copy()
    cm_w[np.isnan(cm_w)] = 50.0

    X_res, y_res = [], []
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j] and np.isfinite(M_base[i, j]):
                residual = M_FULL[i, j] - M_base[i, j]
                feat = list(model_feats[i]) + [cm_w[j], float(obs[:, j].sum())]
                X_res.append(feat)
                y_res.append(residual)

    X_res = np.array(X_res)
    y_res = np.array(y_res)

    # Predict residuals
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    ridge = Ridge(alpha=50.0)
    ridge.fit(X_scaled, y_res)

    # Apply corrections
    M_pred = M_base.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            feat = list(model_feats[i]) + [cm_w[j], float(obs[:, j].sum())]
            feat_scaled = scaler.transform(np.array(feat).reshape(1, -1))
            correction = ridge.predict(feat_scaled)[0]
            M_pred[i, j] = M_base[i, j] + correction

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 90)
    print("  CREATIVE METHODS ROUND 3: FINAL OPTIMIZATION")
    print("  Per-model 50% holdout × 5 seeds + Random 20% holdout")
    print("=" * 90)

    methods = [
        # Baselines
        ("Baseline:Blend(0.6)",          lambda M: predict_blend(M, alpha=0.6)),
        ("R2:KitchenSink",               predict_kitchen_sink),
        ("R2:ConfidenceBlend",           predict_confidence_blend),

        # New methods
        ("SVD-Logit(r=2)",               lambda M: predict_svd_logit(M, rank=2)),
        ("SVD-Logit(r=3)",               lambda M: predict_svd_logit(M, rank=3)),
        ("LogitSVD+LCB(0.5/0.3/0.2)",   predict_logit_svd_breg),
        ("KitchenSinkV2",                predict_kitchen_sink_v2),
        ("SimpleTrio",                   predict_simple_trio),
        ("LogitSVDonly(0.6/0.4)",        predict_logit_svd_only),
        ("GBT-Stacked",                  predict_gbt_stacked),
        ("CatWeightedEnsemble",          predict_cat_weighted_ensemble),
        ("ResidualCorrection",           predict_residual_correction),
    ]

    results = []
    for name, fn in methods:
        t0 = time.time()
        try:
            r = evaluate_extended(fn, name, verbose=True)
            r['time'] = time.time() - t0
            results.append(r)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    print("\n" + "=" * 130)
    print("  ROUND 3 RESULTS (sorted by PM-MedAPE)")
    print("=" * 130)
    results.sort(key=lambda x: x.get('pm_medape', 999))

    header = (f"  {'Method':<28s} {'PM-MedAPE':>9s} {'R-MedAPE':>9s} {'PM-MAE':>7s} "
              f"{'±3pts':>6s} {'MedAPE>50':>9s} {'MedAPE≤50':>9s} "
              f"{'BiAcc':>6s} {'Cov':>5s} {'Time':>6s}")
    print(header)
    print("  " + "─" * 128)

    for r in results:
        print(f"  {r['method']:<28s} {r['pm_medape']:>8.2f}% {r['rand_medape']:>8.2f}% "
              f"{r['pm_mae']:>7.2f} {r['pct_within3']:>5.1f}% "
              f"{r['medape_hi']:>8.2f}% {r['medape_lo']:>8.2f}% "
              f"{r.get('bimodal_acc', float('nan')):>5.1f}% "
              f"{r.get('coverage', 100):>4.1f}% "
              f"{r.get('time', 0):>5.1f}s")

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)
