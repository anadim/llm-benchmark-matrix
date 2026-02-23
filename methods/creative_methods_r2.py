#!/usr/bin/env python3
"""
Creative Methods Round 2: Combining the winners
================================================
Round 1 established:
  - Logit transform: single biggest win (6.61% PM-MedAPE on partial coverage)
  - Category awareness: small but consistent gain
  - GBT+Blend: best full-coverage method (7.46%)
  - Bimodal awareness: minor gain, but logit already handles bimodality

Round 2 combines these:
  1. LogitCatBenchReg: logit + category in one
  2. LogitCatBlend: logit + category + KNN fallback
  3. Adaptive Logit: logit for pct benchmarks, linear for non-pct
  4. GBT-Logit: GBT in logit space
  5. Kitchen Sink: best of everything, properly orchestrated
  6. Fixed MetaLearner: stacking with NaN-safe features
  7. Confidence-Weighted Blend: trust methods differently per cell
"""

import numpy as np
import sys, warnings, os, time
from collections import defaultdict
from sklearn.linear_model import Ridge, LogisticRegression
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
    predict_cat_benchreg, predict_gbt, predict_logit_benchreg,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Logit / inverse-logit transforms
# ══════════════════════════════════════════════════════════════════════════════

def _is_pct_bench(j, M):
    """Heuristic: is this benchmark a percentage (0-100)?"""
    vals = M[~np.isnan(M[:, j]), j]
    if len(vals) == 0:
        return False
    return vals.min() >= -1 and vals.max() <= 101

def _to_logit(x, eps=0.5):
    """Convert percentage [0,100] to logit space."""
    p = np.clip(x, eps, 100 - eps) / 100.0
    return np.log(p / (1 - p))

def _from_logit(z):
    """Convert logit back to percentage [0,100]."""
    return 100.0 / (1 + np.exp(-z))


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 1: LogitCatBenchReg
# ══════════════════════════════════════════════════════════════════════════════

def predict_logit_cat_benchreg(M_train, top_k=5, min_r2=0.2, cat_bonus=0.15):
    """Logit-space BenchReg with category-aware predictor selection."""
    obs = ~np.isnan(M_train)

    # Identify pct benchmarks and transform
    is_pct = np.array([_is_pct_bench(j, M_train) for j in range(N_BENCH)])
    M_work = M_train.copy()
    for j in range(N_BENCH):
        if is_pct[j]:
            valid = obs[:, j]
            M_work[valid, j] = _to_logit(M_train[valid, j])

    M_pred_work = np.full_like(M_work, np.nan)
    M_pred_work[obs] = M_work[obs]

    for j in range(N_BENCH):
        targets_obs = np.where(obs[:, j])[0]
        if len(targets_obs) < 5:
            continue

        target_cat = BENCH_CATS[j]
        correlations = []
        for j2 in range(N_BENCH):
            if j2 == j:
                continue
            shared = obs[:, j] & obs[:, j2]
            if shared.sum() < 5:
                continue
            x, y = M_work[shared, j2], M_work[shared, j]
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

            effective_r2 = r2
            if BENCH_CATS[j2] == target_cat:
                effective_r2 = min(1.0, r2 + cat_bonus)
            correlations.append((j2, r2, effective_r2))

        correlations.sort(key=lambda x: -x[2])
        best = [(j2, r2) for j2, r2, _ in correlations[:top_k] if r2 >= min_r2]
        if not best:
            continue

        for i in range(N_MODELS):
            if not np.isnan(M_train[i, j]):
                continue
            preds, weights = [], []
            for j2, r2 in best:
                if np.isnan(M_work[i, j2]):
                    continue
                shared = obs[:, j] & obs[:, j2]
                if shared.sum() < 5:
                    continue
                x, y = M_work[shared, j2], M_work[shared, j]
                var_x = np.sum((x - x.mean())**2)
                if var_x < 1e-10:
                    continue
                slope = np.sum((x - x.mean()) * (y - y.mean())) / var_x
                intercept = y.mean() - slope * x.mean()
                preds.append(slope * M_work[i, j2] + intercept)
                weights.append(r2)
            if preds:
                pred_val = np.average(preds, weights=weights)
                if is_pct[j]:
                    M_pred_work[i, j] = _from_logit(pred_val)
                else:
                    M_pred_work[i, j] = pred_val

    # Fill remaining with original values
    M_pred_work[obs] = M_train[obs]
    return M_pred_work


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 2: LogitCatBlend (full coverage)
# ══════════════════════════════════════════════════════════════════════════════

def predict_logit_cat_blend(M_train, alpha=0.6):
    """Blend of LogitCatBenchReg + KNN with NaN fallback."""
    M_breg = predict_logit_cat_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            b, k = M_breg[i, j], M_knn[i, j]
            b_ok = np.isfinite(b) if not np.isnan(b) else False
            k_ok = np.isfinite(k) if not np.isnan(k) else False
            if b_ok and k_ok:
                M_pred[i, j] = alpha * b + (1 - alpha) * k
            elif b_ok:
                M_pred[i, j] = b
            elif k_ok:
                M_pred[i, j] = k
            else:
                M_pred[i, j] = col_mean[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 3: Confidence-Weighted Multi-Method Blend
# ══════════════════════════════════════════════════════════════════════════════

def predict_confidence_blend(M_train):
    """Weight each base method by how many shared benchmarks informed its prediction.
    BenchReg is trusted more when the model has many benchmarks observed;
    SVD is trusted more when the matrix is dense; KNN is trusted when there
    are similar models available."""
    obs = ~np.isnan(M_train)

    M_logit_breg = predict_logit_cat_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    M_svd = predict_svd(M_train, rank=2)
    M_gbt = predict_gbt(M_train)

    col_mean = np.nanmean(M_train, axis=0)

    # Model-level features for confidence
    n_obs_per_model = obs.sum(axis=1)
    model_z = np.zeros(N_MODELS)
    cm, cs = col_stats(M_train)
    for i in range(N_MODELS):
        oj = np.where(obs[i])[0]
        if len(oj) >= 2:
            model_z[i] = np.mean((M_train[i, oj] - cm[oj]) / cs[oj])

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue

            preds = []
            weights = []

            # LogitCatBenchReg
            if np.isfinite(M_logit_breg[i, j]):
                # Higher weight when model has many observed benchmarks
                w = 0.4 + 0.1 * min(n_obs_per_model[i] / 20, 1.0)
                preds.append(M_logit_breg[i, j])
                weights.append(w)

            # KNN
            if np.isfinite(M_knn[i, j]):
                w = 0.25
                preds.append(M_knn[i, j])
                weights.append(w)

            # SVD
            if np.isfinite(M_svd[i, j]):
                w = 0.2
                preds.append(M_svd[i, j])
                weights.append(w)

            # GBT
            if np.isfinite(M_gbt[i, j]):
                w = 0.15
                preds.append(M_gbt[i, j])
                weights.append(w)

            if preds:
                weights = np.array(weights)
                weights /= weights.sum()
                M_pred[i, j] = np.average(preds, weights=weights)
            else:
                M_pred[i, j] = col_mean[j]

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 4: Fixed Meta-Learner (NaN-safe stacking)
# ══════════════════════════════════════════════════════════════════════════════

def predict_meta_learner_v2(M_train):
    """Stacking meta-learner that learns optimal weights for base methods.
    NaN-safe: imputes missing base predictions with column mean before feeding to meta-model."""
    obs = ~np.isnan(M_train)

    # Base predictions
    M_breg = predict_logit_cat_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    M_svd2 = predict_svd(M_train, rank=2)
    M_gbt = predict_gbt(M_train)

    bases = [M_breg, M_knn, M_svd2, M_gbt]
    n_bases = len(bases)

    col_mean = np.nanmean(M_train, axis=0)
    col_std = np.nanstd(M_train, axis=0)
    col_std[col_std < 1e-8] = 1.0

    # Build features: base predictions + context features
    def build_meta_features(i, j):
        feat = []
        for base in bases:
            val = base[i, j]
            feat.append(val if np.isfinite(val) else col_mean[j])
        # Disagreement
        valid_preds = [b[i, j] for b in bases if np.isfinite(b[i, j])]
        feat.append(np.std(valid_preds) if len(valid_preds) > 1 else 0.0)
        # Context
        feat.append(float(MODEL_REASONING[i]))
        feat.append(float(obs[i].sum()))
        feat.append(float(obs[:, j].sum()))
        feat.append(col_mean[j])
        feat.append(col_std[j])
        # Category: simple numeric encoding
        cats = sorted(set(BENCH_CATS))
        cat_map = {c: idx for idx, c in enumerate(cats)}
        feat.append(float(cat_map.get(BENCH_CATS[j], 0)))
        return feat

    # Training data: observed entries
    X_meta, y_meta = [], []
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                feat = build_meta_features(i, j)
                if all(np.isfinite(f) for f in feat):
                    X_meta.append(feat)
                    y_meta.append(M_FULL[i, j])

    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)

    # Standardize
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    meta = Ridge(alpha=10.0)
    meta.fit(X_meta_scaled, y_meta)

    # Predict
    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            feat = build_meta_features(i, j)
            if all(np.isfinite(f) for f in feat):
                feat_scaled = scaler.transform(np.array(feat).reshape(1, -1))
                M_pred[i, j] = meta.predict(feat_scaled)[0]
            else:
                M_pred[i, j] = col_mean[j]

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 5: GBT in Logit Space
# ══════════════════════════════════════════════════════════════════════════════

def predict_gbt_logit(M_train, max_depth=3, n_estimators=50, min_samples_leaf=5):
    """GBT per-benchmark in logit space for pct benchmarks."""
    obs = ~np.isnan(M_train)
    is_pct = np.array([_is_pct_bench(j, M_train) for j in range(N_BENCH)])

    M_pred = M_train.copy()
    model_feats = build_model_features(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    for j in range(N_BENCH):
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 10:
            M_pred[np.isnan(M_train[:, j]), j] = col_mean[j] if not np.isnan(col_mean[j]) else 50.0
            continue

        # Build features
        X_bench = M_train.copy()
        for jj in range(N_BENCH):
            nan_mask = np.isnan(X_bench[:, jj])
            X_bench[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0
        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        X_all = np.column_stack([X_bench[:, other_cols], model_feats])

        # Transform target to logit if pct
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
#  METHOD 6: GBT-Logit + LogitBlend ensemble
# ══════════════════════════════════════════════════════════════════════════════

def predict_gbt_logit_blend(M_train, alpha=0.5):
    """Blend GBT-Logit with LogitCatBlend."""
    M_gbt = predict_gbt_logit(M_train)
    M_lcb = predict_logit_cat_blend(M_train)
    obs = ~np.isnan(M_train)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            g, b = M_gbt[i, j], M_lcb[i, j]
            g_ok, b_ok = np.isfinite(g), np.isfinite(b)
            if g_ok and b_ok:
                M_pred[i, j] = alpha * g + (1 - alpha) * b
            elif g_ok:
                M_pred[i, j] = g
            elif b_ok:
                M_pred[i, j] = b
            else:
                M_pred[i, j] = np.nanmean(M_train[:, j])
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 7: Bimodal-Aware LogitCatBlend
# ══════════════════════════════════════════════════════════════════════════════

def predict_bimodal_logit(M_train):
    """For bimodal benchmarks: classify then regress per group.
    For all others: use LogitCatBlend."""
    obs = ~np.isnan(M_train)
    M_pred = predict_logit_cat_blend(M_train)

    model_feats = build_model_features(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    # Mean-impute benchmark features
    M_imp = M_train.copy()
    for jj in range(N_BENCH):
        nan_mask = np.isnan(M_imp[:, jj])
        M_imp[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0

    for j in BIMODAL_BENCH_IDX:
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 6:
            continue

        y_values = M_train[train_idx, j]
        y_class = (y_values > BIMODAL_THRESHOLD).astype(int)
        if y_class.sum() < 2 or (1 - y_class).sum() < 2:
            continue

        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        X_all = np.column_stack([M_imp[:, other_cols], model_feats])

        # Classify
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_all[train_idx], y_class)

        # Separate regressions
        high_idx = train_idx[y_class == 1]
        low_idx = train_idx[y_class == 0]

        missing_idx = np.where(np.isnan(M_train[:, j]))[0]
        for mi in missing_idx:
            x_i = X_all[mi:mi+1]
            pred_class = clf.predict(x_i)[0]
            prob = clf.predict_proba(x_i)[0]

            if pred_class == 1 and len(high_idx) >= 3:
                reg = Ridge(alpha=1.0)
                reg.fit(X_all[high_idx], M_train[high_idx, j])
                pred = reg.predict(x_i)[0]
                # Soft floor at threshold if confident
                if prob[1] > 0.7:
                    pred = max(pred, BIMODAL_THRESHOLD + 1)
                M_pred[mi, j] = pred
            elif pred_class == 0 and len(low_idx) >= 3:
                reg = Ridge(alpha=1.0)
                reg.fit(X_all[low_idx], M_train[low_idx, j])
                pred = reg.predict(x_i)[0]
                if prob[0] > 0.7:
                    pred = min(pred, BIMODAL_THRESHOLD)
                M_pred[mi, j] = max(0, pred)  # Ensure non-negative
            # else: keep LogitCatBlend prediction

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 8: Kitchen Sink — orchestrated best-of-everything
# ══════════════════════════════════════════════════════════════════════════════

def predict_kitchen_sink(M_train):
    """Orchestrated ensemble:
    - LogitCatBenchReg for primary predictions (where available)
    - GBT-Logit as secondary
    - KNN as tertiary
    - SVD as quaternary
    - Bimodal override for bimodal benchmarks
    - Provider correction as final touch

    Weight by method confidence (based on observed training data quantity)."""
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    # Get all base predictions
    M_lcb = predict_logit_cat_benchreg(M_train)
    M_gbt = predict_gbt_logit(M_train)
    M_knn = predict_B2(M_train, k=5)
    M_svd = predict_svd(M_train, rank=2)

    n_obs_per_model = obs.sum(axis=1)
    n_obs_per_bench = obs.sum(axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue

            candidates = []

            # LogitCatBenchReg: highest trust
            if np.isfinite(M_lcb[i, j]):
                # More observations → more confidence
                w = 0.45 if n_obs_per_model[i] >= 10 else 0.35
                candidates.append((M_lcb[i, j], w))

            # GBT-Logit: good for non-linear, needs enough training data
            if np.isfinite(M_gbt[i, j]):
                w = 0.25 if n_obs_per_bench[j] >= 15 else 0.15
                candidates.append((M_gbt[i, j], w))

            # KNN: always available, decent baseline
            if np.isfinite(M_knn[i, j]):
                w = 0.20
                candidates.append((M_knn[i, j], w))

            # SVD: captures global structure
            if np.isfinite(M_svd[i, j]):
                w = 0.10
                candidates.append((M_svd[i, j], w))

            if candidates:
                vals, wts = zip(*candidates)
                wts = np.array(wts)
                wts /= wts.sum()
                M_pred[i, j] = np.average(vals, weights=wts)
            else:
                M_pred[i, j] = col_mean[j]

    # Bimodal override
    model_feats = build_model_features(M_train)
    M_imp = M_train.copy()
    for jj in range(N_BENCH):
        nan_mask = np.isnan(M_imp[:, jj])
        M_imp[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0

    for j in BIMODAL_BENCH_IDX:
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 6:
            continue
        y_class = (M_train[train_idx, j] > BIMODAL_THRESHOLD).astype(int)
        if y_class.sum() < 2 or (1 - y_class).sum() < 2:
            continue

        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        X_all = np.column_stack([M_imp[:, other_cols], model_feats])

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_all[train_idx], y_class)

        missing_idx = np.where(np.isnan(M_train[:, j]))[0]
        for mi in missing_idx:
            x_i = X_all[mi:mi+1]
            prob = clf.predict_proba(x_i)[0]
            # Only override if classifier is very confident
            if prob[0] > 0.85 and M_pred[mi, j] > BIMODAL_THRESHOLD:
                M_pred[mi, j] = min(M_pred[mi, j], BIMODAL_THRESHOLD * 0.5)
            elif prob[1] > 0.85 and M_pred[mi, j] < BIMODAL_THRESHOLD:
                M_pred[mi, j] = max(M_pred[mi, j], BIMODAL_THRESHOLD * 1.5)

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 9: SVD + Logit BenchReg Blend
# ══════════════════════════════════════════════════════════════════════════════

def predict_svd_logit_blend(M_train, alpha=0.5):
    """Simple average of SVD(r=2) and LogitCatBenchReg, with KNN fallback."""
    M_svd = predict_svd(M_train, rank=2)
    M_lcb = predict_logit_cat_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            s, b, k = M_svd[i, j], M_lcb[i, j], M_knn[i, j]
            s_ok = np.isfinite(s)
            b_ok = np.isfinite(b) and not np.isnan(b)
            k_ok = np.isfinite(k)

            preds, weights = [], []
            if b_ok:
                preds.append(b); weights.append(0.5)
            if s_ok:
                preds.append(s); weights.append(0.3)
            if k_ok:
                preds.append(k); weights.append(0.2)

            if preds:
                weights = np.array(weights)
                weights /= weights.sum()
                M_pred[i, j] = np.average(preds, weights=weights)
            else:
                M_pred[i, j] = col_mean[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 10: Adaptive alpha blend (vary alpha by model/bench context)
# ══════════════════════════════════════════════════════════════════════════════

def predict_adaptive_blend(M_train):
    """Like LogitCatBlend but with alpha adapted per cell.
    More BenchReg weight for well-observed models, more KNN for sparse models."""
    M_breg = predict_logit_cat_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    n_obs_per_model = obs.sum(axis=1)
    max_obs = n_obs_per_model.max()

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        # Alpha ranges from 0.4 (sparse) to 0.75 (well-observed)
        obs_frac = n_obs_per_model[i] / max_obs
        alpha = 0.4 + 0.35 * obs_frac

        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            b, k = M_breg[i, j], M_knn[i, j]
            b_ok = np.isfinite(b) and not np.isnan(b)
            k_ok = np.isfinite(k)
            if b_ok and k_ok:
                M_pred[i, j] = alpha * b + (1 - alpha) * k
            elif b_ok:
                M_pred[i, j] = b
            elif k_ok:
                M_pred[i, j] = k
            else:
                M_pred[i, j] = col_mean[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 90)
    print("  CREATIVE METHODS ROUND 2: COMBINING WINNERS")
    print("  Per-model 50% holdout × 5 seeds + Random 20% holdout")
    print("=" * 90)

    methods = [
        # Baselines for reference
        ("Baseline:Blend(0.6)",       lambda M: predict_blend(M, alpha=0.6)),
        ("R1:LogitBenchReg",          predict_logit_benchreg),
        ("R1:CatBenchReg",            predict_cat_benchreg),

        # New combinations
        ("LogitCatBenchReg",          predict_logit_cat_benchreg),
        ("LogitCatBlend(0.6)",        predict_logit_cat_blend),
        ("ConfidenceBlend",           predict_confidence_blend),
        ("MetaLearnerV2",             predict_meta_learner_v2),
        ("GBT-Logit",                 predict_gbt_logit),
        ("GBT-Logit+LCB(0.5)",       predict_gbt_logit_blend),
        ("BimodalLogit",              predict_bimodal_logit),
        ("KitchenSink",               predict_kitchen_sink),
        ("SVD+Logit(0.5/0.3/0.2)",   predict_svd_logit_blend),
        ("AdaptiveBlend",             predict_adaptive_blend),
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
    print("  ROUND 2 RESULTS (sorted by PM-MedAPE)")
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
