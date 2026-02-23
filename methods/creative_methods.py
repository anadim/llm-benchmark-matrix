#!/usr/bin/env python3
"""
Creative Methods for LLM Benchmark Matrix Completion
=====================================================
Explores new prediction approaches that leverage wasted information:
  1. Model metadata (params, reasoning, provider)
  2. Benchmark categories and difficulty structure
  3. Missingness patterns as signal
  4. Non-linear relationships (GBT)
  5. Bimodal benchmark classification
  6. Meta-learner ensembles
  7. Within-family interpolation
"""

import numpy as np
import sys, warnings, os, time
from collections import defaultdict
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_PROVIDERS, MODEL_REASONING,
    MODEL_OPEN, MODEL_PARAMS, MODEL_ACTIVE, BENCH_CATS,
    col_normalize, col_denormalize, col_stats,
    compute_metrics, evaluate_method,
    holdout_random_cells, holdout_per_model,
)
from all_methods import predict_benchreg, predict_B2, predict_blend, predict_svd, predict_B0

# ══════════════════════════════════════════════════════════════════════════════
#  EXTENDED EVALUATION (new metrics)
# ══════════════════════════════════════════════════════════════════════════════

# Bimodal benchmarks
BIMODAL_BENCH_IDS = ['arc_agi_1', 'arc_agi_2', 'imo_2025', 'usamo_2025', 'matharena_apex_2025']
BIMODAL_BENCH_IDX = [BENCH_IDS.index(b) for b in BIMODAL_BENCH_IDS if b in BENCH_IDS]
BIMODAL_THRESHOLD = 10.0  # "can do it" vs "can't do it"


def extended_metrics(actual, predicted, test_set=None):
    """Compute extended metrics: MAE, split MedAPE, bimodal accuracy, % within 3pts."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    valid = ~np.isnan(predicted) & ~np.isnan(actual)
    a, p = actual[valid], predicted[valid]

    if len(a) == 0:
        return {'mae': np.nan, 'medape': np.nan, 'medape_hi': np.nan,
                'medape_lo': np.nan, 'pct_within3': np.nan,
                'bimodal_acc': np.nan, 'bimodal_n': 0, 'coverage': 0.0}

    abs_err = np.abs(p - a)

    # MAE (median absolute error, no denominator)
    mae = np.median(abs_err)

    # MedAPE overall
    nonzero = np.abs(a) > 1e-6
    ape = np.full(len(a), np.nan)
    ape[nonzero] = abs_err[nonzero] / np.abs(a[nonzero]) * 100
    ape_valid = ape[~np.isnan(ape)]
    medape = np.median(ape_valid) if len(ape_valid) > 0 else np.nan

    # Split MedAPE by score > 50 vs <= 50
    hi_mask = a > 50
    lo_mask = a <= 50
    ape_hi = ape_valid[hi_mask[nonzero][:len(ape_valid)]] if hi_mask.any() else np.array([])
    ape_lo = ape_valid[lo_mask[nonzero][:len(ape_valid)]] if lo_mask.any() else np.array([])

    # More precise split
    ape_hi_list, ape_lo_list = [], []
    for idx in range(len(a)):
        if np.abs(a[idx]) > 1e-6:
            this_ape = abs_err[idx] / np.abs(a[idx]) * 100
            if a[idx] > 50:
                ape_hi_list.append(this_ape)
            else:
                ape_lo_list.append(this_ape)
    medape_hi = np.median(ape_hi_list) if ape_hi_list else np.nan
    medape_lo = np.median(ape_lo_list) if ape_lo_list else np.nan

    # % within 3 absolute points
    pct_within3 = np.mean(abs_err <= 3.0) * 100

    # Bimodal classification accuracy
    bimodal_correct = 0
    bimodal_total = 0
    if test_set is not None:
        for idx, (i, j) in enumerate(test_set):
            if not valid[idx]:
                continue
            if j in BIMODAL_BENCH_IDX:
                actual_class = 1 if actual[idx] > BIMODAL_THRESHOLD else 0
                pred_class = 1 if predicted[idx] > BIMODAL_THRESHOLD else 0
                bimodal_total += 1
                if actual_class == pred_class:
                    bimodal_correct += 1
    bimodal_acc = (bimodal_correct / bimodal_total * 100) if bimodal_total > 0 else np.nan

    coverage = np.sum(valid) / len(actual) * 100

    return {
        'mae': mae, 'medape': medape,
        'medape_hi': medape_hi, 'medape_lo': medape_lo,
        'pct_within3': pct_within3,
        'bimodal_acc': bimodal_acc, 'bimodal_n': bimodal_total,
        'coverage': coverage,
    }


def evaluate_extended(predict_fn, method_name, seeds=[42, 123, 456, 789, 1337], verbose=True):
    """Run per-model 50% holdout with 5 seeds + random 20% holdout. Return extended metrics."""

    pm_results = []
    for seed in seeds:
        folds = holdout_per_model(k_frac=0.5, min_scores=8, n_folds=3, seed=seed)
        all_actual, all_pred, all_test = [], [], []
        for M_train, test_set in folds:
            M_pred = predict_fn(M_train)
            for i, j in test_set:
                all_actual.append(M_FULL[i, j])
                all_pred.append(M_pred[i, j])
                all_test.append((i, j))
        em = extended_metrics(all_actual, all_pred, all_test)
        pm_results.append(em)

    # Random holdout (single run with 5 folds)
    folds_r = holdout_random_cells(frac=0.2, n_folds=5, seed=42)
    r_actual, r_pred, r_test = [], [], []
    for M_train, test_set in folds_r:
        M_pred = predict_fn(M_train)
        for i, j in test_set:
            r_actual.append(M_FULL[i, j])
            r_pred.append(M_pred[i, j])
            r_test.append((i, j))
    rand_em = extended_metrics(r_actual, r_pred, r_test)

    # Average per-model results across seeds
    pm_avg = {}
    for key in pm_results[0]:
        vals = [r[key] for r in pm_results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        pm_avg[key] = np.mean(vals) if vals else np.nan

    if verbose:
        print(f"\n  {method_name}:")
        print(f"    PM-MedAPE={pm_avg['medape']:.2f}%  Rand-MedAPE={rand_em['medape']:.2f}%  "
              f"PM-MAE={pm_avg['mae']:.2f}  ±3pts={pm_avg['pct_within3']:.1f}%")
        print(f"    MedAPE>50={pm_avg['medape_hi']:.2f}%  MedAPE≤50={pm_avg['medape_lo']:.2f}%  "
              f"BimodalAcc={pm_avg['bimodal_acc']:.1f}%({pm_avg['bimodal_n']:.0f})  "
              f"Coverage={pm_avg['coverage']:.1f}%")

    return {
        'method': method_name,
        'pm_medape': pm_avg['medape'],
        'rand_medape': rand_em['medape'],
        'pm_mae': pm_avg['mae'],
        'pct_within3': pm_avg['pct_within3'],
        'bimodal_acc': pm_avg['bimodal_acc'],
        'bimodal_n': pm_avg['bimodal_n'],
        'medape_hi': pm_avg['medape_hi'],
        'medape_lo': pm_avg['medape_lo'],
        'coverage': pm_avg['coverage'],
        'rand_mae': rand_em['mae'],
        'rand_pct_within3': rand_em['pct_within3'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build model feature matrix
# ══════════════════════════════════════════════════════════════════════════════

def build_model_features(M_train):
    """Build a feature matrix from model metadata + missingness pattern + aggregate stats.
    Returns (N_MODELS, n_features) array, all finite."""
    obs = ~np.isnan(M_train)
    feats = []

    # 1. log(params) — impute median for missing
    lp = np.log1p(np.nan_to_num(MODEL_PARAMS, nan=0))
    median_lp = np.median(lp[lp > 0]) if (lp > 0).any() else 10.0
    lp[lp == 0] = median_lp
    feats.append(lp)

    # 2. log(active_params) — impute with log(params) or median
    la = np.log1p(np.nan_to_num(MODEL_ACTIVE, nan=0))
    for i in range(N_MODELS):
        if la[i] == 0:
            la[i] = lp[i]  # fallback to total params
    feats.append(la)

    # 3. is_reasoning (binary)
    feats.append(MODEL_REASONING.astype(float))

    # 4. is_open_weight (binary)
    feats.append(MODEL_OPEN.astype(float))

    # 5. Number of benchmarks observed (proxy for model prominence)
    feats.append(obs.sum(axis=1).astype(float))

    # 6. Model's average z-score across observed benchmarks (overall capability)
    col_mean = np.nanmean(M_train, axis=0)
    col_std = np.nanstd(M_train, axis=0)
    col_std[col_std < 1e-8] = 1.0
    model_z = np.zeros(N_MODELS)
    for i in range(N_MODELS):
        obs_j = np.where(obs[i])[0]
        if len(obs_j) >= 2:
            zscores = (M_train[i, obs_j] - col_mean[obs_j]) / col_std[obs_j]
            model_z[i] = np.mean(zscores)
    feats.append(model_z)

    # 7. Model's z-score std (capability variance — specialists vs generalists)
    model_z_std = np.zeros(N_MODELS)
    for i in range(N_MODELS):
        obs_j = np.where(obs[i])[0]
        if len(obs_j) >= 3:
            zscores = (M_train[i, obs_j] - col_mean[obs_j]) / col_std[obs_j]
            model_z_std[i] = np.std(zscores)
    feats.append(model_z_std)

    # 8. Provider one-hot (top providers only: OpenAI, Anthropic, Google, Meta, DeepSeek, Alibaba)
    top_providers = ['OpenAI', 'Anthropic', 'Google', 'DeepSeek', 'Alibaba', 'Meta']
    for prov in top_providers:
        feats.append((MODEL_PROVIDERS == prov).astype(float))

    return np.column_stack(feats)


def build_bench_features():
    """Build per-benchmark feature vector."""
    feats = []

    # 1. Category one-hot
    cats = sorted(set(BENCH_CATS))
    for cat in cats:
        feats.append((BENCH_CATS == cat).astype(float))

    # 2. Number of observed entries (coverage)
    feats.append(OBSERVED.sum(axis=0).astype(float))

    # 3. Mean score
    col_mean = np.nanmean(M_FULL, axis=0)
    col_mean[np.isnan(col_mean)] = 50.0
    feats.append(col_mean)

    # 4. Std of scores
    col_std = np.nanstd(M_FULL, axis=0)
    col_std[np.isnan(col_std)] = 20.0
    feats.append(col_std)

    return np.column_stack(feats)


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 1: Category-Aware BenchReg
# ══════════════════════════════════════════════════════════════════════════════

def predict_cat_benchreg(M_train, top_k=5, min_r2=0.2, cat_bonus=0.15):
    """BenchReg with category-aware predictor selection.
    When two candidate predictors have similar R², prefer the one in the same category."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

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
            x, y = M_train[shared, j2], M_train[shared, j]
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

            # Category bonus: boost R² for same-category benchmarks
            effective_r2 = r2
            if BENCH_CATS[j2] == target_cat:
                effective_r2 = min(1.0, r2 + cat_bonus)

            correlations.append((j2, r2, effective_r2))

        # Sort by effective R² but use real R² for weighting
        correlations.sort(key=lambda x: -x[2])
        best = [(j2, r2) for j2, r2, eff_r2 in correlations[:top_k] if r2 >= min_r2]
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
                weights.append(r2)
            if preds:
                M_pred[i, j] = np.average(preds, weights=weights)
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 2: Category-Aware Blend (CatBenchReg + KNN)
# ══════════════════════════════════════════════════════════════════════════════

def predict_cat_blend(M_train, alpha=0.6):
    """Blend of category-aware BenchReg + KNN with NaN fallback."""
    M_breg = predict_cat_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            b, k = M_breg[i, j], M_knn[i, j]
            b_ok, k_ok = np.isfinite(b), np.isfinite(k)
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
#  METHOD 3: Metadata-Enhanced KNN
# ══════════════════════════════════════════════════════════════════════════════

def predict_meta_knn(M_train, k=5, meta_weight=0.3):
    """KNN that uses both benchmark score similarity and model metadata similarity."""
    M_norm, cm, cs = col_normalize(M_train)
    obs = ~np.isnan(M_norm)
    M_pred = M_train.copy()

    # Build metadata similarity
    model_feats = build_model_features(M_train)
    scaler = StandardScaler()
    model_feats_scaled = scaler.fit_transform(model_feats)

    # Metadata cosine similarity
    norms = np.linalg.norm(model_feats_scaled, axis=1, keepdims=True) + 1e-10
    model_feats_unit = model_feats_scaled / norms

    for i in range(N_MODELS):
        missing = np.where(np.isnan(M_train[i]))[0]
        if len(missing) == 0:
            continue

        # Score-based similarity (existing KNN approach)
        shared_all = obs[i]
        score_sims = np.full(N_MODELS, -999.0)
        for k2 in range(N_MODELS):
            if k2 == i:
                continue
            shared = shared_all & obs[k2]
            if shared.sum() < 3:
                continue
            a, b = M_norm[i, shared], M_norm[k2, shared]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
            score_sims[k2] = np.dot(a, b) / denom

        # Metadata similarity
        meta_sims = model_feats_unit @ model_feats_unit[i]
        meta_sims[i] = -999.0

        # Combined similarity
        has_score = score_sims > -999
        combined = np.full(N_MODELS, -999.0)
        for k2 in range(N_MODELS):
            if has_score[k2]:
                combined[k2] = (1 - meta_weight) * score_sims[k2] + meta_weight * meta_sims[k2]
            # If no score sim (too few shared benchmarks), use metadata alone but penalize
            elif meta_sims[k2] > 0.5:
                combined[k2] = meta_weight * meta_sims[k2] * 0.5  # heavy penalty

        top_k = np.argsort(combined)[-k:]
        top_k = top_k[combined[top_k] > -999]
        if len(top_k) == 0:
            col_mean = np.nanmean(M_train, axis=0)
            for j in missing:
                M_pred[i, j] = col_mean[j]
            continue

        weights = np.maximum(combined[top_k], 0.01)
        weights /= weights.sum()
        for j in missing:
            vals, ws = [], []
            for idx, ki in enumerate(top_k):
                if not np.isnan(M_norm[ki, j]):
                    vals.append(M_norm[ki, j])
                    ws.append(weights[idx])
            if vals:
                pred_norm = np.average(vals, weights=ws)
                M_pred[i, j] = pred_norm * cs[j] + cm[j]
            else:
                M_pred[i, j] = cm[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 4: GBT Per-Benchmark Predictor
# ══════════════════════════════════════════════════════════════════════════════

def predict_gbt(M_train, max_depth=3, n_estimators=50, min_samples_leaf=5):
    """For each benchmark, train a GBT using all other benchmarks + model metadata as features."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    model_feats = build_model_features(M_train)

    for j in range(N_BENCH):
        # Training: models that have this benchmark observed
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 10:
            # Fallback to column mean
            M_pred[np.isnan(M_train[:, j]), j] = np.nanmean(M_train[:, j])
            continue

        # Features: other benchmark scores (imputed with column mean) + model metadata
        col_mean = np.nanmean(M_train, axis=0)
        X_bench = M_train.copy()
        for jj in range(N_BENCH):
            nan_mask = np.isnan(X_bench[:, jj])
            X_bench[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0

        # Remove target column, add metadata
        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        X_all = np.column_stack([X_bench[:, other_cols], model_feats])

        X_train = X_all[train_idx]
        y_train = M_train[train_idx, j]

        gbt = GradientBoostingRegressor(
            max_depth=max_depth, n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf, learning_rate=0.1,
            random_state=42
        )
        gbt.fit(X_train, y_train)

        # Predict missing
        missing_idx = np.where(np.isnan(M_train[:, j]))[0]
        if len(missing_idx) > 0:
            X_test = X_all[missing_idx]
            M_pred[missing_idx, j] = gbt.predict(X_test)

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 5: Hierarchical Category Model
# ══════════════════════════════════════════════════════════════════════════════

def predict_hierarchical(M_train):
    """Two-stage: (1) estimate per-category ability, (2) refine within category via BenchReg."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    # Stage 1: For each (model, category), compute ability = mean z-score in that category
    col_mean = np.nanmean(M_train, axis=0)
    col_std = np.nanstd(M_train, axis=0)
    col_std[col_std < 1e-8] = 1.0

    cats = sorted(set(BENCH_CATS))
    cat_idx = {cat: [j for j in range(N_BENCH) if BENCH_CATS[j] == cat] for cat in cats}

    # Model ability per category
    model_cat_ability = {}  # (model_idx, cat) -> mean z-score
    model_overall_z = np.zeros(N_MODELS)

    for i in range(N_MODELS):
        obs_j = np.where(obs[i])[0]
        if len(obs_j) < 2:
            continue
        zscores = (M_train[i, obs_j] - col_mean[obs_j]) / col_std[obs_j]
        model_overall_z[i] = np.mean(zscores)

        for cat in cats:
            cat_j = [j for j in cat_idx[cat] if obs[i, j]]
            if cat_j:
                cat_z = np.mean((M_train[i, cat_j] - col_mean[cat_j]) / col_std[cat_j])
                model_cat_ability[(i, cat)] = cat_z

    # Stage 2: For missing entries, use category ability as prior, refine with BenchReg
    M_breg = predict_benchreg(M_train)

    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if not np.isnan(M_train[i, j]):
                continue

            cat = BENCH_CATS[j]
            breg_ok = np.isfinite(M_breg[i, j])

            if breg_ok:
                # Use BenchReg as primary
                M_pred[i, j] = M_breg[i, j]
            else:
                # Fallback: use category ability
                if (i, cat) in model_cat_ability:
                    z = model_cat_ability[(i, cat)]
                else:
                    z = model_overall_z[i]
                M_pred[i, j] = col_mean[j] + z * col_std[j]

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 6: Bimodal-Aware Predictor
# ══════════════════════════════════════════════════════════════════════════════

def predict_bimodal_aware(M_train, base_fn=None):
    """For bimodal benchmarks: classify first (above/below threshold), then regress per group.
    For non-bimodal: use base prediction method."""
    if base_fn is None:
        base_fn = lambda M: predict_blend(M, alpha=0.6)

    obs = ~np.isnan(M_train)
    M_pred = base_fn(M_train)

    model_feats = build_model_features(M_train)

    for j in BIMODAL_BENCH_IDX:
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 6:
            continue

        y_values = M_train[train_idx, j]
        y_class = (y_values > BIMODAL_THRESHOLD).astype(int)

        # Need both classes represented
        if y_class.sum() < 2 or (1 - y_class).sum() < 2:
            continue

        # Features: other benchmarks (mean-imputed) + model metadata
        col_mean = np.nanmean(M_train, axis=0)
        X_bench = M_train.copy()
        for jj in range(N_BENCH):
            nan_mask = np.isnan(X_bench[:, jj])
            X_bench[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0
        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        X_all = np.column_stack([X_bench[:, other_cols], model_feats])

        X_train = X_all[train_idx]

        # Stage 1: Classify
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_train, y_class)

        # Stage 2: Separate regressions for each group
        high_idx = train_idx[y_class == 1]
        low_idx = train_idx[y_class == 0]

        reg_high = Ridge(alpha=1.0)
        reg_low = Ridge(alpha=1.0)

        if len(high_idx) >= 3:
            reg_high.fit(X_all[high_idx], M_train[high_idx, j])
        if len(low_idx) >= 3:
            reg_low.fit(X_all[low_idx], M_train[low_idx, j])

        # Predict missing
        missing_idx = np.where(np.isnan(M_train[:, j]))[0]
        for i in missing_idx:
            x_i = X_all[i:i+1]
            pred_class = clf.predict(x_i)[0]
            prob = clf.predict_proba(x_i)[0]

            if pred_class == 1 and len(high_idx) >= 3:
                # Predicted "can do it" — use high regression
                pred = reg_high.predict(x_i)[0]
                # Ensure prediction is above threshold if confident
                if prob[1] > 0.7:
                    pred = max(pred, BIMODAL_THRESHOLD + 1)
            elif pred_class == 0 and len(low_idx) >= 3:
                # Predicted "can't do it" — use low regression
                pred = reg_low.predict(x_i)[0]
                if prob[0] > 0.7:
                    pred = min(pred, BIMODAL_THRESHOLD)
            else:
                pred = M_pred[i, j]  # fallback to base

            M_pred[i, j] = pred

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 7: Meta-Learner Ensemble (Stacking)
# ══════════════════════════════════════════════════════════════════════════════

def predict_meta_learner(M_train):
    """Meta-learner that selects/weights base methods per (model, benchmark) context.
    Uses features about the model, benchmark, and base predictions to learn optimal weighting."""
    obs = ~np.isnan(M_train)

    # Get base predictions
    M_breg = predict_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    M_svd2 = predict_svd(M_train, rank=2)
    M_svd5 = predict_svd(M_train, rank=5)

    bases = [M_breg, M_knn, M_svd2, M_svd5]
    n_bases = len(bases)

    # Build training data from observed entries
    # For each observed (i,j), compute LOO-ish residual
    # Actually, we can't do full LOO here (too expensive).
    # Instead, use the predictions from each method and compute errors on observed entries.
    # The meta-learner learns which method to trust based on context.

    model_feats = build_model_features(M_train)
    col_mean = np.nanmean(M_train, axis=0)
    col_std = np.nanstd(M_train, axis=0)
    col_std[col_std < 1e-8] = 1.0

    # Build features for each (model, benchmark) cell
    X_meta, y_meta = [], []
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if not obs[i, j]:
                continue

            # Features:
            feat = []
            # 1. Base predictions for this cell
            for base in bases:
                val = base[i, j] if np.isfinite(base[i, j]) else col_mean[j]
                feat.append(val)
            # 2. Variance among base predictions (disagreement)
            base_vals = [b[i, j] for b in bases if np.isfinite(b[i, j])]
            feat.append(np.std(base_vals) if len(base_vals) > 1 else 0.0)
            # 3. Model metadata: is_reasoning, n_obs, mean_z
            feat.append(float(MODEL_REASONING[i]))
            feat.append(float(obs[i].sum()))
            feat.append(model_feats[i, 5])  # mean z-score
            # 4. Benchmark features: category is implicit in column, use coverage + mean
            feat.append(float(obs[:, j].sum()))
            feat.append(col_mean[j])
            feat.append(col_std[j])

            X_meta.append(feat)
            y_meta.append(M_FULL[i, j])

    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)

    # Train meta-learner (Ridge regression on features → score)
    meta_model = Ridge(alpha=10.0)
    meta_model.fit(X_meta, y_meta)

    # Predict missing cells
    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            feat = []
            for base in bases:
                val = base[i, j] if np.isfinite(base[i, j]) else col_mean[j]
                feat.append(val)
            base_vals = [b[i, j] for b in bases if np.isfinite(b[i, j])]
            feat.append(np.std(base_vals) if len(base_vals) > 1 else 0.0)
            feat.append(float(MODEL_REASONING[i]))
            feat.append(float(obs[i].sum()))
            feat.append(model_feats[i, 5])
            feat.append(float(obs[:, j].sum()))
            feat.append(col_mean[j])
            feat.append(col_std[j])

            pred = meta_model.predict(np.array(feat).reshape(1, -1))[0]
            M_pred[i, j] = pred

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 8: Within-Family Log-Linear Interpolation + Residual Completion
# ══════════════════════════════════════════════════════════════════════════════

def predict_family_interpolation(M_train, base_fn=None):
    """For model families with known sizes, interpolate using log-linear scaling.
    Then apply matrix completion on the residuals."""
    if base_fn is None:
        base_fn = lambda M: predict_blend(M, alpha=0.6)

    obs = ~np.isnan(M_train)
    M_pred = base_fn(M_train)

    # Identify families: same provider + similar naming pattern
    # Group by (provider, architecture, reasoning) as proxy for family
    families = defaultdict(list)
    for i in range(N_MODELS):
        if np.isnan(MODEL_PARAMS[i]):
            continue
        key = (str(MODEL_PROVIDERS[i]), bool(MODEL_REASONING[i]))
        families[key].append(i)

    # Only use families with 3+ models with known params
    for key, members in families.items():
        if len(members) < 3:
            continue

        params = MODEL_PARAMS[np.array(members)]
        log_params = np.log(params + 1)

        for j in range(N_BENCH):
            # Which members have this benchmark observed?
            known = [(idx, members[idx]) for idx in range(len(members)) if obs[members[idx], j]]
            if len(known) < 2:
                continue

            # Fit log-linear: score = a * log(params) + b
            x = np.array([log_params[idx] for idx, _ in known])
            y = np.array([M_train[mi, j] for _, mi in known])

            if np.std(x) < 1e-6:
                continue

            # Simple linear regression
            slope = np.sum((x - x.mean()) * (y - y.mean())) / (np.sum((x - x.mean())**2) + 1e-10)
            intercept = y.mean() - slope * x.mean()

            # Predict missing members in this family
            for idx in range(len(members)):
                mi = members[idx]
                if np.isnan(M_train[mi, j]):
                    family_pred = slope * log_params[idx] + intercept
                    base_pred = M_pred[mi, j]
                    # Average family prediction with base prediction (trust both)
                    if np.isfinite(base_pred):
                        M_pred[mi, j] = 0.4 * family_pred + 0.6 * base_pred
                    else:
                        M_pred[mi, j] = family_pred

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 9: GBT + Blend Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def predict_gbt_blend(M_train, alpha=0.5):
    """Average of GBT and Blend predictions."""
    M_gbt = predict_gbt(M_train)
    M_blend = predict_blend(M_train, alpha=0.6)
    obs = ~np.isnan(M_train)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            g, b = M_gbt[i, j], M_blend[i, j]
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
#  METHOD 10: Missingness-Aware KNN
# ══════════════════════════════════════════════════════════════════════════════

def predict_miss_knn(M_train, k=5, miss_weight=0.2):
    """KNN that uses missingness pattern similarity alongside score similarity.
    Models with similar 'which benchmarks were run' patterns are likely similar models."""
    M_norm, cm, cs = col_normalize(M_train)
    obs = ~np.isnan(M_norm)
    M_pred = M_train.copy()

    # Missingness patterns (binary: was this benchmark observed?)
    miss_pattern = obs.astype(float)

    for i in range(N_MODELS):
        missing = np.where(np.isnan(M_train[i]))[0]
        if len(missing) == 0:
            continue

        shared_all = obs[i]
        sims = np.full(N_MODELS, -999.0)
        for k2 in range(N_MODELS):
            if k2 == i:
                continue
            # Score similarity
            shared = shared_all & obs[k2]
            score_sim = -999.0
            if shared.sum() >= 3:
                a, b = M_norm[i, shared], M_norm[k2, shared]
                denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
                score_sim = np.dot(a, b) / denom

            # Missingness similarity (Jaccard)
            intersection = np.sum(obs[i] & obs[k2])
            union = np.sum(obs[i] | obs[k2])
            miss_sim = intersection / (union + 1e-10)

            if score_sim > -999:
                sims[k2] = (1 - miss_weight) * score_sim + miss_weight * miss_sim
            elif miss_sim > 0.5:
                sims[k2] = miss_weight * miss_sim * 0.3  # penalized fallback

        top_k = np.argsort(sims)[-k:]
        top_k = top_k[sims[top_k] > -999]
        if len(top_k) == 0:
            for j in missing:
                M_pred[i, j] = cm[j]
            continue

        weights = np.maximum(sims[top_k], 0.01)
        weights /= weights.sum()
        for j in missing:
            vals, ws = [], []
            for idx, ki in enumerate(top_k):
                if not np.isnan(M_norm[ki, j]):
                    vals.append(M_norm[ki, j])
                    ws.append(weights[idx])
            if vals:
                pred_norm = np.average(vals, weights=ws)
                M_pred[i, j] = pred_norm * cs[j] + cm[j]
            else:
                M_pred[i, j] = cm[j]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 11: Logit-Space BenchReg (for percentage benchmarks)
# ══════════════════════════════════════════════════════════════════════════════

def predict_logit_benchreg(M_train, top_k=5, min_r2=0.2):
    """BenchReg in logit space for percentage benchmarks, normal space for others.
    logit(p) = log(p / (1-p)) linearizes sigmoid-shaped relationships."""
    obs = ~np.isnan(M_train)

    # Identify which benchmarks are percentages (0-100 range)
    is_pct = np.zeros(N_BENCH, dtype=bool)
    for j in range(N_BENCH):
        vals = M_train[obs[:, j], j]
        if len(vals) > 0 and vals.min() >= 0 and vals.max() <= 100:
            is_pct[j] = True

    # Transform to logit space where applicable
    M_logit = M_train.copy()
    for j in range(N_BENCH):
        if is_pct[j]:
            valid = obs[:, j]
            # Clip to (0.5, 99.5) to avoid infinities
            clipped = np.clip(M_train[valid, j], 0.5, 99.5) / 100.0
            M_logit[valid, j] = np.log(clipped / (1 - clipped))

    # Run BenchReg in transformed space
    M_pred_logit = predict_benchreg(M_logit, top_k=top_k, min_r2=min_r2)

    # Transform back
    M_pred = M_pred_logit.copy()
    for j in range(N_BENCH):
        if is_pct[j]:
            missing = np.isnan(M_train[:, j])
            if missing.any():
                logit_vals = M_pred_logit[missing, j]
                # Inverse logit: p = 1 / (1 + exp(-x))
                M_pred[missing, j] = 100.0 / (1 + np.exp(-logit_vals))

    M_pred[obs] = M_train[obs]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 12: Logit Blend
# ══════════════════════════════════════════════════════════════════════════════

def predict_logit_blend(M_train, alpha=0.6):
    """Blend of logit-space BenchReg + KNN."""
    M_breg = predict_logit_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            b, k = M_breg[i, j], M_knn[i, j]
            b_ok, k_ok = np.isfinite(b), np.isfinite(k)
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
#  METHOD 13: Provider-Bias Correction
# ══════════════════════════════════════════════════════════════════════════════

def predict_provider_corrected(M_train, base_fn=None):
    """Applies provider × category bias correction on top of base predictions.
    If Google models systematically outperform on multimodal benchmarks relative to
    their general capability, capture that bias."""
    if base_fn is None:
        base_fn = lambda M: predict_blend(M, alpha=0.6)

    obs = ~np.isnan(M_train)
    M_base = base_fn(M_train)

    # Compute residuals on observed entries: actual - base_prediction
    # Group by (provider, category)
    bias = defaultdict(list)
    for i in range(N_MODELS):
        prov = str(MODEL_PROVIDERS[i])
        for j in range(N_BENCH):
            if obs[i, j] and np.isfinite(M_base[i, j]):
                cat = BENCH_CATS[j]
                residual = M_FULL[i, j] - M_base[i, j]
                bias[(prov, cat)].append(residual)

    # Compute median bias per (provider, category), only if enough data
    bias_correction = {}
    for key, residuals in bias.items():
        if len(residuals) >= 3:
            bias_correction[key] = np.median(residuals)

    # Apply corrections
    M_pred = M_base.copy()
    for i in range(N_MODELS):
        prov = str(MODEL_PROVIDERS[i])
        for j in range(N_BENCH):
            if np.isnan(M_train[i, j]):
                cat = BENCH_CATS[j]
                corr = bias_correction.get((prov, cat), 0.0)
                M_pred[i, j] = M_base[i, j] + corr

    M_pred[obs] = M_train[obs]
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 14: Multi-Ridge (all benchmarks + metadata in one Ridge per target)
# ══════════════════════════════════════════════════════════════════════════════

def predict_multi_ridge(M_train, alpha=10.0):
    """For each benchmark, fit Ridge regression using ALL other benchmarks + metadata.
    Unlike BenchReg which picks top-5, this uses all features but relies on Ridge
    regularization to handle irrelevant ones."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    model_feats = build_model_features(M_train)
    col_mean = np.nanmean(M_train, axis=0)

    for j in range(N_BENCH):
        train_idx = np.where(obs[:, j])[0]
        if len(train_idx) < 8:
            M_pred[np.isnan(M_train[:, j]), j] = col_mean[j] if not np.isnan(col_mean[j]) else 50.0
            continue

        # Impute missing benchmark scores for feature construction
        M_imp = M_train.copy()
        for jj in range(N_BENCH):
            nan_mask = np.isnan(M_imp[:, jj])
            M_imp[nan_mask, jj] = col_mean[jj] if not np.isnan(col_mean[jj]) else 50.0

        other_cols = [jj for jj in range(N_BENCH) if jj != j]
        X = np.column_stack([M_imp[:, other_cols], model_feats])
        y = M_train[:, j]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        y_train = y[train_idx]

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        missing_idx = np.where(np.isnan(M_train[:, j]))[0]
        if len(missing_idx) > 0:
            X_test = scaler.transform(X[missing_idx])
            M_pred[missing_idx, j] = ridge.predict(X_test)

    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 15: Multi-Ridge + Blend
# ══════════════════════════════════════════════════════════════════════════════

def predict_ridge_blend(M_train, alpha=0.5):
    """Blend Multi-Ridge with BenchReg+KNN."""
    M_ridge = predict_multi_ridge(M_train)
    M_blend = predict_blend(M_train, alpha=0.6)
    obs = ~np.isnan(M_train)

    M_pred = M_train.copy()
    for i in range(N_MODELS):
        for j in range(N_BENCH):
            if obs[i, j]:
                continue
            r, b = M_ridge[i, j], M_blend[i, j]
            r_ok, b_ok = np.isfinite(r), np.isfinite(b)
            if r_ok and b_ok:
                M_pred[i, j] = alpha * r + (1 - alpha) * b
            elif r_ok:
                M_pred[i, j] = r
            elif b_ok:
                M_pred[i, j] = b
            else:
                M_pred[i, j] = np.nanmean(M_train[:, j])
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN: EVALUATE ALL METHODS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 90)
    print("  CREATIVE METHODS EVALUATION")
    print("  Per-model 50% holdout × 5 seeds + Random 20% holdout")
    print("=" * 90)

    methods = [
        # Baselines
        ("B0:BenchMean",          predict_B0),
        ("Baseline:Blend(0.6)",   lambda M: predict_blend(M, alpha=0.6)),
        ("Baseline:BenchReg",     predict_benchreg),
        ("Baseline:KNN(k=5)",     lambda M: predict_B2(M, k=5)),
        ("Baseline:SVD(r=2)",     lambda M: predict_svd(M, rank=2)),

        # New methods
        ("CatBenchReg",           predict_cat_benchreg),
        ("CatBlend(0.6)",         predict_cat_blend),
        ("MetaKNN(w=0.3)",        lambda M: predict_meta_knn(M, meta_weight=0.3)),
        ("GBT(d3,n50)",           predict_gbt),
        ("Hierarchical",          predict_hierarchical),
        ("BimodalAware",          predict_bimodal_aware),
        ("MetaLearner",           predict_meta_learner),
        ("FamilyInterp",          predict_family_interpolation),
        ("GBT+Blend(0.5)",       predict_gbt_blend),
        ("MissKNN(w=0.2)",        predict_miss_knn),
        ("LogitBenchReg",         predict_logit_benchreg),
        ("LogitBlend(0.6)",       predict_logit_blend),
        ("ProviderCorrected",     predict_provider_corrected),
        ("MultiRidge(α=10)",      predict_multi_ridge),
        ("RidgeBlend(0.5)",       predict_ridge_blend),
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
    print("  RESULTS TABLE (sorted by PM-MedAPE)")
    print("=" * 130)
    results.sort(key=lambda x: x.get('pm_medape', 999))

    header = (f"  {'Method':<25s} {'PM-MedAPE':>9s} {'R-MedAPE':>9s} {'PM-MAE':>7s} "
              f"{'±3pts':>6s} {'MedAPE>50':>9s} {'MedAPE≤50':>9s} "
              f"{'BiAcc':>6s} {'Cov':>5s} {'Time':>6s}")
    print(header)
    print("  " + "─" * 128)

    for r in results:
        print(f"  {r['method']:<25s} {r['pm_medape']:>8.2f}% {r['rand_medape']:>8.2f}% "
              f"{r['pm_mae']:>7.2f} {r['pct_within3']:>5.1f}% "
              f"{r['medape_hi']:>8.2f}% {r['medape_lo']:>8.2f}% "
              f"{r.get('bimodal_acc', float('nan')):>5.1f}% "
              f"{r.get('coverage', 100):>4.1f}% "
              f"{r.get('time', 0):>5.1f}s")

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)
