#!/usr/bin/env python3
"""
Quick before/after comparison of audit fixes.
Runs LogitSVD Blend on both pre-audit and post-audit matrices.
"""
import numpy as np
import sys, os, warnings, json

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    holdout_per_model,
)
from all_methods import predict_logit_svd_blend

# Bimodal setup
BIMODAL_IDS = ['arc_agi_1', 'arc_agi_2', 'imo_2025', 'usamo_2025', 'matharena_apex_2025']
BIMODAL_IDX = [BENCH_IDS.index(b) for b in BIMODAL_IDS if b in BENCH_IDS]
BIMODAL_THRESHOLD = 10.0


def compute_metrics(actual, predicted, test_set):
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    valid = ~np.isnan(p) & ~np.isnan(a)
    a, p = a[valid], p[valid]
    ts_valid = [ts for ts, v in zip(test_set, valid) if v]

    if len(a) == 0:
        return {}

    abs_err = np.abs(p - a)
    nonzero = np.abs(a) > 1e-6
    ape = abs_err[nonzero] / np.abs(a[nonzero]) * 100

    # Bimodal
    bc, bt = 0, 0
    for k, (i, j) in enumerate(ts_valid):
        if j in BIMODAL_IDX:
            bt += 1
            if (actual[k] > BIMODAL_THRESHOLD) == (predicted[k] > BIMODAL_THRESHOLD):
                bc += 1

    return {
        'MedAPE': float(np.median(ape)) if len(ape) > 0 else np.nan,
        'MAE': float(np.median(abs_err)),
        'Within_3': float(np.mean(abs_err <= 3.0) * 100),
        'Within_5': float(np.mean(abs_err <= 5.0) * 100),
        'BimodalAcc': float(bc / bt * 100) if bt > 0 else np.nan,
        'Coverage': float(np.sum(valid) / len(actual) * 100),
        'N': int(np.sum(valid)),
    }


def run_eval(label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Matrix: {N_MODELS}x{N_BENCH}, observed: {int(OBSERVED.sum())}")
    print(f"{'='*60}")

    # holdout_per_model returns list of (M_train, test_cells) folds
    folds = holdout_per_model(k_frac=0.5, min_scores=8, n_folds=3, seed=42)

    all_actual, all_pred, all_ts = [], [], []

    for fold_idx, (M_train, test_cells) in enumerate(folds):
        pred_matrix = predict_logit_svd_blend(M_train)

        for (i, j) in test_cells:
            all_actual.append(M_FULL[i, j])
            all_pred.append(pred_matrix[i, j])
            all_ts.append((i, j))
        print(f"  Fold {fold_idx+1}/3: {len(test_cells)} test cells")

    metrics = compute_metrics(all_actual, all_pred, all_ts)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>12s}: {v:.2f}")
        else:
            print(f"  {k:>12s}: {v}")

    return metrics


# ── Run on current (post-audit) matrix ──
print("Running LogitSVD Blend evaluation (3 seeds, per-model leave-50%-out)...")
post = run_eval("POST-AUDIT MATRIX")

# ── Load pre-audit results if available ──
pre_results_path = os.path.join(REPO_ROOT, 'results', 'logit_svd_eval.json')
if os.path.exists(pre_results_path):
    with open(pre_results_path) as f:
        pre_data = json.load(f)
    # Extract per-model metrics
    pm = pre_data.get('per_model_holdout', {})
    pre = {
        'MedAPE': pm.get('medape', np.nan),
        'MAE': pm.get('mae', np.nan),
        'Within_3': pm.get('within3', np.nan),
        'Within_5': pm.get('within5', np.nan),
        'BimodalAcc': pm.get('bimodal_acc', np.nan),
        'Coverage': pm.get('coverage', np.nan),
    }
    print(f"\n{'='*60}")
    print(f"  BEFORE/AFTER COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':>12s}  {'Pre-audit':>10s}  {'Post-audit':>10s}  {'Delta':>8s}")
    print(f"  {'-'*46}")
    for k in ['MedAPE', 'MAE', 'Within_3', 'Within_5', 'BimodalAcc']:
        pre_v = pre.get(k, np.nan)
        post_v = post.get(k, np.nan)
        if not (np.isnan(pre_v) or np.isnan(post_v)):
            delta = post_v - pre_v
            sign = '+' if delta >= 0 else ''
            print(f"  {k:>12s}  {pre_v:>10.2f}  {post_v:>10.2f}  {sign}{delta:>7.2f}")
        else:
            print(f"  {k:>12s}  {'N/A':>10s}  {post_v:>10.2f}  {'N/A':>8s}")
else:
    print("\nNo pre-audit results file found at", pre_results_path)

# Save post-audit results
post_path = os.path.join(REPO_ROOT, 'results', 'logit_svd_eval_post_audit.json')
with open(post_path, 'w') as f:
    json.dump({'post_audit': post, 'n_seeds': 3, 'method': 'logit_svd_blend'}, f, indent=2)
print(f"\nSaved post-audit results to {post_path}")
