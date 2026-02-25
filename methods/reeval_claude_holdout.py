#!/usr/bin/env python3
"""
Re-evaluate statistical baselines on the post-audit matrix using the same
holdout as the Claude predictor experiment (random 20%, seed=42, 276 cells).

This lets us update the comparison table without re-running Claude.
"""
import numpy as np
import sys, os, warnings, json

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, holdout_random_cells,
)
from all_methods import predict_logit_svd_blend, predict_benchreg

# ── Bimodal setup (same as claude_predictor.py) ──
BIMODAL_IDS = ['arc_agi_1', 'arc_agi_2', 'imo_2025', 'usamo_2025', 'matharena_apex_2025']
BIMODAL_IDX = set()
for b in BIMODAL_IDS:
    if b in BENCH_IDS:
        BIMODAL_IDX.add(BENCH_IDS.index(b))
BIMODAL_THRESHOLD = 10.0


def compute_metrics(actual, predicted, test_cells):
    """Compute extended metrics matching claude_analysis.json format."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    valid = ~np.isnan(p) & ~np.isnan(a)
    a_v, p_v = a[valid], p[valid]
    cells_v = [c for c, v in zip(test_cells, valid) if v]

    abs_err = np.abs(p_v - a_v)
    nonzero = np.abs(a_v) > 1e-6
    ape = abs_err[nonzero] / np.abs(a_v[nonzero]) * 100

    # Split hi/lo scoring
    hi = a_v[nonzero] > 50
    ape_hi = ape[hi] if hi.any() else np.array([])
    ape_lo = ape[~hi] if (~hi).any() else np.array([])

    # Bimodal accuracy
    bc, bt = 0, 0
    for k, (i, j) in enumerate(cells_v):
        if j in BIMODAL_IDX:
            bt += 1
            if (a_v[k] > BIMODAL_THRESHOLD) == (p_v[k] > BIMODAL_THRESHOLD):
                bc += 1

    return {
        'medape': float(np.median(ape)) if len(ape) > 0 else None,
        'mae': float(np.mean(abs_err)),
        'within3': float(np.mean(abs_err <= 3.0) * 100),
        'within5': float(np.mean(abs_err <= 5.0) * 100),
        'medape_hi': float(np.median(ape_hi)) if len(ape_hi) > 0 else None,
        'medape_lo': float(np.median(ape_lo)) if len(ape_lo) > 0 else None,
        'bimodal_acc': float(bc / bt * 100) if bt > 0 else None,
        'bimodal_n': bt,
        'coverage': float(np.sum(valid) / len(a) * 100),
        'n': int(np.sum(valid)),
    }


# ── Same holdout as Claude experiment ──
print("Generating holdout (random 20%, seed=42)...")
folds = holdout_random_cells(frac=0.2, n_folds=1, seed=42)
M_train, test_set = folds[0]
print(f"  {len(test_set)} holdout cells")

# ── Check which holdout cells were affected by audit ──
# Load pre-audit matrix for comparison
pre_audit_path = os.path.join(REPO_ROOT, 'data', 'llm_benchmark_matrix_BEFORE_AUDIT.xlsx')
if os.path.exists(pre_audit_path):
    import pandas as pd
    df_pre = pd.read_excel(pre_audit_path, index_col=0)
    META_COLS = ['Provider', 'Release Date', 'Parameters (B)', 'Active Parameters (B)',
                 'Context Window', 'Open Weights', 'Reasoning']
    for col in META_COLS:
        if col in df_pre.columns:
            df_pre = df_pre.drop(columns=[col])

    changed_in_holdout = []
    for i, j in test_set:
        mid = MODEL_IDS[i]
        bid = BENCH_IDS[j]
        new_val = M_FULL[i, j]
        if mid in df_pre.index and bid in df_pre.columns:
            old_val = df_pre.loc[mid, bid]
            if not np.isnan(old_val) and not np.isnan(new_val) and abs(old_val - new_val) > 0.01:
                changed_in_holdout.append((mid, bid, old_val, new_val))
            elif np.isnan(new_val) and not np.isnan(old_val):
                changed_in_holdout.append((mid, bid, old_val, 'REMOVED'))

    if changed_in_holdout:
        print(f"\n  {len(changed_in_holdout)} holdout cells changed by audit:")
        for mid, bid, old, new in changed_in_holdout:
            print(f"    {mid} / {bid}: {old} → {new}")
    else:
        print("\n  No holdout cells were affected by the audit.")

# ── Run baselines ──
print("\nRunning LogitSVD Blend...")
M_logit = predict_logit_svd_blend(M_train)
actual_l, pred_l, cells_l = [], [], []
for i, j in test_set:
    actual_l.append(M_FULL[i, j])
    pred_l.append(M_logit[i, j])
    cells_l.append((i, j))
logit_metrics = compute_metrics(actual_l, pred_l, cells_l)

print("Running BenchReg (no logit)...")
M_breg = predict_benchreg(M_train)
actual_b, pred_b, cells_b = [], [], []
for i, j in test_set:
    actual_b.append(M_FULL[i, j])
    pred_b.append(M_breg[i, j])
    cells_b.append((i, j))
breg_metrics = compute_metrics(actual_b, pred_b, cells_b)

# ── Load pre-audit Claude results ──
claude_path = os.path.join(REPO_ROOT, 'results', 'claude_analysis.json')
with open(claude_path) as f:
    claude_data = json.load(f)

# ── Print comparison table ──
print(f"\n{'='*80}")
print("  COMPARISON TABLE (same 276-cell holdout)")
print(f"{'='*80}")
print(f"  {'Method':<30s} {'MedAPE':>8s} {'±5 pts':>8s} {'BiAcc':>8s} {'Cost':>8s}")
print(f"  {'-'*62}")

rows = [
    ("Claude Sonnet (full matrix)", claude_data['full_matrix']['all'], claude_data['full_matrix']['cost']),
    ("LogitSVD Blend", logit_metrics, 0),
    ("Claude Sonnet (row-only)", claude_data['row_only']['all'], claude_data['row_only']['cost']),
    ("BenchReg (no logit)", breg_metrics, 0),
]

for name, m, cost in rows:
    medape = m['medape']
    within5 = m['within5']
    bimodal = m['bimodal_acc']
    cost_str = f"${cost:.2f}" if cost > 0 else "~$0"
    print(f"  {name:<30s} {medape:>7.1f}% {within5:>7.1f}% {bimodal:>7.1f}% {cost_str:>8s}")

# ── Also print pre-audit baseline numbers for comparison ──
print(f"\n  PRE-AUDIT baseline comparison:")
print(f"  {'Method':<30s} {'Pre MedAPE':>10s} {'Post MedAPE':>11s} {'Delta':>8s}")
print(f"  {'-'*62}")
pre_logit = claude_data['baselines']['LogitSVD Blend']
pre_breg = claude_data['baselines']['BenchReg']

for name, pre, post in [
    ("LogitSVD Blend", pre_logit, logit_metrics),
    ("BenchReg (no logit)", pre_breg, breg_metrics),
]:
    delta = post['medape'] - pre['medape']
    sign = '+' if delta >= 0 else ''
    print(f"  {name:<30s} {pre['medape']:>9.2f}% {post['medape']:>10.2f}% {sign}{delta:>7.2f}")

# ── Save results ──
out = {
    'holdout': {'seed': 42, 'frac': 0.2, 'n_cells': len(test_set)},
    'post_audit': {
        'LogitSVD Blend': logit_metrics,
        'BenchReg': breg_metrics,
    },
    'pre_audit_claude': {
        'full_matrix': claude_data['full_matrix']['all'],
        'row_only': claude_data['row_only']['all'],
    },
    'note': 'Claude predictions are from pre-audit matrix. Statistical methods re-run on post-audit matrix.'
}
out_path = os.path.join(REPO_ROOT, 'results', 'claude_comparison_post_audit.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
