#!/usr/bin/env python3
"""
Full Evaluation of BenchPress and All Components
=================================================

Produces a comprehensive results JSON covering:
1. BenchPress (LogitSVD Blend) — the recommended method
2. SVD-only at different ranks (1, 2, 3, 5, 8) — raw and logit space
3. LogitBenchReg only (regression component)
4. Blend ratio sweep: alpha in [0.0, 0.1, ..., 1.0]
5. Per-model holdout at different hiding fractions (10%, 20%, 30%, 40%, 50%)
6. Column Mean baseline
7. All Claude Sonnet prediction accuracies (from saved results)

All evaluated with per-model leave-50%-out (seed=42, 3 folds) as the primary metric.
"""

import sys, os, io, json, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'data'))
sys.path.insert(0, os.path.join(REPO, 'methods'))

_old = sys.stdout; sys.stdout = io.StringIO()
from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, holdout_per_model,
)
from all_methods import (
    predict_logit_svd_blend, predict_logit_benchreg, predict_svd_logit,
    predict_svd, predict_B0, predict_B1, predict_B2, predict_B3,
)
sys.stdout = _old


def evaluate(predict_fn, k_frac=0.5, min_scores=8, seed=42, n_folds=3, label=""):
    """Run per-model holdout evaluation and return metrics dict."""
    folds = holdout_per_model(k_frac=k_frac, min_scores=min_scores,
                              n_folds=n_folds, seed=seed)
    all_apes = []
    all_abs = []
    n_missing = 0

    for M_train, test_set in folds:
        M_pred = predict_fn(M_train)
        for i, j in test_set:
            actual = M_FULL[i, j]
            pred = M_pred[i, j]
            if np.isfinite(pred) and abs(actual) > 1e-6:
                all_apes.append(abs(pred - actual) / abs(actual) * 100)
                all_abs.append(abs(pred - actual))
            else:
                n_missing += 1

    n = len(all_apes)
    if n == 0:
        return {'label': label, 'n': 0}

    return {
        'label': label,
        'MedAPE': round(float(np.median(all_apes)), 4),
        'MedianAE': round(float(np.median(all_abs)), 4),
        'MeanAE': round(float(np.mean(all_abs)), 4),
        'Within_3': round(sum(1 for e in all_abs if e <= 3) / n * 100, 2),
        'Within_5': round(sum(1 for e in all_abs if e <= 5) / n * 100, 2),
        'Within_10': round(sum(1 for e in all_abs if e <= 10) / n * 100, 2),
        'Coverage': round(n / (n + n_missing) * 100, 2),
        'N': n,
        'N_missing': n_missing,
    }


def evaluate_multi_seed(predict_fn, k_frac=0.5, seeds=[42, 43, 44], label=""):
    """Run evaluation across multiple seeds and return per-seed + pooled metrics."""
    per_seed = []
    all_apes = []
    all_abs = []

    for seed in seeds:
        folds = holdout_per_model(k_frac=k_frac, min_scores=8,
                                  n_folds=3, seed=seed)
        seed_apes = []
        seed_abs = []
        for M_train, test_set in folds:
            M_pred = predict_fn(M_train)
            for i, j in test_set:
                actual = M_FULL[i, j]
                pred = M_pred[i, j]
                if np.isfinite(pred) and abs(actual) > 1e-6:
                    ape = abs(pred - actual) / abs(actual) * 100
                    seed_apes.append(ape)
                    seed_abs.append(abs(pred - actual))
                    all_apes.append(ape)
                    all_abs.append(abs(pred - actual))

        per_seed.append({
            'seed': seed,
            'MedAPE': round(float(np.median(seed_apes)), 4),
            'MedianAE': round(float(np.median(seed_abs)), 4),
            'N': len(seed_apes),
        })

    n = len(all_apes)
    return {
        'label': label,
        'pooled': {
            'MedAPE': round(float(np.median(all_apes)), 4),
            'MedianAE': round(float(np.median(all_abs)), 4),
            'MeanAE': round(float(np.mean(all_abs)), 4),
            'Within_3': round(sum(1 for e in all_abs if e <= 3) / n * 100, 2),
            'Within_5': round(sum(1 for e in all_abs if e <= 5) / n * 100, 2),
            'N': n,
        },
        'per_seed': per_seed,
    }


results = {}
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. BenchPress (recommended method) — multi-seed
# ══════════════════════════════════════════════════════════════════════════════
print("1. BenchPress (LogitSVD Blend, alpha=0.6) — multi-seed evaluation...")
results['benchpress'] = evaluate_multi_seed(
    predict_logit_svd_blend, k_frac=0.5, label="BenchPress (0.6×LogitBenchReg + 0.4×SVD-Logit r=2)")
print(f"   MedAPE = {results['benchpress']['pooled']['MedAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SVD-only at different ranks — raw space
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. SVD (raw space) at different ranks...")
results['svd_raw'] = {}
for rank in [1, 2, 3, 5, 8]:
    fn = lambda M, r=rank: predict_svd(M, rank=r)
    r = evaluate(fn, label=f"SVD raw rank={rank}")
    results['svd_raw'][f'rank_{rank}'] = r
    print(f"   rank={rank}: MedAPE={r['MedAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SVD-Logit at different ranks
# ══════════════════════════════════════════════════════════════════════════════
print("\n3. SVD-Logit (logit space) at different ranks...")
results['svd_logit'] = {}
for rank in [1, 2, 3, 5, 8]:
    fn = lambda M, r=rank: predict_svd_logit(M, rank=r)
    r = evaluate(fn, label=f"SVD-Logit rank={rank}")
    results['svd_logit'][f'rank_{rank}'] = r
    print(f"   rank={rank}: MedAPE={r['MedAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LogitBenchReg only (regression component)
# ══════════════════════════════════════════════════════════════════════════════
print("\n4. LogitBenchReg only...")
results['logit_benchreg'] = evaluate(predict_logit_benchreg, label="LogitBenchReg only")
print(f"   MedAPE = {results['logit_benchreg']['MedAPE']}%, Coverage = {results['logit_benchreg']['Coverage']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Blend ratio sweep: alpha × LogitBenchReg + (1-alpha) × SVD-Logit
# ══════════════════════════════════════════════════════════════════════════════
print("\n5. Blend ratio sweep (alpha = weight on LogitBenchReg)...")
results['blend_sweep'] = {}
for alpha_10 in range(0, 11):
    alpha = alpha_10 / 10.0
    fn = lambda M, a=alpha: predict_logit_svd_blend(M, alpha=a)
    r = evaluate(fn, label=f"Blend alpha={alpha:.1f}")
    results['blend_sweep'][f'alpha_{alpha_10:02d}'] = r
    print(f"   alpha={alpha:.1f}: MedAPE={r['MedAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6. BenchPress at different hiding fractions
# ══════════════════════════════════════════════════════════════════════════════
print("\n6. BenchPress at different hiding fractions...")
results['hiding_fractions'] = {}
for frac_pct in [10, 20, 30, 40, 50]:
    frac = frac_pct / 100.0
    r = evaluate(predict_logit_svd_blend, k_frac=frac,
                 label=f"BenchPress hide={frac_pct}%")
    results['hiding_fractions'][f'hide_{frac_pct}'] = r
    print(f"   hide={frac_pct}%: MedAPE={r['MedAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Each component at different hiding fractions
# ══════════════════════════════════════════════════════════════════════════════
print("\n7. Components at different hiding fractions...")
results['components_by_fraction'] = {}
methods = {
    'benchpress': predict_logit_svd_blend,
    'svd_logit_r2': lambda M: predict_svd_logit(M, rank=2),
    'logit_benchreg': predict_logit_benchreg,
    'column_mean': predict_B0,
}
for name, fn in methods.items():
    results['components_by_fraction'][name] = {}
    for frac_pct in [10, 20, 30, 40, 50]:
        frac = frac_pct / 100.0
        r = evaluate(fn, k_frac=frac, label=f"{name} hide={frac_pct}%")
        results['components_by_fraction'][name][f'hide_{frac_pct}'] = r
    medapes = [results['components_by_fraction'][name][f'hide_{p}']['MedAPE']
               for p in [10, 20, 30, 40, 50]]
    print(f"   {name:<20s}: {', '.join(f'{m:.1f}%' for m in medapes)}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Baselines
# ══════════════════════════════════════════════════════════════════════════════
print("\n8. Baselines (50% holdout)...")
results['baselines'] = {}
baselines = {
    'B0_column_mean': ('Column Mean', predict_B0),
    'B1_model_normalized': ('Model-Normalized', predict_B1),
    'B2_knn': ('KNN k=5', predict_B2),
    'B3_bench_knn': ('Bench-KNN k=5', predict_B3),
}
for key, (name, fn) in baselines.items():
    r = evaluate(fn, label=name)
    results['baselines'][key] = r
    print(f"   {name:<25s}: MedAPE={r['MedAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Claude Sonnet predictions (load from saved results)
# ══════════════════════════════════════════════════════════════════════════════
print("\n9. Claude Sonnet predictions (from saved JSON)...")
results['claude_predictions'] = {}

claude_files = {
    'claude_eval': os.path.join(REPO, 'results', 'claude_eval.json'),
    'claude_comparison': os.path.join(REPO, 'results', 'claude_comparison_post_audit.json'),
    'claude_analysis': os.path.join(REPO, 'results', 'claude_analysis.json'),
    'claude_vs_algorithm': os.path.join(REPO, 'results', 'claude_vs_algorithm_phase.json'),
    'claude_opus_nothinking': os.path.join(REPO, 'results', 'claude_opus_nothinking.json'),
    'claude_opus_rowonly': os.path.join(REPO, 'results', 'claude_opus_rowonly.json'),
}
for key, path in claude_files.items():
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        results['claude_predictions'][key] = data
        print(f"   Loaded {key}")
    else:
        print(f"   MISSING: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Matrix statistics
# ══════════════════════════════════════════════════════════════════════════════
results['matrix_stats'] = {
    'n_models': int(N_MODELS),
    'n_benchmarks': int(N_BENCH),
    'observed_cells': int(OBSERVED.sum()),
    'total_cells': int(N_MODELS * N_BENCH),
    'fill_rate': round(float(OBSERVED.sum() / (N_MODELS * N_BENCH) * 100), 2),
    'missing_cells': int(N_MODELS * N_BENCH - OBSERVED.sum()),
}

elapsed = time.time() - t0
results['meta'] = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'elapsed_seconds': round(elapsed, 1),
    'primary_metric': 'MedAPE = median(|pred - actual| / |actual| × 100) over held-out cells',
    'primary_holdout': 'per-model leave-50%-out, 3 folds, seed=42',
    'MedianAE_note': 'MedianAE = median(|pred - actual|) in raw score points',
    'MeanAE_note': 'MeanAE = mean(|pred - actual|) — inflated by non-percentage benchmarks (Elo, Codeforces)',
    'method_name': 'BenchPress = 0.6 × LogitBenchReg + 0.4 × SVD-Logit(rank=2)',
}

# Save results
out_path = os.path.join(REPO, 'results', 'full_evaluation.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
print(f"Total time: {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Method':<35s} {'MedAPE':>8s} {'MedianAE':>10s} {'±3pts':>7s} {'±5pts':>7s} {'Cov':>6s}")
print("-" * 80)

# BenchPress
bp = results['benchpress']['pooled']
print(f"{'BenchPress (recommended)':<35s} {bp['MedAPE']:>7.2f}% {bp['MedianAE']:>9.2f} {bp['Within_3']:>6.1f}% {bp['Within_5']:>6.1f}%  100%")

# LogitBenchReg
lb = results['logit_benchreg']
print(f"{'LogitBenchReg only':<35s} {lb['MedAPE']:>7.2f}% {lb['MedianAE']:>9.2f} {lb['Within_3']:>6.1f}% {lb['Within_5']:>6.1f}% {lb['Coverage']:>5.1f}%")

# SVD-Logit ranks
for rank in [1, 2, 3, 5, 8]:
    sv = results['svd_logit'][f'rank_{rank}']
    print(f"{'SVD-Logit rank=' + str(rank):<35s} {sv['MedAPE']:>7.2f}% {sv['MedianAE']:>9.2f} {sv['Within_3']:>6.1f}% {sv['Within_5']:>6.1f}% {sv['Coverage']:>5.1f}%")

# SVD raw ranks
for rank in [2, 3]:
    sv = results['svd_raw'][f'rank_{rank}']
    print(f"{'SVD raw rank=' + str(rank):<35s} {sv['MedAPE']:>7.2f}% {sv['MedianAE']:>9.2f} {sv['Within_3']:>6.1f}% {sv['Within_5']:>6.1f}% {sv['Coverage']:>5.1f}%")

# Baselines
for key, (name, _) in baselines.items():
    b = results['baselines'][key]
    print(f"{name:<35s} {b['MedAPE']:>7.2f}% {b['MedianAE']:>9.2f} {b['Within_3']:>6.1f}% {b['Within_5']:>6.1f}% {b['Coverage']:>5.1f}%")

# Blend sweep summary
print("\n" + "=" * 80)
print("BLEND RATIO SWEEP (alpha × LogitBenchReg + (1-alpha) × SVD-Logit)")
print("=" * 80)
print(f"{'alpha':>6s} {'MedAPE':>8s} {'MedianAE':>10s} {'Coverage':>10s}")
print("-" * 40)
for alpha_10 in range(0, 11):
    r = results['blend_sweep'][f'alpha_{alpha_10:02d}']
    marker = " ◄ default" if alpha_10 == 6 else ""
    print(f"  {alpha_10/10:.1f}   {r['MedAPE']:>7.2f}%  {r['MedianAE']:>9.2f}  {r['Coverage']:>8.1f}%{marker}")

# Hiding fractions
print("\n" + "=" * 80)
print("HIDING FRACTION COMPARISON (MedAPE)")
print("=" * 80)
print(f"{'Method':<25s} {'10%':>8s} {'20%':>8s} {'30%':>8s} {'40%':>8s} {'50%':>8s}")
print("-" * 70)
for name in ['benchpress', 'svd_logit_r2', 'logit_benchreg', 'column_mean']:
    vals = []
    for p in [10, 20, 30, 40, 50]:
        vals.append(f"{results['components_by_fraction'][name][f'hide_{p}']['MedAPE']:>6.2f}%")
    print(f"  {name:<23s} {'  '.join(vals)}")
