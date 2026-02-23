#!/usr/bin/env python3
"""
Evaluation Harness for LLM Benchmark Matrix Completion
=======================================================
Implements 6 holdout strategies, 7 metrics, and significance testing.
All prediction methods plug into this harness for fair comparison.
"""

import numpy as np
import sys, warnings, json, os
from collections import defaultdict
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))
from build_benchmark_matrix import MODELS, BENCHMARKS, DATA

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

MODEL_IDS   = [m[0] for m in MODELS]
BENCH_IDS   = [b[0] for b in BENCHMARKS]
MODEL_NAMES = {m[0]: m[1] for m in MODELS}
BENCH_NAMES = {b[0]: b[1] for b in BENCHMARKS}
MODEL_IDX   = {m: i for i, m in enumerate(MODEL_IDS)}
BENCH_IDX   = {b: i for i, b in enumerate(BENCH_IDS)}
N_MODELS    = len(MODEL_IDS)
N_BENCH     = len(BENCH_IDS)

# Metadata arrays
MODEL_PROVIDERS  = np.array([m[2] for m in MODELS])
MODEL_REASONING  = np.array([m[7] if len(m) > 7 else False for m in MODELS], dtype=bool)
MODEL_OPEN       = np.array([m[8] if len(m) > 8 else False for m in MODELS], dtype=bool)
MODEL_PARAMS     = np.array([m[4] if len(m) > 4 and m[4] is not None else np.nan for m in MODELS], dtype=float)
MODEL_ACTIVE     = np.array([m[5] if len(m) > 5 and m[5] is not None else np.nan for m in MODELS], dtype=float)
BENCH_CATS       = np.array([b[2] for b in BENCHMARKS])

# Build full matrix
M_FULL = np.full((N_MODELS, N_BENCH), np.nan)
for mid, bid, score, url in DATA:
    if mid in MODEL_IDX and bid in BENCH_IDX:
        M_FULL[MODEL_IDX[mid], BENCH_IDX[bid]] = score

OBSERVED = ~np.isnan(M_FULL)
print(f"Matrix: {N_MODELS}×{N_BENCH}, observed: {OBSERVED.sum()}, fill: {OBSERVED.sum()/(N_MODELS*N_BENCH)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(actual, predicted, label=""):
    """Compute all 7 metrics. actual and predicted are 1-D arrays of same length."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    # Filter out NaN predictions and zero actuals (can't compute APE)
    valid = ~np.isnan(predicted) & ~np.isnan(actual)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return {'n': 0, 'medape': np.nan, 'meanape': np.nan, 'rmse': np.nan,
                'r2': np.nan, 'pct5': np.nan, 'pct10': np.nan, 'worst': []}

    # Absolute errors
    abs_err = np.abs(predicted - actual)

    # APE: avoid division by zero
    nonzero = np.abs(actual) > 1e-6
    ape = np.full(len(actual), np.nan)
    ape[nonzero] = abs_err[nonzero] / np.abs(actual[nonzero])
    ape_valid = ape[~np.isnan(ape)]

    medape  = np.median(ape_valid) * 100 if len(ape_valid) > 0 else np.nan
    meanape = np.mean(ape_valid) * 100 if len(ape_valid) > 0 else np.nan

    # RMSE
    rmse = np.sqrt(np.mean(abs_err**2))

    # R²
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    # % within thresholds (using APE)
    pct5  = np.mean(ape_valid < 0.05) * 100 if len(ape_valid) > 0 else np.nan
    pct10 = np.mean(ape_valid < 0.10) * 100 if len(ape_valid) > 0 else np.nan

    # Worst 10
    worst_idx = np.argsort(ape)[::-1][:10]
    worst = [(actual[i], predicted[i], ape[i]*100 if not np.isnan(ape[i]) else 999) for i in worst_idx]

    return {
        'n': len(actual), 'medape': medape, 'meanape': meanape,
        'rmse': rmse, 'r2': r2, 'pct5': pct5, 'pct10': pct10,
        'worst': worst, 'abs_errors': abs_err, 'ape': ape_valid,
    }


def print_metrics(m, label=""):
    """Pretty-print a metrics dict."""
    if m['n'] == 0:
        print(f"  {label}: NO PREDICTIONS")
        return
    print(f"  {label}: MedAPE={m['medape']:.1f}%  MeanAPE={m['meanape']:.1f}%  "
          f"RMSE={m['rmse']:.1f}  R²={m['r2']:.3f}  "
          f"<5%={m['pct5']:.0f}%  <10%={m['pct10']:.0f}%  (n={m['n']})")


def significance_test(errors_a, errors_b):
    """Wilcoxon signed-rank test. Returns p-value and direction."""
    n = min(len(errors_a), len(errors_b))
    if n < 10:
        return np.nan, "insufficient"
    a, b = errors_a[:n], errors_b[:n]
    try:
        stat, p = sp_stats.wilcoxon(a, b)
        better = "A" if np.median(a) < np.median(b) else "B"
        return p, better
    except Exception:
        return np.nan, "error"


# ══════════════════════════════════════════════════════════════════════════════
#  HOLDOUT STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def holdout_random_cells(frac=0.2, n_folds=5, seed=42):
    """Strategy A: Randomly hide frac of all observed cells."""
    rng = np.random.RandomState(seed)
    obs_idx = list(zip(*np.where(OBSERVED)))
    folds = []
    for fold in range(n_folds):
        rng.shuffle(obs_idx)
        n_hide = int(len(obs_idx) * frac)
        test_set = obs_idx[:n_hide]
        M_train = M_FULL.copy()
        for i, j in test_set:
            M_train[i, j] = np.nan
        folds.append((M_train, test_set))
    return folds


def holdout_per_model(k_frac=0.5, min_scores=8, n_folds=3, seed=42):
    """Strategy B: For each model with ≥min_scores, hide k_frac of its scores."""
    rng = np.random.RandomState(seed)
    folds = []
    for fold in range(n_folds):
        M_train = M_FULL.copy()
        test_set = []
        for i in range(N_MODELS):
            obs_j = np.where(OBSERVED[i])[0]
            if len(obs_j) < min_scores:
                continue
            rng.shuffle(obs_j)
            n_hide = max(1, int(len(obs_j) * k_frac))
            # Rotate by fold to get different splits
            start = (fold * n_hide) % len(obs_j)
            if start + n_hide <= len(obs_j):
                hidden = obs_j[start:start+n_hide]
            else:
                hidden = np.concatenate([obs_j[start:], obs_j[:start+n_hide-len(obs_j)]])
            for j in hidden:
                M_train[i, j] = np.nan
                test_set.append((i, j))
        folds.append((M_train, test_set))
    return folds


def holdout_leave_one_benchmark(min_scores=5):
    """Strategy C: Leave one entire benchmark column out."""
    folds = []
    for j in range(N_BENCH):
        n_obs = OBSERVED[:, j].sum()
        if n_obs < min_scores:
            continue
        M_train = M_FULL.copy()
        test_set = []
        for i in range(N_MODELS):
            if OBSERVED[i, j]:
                M_train[i, j] = np.nan
                test_set.append((i, j))
        folds.append((M_train, test_set))
    return folds


def holdout_cold_start(n_reveal=3, min_scores=8, seed=42):
    """Strategy D: Reveal only n_reveal most common benchmarks per model."""
    rng = np.random.RandomState(seed)
    # Find the n_reveal most commonly-filled benchmarks
    bench_coverage = OBSERVED.sum(axis=0)
    top_benches = np.argsort(-bench_coverage)[:n_reveal]

    folds = []
    M_train = M_FULL.copy()
    test_set = []
    for i in range(N_MODELS):
        obs_j = np.where(OBSERVED[i])[0]
        if len(obs_j) < min_scores:
            continue
        for j in obs_j:
            if j not in top_benches:
                M_train[i, j] = np.nan
                test_set.append((i, j))
    folds.append((M_train, test_set))
    return folds


def holdout_stratified_difficulty(frac=0.2, seed=42):
    """Strategy E: Hold out from easy and hard benchmarks separately."""
    # Use per-benchmark coverage as proxy for difficulty (sparse = harder)
    bench_coverage = OBSERVED.sum(axis=0)
    median_cov = np.median(bench_coverage)
    easy_benches = set(np.where(bench_coverage >= median_cov)[0])
    hard_benches = set(np.where(bench_coverage < median_cov)[0])

    rng = np.random.RandomState(seed)
    folds = []

    for bench_group, label in [(easy_benches, "easy"), (hard_benches, "hard")]:
        obs_in_group = [(i, j) for i, j in zip(*np.where(OBSERVED)) if j in bench_group]
        rng.shuffle(obs_in_group)
        n_hide = int(len(obs_in_group) * frac)
        test_set = obs_in_group[:n_hide]
        M_train = M_FULL.copy()
        for i, j in test_set:
            M_train[i, j] = np.nan
        folds.append((M_train, test_set, label))

    return folds


def holdout_leave_one_provider():
    """Strategy F: Leave one provider out entirely."""
    providers = sorted(set(MODEL_PROVIDERS))
    folds = []
    for prov in providers:
        prov_models = np.where(MODEL_PROVIDERS == prov)[0]
        # Need enough observed entries to be meaningful
        n_obs = sum(OBSERVED[i].sum() for i in prov_models)
        if n_obs < 10:
            continue
        M_train = M_FULL.copy()
        test_set = []
        for i in prov_models:
            for j in range(N_BENCH):
                if OBSERVED[i, j]:
                    M_train[i, j] = np.nan
                    test_set.append((i, j))
        folds.append((M_train, test_set, prov))
    return folds


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_method(predict_fn, holdout_folds, method_name=""):
    """
    Run a prediction method on holdout folds and collect metrics.

    predict_fn: callable(M_train) -> M_pred (full matrix with predictions filled in)
    holdout_folds: list of (M_train, test_set, [optional_label])
    """
    all_actual = []
    all_pred = []
    fold_metrics = []

    for fold_data in holdout_folds:
        if len(fold_data) == 3:
            M_train, test_set, label = fold_data
        else:
            M_train, test_set = fold_data
            label = ""

        M_pred = predict_fn(M_train)

        actual = [M_FULL[i, j] for i, j in test_set]
        predicted = [M_pred[i, j] for i, j in test_set]

        all_actual.extend(actual)
        all_pred.extend(predicted)

        m = compute_metrics(actual, predicted)
        m['label'] = label
        fold_metrics.append(m)

    overall = compute_metrics(all_actual, all_pred)
    return overall, fold_metrics


def run_full_evaluation(predict_fn, method_name, verbose=True):
    """Run a method through all holdout strategies. Returns summary dict."""
    results = {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"  EVALUATING: {method_name}")
        print(f"{'='*80}")

    # A) Random cell holdout
    folds = holdout_random_cells(frac=0.2, n_folds=5, seed=42)
    overall, fold_m = evaluate_method(predict_fn, folds)
    results['random'] = overall
    medapes = [f['medape'] for f in fold_m]
    if verbose:
        print(f"\n  [A] Random cell (5-fold): MedAPE = {np.mean(medapes):.1f}% ± {np.std(medapes):.1f}%")
        print_metrics(overall, "    Overall")

    # B) Per-model leave-50%-out
    for k_frac, label in [(0.5, "50%"), (0.2, "20%")]:
        folds = holdout_per_model(k_frac=k_frac, min_scores=8, n_folds=3, seed=42)
        overall, fold_m = evaluate_method(predict_fn, folds)
        results[f'per_model_{label}'] = overall
        if verbose:
            print_metrics(overall, f"  [B] Per-model hide-{label}")

    # C) Leave-one-benchmark-out
    folds = holdout_leave_one_benchmark(min_scores=5)
    overall, fold_m = evaluate_method(predict_fn, folds)
    results['leave_bench'] = overall
    if verbose:
        print_metrics(overall, "  [C] Leave-one-benchmark")

    # D) Cold-start (3 benchmarks)
    folds = holdout_cold_start(n_reveal=3, min_scores=8, seed=42)
    overall, fold_m = evaluate_method(predict_fn, folds)
    results['cold_start'] = overall
    if verbose:
        print_metrics(overall, "  [D] Cold-start (3 known)")

    # E) Stratified difficulty
    folds = holdout_stratified_difficulty(frac=0.2, seed=42)
    for overall_fold, fold_m_i, label in [(folds[0][:2], None, "easy"), (folds[1][:2], None, "hard")]:
        o, _ = evaluate_method(predict_fn, [(overall_fold[0] if isinstance(overall_fold, tuple) else overall_fold, overall_fold[1] if isinstance(overall_fold, tuple) else [])])
    # Actually, stratified returns tuples of 3
    strat_results = {}
    for M_train, test_set, label in folds:
        M_pred = predict_fn(M_train)
        actual = [M_FULL[i, j] for i, j in test_set]
        predicted = [M_pred[i, j] for i, j in test_set]
        m = compute_metrics(actual, predicted)
        strat_results[label] = m
        if verbose:
            print_metrics(m, f"  [E] Stratified ({label})")
    results['stratified'] = strat_results

    # F) Leave-one-provider-out (summarize top-5 largest providers)
    folds = holdout_leave_one_provider()
    prov_results = {}
    for M_train, test_set, prov in folds:
        M_pred = predict_fn(M_train)
        actual = [M_FULL[i, j] for i, j in test_set]
        predicted = [M_pred[i, j] for i, j in test_set]
        m = compute_metrics(actual, predicted)
        prov_results[prov] = m
    results['leave_provider'] = prov_results
    if verbose:
        # Show top 5 by number of test entries
        top_provs = sorted(prov_results.items(), key=lambda x: -x[1]['n'])[:5]
        for prov, m in top_provs:
            print_metrics(m, f"  [F] Leave-{prov}-out")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY: Column normalization
# ══════════════════════════════════════════════════════════════════════════════

def col_stats(M):
    """Column means and stds ignoring NaN."""
    col_mean = np.nanmean(M, axis=0)
    col_std  = np.nanstd(M, axis=0)
    col_std[col_std < 1e-8] = 1.0
    return col_mean, col_std

def col_normalize(M):
    """Z-score normalize columns."""
    cm, cs = col_stats(M)
    M_norm = (M - cm) / cs
    M_norm[np.isnan(M)] = np.nan
    return M_norm, cm, cs

def col_denormalize(M_norm, cm, cs):
    return M_norm * cs + cm


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nEvaluation harness loaded successfully.")
    print(f"Models: {N_MODELS}, Benchmarks: {N_BENCH}, Observed: {OBSERVED.sum()}")

    # Quick test with benchmark mean baseline
    def benchmark_mean_predict(M_train):
        M_pred = M_train.copy()
        col_mean = np.nanmean(M_train, axis=0)
        for i in range(N_MODELS):
            for j in range(N_BENCH):
                if np.isnan(M_pred[i, j]):
                    M_pred[i, j] = col_mean[j]
        return M_pred

    print("\n--- Quick test: Benchmark Mean baseline ---")
    folds = holdout_random_cells(frac=0.2, n_folds=3, seed=42)
    overall, fold_m = evaluate_method(benchmark_mean_predict, folds)
    print_metrics(overall, "B0 Benchmark Mean")
