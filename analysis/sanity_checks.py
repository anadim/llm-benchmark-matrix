#!/usr/bin/env python3
"""
Sanity checks for SVD / latent factor analysis in LLM benchmark matrix completion.
Questions 6, 7, 8, 9, 11.
"""

import numpy as np
import sys, warnings, os
from collections import defaultdict

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_PROVIDERS, MODEL_REASONING,
    col_normalize, col_denormalize, col_stats,
    compute_metrics, print_metrics, evaluate_method,
    holdout_per_model, holdout_random_cells,
)
from all_methods import predict_svd, predict_benchreg, predict_blend, predict_B0, predict_B2

# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION 6: SVD CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

def question_6():
    print("\n" + "=" * 80)
    print("  QUESTION 6: SVD (SOFT-IMPUTE) CONVERGENCE — RANK 3")
    print("=" * 80)

    obs = ~np.isnan(M_FULL)
    cm, cs = col_stats(M_FULL)

    # Confirm normalization: z-score per benchmark column
    print("\n  NORMALIZATION CONFIRMATION:")
    print(f"  Method: z-score per benchmark column (subtract col mean, divide by col std)")
    print(f"  Number of benchmark columns: {N_BENCH}")
    print(f"  Sample column means (first 5): {cm[:5]}")
    print(f"  Sample column stds  (first 5): {cs[:5]}")

    # z-score normalize
    M_norm = (M_FULL - cm) / cs
    M_norm[np.isnan(M_FULL)] = np.nan

    # Verify that after normalization, columns have mean~0, std~1
    for j in range(min(5, N_BENCH)):
        col = M_norm[:, j]
        valid = col[~np.isnan(col)]
        print(f"    Col {j} ({BENCH_NAMES[BENCH_IDS[j]][:20]}): mean={np.mean(valid):.6f}, std={np.std(valid):.6f}")

    # Run Soft-Impute rank-3 with convergence trace
    print(f"\n  CONVERGENCE TRACE (Soft-Impute, rank=3, tol=1e-4):")
    M_imp = M_norm.copy()
    M_imp[np.isnan(M_imp)] = 0

    rank = 3
    max_iter = 100
    tol = 1e-4
    diffs = []

    print(f"  {'Iter':>5s}  {'Diff':>12s}  {'Converged?':>10s}")
    for it in range(max_iter):
        M_old = M_imp.copy()
        U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)
        U_r = U[:, :rank]
        s_r = s[:rank]
        Vt_r = Vt[:rank, :]
        M_approx = U_r @ np.diag(s_r) @ Vt_r
        M_imp = np.where(obs, M_norm, M_approx)
        M_imp[np.isnan(M_imp)] = 0
        diff = np.sqrt(np.mean((M_imp - M_old) ** 2))
        diffs.append(diff)
        converged = diff < tol
        if it < 20 or it % 10 == 0 or converged:
            print(f"  {it + 1:>5d}  {diff:>12.6f}  {'YES' if converged else ''}")
        if converged:
            print(f"\n  CONVERGED at iteration {it + 1} (diff={diff:.2e} < tol={tol:.0e})")
            break
    else:
        print(f"\n  DID NOT CONVERGE after {max_iter} iterations (final diff={diffs[-1]:.2e})")

    print(f"\n  Total iterations: {len(diffs)}")
    print(f"  Initial diff: {diffs[0]:.6f}")
    print(f"  Final diff:   {diffs[-1]:.6f}")
    print(f"  Ratio first/last: {diffs[0] / diffs[-1]:.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION 7: RANK 2 vs 3 (and 1, 4, 5)
# ══════════════════════════════════════════════════════════════════════════════

def question_7():
    print("\n" + "=" * 80)
    print("  QUESTION 7: RANK COMPARISON (1, 2, 3, 4, 5) — Per-model 50% holdout, 3-fold")
    print("=" * 80)

    ranks = [1, 2, 3, 4, 5]
    results = {}

    print(f"\n  {'Rank':>4s}  {'MedAPE':>8s}  {'MeanAPE':>9s}  {'RMSE':>8s}  {'R2':>7s}  {'<5%':>5s}  {'<10%':>6s}  {'n':>6s}")
    print(f"  {'----':>4s}  {'------':>8s}  {'-------':>9s}  {'----':>8s}  {'--':>7s}  {'---':>5s}  {'----':>6s}  {'---':>6s}")

    for r in ranks:
        predict_fn = lambda M, rank=r: predict_svd(M, rank=rank)
        folds = holdout_per_model(k_frac=0.5, min_scores=8, n_folds=3, seed=42)
        overall, fold_m = evaluate_method(predict_fn, folds)
        results[r] = overall
        print(f"  {r:>4d}  {overall['medape']:>7.1f}%  {overall['meanape']:>8.1f}%  "
              f"{overall['rmse']:>8.1f}  {overall['r2']:>7.3f}  {overall['pct5']:>4.0f}%  "
              f"{overall['pct10']:>5.0f}%  {overall['n']:>6d}")

    # Analysis
    print(f"\n  ANALYSIS:")
    for r in [2, 3, 4, 5]:
        diff = results[r]['medape'] - results[r - 1]['medape']
        better = "BETTER" if diff < 0 else "WORSE" if diff > 0 else "SAME"
        print(f"    Rank {r} vs Rank {r - 1}: MedAPE change = {diff:+.1f}pp ({better})")

    print(f"\n  Does rank 2 beat rank 1? {'YES' if results[2]['medape'] < results[1]['medape'] else 'NO'} "
          f"(MedAPE: {results[1]['medape']:.1f}% -> {results[2]['medape']:.1f}%)")
    print(f"  Does rank 3 beat rank 2? {'YES' if results[3]['medape'] < results[2]['medape'] else 'NO'} "
          f"(MedAPE: {results[2]['medape']:.1f}% -> {results[3]['medape']:.1f}%)")
    print(f"\n  Even though factor 3 explains only 5.7% of variance,")
    if results[3]['medape'] < results[2]['medape']:
        print(f"  it DOES add real predictive signal in holdout evaluation (rank 3 < rank 2 MedAPE).")
    else:
        print(f"  it does NOT add predictive signal in holdout (rank 3 >= rank 2 MedAPE).")
        print(f"  This suggests factor 3's 5.7% variance is noise or overfitting in this holdout regime.")


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION 8: NEGATIVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def question_8():
    print("\n" + "=" * 80)
    print("  QUESTION 8: NEGATIVE PREDICTIONS CHECK")
    print("=" * 80)

    # --- SVD rank-3 on full matrix ---
    print("\n  A) SVD rank-3 on full matrix:")
    M_svd = predict_svd(M_FULL, rank=3)
    obs = ~np.isnan(M_FULL)

    # Only look at cells that were missing (i.e., predicted cells)
    missing_mask = np.isnan(M_FULL)
    predicted_cells = M_svd[missing_mask]
    n_predicted = len(predicted_cells)
    n_negative = np.sum(predicted_cells < 0)
    frac_negative = n_negative / n_predicted if n_predicted > 0 else 0

    print(f"  Total predicted (missing) cells: {n_predicted}")
    print(f"  Negative predictions: {n_negative}")
    print(f"  Fraction negative: {frac_negative * 100:.2f}%")

    # Show examples of negative predictions
    if n_negative > 0:
        print(f"\n  Examples of negative SVD predictions:")
        print(f"  {'Model':<35s}  {'Benchmark':<25s}  {'Predicted':>10s}")
        neg_examples = []
        for i in range(N_MODELS):
            for j in range(N_BENCH):
                if np.isnan(M_FULL[i, j]) and M_svd[i, j] < 0:
                    neg_examples.append((i, j, M_svd[i, j]))
        # Sort by most negative
        neg_examples.sort(key=lambda x: x[2])
        for i, j, val in neg_examples[:20]:
            print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s}  {BENCH_NAMES[BENCH_IDS[j]]:<25s}  {val:>10.1f}")

    # Also check: do observed cells get changed to negative? (they shouldn't)
    obs_neg = np.sum(M_svd[obs] < 0)
    print(f"\n  Observed cells that became negative (should be 0): {obs_neg}")

    # --- BenchReg ---
    print(f"\n  B) BenchReg on full matrix:")
    M_breg = predict_benchreg(M_FULL)
    predicted_breg = M_breg[missing_mask]
    n_pred_breg = np.sum(~np.isnan(predicted_breg))
    n_neg_breg = np.sum(predicted_breg < 0)
    frac_neg_breg = n_neg_breg / n_pred_breg if n_pred_breg > 0 else 0
    print(f"  Total predicted cells: {n_pred_breg}")
    print(f"  Negative predictions: {n_neg_breg}")
    print(f"  Fraction negative: {frac_neg_breg * 100:.2f}%")

    if n_neg_breg > 0:
        print(f"\n  Examples of negative BenchReg predictions:")
        print(f"  {'Model':<35s}  {'Benchmark':<25s}  {'Predicted':>10s}")
        neg_br = []
        for i in range(N_MODELS):
            for j in range(N_BENCH):
                if np.isnan(M_FULL[i, j]) and not np.isnan(M_breg[i, j]) and M_breg[i, j] < 0:
                    neg_br.append((i, j, M_breg[i, j]))
        neg_br.sort(key=lambda x: x[2])
        for i, j, val in neg_br[:15]:
            print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s}  {BENCH_NAMES[BENCH_IDS[j]]:<25s}  {val:>10.1f}")

    # --- Blend ---
    print(f"\n  C) Blend (BenchReg + KNN, alpha=0.6) on full matrix:")
    M_bl = predict_blend(M_FULL)
    predicted_bl = M_bl[missing_mask]
    n_pred_bl = np.sum(~np.isnan(predicted_bl))
    n_neg_bl = np.sum(predicted_bl < 0)
    frac_neg_bl = n_neg_bl / n_pred_bl if n_pred_bl > 0 else 0
    print(f"  Total predicted cells: {n_pred_bl}")
    print(f"  Negative predictions: {n_neg_bl}")
    print(f"  Fraction negative: {frac_neg_bl * 100:.2f}%")

    if n_neg_bl > 0:
        print(f"\n  Examples of negative Blend predictions:")
        print(f"  {'Model':<35s}  {'Benchmark':<25s}  {'Predicted':>10s}")
        neg_bl = []
        for i in range(N_MODELS):
            for j in range(N_BENCH):
                if np.isnan(M_FULL[i, j]) and not np.isnan(M_bl[i, j]) and M_bl[i, j] < 0:
                    neg_bl.append((i, j, M_bl[i, j]))
        neg_bl.sort(key=lambda x: x[2])
        for i, j, val in neg_bl[:15]:
            print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s}  {BENCH_NAMES[BENCH_IDS[j]]:<25s}  {val:>10.1f}")

    print(f"\n  SUMMARY:")
    print(f"  {'Method':<20s}  {'Neg preds':>10s}  {'Total preds':>12s}  {'% Negative':>11s}")
    print(f"  {'SVD rank-3':<20s}  {n_negative:>10d}  {n_predicted:>12d}  {frac_negative * 100:>10.2f}%")
    print(f"  {'BenchReg':<20s}  {n_neg_breg:>10d}  {n_pred_breg:>12d}  {frac_neg_breg * 100:>10.2f}%")
    print(f"  {'Blend':<20s}  {n_neg_bl:>10d}  {n_pred_bl:>12d}  {frac_neg_bl * 100:>10.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION 9: DATA EFFICIENCY NON-MONOTONICITY
# ══════════════════════════════════════════════════════════════════════════════

def predict_B0_local(M_train):
    """Local version of benchmark mean."""
    M_pred = M_train.copy()
    col_mean = np.nanmean(M_train, axis=0)
    for j in range(N_BENCH):
        mask = np.isnan(M_pred[:, j])
        M_pred[mask, j] = col_mean[j] if not np.isnan(col_mean[j]) else 0
    return M_pred


def question_9():
    print("\n" + "=" * 80)
    print("  QUESTION 9: DATA EFFICIENCY NON-MONOTONICITY CHECK")
    print("=" * 80)

    # First, check the original code to understand the seed issue
    print("\n  ORIGINAL CODE ANALYSIS:")
    print("  The original analyze_data_efficiency() uses a SINGLE rng = RandomState(42)")
    print("  that is shared across all fill rates. Each fill rate does rng.shuffle(obs_cells),")
    print("  which mutates the rng state. So the different fill rates use DIFFERENT random")
    print("  orderings (state changes with each shuffle). Additionally, the test set rng2")
    print("  is reset with seed=123 each time, but the TRAINING set changes with the rng state.")
    print("  This means: (a) different random masks at each fill rate, and")
    print("  (b) the sequential dependency means results are path-dependent on fill_rate order.")
    print("  This explains the non-monotonicity as NOISE from different random subsets.")

    obs_cells_master = list(zip(*np.where(OBSERVED)))
    fill_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.34]
    seeds = [42, 123, 456, 789, 1337]
    n_seeds = len(seeds)

    methods = {
        'BenchMean': predict_B0_local,
        'KNN': lambda M: predict_B2(M, k=5),
        'BenchReg': lambda M: predict_benchreg(M),
        'SVD3': lambda M: predict_svd(M, rank=3),
        'Blend': lambda M: predict_blend(M, 0.6),
    }

    # results[method][fill_rate] = list of MedAPEs across seeds
    results_all = {m: {fr: [] for fr in fill_rates} for m in methods}

    print(f"\n  Running {n_seeds} seeds x {len(fill_rates)} fill rates x {len(methods)} methods...")
    print(f"  (This may take a minute...)")

    for seed in seeds:
        for target_fill in fill_rates:
            rng = np.random.RandomState(seed)
            obs_cells = list(obs_cells_master)  # fresh copy each time
            rng.shuffle(obs_cells)
            target_n = int(target_fill * N_MODELS * N_BENCH)
            keep = set(map(tuple, obs_cells[:min(target_n, len(obs_cells))]))

            M_reduced = np.full_like(M_FULL, np.nan)
            for i, j in keep:
                M_reduced[i, j] = M_FULL[i, j]

            actual_fill = (~np.isnan(M_reduced)).sum() / (N_MODELS * N_BENCH)

            # Hold out 20% for testing
            remaining_obs = list(keep)
            rng2 = np.random.RandomState(seed + 1000)  # different but deterministic
            rng2.shuffle(remaining_obs)
            n_test = int(len(remaining_obs) * 0.2)
            test_set = remaining_obs[:n_test]

            M_train = M_reduced.copy()
            for i, j in test_set:
                M_train[i, j] = np.nan

            actual_vals = [M_FULL[i, j] for i, j in test_set]

            for mname, mfn in methods.items():
                M_pred = mfn(M_train)
                predicted = [M_pred[i, j] for i, j in test_set]
                m = compute_metrics(actual_vals, predicted)
                results_all[mname][target_fill].append(m['medape'])

    # Print results: mean +/- std
    print(f"\n  DATA EFFICIENCY TABLE (mean +/- std over {n_seeds} seeds):")
    header = f"  {'FillRate':>8s}"
    for mname in methods:
        header += f"  {mname:>16s}"
    print(header)
    print(f"  {'--------':>8s}" + f"  {'----------------':>16s}" * len(methods))

    for fr in fill_rates:
        row = f"  {fr * 100:>7.0f}%"
        for mname in methods:
            vals = results_all[mname][fr]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            row += f"  {mean_v:>6.1f} +/- {std_v:>4.1f}"
        print(row)

    # Check for remaining non-monotonicity in the averaged results
    print(f"\n  MONOTONICITY CHECK (should decrease with more data):")
    for mname in methods:
        means = [np.mean(results_all[mname][fr]) for fr in fill_rates]
        monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
        violations = []
        for i in range(len(fill_rates) - 1):
            if means[i] < means[i + 1]:
                violations.append(f"{fill_rates[i]*100:.0f}%->{fill_rates[i+1]*100:.0f}% (+{means[i+1]-means[i]:.1f}pp)")
        if monotonic:
            print(f"    {mname:<12s}: MONOTONIC (error decreases with more data)")
        else:
            print(f"    {mname:<12s}: NON-MONOTONIC — violations: {', '.join(violations)}")
            print(f"      (but note std dev — violations within noise band suggest this is sampling noise)")


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION 11: CORRELATION SAMPLE SIZES
# ══════════════════════════════════════════════════════════════════════════════

def question_11():
    print("\n" + "=" * 80)
    print("  QUESTION 11: CORRELATION PAIR SAMPLE SIZES")
    print("=" * 80)

    # Compute pairwise correlations (same as in redundancy analysis)
    corr_pairs = []
    for j1 in range(N_BENCH):
        for j2 in range(j1 + 1, N_BENCH):
            shared = OBSERVED[:, j1] & OBSERVED[:, j2]
            n_shared = shared.sum()
            if n_shared < 5:
                r = 0
            else:
                r = np.corrcoef(M_FULL[shared, j1], M_FULL[shared, j2])[0, 1]
                if np.isnan(r):
                    r = 0
            corr_pairs.append((r, j1, j2, n_shared))

    # Sort by correlation (descending)
    corr_pairs.sort(key=lambda x: -x[0])

    # Top 15
    print(f"\n  Top 15 most correlated benchmark pairs:")
    print(f"  {'#':>3s}  {'Benchmark A':<30s}  {'Benchmark B':<30s}  {'Corr':>6s}  {'N_shared':>8s}  {'Flag':>10s}")
    print(f"  {'---':>3s}  {'-----':>30s}  {'-----':>30s}  {'----':>6s}  {'--------':>8s}  {'----':>10s}")

    flagged = []
    for idx, (r, j1, j2, n_shared) in enumerate(corr_pairs[:15]):
        flag = "*** < 10 ***" if n_shared < 10 else ""
        if n_shared < 10:
            flagged.append((BENCH_NAMES[BENCH_IDS[j1]], BENCH_NAMES[BENCH_IDS[j2]], n_shared, r))
        print(f"  {idx + 1:>3d}  {BENCH_NAMES[BENCH_IDS[j1]]:<30s}  {BENCH_NAMES[BENCH_IDS[j2]]:<30s}  {r:>6.3f}  {n_shared:>8d}  {flag}")

    if flagged:
        print(f"\n  WARNING: {len(flagged)} pairs have fewer than 10 shared models!")
        print(f"  These correlations are UNRELIABLE and should be treated with caution:")
        for bA, bB, n, r in flagged:
            print(f"    - {bA} vs {bB}: only {n} shared models (r={r:.3f})")
            print(f"      With n={n}, 95% CI for r is approximately +/- {1.96 / np.sqrt(n - 3):.2f}" if n > 3 else f"      Cannot compute CI with n={n}")
    else:
        print(f"\n  All top-15 pairs have >= 10 shared models. No flags.")

    # Also show distribution of sample sizes across all pairs
    all_n = [x[3] for x in corr_pairs]
    print(f"\n  Sample size distribution across ALL {len(corr_pairs)} benchmark pairs:")
    print(f"    Min: {min(all_n)}, Max: {max(all_n)}, Median: {np.median(all_n):.0f}, Mean: {np.mean(all_n):.1f}")
    print(f"    Pairs with n < 5:  {sum(1 for n in all_n if n < 5)}")
    print(f"    Pairs with n < 10: {sum(1 for n in all_n if n < 10)}")
    print(f"    Pairs with n < 20: {sum(1 for n in all_n if n < 20)}")
    print(f"    Pairs with n >= 20: {sum(1 for n in all_n if n >= 20)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    question_6()
    question_7()
    question_8()
    question_9()
    question_11()
    print("\n" + "=" * 80)
    print("  ALL SANITY CHECKS COMPLETE")
    print("=" * 80)
