#!/usr/bin/env python3
"""
Diagnostics for LLM Benchmark Matrix Completion
================================================
Runs: intrinsic dimensionality, benchmark redundancy, minimum eval set,
data efficiency, scaling laws, provider effects, reasoning mode analysis.
"""

import numpy as np
import sys, warnings, json, os
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_PROVIDERS, MODEL_REASONING,
    MODEL_OPEN, MODEL_PARAMS, BENCH_CATS,
    col_normalize, col_stats, compute_metrics,
    holdout_random_cells, holdout_per_model, evaluate_method,
)
from all_methods import predict_benchreg, predict_B2, predict_svd, predict_blend


# ══════════════════════════════════════════════════════════════════════════════
#  3.4: INTRINSIC DIMENSIONALITY
# ══════════════════════════════════════════════════════════════════════════════

def analyze_intrinsic_dimensionality():
    print("\n" + "="*80)
    print("  3.4: INTRINSIC DIMENSIONALITY")
    print("="*80)

    # Impute missing with column means, then normalize
    cm, cs = col_stats(M_FULL)
    M_imp = M_FULL.copy()
    for j in range(N_BENCH):
        M_imp[np.isnan(M_imp[:, j]), j] = cm[j]
    M_norm = (M_imp - cm) / cs

    U, s, Vt = np.linalg.svd(M_norm, full_matrices=False)

    # Variance explained
    var_explained = s**2 / np.sum(s**2)
    cum_var = np.cumsum(var_explained)

    print("\n  Singular values and cumulative variance explained:")
    print(f"  {'Rank':>4s}  {'SingVal':>8s}  {'VarExpl':>8s}  {'CumVar':>8s}  {'Bar'}")
    for i in range(min(15, len(s))):
        bar = '█' * int(var_explained[i] * 100)
        print(f"  {i+1:>4d}  {s[i]:>8.1f}  {var_explained[i]*100:>7.1f}%  {cum_var[i]*100:>7.1f}%  {bar}")

    # Find elbow
    for target in [0.8, 0.9, 0.95, 0.99]:
        rank = np.searchsorted(cum_var, target) + 1
        print(f"  Rank for {target*100:.0f}% variance: {rank}")

    # Repeat excluding agentic benchmarks
    agentic_mask = BENCH_CATS != 'Agentic'
    M_noagent = M_norm[:, agentic_mask]
    U2, s2, Vt2 = np.linalg.svd(M_noagent, full_matrices=False)
    var2 = s2**2 / np.sum(s2**2)
    cum2 = np.cumsum(var2)
    print(f"\n  Without agentic benchmarks ({agentic_mask.sum()} remaining):")
    for target in [0.8, 0.9, 0.95]:
        rank = np.searchsorted(cum2, target) + 1
        print(f"  Rank for {target*100:.0f}% variance: {rank}")

    # Repeat excluding competition math (bimodal)
    comp_ids = {'imo_2025', 'usamo_2025', 'matharena_apex_2025', 'arc_agi_1', 'arc_agi_2'}
    noncomp_mask = np.array([b not in comp_ids for b in BENCH_IDS])
    M_nocomp = M_norm[:, noncomp_mask]
    U3, s3, Vt3 = np.linalg.svd(M_nocomp, full_matrices=False)
    var3 = s3**2 / np.sum(s3**2)
    cum3 = np.cumsum(var3)
    print(f"\n  Without bimodal benchmarks ({noncomp_mask.sum()} remaining):")
    for target in [0.8, 0.9, 0.95]:
        rank = np.searchsorted(cum3, target) + 1
        print(f"  Rank for {target*100:.0f}% variance: {rank}")

    # Interpret top factors
    print(f"\n  Top 5 latent factors — benchmark loadings (from V^T):")
    for f in range(min(5, Vt.shape[0])):
        loadings = Vt[f, :]
        top_pos = np.argsort(-loadings)[:5]
        top_neg = np.argsort(loadings)[:3]
        pos_str = ", ".join(f"{BENCH_NAMES[BENCH_IDS[j]]}({loadings[j]:+.2f})" for j in top_pos)
        neg_str = ", ".join(f"{BENCH_NAMES[BENCH_IDS[j]]}({loadings[j]:+.2f})" for j in top_neg)
        print(f"\n  Factor {f+1} ({var_explained[f]*100:.1f}% var):")
        print(f"    Positive: {pos_str}")
        print(f"    Negative: {neg_str}")

    return s, var_explained, cum_var, Vt


# ══════════════════════════════════════════════════════════════════════════════
#  3.2: BENCHMARK REDUNDANCY (correlation clustering)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_benchmark_redundancy():
    print("\n" + "="*80)
    print("  3.2: BENCHMARK REDUNDANCY")
    print("="*80)

    # Compute pairwise correlations between benchmarks
    corr = np.full((N_BENCH, N_BENCH), np.nan)
    n_shared = np.zeros((N_BENCH, N_BENCH), dtype=int)
    for j1 in range(N_BENCH):
        for j2 in range(j1, N_BENCH):
            shared = OBSERVED[:, j1] & OBSERVED[:, j2]
            n_shared[j1, j2] = n_shared[j2, j1] = int(shared.sum())
            if shared.sum() < 5:
                corr[j1, j2] = corr[j2, j1] = 0
                continue
            r = np.corrcoef(M_FULL[shared, j1], M_FULL[shared, j2])[0, 1]
            if np.isnan(r):
                r = 0
            corr[j1, j2] = corr[j2, j1] = r

    # Most correlated pairs (require n_shared >= 20 for reliable claims)
    pairs = []
    pairs_all = []
    for j1 in range(N_BENCH):
        for j2 in range(j1+1, N_BENCH):
            pairs_all.append((corr[j1, j2], j1, j2, n_shared[j1, j2]))
            if n_shared[j1, j2] >= 20:
                pairs.append((corr[j1, j2], j1, j2, n_shared[j1, j2]))
    pairs.sort(key=lambda x: -x[0])
    pairs_all.sort(key=lambda x: -x[0])

    # Report sample size stats
    all_n = [ns for _, _, _, ns in pairs_all]
    print(f"\n  Pairwise sample sizes: median={np.median(all_n):.0f}, "
          f"<10: {sum(1 for n in all_n if n<10)}/{len(all_n)} ({100*sum(1 for n in all_n if n<10)/len(all_n):.0f}%), "
          f">=20: {sum(1 for n in all_n if n>=20)}/{len(all_n)} ({100*sum(1 for n in all_n if n>=20)/len(all_n):.0f}%)")

    print("\n  Most correlated benchmark pairs (n_shared >= 20 for reliability):")
    print(f"  {'Benchmark A':<30s}  {'Benchmark B':<30s}  {'Corr':>6s}  {'n':>4s}")
    for r, j1, j2, ns in pairs[:15]:
        print(f"  {BENCH_NAMES[BENCH_IDS[j1]]:<30s}  {BENCH_NAMES[BENCH_IDS[j2]]:<30s}  {r:>6.3f}  {ns:>4d}")

    print("\n  Least correlated / anti-correlated pairs:")
    for r, j1, j2, ns in pairs[-10:]:
        print(f"  {BENCH_NAMES[BENCH_IDS[j1]]:<30s}  {BENCH_NAMES[BENCH_IDS[j2]]:<30s}  {r:>6.3f}  {ns:>4d}")

    # Hierarchical clustering
    # Convert correlation to distance
    dist = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=0.4, criterion='distance')  # correlation > 0.6 within cluster

    print(f"\n  Hierarchical clustering (corr > 0.6 within cluster):")
    cluster_map = defaultdict(list)
    for j, c in enumerate(clusters):
        cluster_map[c].append(j)
    for c in sorted(cluster_map.keys(), key=lambda c: -len(cluster_map[c])):
        members = cluster_map[c]
        if len(members) >= 2:
            names = [BENCH_NAMES[BENCH_IDS[j]] for j in members]
            cats = [BENCH_CATS[j] for j in members]
            print(f"    Cluster {c} ({len(members)} benchmarks): {', '.join(names)}")
            print(f"      Categories: {', '.join(set(cats))}")

    return corr, clusters


# ══════════════════════════════════════════════════════════════════════════════
#  3.3: MINIMUM EVAL SET (which benchmarks are most informative?)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_minimum_eval_set():
    print("\n" + "="*80)
    print("  3.3: MINIMUM EVAL SET — Which 5 benchmarks predict the rest best?")
    print("="*80)

    # Greedy forward selection: pick the benchmark that most reduces prediction error
    # Use a simplified approach: for each candidate set, do SVD completion on just those benchmarks
    # and measure how well we predict the rest.

    # First, measure information value of each benchmark individually
    # (how much does removing it increase error?)
    print("\n  Individual benchmark information value (drop-one analysis):")

    base_folds = holdout_random_cells(frac=0.2, n_folds=3, seed=42)
    base_overall, _ = evaluate_method(lambda M: predict_blend(M, 0.6), base_folds)
    base_medape = base_overall['medape']
    print(f"  Baseline MedAPE (all benchmarks): {base_medape:.1f}%")

    # For efficiency, use a simple approach: greedy selection based on coverage × correlation
    # Pick benchmarks that (a) many models have, (b) correlate with many other benchmarks

    bench_info = []
    for j in range(N_BENCH):
        coverage = OBSERVED[:, j].sum()
        # Average absolute correlation with other benchmarks
        avg_corr = 0
        n_corr = 0
        for j2 in range(N_BENCH):
            if j2 == j:
                continue
            shared = OBSERVED[:, j] & OBSERVED[:, j2]
            if shared.sum() < 5:
                continue
            r = np.corrcoef(M_FULL[shared, j], M_FULL[shared, j2])[0, 1]
            if not np.isnan(r):
                avg_corr += abs(r)
                n_corr += 1
        avg_corr = avg_corr / max(n_corr, 1)
        # Information value = coverage × avg_correlation
        info = coverage * avg_corr
        bench_info.append((j, BENCH_NAMES[BENCH_IDS[j]], coverage, avg_corr, info))

    bench_info.sort(key=lambda x: -x[4])

    print(f"\n  {'Rank':>4s}  {'Benchmark':<35s}  {'Coverage':>8s}  {'AvgCorr':>8s}  {'InfoValue':>9s}")
    for rank, (j, name, cov, corr, info) in enumerate(bench_info[:20]):
        print(f"  {rank+1:>4d}  {name:<35s}  {cov:>8d}  {corr:>8.3f}  {info:>9.1f}")

    # Greedy forward selection using correlation-based prediction
    # For each target benchmark, predict from selected set via ridge regression on models
    # that have both the selected benchmarks and the target
    print(f"\n  Greedy forward selection (best subsets):")
    selected = []
    remaining = list(range(N_BENCH))

    def predict_from_subset(selected_j):
        """Predict all benchmarks from a subset using per-target ridge regression."""
        from sklearn.linear_model import Ridge
        M_pred = np.full_like(M_FULL, np.nan)
        # Copy known benchmarks
        for jj in selected_j:
            M_pred[:, jj] = M_FULL[:, jj]

        for j_target in range(N_BENCH):
            if j_target in selected_j:
                continue
            # Find models that have both all selected benchmarks AND target
            has_selected = np.ones(N_MODELS, dtype=bool)
            for jj in selected_j:
                has_selected &= OBSERVED[:, jj]
            has_target = OBSERVED[:, j_target]
            train_mask = has_selected & has_target

            if train_mask.sum() < 3:
                # Fall back to benchmark mean
                M_pred[:, j_target] = np.nanmean(M_FULL[:, j_target])
                continue

            X_train = M_FULL[train_mask][:, selected_j]
            y_train = M_FULL[train_mask, j_target]

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)

            # Predict for all models that have the selected benchmarks
            for i in range(N_MODELS):
                if has_selected[i]:
                    x = M_FULL[i, selected_j].reshape(1, -1)
                    M_pred[i, j_target] = ridge.predict(x)[0]
                else:
                    M_pred[i, j_target] = np.nanmean(M_FULL[:, j_target])

        return M_pred

    for step in range(10):
        best_j = None
        best_err = np.inf

        for j in remaining:
            candidate = selected + [j]
            M_pred = predict_from_subset(candidate)

            # Measure prediction quality on all OTHER observed entries
            actuals, preds = [], []
            for i in range(N_MODELS):
                for j2 in range(N_BENCH):
                    if j2 in candidate:
                        continue
                    if OBSERVED[i, j2] and not np.isnan(M_pred[i, j2]):
                        if abs(M_FULL[i, j2]) > 1e-6:
                            actuals.append(M_FULL[i, j2])
                            preds.append(M_pred[i, j2])
            if actuals:
                ape = np.abs((np.array(preds) - np.array(actuals)) / np.array(actuals))
                err = np.median(ape)
            else:
                err = np.inf

            if err < best_err:
                best_err = err
                best_j = j

        if best_j is not None:
            selected.append(best_j)
            remaining.remove(best_j)
            names = [BENCH_NAMES[BENCH_IDS[j]] for j in selected]
            print(f"    Step {step+1}: +{BENCH_NAMES[BENCH_IDS[best_j]]:<30s}  "
                  f"In-sample MedAPE={best_err*100:.1f}%  Set: {', '.join(names)}")

    # ── Proper holdout evaluation of top-5 set ──
    # The in-sample MedAPE above is OPTIMISTIC because ridge trains and evaluates
    # on the same observed entries. Compute proper holdout here.
    from sklearn.linear_model import Ridge as _Ridge
    top5 = selected[:5]
    print(f"\n  Proper holdout evaluation of top-5 set: {[BENCH_NAMES[BENCH_IDS[j]] for j in top5]}")

    def predict_5bench_ridge(M_train, sel_idx=top5):
        """Predict all benchmarks from 5-benchmark subset using per-target ridge."""
        obs_t = ~np.isnan(M_train)
        M_pred = M_train.copy()
        for j_target in range(N_BENCH):
            if j_target in sel_idx:
                continue
            has_selected = np.ones(N_MODELS, dtype=bool)
            for jj in sel_idx:
                has_selected &= obs_t[:, jj]
            has_target = obs_t[:, j_target]
            train_mask = has_selected & has_target
            if train_mask.sum() < 3:
                M_pred[:, j_target] = np.where(
                    np.isnan(M_pred[:, j_target]),
                    np.nanmean(M_train[:, j_target]),
                    M_pred[:, j_target])
                continue
            X_train = M_train[train_mask][:, sel_idx]
            y_train = M_train[train_mask, j_target]
            ridge = _Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            for i in range(N_MODELS):
                if np.isnan(M_train[i, j_target]) and has_selected[i]:
                    M_pred[i, j_target] = ridge.predict(M_train[i, sel_idx].reshape(1, -1))[0]
        return M_pred

    folds_r = holdout_random_cells(frac=0.2, n_folds=3, seed=42)
    overall_5r, _ = evaluate_method(predict_5bench_ridge, folds_r)
    overall_br, _ = evaluate_method(lambda M: predict_blend(M, 0.6), folds_r)

    folds_m = holdout_per_model(k_frac=0.5, min_scores=8, n_folds=3, seed=42)
    overall_5m, _ = evaluate_method(predict_5bench_ridge, folds_m)
    overall_bm, _ = evaluate_method(lambda M: predict_blend(M, 0.6), folds_m)

    print(f"    Random 20% holdout:     5-bench ridge = {overall_5r['medape']:.1f}%   vs  BenchReg+KNN blend = {overall_br['medape']:.1f}%")
    print(f"    Per-model 50% holdout:  5-bench ridge = {overall_5m['medape']:.1f}%   vs  BenchReg+KNN blend = {overall_bm['medape']:.1f}%")
    print(f"    (NOTE: in-sample number above is optimistic; holdout is the reliable metric)")

    return bench_info, selected


# ══════════════════════════════════════════════════════════════════════════════
#  3.1: DATA EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════

def analyze_data_efficiency():
    print("\n" + "="*80)
    print("  3.1: DATA EFFICIENCY — How does accuracy scale with fill rate?")
    print("="*80)

    rng = np.random.RandomState(42)
    obs_cells = list(zip(*np.where(OBSERVED)))

    fill_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.34]
    print(f"\n  {'FillRate':>8s}  {'B0-Mean':>8s}  {'KNN':>8s}  {'BReg':>8s}  {'SVD3':>8s}  {'Blend':>8s}")

    for target_fill in fill_rates:
        target_n = int(target_fill * N_MODELS * N_BENCH)
        # Keep only target_n randomly selected observed cells
        rng.shuffle(obs_cells)
        keep = set(obs_cells[:min(target_n, len(obs_cells))])

        M_reduced = np.full_like(M_FULL, np.nan)
        for i, j in keep:
            M_reduced[i, j] = M_FULL[i, j]

        actual_fill = (~np.isnan(M_reduced)).sum() / (N_MODELS * N_BENCH)

        # Hide 20% of remaining for testing
        remaining_obs = [(i, j) for i, j in keep]
        rng2 = np.random.RandomState(123)
        rng2.shuffle(remaining_obs)
        n_test = int(len(remaining_obs) * 0.2)
        test_set = remaining_obs[:n_test]

        M_train = M_reduced.copy()
        for i, j in test_set:
            M_train[i, j] = np.nan

        methods = {
            'B0-Mean': lambda M: predict_B0_local(M),
            'KNN': lambda M: predict_B2(M, k=5),
            'BReg': lambda M: predict_benchreg(M),
            'SVD3': lambda M: predict_svd(M, rank=3),
            'Blend': lambda M: predict_blend(M, 0.6),
        }

        actual = [M_FULL[i, j] for i, j in test_set]
        row_strs = [f"{actual_fill*100:>7.1f}%"]
        for mname, mfn in methods.items():
            M_pred = mfn(M_train)
            predicted = [M_pred[i, j] for i, j in test_set]
            m = compute_metrics(actual, predicted)
            row_strs.append(f"{m['medape']:>7.1f}%")
        print(f"  {'  '.join(row_strs)}")


def predict_B0_local(M_train):
    """Local version of benchmark mean."""
    M_pred = M_train.copy()
    col_mean = np.nanmean(M_train, axis=0)
    for j in range(N_BENCH):
        mask = np.isnan(M_pred[:, j])
        M_pred[mask, j] = col_mean[j] if not np.isnan(col_mean[j]) else 0
    return M_pred


# ══════════════════════════════════════════════════════════════════════════════
#  3.5: SCALING LAWS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_scaling_laws():
    print("\n" + "="*80)
    print("  3.5: SCALING LAWS — Score vs log(params) within model families")
    print("="*80)

    # Group models by family
    families = {
        'Qwen3': [m for m in MODEL_IDS if m.startswith('qwen3-') and not m.startswith('qwen3.5')],
        'DeepSeek-R1-Distill': [m for m in MODEL_IDS if 'distill' in m],
        'GPT-4.1': [m for m in MODEL_IDS if m.startswith('gpt-4.1')],
    }

    for family_name, members in families.items():
        member_idx = [MODEL_IDS.index(m) for m in members if m in MODEL_IDS]
        params = [MODEL_PARAMS[i] for i in member_idx]

        # Filter to models with known params
        valid = [(i, p) for i, p in zip(member_idx, params) if not np.isnan(p)]
        if len(valid) < 3:
            continue

        print(f"\n  Family: {family_name} ({len(valid)} models with known params)")
        models_sorted = sorted(valid, key=lambda x: x[1])

        for j in range(N_BENCH):
            # Need at least 3 data points
            x_log, y = [], []
            for i, p in models_sorted:
                if OBSERVED[i, j]:
                    x_log.append(np.log(p))
                    y.append(M_FULL[i, j])
            if len(x_log) < 3:
                continue
            x_arr = np.array(x_log)
            y_arr = np.array(y)
            # Fit linear: score = a + b * log(params)
            slope = np.sum((x_arr - x_arr.mean()) * (y_arr - y_arr.mean())) / (np.sum((x_arr - x_arr.mean())**2) + 1e-10)
            intercept = y_arr.mean() - slope * x_arr.mean()
            y_hat = slope * x_arr + intercept
            ss_res = np.sum((y_arr - y_hat)**2)
            ss_tot = np.sum((y_arr - y_arr.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

            if abs(r2) > 0.7:
                model_strs = [f"{MODEL_NAMES[MODEL_IDS[i]]}({M_FULL[i,j]:.0f})" for i, p in models_sorted if OBSERVED[i, j]]
                print(f"    {BENCH_NAMES[BENCH_IDS[j]]:<30s}  R²={r2:.2f}  slope={slope:.1f}  "
                      f"[{', '.join(model_strs)}]")


# ══════════════════════════════════════════════════════════════════════════════
#  3.6-3.7: PROVIDER EFFECTS AND REASONING MODE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_provider_and_reasoning():
    print("\n" + "="*80)
    print("  3.6-3.7: PROVIDER EFFECTS & REASONING MODE ANALYSIS")
    print("="*80)

    # Compare reasoning vs non-reasoning models
    cm, cs = col_stats(M_FULL)
    M_norm = (M_FULL - cm) / cs
    M_norm[np.isnan(M_FULL)] = np.nan

    reasoning_idx = np.where(MODEL_REASONING)[0]
    nonreason_idx = np.where(~MODEL_REASONING)[0]

    print(f"\n  Reasoning models: {len(reasoning_idx)}, Non-reasoning: {len(nonreason_idx)}")

    # For each benchmark, compare mean z-score of reasoning vs non-reasoning
    print(f"\n  Benchmarks where reasoning models score MUCH higher (z-score gap > 0.5):")
    print(f"  {'Benchmark':<35s}  {'Reasoning':>10s}  {'NonReason':>10s}  {'Gap':>6s}")
    gaps = []
    for j in range(N_BENCH):
        r_scores = M_norm[reasoning_idx, j]
        nr_scores = M_norm[nonreason_idx, j]
        r_mean = np.nanmean(r_scores) if np.sum(~np.isnan(r_scores)) > 2 else np.nan
        nr_mean = np.nanmean(nr_scores) if np.sum(~np.isnan(nr_scores)) > 2 else np.nan
        if not np.isnan(r_mean) and not np.isnan(nr_mean):
            gap = r_mean - nr_mean
            gaps.append((j, gap, r_mean, nr_mean))

    gaps.sort(key=lambda x: -x[1])
    for j, gap, r_mean, nr_mean in gaps[:10]:
        print(f"  {BENCH_NAMES[BENCH_IDS[j]]:<35s}  {r_mean:>+10.2f}  {nr_mean:>+10.2f}  {gap:>+6.2f}")
    print(f"\n  Benchmarks where reasoning models DON'T help much:")
    for j, gap, r_mean, nr_mean in gaps[-5:]:
        print(f"  {BENCH_NAMES[BENCH_IDS[j]]:<35s}  {r_mean:>+10.2f}  {nr_mean:>+10.2f}  {gap:>+6.2f}")

    # Provider analysis: for each provider, compute average residual
    # (how much does the provider's models deviate from expected based on latent factors?)
    print(f"\n  Provider average z-scores per benchmark category:")
    providers = sorted(set(MODEL_PROVIDERS))
    cats = sorted(set(BENCH_CATS))
    print(f"  {'Provider':<20s}", end="")
    for cat in cats:
        print(f"  {cat[:8]:>8s}", end="")
    print()
    for prov in providers:
        prov_idx = np.where(MODEL_PROVIDERS == prov)[0]
        if len(prov_idx) < 2:
            continue
        print(f"  {prov:<20s}", end="")
        for cat in cats:
            cat_benches = np.where(BENCH_CATS == cat)[0]
            vals = M_norm[np.ix_(prov_idx, cat_benches)]
            avg = np.nanmean(vals)
            if np.isnan(avg):
                print(f"  {'---':>8s}", end="")
            else:
                print(f"  {avg:>+8.2f}", end="")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  SURPRISING MODELS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_surprising_models():
    print("\n" + "="*80)
    print("  SURPRISING MODELS — largest deviations from low-rank structure")
    print("="*80)

    # Compute raw rank-3 SVD reconstruction (NOT preserving observed entries)
    # This shows which models deviate most from the low-rank assumption
    cm, cs = col_stats(M_FULL)
    M_norm = (M_FULL - cm) / cs
    M_norm[np.isnan(M_FULL)] = np.nan

    # Impute missing with 0 (column mean in z-space), iterate Soft-Impute
    M_imp = M_norm.copy()
    M_imp[np.isnan(M_imp)] = 0
    obs = ~np.isnan(M_norm)
    for it in range(100):
        M_old = M_imp.copy()
        U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)
        M_approx = (U[:, :3] * s[:3]) @ Vt[:3, :]
        M_imp = np.where(obs, M_norm, M_approx)
        if np.sqrt(np.mean((M_imp - M_old)**2)) < 1e-4:
            break

    # M_approx is the raw rank-3 reconstruction — compare against actual observed values
    M_recon = M_approx * cs + cm  # denormalize

    model_surprise = []
    for i in range(N_MODELS):
        obs_j = np.where(OBSERVED[i])[0]
        if len(obs_j) < 5:
            continue
        actual = M_FULL[i, obs_j]
        predicted = M_recon[i, obs_j]
        nonzero = np.abs(actual) > 1e-6
        if nonzero.sum() < 3:
            continue
        ape = np.abs(predicted[nonzero] - actual[nonzero]) / np.abs(actual[nonzero])
        medape = np.median(ape) * 100

        # Find the most surprising individual scores
        worst_j = obs_j[nonzero][np.argmax(ape)]
        model_surprise.append((i, medape, worst_j, ape.max()*100))

    model_surprise.sort(key=lambda x: -x[1])
    print(f"\n  Models that deviate most from rank-3 SVD reconstruction:")
    print(f"  {'Model':<40s}  {'MedAPE':>8s}  {'Biggest surprise':>40s}  {'Error':>8s}")
    for i, medape, worst_j, worst_err in model_surprise[:15]:
        actual = M_FULL[i, worst_j]
        predicted = M_recon[i, worst_j]
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<40s}  {medape:>7.1f}%  "
              f"{BENCH_NAMES[BENCH_IDS[worst_j]]} (act={actual:.0f}, exp={predicted:.0f})"
              f"  {worst_err:>7.1f}%")

    # Also show most surprising individual entries
    print(f"\n  Most surprising individual (model, benchmark) entries:")
    print(f"  {'Model':<30s}  {'Benchmark':<25s}  {'Actual':>8s}  {'Expected':>8s}  {'APE':>8s}")
    all_entries = []
    for i in range(N_MODELS):
        obs_j = np.where(OBSERVED[i])[0]
        for j in obs_j:
            if abs(M_FULL[i, j]) > 1e-6:
                ape = abs(M_recon[i, j] - M_FULL[i, j]) / abs(M_FULL[i, j]) * 100
                all_entries.append((i, j, M_FULL[i, j], M_recon[i, j], ape))
    all_entries.sort(key=lambda x: -x[4])
    for i, j, actual, expected, ape in all_entries[:20]:
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<30s}  {BENCH_NAMES[BENCH_IDS[j]]:<25s}  "
              f"{actual:>8.1f}  {expected:>8.1f}  {ape:>7.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    s, var_explained, cum_var, Vt = analyze_intrinsic_dimensionality()
    corr, clusters = analyze_benchmark_redundancy()
    bench_info, min_eval_set = analyze_minimum_eval_set()
    analyze_data_efficiency()
    analyze_scaling_laws()
    analyze_provider_and_reasoning()
    analyze_surprising_models()

    # Save latent factors
    import csv
    with open(os.path.join(REPO_ROOT, 'results', 'latent_factors.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['benchmark', 'factor1', 'factor2', 'factor3', 'factor4', 'factor5'])
        for j in range(N_BENCH):
            row = [BENCH_NAMES[BENCH_IDS[j]]]
            for k in range(min(5, Vt.shape[0])):
                row.append(f"{Vt[k, j]:.3f}")
            writer.writerow(row)
    print(f"\n  Latent factors saved to {os.path.join(REPO_ROOT, 'results', 'latent_factors.csv')}")
    print("\n  Done.")
