#!/usr/bin/env python3
"""
Final evaluation of LogitSVD Blend vs baselines.
Produces all numbers needed for the report and README update.
"""

import numpy as np
import sys, warnings, os, time, csv, json

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, BENCH_CATS,
    holdout_random_cells, holdout_per_model,
)
from all_methods import (
    predict_B0, predict_benchreg, predict_B2, predict_blend, predict_svd,
    predict_logit_benchreg, predict_svd_logit, predict_logit_svd_blend,
)

# ── Bimodal benchmarks ──
BIMODAL_IDS = ['arc_agi_1', 'arc_agi_2', 'imo_2025', 'usamo_2025', 'matharena_apex_2025']
BIMODAL_IDX = [BENCH_IDS.index(b) for b in BIMODAL_IDS if b in BENCH_IDS]
BIMODAL_THRESHOLD = 10.0


def full_metrics(actual, predicted, test_set):
    """Compute all metrics requested in the task spec."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    valid = ~np.isnan(p) & ~np.isnan(a)
    a, p = a[valid], p[valid]
    test_set_valid = [ts for ts, v in zip(test_set, valid) if v] if test_set else []

    if len(a) == 0:
        return {k: np.nan for k in ['medape', 'mae', 'within3', 'within5',
                                     'medape_hi', 'medape_lo', 'bimodal_acc',
                                     'bimodal_n', 'coverage', 'n']}

    abs_err = np.abs(p - a)
    mae = np.median(abs_err)
    within3 = np.mean(abs_err <= 3.0) * 100
    within5 = np.mean(abs_err <= 5.0) * 100

    nonzero = np.abs(a) > 1e-6
    ape = abs_err[nonzero] / np.abs(a[nonzero]) * 100
    medape = np.median(ape) if len(ape) > 0 else np.nan

    # Split by score > 50 vs <= 50
    hi_apes = [abs_err[k] / np.abs(a[k]) * 100 for k in range(len(a))
               if np.abs(a[k]) > 1e-6 and a[k] > 50]
    lo_apes = [abs_err[k] / np.abs(a[k]) * 100 for k in range(len(a))
               if np.abs(a[k]) > 1e-6 and a[k] <= 50]
    medape_hi = np.median(hi_apes) if hi_apes else np.nan
    medape_lo = np.median(lo_apes) if lo_apes else np.nan

    # Bimodal classification
    bimodal_correct, bimodal_total = 0, 0
    for idx, (i, j) in enumerate(test_set_valid):
        if j in BIMODAL_IDX:
            bimodal_total += 1
            a_cls = 1 if actual[idx] > BIMODAL_THRESHOLD else 0
            p_cls = 1 if predicted[idx] > BIMODAL_THRESHOLD else 0
            if a_cls == p_cls:
                bimodal_correct += 1
    bimodal_acc = (bimodal_correct / bimodal_total * 100) if bimodal_total > 0 else np.nan

    return {
        'medape': medape, 'mae': mae, 'within3': within3, 'within5': within5,
        'medape_hi': medape_hi, 'medape_lo': medape_lo,
        'bimodal_acc': bimodal_acc, 'bimodal_n': bimodal_total,
        'coverage': np.sum(valid) / len(actual) * 100 if len(actual) > 0 else 0,
        'n': len(a),
    }


def run_multi_seed(predict_fn, holdout_fn, seeds, **holdout_kwargs):
    """Run holdout evaluation across multiple seeds, return per-seed and averaged metrics."""
    all_seed_metrics = []
    for seed in seeds:
        folds = holdout_fn(seed=seed, **holdout_kwargs)
        all_a, all_p, all_ts = [], [], []
        for M_train, test_set in folds:
            M_pred = predict_fn(M_train)
            for i, j in test_set:
                all_a.append(M_FULL[i, j])
                all_p.append(M_pred[i, j])
                all_ts.append((i, j))
        m = full_metrics(all_a, all_p, all_ts)
        all_seed_metrics.append(m)

    # Average across seeds
    avg = {}
    for key in all_seed_metrics[0]:
        vals = [m[key] for m in all_seed_metrics
                if not (isinstance(m[key], float) and np.isnan(m[key]))]
        avg[key] = np.mean(vals) if vals else np.nan
        if len(vals) > 1:
            avg[key + '_std'] = np.std(vals)
    return avg, all_seed_metrics


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEEDS = [42, 123, 456, 789, 1337]

    methods = [
        ("B0: Benchmark Mean",         predict_B0),
        ("BenchReg",                    predict_benchreg),
        ("SVD(r=2)",                    lambda M: predict_svd(M, rank=2)),
        ("BenchReg+KNN Blend(0.6)",     lambda M: predict_blend(M, alpha=0.6)),
        ("LogitBenchReg",               predict_logit_benchreg),
        ("SVD-Logit(r=2)",              lambda M: predict_svd_logit(M, rank=2)),
        ("LogitSVD Blend (0.6/0.4)",    predict_logit_svd_blend),
    ]

    results = {}

    # ── Per-model 50% holdout, 5 seeds ──
    print("=" * 100)
    print("  PRIMARY: Per-model leave-50%-out, 5 seeds")
    print("=" * 100)

    for name, fn in methods:
        t0 = time.time()
        avg, seeds_m = run_multi_seed(
            fn, holdout_per_model, SEEDS,
            k_frac=0.5, min_scores=8, n_folds=3
        )
        elapsed = time.time() - t0
        results[name] = {'pm': avg, 'time': elapsed}

        print(f"\n  {name}:")
        print(f"    MedAPE  = {avg['medape']:.2f}% ± {avg.get('medape_std', 0):.2f}%")
        print(f"    MAE     = {avg['mae']:.2f}")
        print(f"    ±3 pts  = {avg['within3']:.1f}%")
        print(f"    ±5 pts  = {avg['within5']:.1f}%")
        print(f"    APE>50  = {avg['medape_hi']:.2f}%")
        print(f"    APE≤50  = {avg['medape_lo']:.2f}%")
        print(f"    BiAcc   = {avg['bimodal_acc']:.1f}% (n={avg['bimodal_n']:.0f})")
        print(f"    Coverage= {avg['coverage']:.1f}%")
        print(f"    [{elapsed:.1f}s]")

    # ── Random 20% holdout, 5 seeds ──
    print("\n" + "=" * 100)
    print("  SECONDARY: Random 20% holdout, 5 seeds")
    print("=" * 100)

    for name, fn in methods:
        t0 = time.time()
        avg, seeds_m = run_multi_seed(
            fn, holdout_random_cells, SEEDS,
            frac=0.2, n_folds=5
        )
        results[name]['rand'] = avg

        print(f"\n  {name}:")
        print(f"    MedAPE  = {avg['medape']:.2f}%")
        print(f"    MAE     = {avg['mae']:.2f}")
        print(f"    ±3 pts  = {avg['within3']:.1f}%")
        print(f"    ±5 pts  = {avg['within5']:.1f}%")

    # ── Clean comparison table ──
    print("\n\n" + "=" * 140)
    print("  COMPARISON TABLE")
    print("=" * 140)

    print(f"\n  Per-model leave-50%-out (primary), averaged over 5 seeds:")
    print(f"  {'Method':<30s} {'MedAPE':>8s} {'MAE':>6s} {'±3pts':>6s} {'±5pts':>6s} "
          f"{'APE>50':>7s} {'APE≤50':>8s} {'BiAcc':>6s} {'Cov':>5s}")
    print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*8} {'─'*6} {'─'*5}")

    sorted_methods = sorted(methods, key=lambda x: results[x[0]]['pm']['medape'])
    for name, _ in sorted_methods:
        pm = results[name]['pm']
        print(f"  {name:<30s} {pm['medape']:>7.2f}% {pm['mae']:>6.2f} {pm['within3']:>5.1f}% "
              f"{pm['within5']:>5.1f}% {pm['medape_hi']:>6.2f}% {pm['medape_lo']:>7.2f}% "
              f"{pm['bimodal_acc']:>5.1f}% {pm['coverage']:>4.1f}%")

    print(f"\n  Random 20% holdout (secondary), averaged over 5 seeds:")
    print(f"  {'Method':<30s} {'MedAPE':>8s} {'MAE':>6s} {'±3pts':>6s} {'±5pts':>6s}")
    print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")
    for name, _ in sorted_methods:
        rd = results[name]['rand']
        print(f"  {name:<30s} {rd['medape']:>7.2f}% {rd['mae']:>6.2f} {rd['within3']:>5.1f}% "
              f"{rd['within5']:>5.1f}%")

    # ── Improvement summary ──
    baseline = results['BenchReg+KNN Blend(0.6)']['pm']
    best = results['LogitSVD Blend (0.6/0.4)']['pm']
    print(f"\n  ── Improvement: LogitSVD Blend vs old Blend ──")
    print(f"  MedAPE:  {baseline['medape']:.2f}% → {best['medape']:.2f}% "
          f"({(1 - best['medape']/baseline['medape'])*100:.1f}% relative improvement)")
    print(f"  MAE:     {baseline['mae']:.2f} → {best['mae']:.2f} "
          f"({(1 - best['mae']/baseline['mae'])*100:.1f}%↓)")
    print(f"  ±3pts:   {baseline['within3']:.1f}% → {best['within3']:.1f}% "
          f"(+{best['within3']-baseline['within3']:.1f}pp)")
    print(f"  ±5pts:   {baseline['within5']:.1f}% → {best['within5']:.1f}% "
          f"(+{best['within5']-baseline['within5']:.1f}pp)")
    print(f"  BiAcc:   {baseline['bimodal_acc']:.1f}% → {best['bimodal_acc']:.1f}% "
          f"(+{best['bimodal_acc']-baseline['bimodal_acc']:.1f}pp)")

    # ── Save results as JSON ──
    out = {}
    for name, _ in methods:
        out[name] = {
            'pm_medape': round(results[name]['pm']['medape'], 2),
            'pm_mae': round(results[name]['pm']['mae'], 2),
            'pm_within3': round(results[name]['pm']['within3'], 1),
            'pm_within5': round(results[name]['pm']['within5'], 1),
            'pm_medape_hi': round(results[name]['pm']['medape_hi'], 2),
            'pm_medape_lo': round(results[name]['pm']['medape_lo'], 2),
            'pm_bimodal_acc': round(results[name]['pm']['bimodal_acc'], 1),
            'pm_coverage': round(results[name]['pm']['coverage'], 1),
            'rand_medape': round(results[name]['rand']['medape'], 2),
            'rand_mae': round(results[name]['rand']['mae'], 2),
            'rand_within3': round(results[name]['rand']['within3'], 1),
            'rand_within5': round(results[name]['rand']['within5'], 1),
        }
    json_path = os.path.join(REPO_ROOT, 'results', 'logit_svd_eval.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE TRANSITION CURVE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE TRANSITION: LogitSVD Blend accuracy vs number of known scores")
    print("=" * 100)

    phase_seeds = [42, 123, 456, 789, 1337]
    # For each model with ≥10 known scores: hide all, reveal 1..N, predict rest
    eligible = [i for i in range(N_MODELS) if OBSERVED[i].sum() >= 10]
    max_reveal = 20  # up to 20 known scores

    phase_data = {k: [] for k in range(1, max_reveal + 1)}  # n_known → list of MedAPEs

    for seed in phase_seeds:
        rng = np.random.RandomState(seed)
        for i in eligible:
            obs_j = np.where(OBSERVED[i])[0].copy()
            rng.shuffle(obs_j)
            n_total = len(obs_j)

            for n_reveal in range(1, min(max_reveal + 1, n_total)):
                revealed = obs_j[:n_reveal]
                hidden = obs_j[n_reveal:]

                M_train = M_FULL.copy()
                M_train[i, :] = np.nan  # hide all for this model
                for j in revealed:
                    M_train[i, j] = M_FULL[i, j]

                M_pred = predict_logit_svd_blend(M_train)

                apes = []
                abs_errs = []
                for j in hidden:
                    actual = M_FULL[i, j]
                    pred = M_pred[i, j]
                    if np.isfinite(pred) and abs(actual) > 1e-6:
                        apes.append(abs(pred - actual) / abs(actual) * 100)
                        abs_errs.append(abs(pred - actual))

                if apes:
                    phase_data[n_reveal].append({
                        'medape': np.median(apes),
                        'mae': np.median(abs_errs),
                        'within3': np.mean([e <= 3 for e in abs_errs]) * 100,
                        'within5': np.mean([e <= 5 for e in abs_errs]) * 100,
                    })

    phase_path = os.path.join(REPO_ROOT, 'results', 'phase_transition.csv')
    with open(phase_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_known', 'mean_medape', 'std_medape', 'mean_mae', 'std_mae',
                         'mean_within3', 'mean_within5'])
        for n_known in range(1, max_reveal + 1):
            data = phase_data[n_known]
            if data:
                medapes = [d['medape'] for d in data]
                maes = [d['mae'] for d in data]
                w3 = [d['within3'] for d in data]
                w5 = [d['within5'] for d in data]
                writer.writerow([
                    n_known,
                    f"{np.mean(medapes):.2f}", f"{np.std(medapes):.2f}",
                    f"{np.mean(maes):.2f}", f"{np.std(maes):.2f}",
                    f"{np.mean(w3):.1f}", f"{np.mean(w5):.1f}",
                ])
                print(f"  n_known={n_known:2d}: MedAPE={np.mean(medapes):6.1f}% ± {np.std(medapes):5.1f}%  "
                      f"MAE={np.mean(maes):5.1f}  ±3pts={np.mean(w3):5.1f}%  ±5pts={np.mean(w5):5.1f}%")

    print(f"\n  Phase transition saved to {phase_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  REGENERATE best_predictions.csv
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  REGENERATING best_predictions.csv with LogitSVD Blend")
    print("=" * 100)

    from build_benchmark_matrix import BENCHMARKS

    bench_is_pct = []
    for b in BENCHMARKS:
        metric = b[3].lower() if len(b) > 3 else ""
        bench_is_pct.append("%" in metric or "pass@" in metric.replace("%", ""))
    bench_is_pct = np.array(bench_is_pct)

    M_pred_best = predict_logit_svd_blend(M_FULL)

    # Clamp
    for j in range(N_BENCH):
        if bench_is_pct[j]:
            M_pred_best[:, j] = np.clip(M_pred_best[:, j], 0.0, 100.0)
        else:
            obs_vals = M_FULL[OBSERVED[:, j], j]
            if len(obs_vals) > 0:
                lo = max(0, obs_vals.min() - 200)
                hi = obs_vals.max() + 200
                M_pred_best[:, j] = np.clip(M_pred_best[:, j], lo, hi)

    pred_path = os.path.join(REPO_ROOT, 'results', 'best_predictions.csv')
    n_written = 0
    n_nan = 0
    with open(pred_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'benchmark', 'predicted_score', 'method'])
        for i in range(N_MODELS):
            for j in range(N_BENCH):
                if not OBSERVED[i, j]:
                    val = M_pred_best[i, j]
                    if np.isfinite(val):
                        writer.writerow([MODEL_NAMES[MODEL_IDS[i]], BENCH_NAMES[BENCH_IDS[j]],
                                        f"{val:.1f}", 'LogitSVD(0.6/0.4)'])
                        n_written += 1
                    else:
                        writer.writerow([MODEL_NAMES[MODEL_IDS[i]], BENCH_NAMES[BENCH_IDS[j]],
                                        '', 'no_prediction'])
                        n_nan += 1

    total_missing = (~OBSERVED).sum()
    print(f"  Written {n_written}/{total_missing} cells ({n_nan} NaN)")

    # Verify: no negatives, no pct > 100
    vals = []
    with open(pred_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['predicted_score']:
                vals.append(float(row['predicted_score']))
    vals = np.array(vals)
    print(f"  Negatives: {(vals < 0).sum()}")
    print(f"  > 100 (on pct benchmarks): checked via clamp")
    print(f"  Range: [{vals.min():.1f}, {vals.max():.1f}]")

    print(f"\n  Saved to {pred_path}")
    print("\n" + "=" * 100)
    print("  ALL DONE")
    print("=" * 100)
