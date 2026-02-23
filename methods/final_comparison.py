#!/usr/bin/env python3
"""
Final Comprehensive Comparison
==============================
All top methods from Rounds 1-3 + baselines in one clean evaluation.
"""

import numpy as np
import sys, warnings, os, time

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_PROVIDERS, MODEL_REASONING,
    MODEL_PARAMS, BENCH_CATS,
    holdout_random_cells, holdout_per_model,
)
from all_methods import predict_benchreg, predict_B2, predict_blend, predict_svd, predict_B0, predict_B1
from creative_methods import (
    extended_metrics, evaluate_extended, BIMODAL_BENCH_IDX, BIMODAL_THRESHOLD,
    predict_cat_benchreg, predict_cat_blend, predict_gbt, predict_logit_benchreg,
    predict_logit_blend, predict_gbt_blend,
)
from creative_methods_r2 import (
    predict_logit_cat_benchreg, predict_logit_cat_blend,
    predict_confidence_blend, predict_kitchen_sink, predict_gbt_logit,
)
from creative_methods_r3 import (
    predict_svd_logit, predict_logit_svd_breg, predict_kitchen_sink_v2,
    predict_logit_svd_only, predict_simple_trio,
)


if __name__ == "__main__":
    print("=" * 100)
    print("  FINAL COMPREHENSIVE COMPARISON")
    print("  Per-model 50% holdout × 5 seeds + Random 20% holdout")
    print("=" * 100)

    methods = [
        # ── Baselines ──
        ("B0:BenchmarkMean",         predict_B0),
        ("B1:ModelNormalized",       predict_B1),
        ("B2:KNN(k=5)",             lambda M: predict_B2(M, k=5)),
        ("Baseline:BenchReg",        predict_benchreg),
        ("Baseline:SVD(r=2)",        lambda M: predict_svd(M, rank=2)),
        ("Baseline:Blend(0.6)",      lambda M: predict_blend(M, alpha=0.6)),

        # ── Round 1 winners ──
        ("R1:LogitBenchReg",         predict_logit_benchreg),
        ("R1:CatBenchReg",           predict_cat_benchreg),
        ("R1:GBT+Blend(0.5)",       predict_gbt_blend),
        ("R1:LogitBlend(0.6)",       predict_logit_blend),

        # ── Round 2 winners ──
        ("R2:LogitCatBenchReg",      predict_logit_cat_benchreg),
        ("R2:ConfidenceBlend",       predict_confidence_blend),
        ("R2:KitchenSink",           predict_kitchen_sink),
        ("R2:LogitCatBlend(0.6)",    predict_logit_cat_blend),

        # ── Round 3 winners ──
        ("★LogitSVDonly(0.6/0.4)",   predict_logit_svd_only),
        ("★KitchenSinkV2",           predict_kitchen_sink_v2),
        ("R3:LogitSVD+LCB",          predict_logit_svd_breg),
        ("R3:SimpleTrio",            predict_simple_trio),
        ("R3:SVD-Logit(r=2)",        lambda M: predict_svd_logit(M, rank=2)),
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

    # ── PRINT COMPREHENSIVE TABLE ──
    print("\n\n" + "=" * 145)
    print("  FINAL RESULTS TABLE (sorted by PM-MedAPE)")
    print("=" * 145)

    header = (f"  {'#':>2s} {'Method':<28s} {'PM-MedAPE':>9s} {'R-MedAPE':>9s} {'PM-MAE':>7s} "
              f"{'±3pts':>6s} {'APE>50':>7s} {'APE≤50':>8s} "
              f"{'BiAcc':>6s}{'(n)':>5s} {'Cov':>5s} {'Time':>6s}")
    print(header)
    print("  " + "─" * 143)

    results.sort(key=lambda x: x.get('pm_medape', 999))

    for rank, r in enumerate(results):
        marker = "→" if r['method'].startswith("★") else " "
        print(f" {marker}{rank+1:>2d} {r['method']:<28s} {r['pm_medape']:>8.2f}% {r['rand_medape']:>8.2f}% "
              f"{r['pm_mae']:>7.2f} {r['pct_within3']:>5.1f}% "
              f"{r['medape_hi']:>6.2f}% {r['medape_lo']:>7.2f}% "
              f"{r.get('bimodal_acc', float('nan')):>5.1f}%{int(r.get('bimodal_n', 0)):>4d} "
              f"{r.get('coverage', 100):>4.1f}% "
              f"{r.get('time', 0):>5.1f}s")

    # ── RELATIVE IMPROVEMENT ──
    baseline_medape = next(r['pm_medape'] for r in results if r['method'] == 'Baseline:Blend(0.6)')
    print(f"\n  Baseline Blend PM-MedAPE: {baseline_medape:.2f}%")
    print(f"\n  {'Method':<28s}  {'PM-MedAPE':>9s}  {'Δ vs Blend':>10s}  {'Rel Δ':>7s}")
    print(f"  {'─'*28}  {'─'*9}  {'─'*10}  {'─'*7}")
    for r in results:
        delta = r['pm_medape'] - baseline_medape
        rel_delta = delta / baseline_medape * 100
        arrow = "↓" if delta < 0 else "↑"
        print(f"  {r['method']:<28s}  {r['pm_medape']:>8.2f}%  {delta:>+9.2f}%  {arrow}{abs(rel_delta):>5.1f}%")

    # ── ANALYSIS: What information helps? ──
    print(f"\n\n{'='*100}")
    print(f"  ANALYSIS: WHAT INFORMATION HELPS?")
    print(f"{'='*100}")

    # 1. Logit transform impact
    breg = next(r for r in results if r['method'] == 'Baseline:BenchReg')
    logit_breg = next(r for r in results if r['method'] == 'R1:LogitBenchReg')
    print(f"\n  [1] LOGIT TRANSFORM (biggest single win)")
    print(f"      BenchReg:       {breg['pm_medape']:.2f}% PM-MedAPE")
    print(f"      LogitBenchReg:  {logit_breg['pm_medape']:.2f}% PM-MedAPE")
    print(f"      Improvement:    {(1 - logit_breg['pm_medape']/breg['pm_medape'])*100:.1f}% relative")
    print(f"      Bimodal Acc:    {breg.get('bimodal_acc', 0):.1f}% → {logit_breg.get('bimodal_acc', 0):.1f}%")

    # 2. Category awareness
    logit_cat = next(r for r in results if r['method'] == 'R2:LogitCatBenchReg')
    print(f"\n  [2] CATEGORY AWARENESS (small but consistent)")
    print(f"      LogitBenchReg:      {logit_breg['pm_medape']:.2f}%")
    print(f"      LogitCatBenchReg:   {logit_cat['pm_medape']:.2f}%")
    print(f"      Improvement:        {(1 - logit_cat['pm_medape']/logit_breg['pm_medape'])*100:.1f}% relative")

    # 3. SVD in logit space
    svd_normal = next(r for r in results if r['method'] == 'Baseline:SVD(r=2)')
    svd_logit = next(r for r in results if r['method'] == 'R3:SVD-Logit(r=2)')
    print(f"\n  [3] SVD IN LOGIT SPACE (major improvement to SVD)")
    print(f"      SVD(r=2):       {svd_normal['pm_medape']:.2f}% PM-MedAPE, BiAcc={svd_normal.get('bimodal_acc', 0):.1f}%")
    print(f"      SVD-Logit(r=2): {svd_logit['pm_medape']:.2f}% PM-MedAPE, BiAcc={svd_logit.get('bimodal_acc', 0):.1f}%")
    print(f"      Improvement:    {(1 - svd_logit['pm_medape']/svd_normal['pm_medape'])*100:.1f}% relative")

    # 4. KNN value
    knn = next(r for r in results if r['method'] == 'B2:KNN(k=5)')
    logit_svd_only = next(r for r in results if r['method'] == '★LogitSVDonly(0.6/0.4)')
    logit_cat_blend = next(r for r in results if r['method'] == 'R2:LogitCatBlend(0.6)')
    print(f"\n  [4] DOES KNN HELP? (surprising answer: NO)")
    print(f"      LogitCatBenchReg + KNN:        {logit_cat_blend['pm_medape']:.2f}% (LogitCatBlend)")
    print(f"      LogitCatBenchReg + SVD-Logit:   {logit_svd_only['pm_medape']:.2f}% (LogitSVDonly)")
    print(f"      KNN adds noise; SVD-Logit is a BETTER complement to BenchReg")

    # 5. Full-coverage champion
    best_full = next(r for r in results if r['coverage'] >= 99.5)
    baseline_blend = next(r for r in results if r['method'] == 'Baseline:Blend(0.6)')
    print(f"\n  [5] BEST FULL-COVERAGE METHOD")
    print(f"      Baseline Blend:    {baseline_blend['pm_medape']:.2f}% (α*BenchReg + (1-α)*KNN)")
    print(f"      {best_full['method']}: {best_full['pm_medape']:.2f}%")
    print(f"      Total improvement: {(1 - best_full['pm_medape']/baseline_blend['pm_medape'])*100:.1f}% relative")
    print(f"      MAE:    {baseline_blend['pm_mae']:.2f} → {best_full['pm_mae']:.2f} ({(1-best_full['pm_mae']/baseline_blend['pm_mae'])*100:.1f}%↓)")
    print(f"      ±3pts:  {baseline_blend['pct_within3']:.1f}% → {best_full['pct_within3']:.1f}% (+{best_full['pct_within3']-baseline_blend['pct_within3']:.1f}pp)")
    print(f"      BiAcc:  {baseline_blend.get('bimodal_acc',0):.1f}% → {best_full.get('bimodal_acc',0):.1f}% (+{best_full.get('bimodal_acc',0)-baseline_blend.get('bimodal_acc',0):.1f}pp)")

    print(f"\n{'='*100}")
    print(f"  DONE")
    print(f"{'='*100}")
