#!/usr/bin/env python3
"""
Component-level analysis of BenchPress prediction methods.
Tests each component independently and the blend at different holdout fractions.

Experiments:
  1. BenchPress (full blend) at different per-model hiding fractions (10-50%)
  2. SVD-Logit only at different ranks (1,2,3,5,8)
  3. LogitBenchReg only (no SVD)
  4. Summary table: all methods x all hiding fractions
"""
import sys, os, io, time
import numpy as np

REPO = '/Users/anadim/llm-benchmark-matrix'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'data'))
sys.path.insert(0, os.path.join(REPO, 'methods'))

# Suppress the print from evaluation_harness import
_old = sys.stdout; sys.stdout = io.StringIO()
from evaluation_harness import M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS
from all_methods import (
    predict_logit_svd_blend,
    predict_svd_logit,
    predict_logit_benchreg,
    predict_B0,
)
sys.stdout = _old

print(f"Matrix loaded: {N_MODELS} models x {N_BENCH} benchmarks, "
      f"{OBSERVED.sum()} observed cells ({OBSERVED.sum()/(N_MODELS*N_BENCH)*100:.1f}% fill)")
print()


# ---------------------------------------------------------------------------
#  Holdout + evaluation logic
# ---------------------------------------------------------------------------

def run_holdout_experiment(predict_fn, frac_hide, n_seeds=3, min_scores=8, label=""):
    """
    Per-model holdout: for each model with >= min_scores observed benchmarks,
    hide frac_hide of its scores, predict them, compute error statistics.

    Returns dict with MedAPE, MAE, within-3pt, within-5pt, and cell count.
    """
    all_apes = []
    all_abs_errs = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)

        M_train = M_FULL.copy()
        test_set = []

        for i in range(N_MODELS):
            obs_j = np.where(OBSERVED[i])[0]
            if len(obs_j) < min_scores:
                continue

            n_hide = max(1, int(len(obs_j) * frac_hide))
            n_hide = min(n_hide, len(obs_j) - 2)   # keep at least 2 revealed

            perm = obs_j.copy()
            rng.shuffle(perm)
            hidden = perm[:n_hide]

            for j in hidden:
                M_train[i, j] = np.nan
                test_set.append((i, j))

        # Single prediction call for this fold (much faster than per-model)
        M_pred = predict_fn(M_train)

        for i, j in test_set:
            actual = M_FULL[i, j]
            pred = M_pred[i, j]
            if np.isfinite(pred) and abs(actual) > 1e-6:
                ape = abs(pred - actual) / abs(actual) * 100
                all_apes.append(ape)
                all_abs_errs.append(abs(pred - actual))

    if all_apes:
        medape = np.median(all_apes)
        mae = np.mean(all_abs_errs)
        within3 = sum(1 for e in all_abs_errs if e <= 3) / len(all_abs_errs) * 100
        within5 = sum(1 for e in all_abs_errs if e <= 5) / len(all_abs_errs) * 100
        n_cells = len(all_apes)
    else:
        medape = mae = within3 = within5 = float('nan')
        n_cells = 0

    return {
        'label': label,
        'medape': medape,
        'mae': mae,
        'within3': within3,
        'within5': within5,
        'n_cells': n_cells,
    }


# ---------------------------------------------------------------------------
#  EXPERIMENT 1: BenchPress at different hiding fractions
# ---------------------------------------------------------------------------
print("=" * 78)
print("EXPERIMENT 1: BenchPress (full blend) at different hiding fractions")
print("=" * 78)
t0 = time.time()
for frac in [0.10, 0.20, 0.30, 0.40, 0.50]:
    r = run_holdout_experiment(predict_logit_svd_blend, frac, n_seeds=3,
                               label=f"BenchPress hide={int(frac*100)}%")
    print(f"  {r['label']:<30s}  MedAPE={r['medape']:6.2f}%  MAE={r['mae']:5.2f}  "
          f"within3={r['within3']:5.1f}%  within5={r['within5']:5.1f}%  n={r['n_cells']}")
print(f"  (elapsed: {time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
#  EXPERIMENT 2: SVD-Logit only at different ranks (50% holdout)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("EXPERIMENT 2: SVD-Logit only at different ranks (50% holdout)")
print("=" * 78)
t0 = time.time()
for rank in [1, 2, 3, 5, 8]:
    fn = lambda M, r=rank: predict_svd_logit(M, rank=r)
    r = run_holdout_experiment(fn, 0.50, n_seeds=3,
                               label=f"SVD-Logit rank={rank}")
    print(f"  {r['label']:<30s}  MedAPE={r['medape']:6.2f}%  MAE={r['mae']:5.2f}  "
          f"within3={r['within3']:5.1f}%  within5={r['within5']:5.1f}%  n={r['n_cells']}")
print(f"  (elapsed: {time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
#  EXPERIMENT 3: LogitBenchReg only (50% holdout)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("EXPERIMENT 3: LogitBenchReg only (50% holdout)")
print("=" * 78)
t0 = time.time()
r = run_holdout_experiment(predict_logit_benchreg, 0.50, n_seeds=3,
                           label="LogitBenchReg only")
print(f"  {r['label']:<30s}  MedAPE={r['medape']:6.2f}%  MAE={r['mae']:5.2f}  "
      f"within3={r['within3']:5.1f}%  within5={r['within5']:5.1f}%  n={r['n_cells']}")
print(f"  (elapsed: {time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
#  EXPERIMENT 4: All methods x all hiding fractions (summary table)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("EXPERIMENT 4: All methods at all hiding fractions  (MedAPE %)")
print("=" * 78)

fracs = [0.10, 0.20, 0.30, 0.40, 0.50]
methods = [
    ("BenchPress (blend)",       predict_logit_svd_blend),
    ("SVD-Logit r=2",            lambda M: predict_svd_logit(M, rank=2)),
    ("LogitBenchReg",            predict_logit_benchreg),
    ("Column Mean (B0)",         predict_B0),
]

header = f"  {'Method':<28s}" + "".join(f"  {int(f*100):>3d}%" for f in fracs)
print(header)
print("  " + "-" * (28 + 7 * len(fracs)))

t0 = time.time()
all_results = {}
for name, fn in methods:
    row_vals = []
    for frac in fracs:
        r = run_holdout_experiment(fn, frac, n_seeds=3, label=name)
        row_vals.append(r['medape'])
        all_results[(name, frac)] = r
    cells = "".join(f"  {v:5.1f}%" for v in row_vals)
    print(f"  {name:<28s}{cells}")

print(f"  (elapsed: {time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
#  EXPERIMENT 5: Detailed comparison at 20% and 50%
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("EXPERIMENT 5: Detailed comparison at 20% and 50% hiding")
print("=" * 78)

for frac in [0.20, 0.50]:
    print(f"\n  --- Hiding {int(frac*100)}% ---")
    print(f"  {'Method':<28s} {'MedAPE':>8s} {'MAE':>7s} {'w/in 3':>8s} {'w/in 5':>8s} {'n':>7s}")
    print(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for name, fn in methods:
        key = (name, frac)
        if key in all_results:
            r = all_results[key]
        else:
            r = run_holdout_experiment(fn, frac, n_seeds=3, label=name)
        print(f"  {name:<28s} {r['medape']:7.2f}% {r['mae']:6.2f} {r['within3']:7.1f}% "
              f"{r['within5']:7.1f}% {r['n_cells']:>7d}")


# ---------------------------------------------------------------------------
#  EXPERIMENT 6: SVD rank sweep at 20% hiding too
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("EXPERIMENT 6: SVD rank sweep at 20% and 50% hiding")
print("=" * 78)
print(f"  {'Rank':<10s} {'MedAPE@20%':>12s} {'MedAPE@50%':>12s}")
print(f"  {'-'*10} {'-'*12} {'-'*12}")
t0 = time.time()
for rank in [1, 2, 3, 5, 8]:
    fn = lambda M, r=rank: predict_svd_logit(M, rank=r)
    r20 = run_holdout_experiment(fn, 0.20, n_seeds=3, label=f"r={rank}")
    r50 = run_holdout_experiment(fn, 0.50, n_seeds=3, label=f"r={rank}")
    print(f"  rank={rank:<5d} {r20['medape']:11.2f}% {r50['medape']:11.2f}%")
print(f"  (elapsed: {time.time()-t0:.0f}s)")


print()
print("=" * 78)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 78)
