#!/usr/bin/env python3
"""
Phase Transition Curve for LLM Benchmark Matrix Completion
===========================================================
For each model with >= 15 known scores:
  1. Hide ALL scores for that model
  2. Reveal 1 score at a time (random order)
  3. At each step, predict all remaining hidden scores using BenchReg+KNN blend
  4. Record MedAPE of predictions vs actual hidden scores

Average across qualifying models and 5 random seeds.
Output: results/phase_transition.csv
"""

import numpy as np
import sys, warnings, time, csv, os
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, MODEL_NAMES
from methods.all_methods import predict_benchreg, predict_B2

# --------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------
MIN_KNOWN = 15        # models must have >= 15 known scores
MAX_N_KNOWN = 25      # stop revealing at 25
MAX_MODELS = 30       # sample at most 30 qualifying models
SEEDS = list(range(5))
ALPHA = 0.6           # blend weight: alpha * BenchReg + (1-alpha) * KNN

# --------------------------------------------------------------------------
#  Find qualifying models
# --------------------------------------------------------------------------
model_known_counts = OBSERVED.sum(axis=1).astype(int)
qualifying = [i for i in range(N_MODELS) if model_known_counts[i] >= MIN_KNOWN]
print(f"Qualifying models (>= {MIN_KNOWN} known scores): {len(qualifying)}")

# If too many, sample
rng_sample = np.random.RandomState(99)
if len(qualifying) > MAX_MODELS:
    qualifying = sorted(rng_sample.choice(qualifying, MAX_MODELS, replace=False).tolist())
    print(f"Sampled down to {len(qualifying)} models")

for idx in qualifying:
    print(f"  {MODEL_NAMES.get(MODEL_IDS[idx], MODEL_IDS[idx])}: {model_known_counts[idx]} known")

# --------------------------------------------------------------------------
#  Simplified predict_blend for a single model row
#  Instead of recomputing on the full matrix each time, we do full-matrix
#  prediction but only care about row i.
# --------------------------------------------------------------------------
def predict_blend_for_model(M_train, alpha=ALPHA):
    """BenchReg+KNN blend. Falls back to KNN if blend returns NaN."""
    M_breg = predict_benchreg(M_train)
    M_knn = predict_B2(M_train, k=5)
    obs = ~np.isnan(M_train)
    M_pred = alpha * M_breg + (1 - alpha) * M_knn
    # Where BenchReg returned NaN, fall back to KNN
    nan_mask = np.isnan(M_pred) & ~obs
    M_pred[nan_mask] = M_knn[nan_mask]
    # Where still NaN, use column mean
    still_nan = np.isnan(M_pred) & ~obs
    col_mean = np.nanmean(M_train, axis=0)
    for j in range(N_BENCH):
        M_pred[still_nan[:, j], j] = col_mean[j] if not np.isnan(col_mean[j]) else 50.0
    M_pred[obs] = M_train[obs]
    return M_pred


# --------------------------------------------------------------------------
#  Main computation
# --------------------------------------------------------------------------
# results[n_known] = list of MedAPE values (one per model per seed)
from collections import defaultdict
results = defaultdict(list)

total_runs = len(qualifying) * len(SEEDS)
run_count = 0
t_start = time.time()

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    for model_i in qualifying:
        run_count += 1

        # Get indices of known benchmarks for this model
        known_js = np.where(OBSERVED[model_i])[0].copy()
        rng.shuffle(known_js)
        n_total = len(known_js)

        # Cap at MAX_N_KNOWN
        max_reveal = min(n_total - 1, MAX_N_KNOWN)  # need at least 1 hidden

        # Start with ALL scores hidden for this model
        M_base = M_FULL.copy()
        M_base[model_i, :] = np.nan  # hide all scores for this model

        # Reveal one at a time
        for step in range(max_reveal):
            n_known = step + 1
            j_reveal = known_js[step]
            M_base[model_i, j_reveal] = M_FULL[model_i, j_reveal]

            # Which benchmarks are still hidden?
            hidden_js = known_js[step + 1:]
            if len(hidden_js) == 0:
                break

            # Predict
            M_pred = predict_blend_for_model(M_base)

            # Compute MedAPE on hidden scores
            actual = M_FULL[model_i, hidden_js]
            predicted = M_pred[model_i, hidden_js]

            # APE
            nonzero = np.abs(actual) > 1e-6
            if nonzero.sum() == 0:
                continue
            ape = np.abs(predicted[nonzero] - actual[nonzero]) / np.abs(actual[nonzero])
            medape = np.median(ape) * 100

            results[n_known].append(medape)

        # Progress
        if run_count % 10 == 0 or run_count == total_runs:
            elapsed = time.time() - t_start
            rate = run_count / elapsed if elapsed > 0 else 0
            eta = (total_runs - run_count) / rate if rate > 0 else 0
            print(f"  Progress: {run_count}/{total_runs} runs done "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")


# --------------------------------------------------------------------------
#  Aggregate and output
# --------------------------------------------------------------------------
print(f"\nTotal time: {time.time() - t_start:.1f}s")
print(f"\n{'n_known':>8s} {'mean_medape':>12s} {'std_medape':>11s} {'n_models':>9s}")
print(f"{'─'*8} {'─'*12} {'─'*11} {'─'*9}")

rows = []
for n_known in sorted(results.keys()):
    vals = results[n_known]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    n_models = len(vals)
    rows.append({
        'n_known': n_known,
        'mean_medape': round(mean_val, 2),
        'std_medape': round(std_val, 2),
        'n_models': n_models,
    })
    print(f"{n_known:>8d} {mean_val:>11.2f}% {std_val:>10.2f}% {n_models:>9d}")

# Write CSV
outpath = os.path.join(REPO_ROOT, 'results', 'phase_transition.csv')
with open(outpath, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['n_known', 'mean_medape', 'std_medape', 'n_models'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"\nCSV written to {outpath}")
