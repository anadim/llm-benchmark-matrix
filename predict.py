#!/usr/bin/env python3
"""
LLM Benchmark Score Predictor
==============================

Predict missing benchmark scores for LLM models using LogitSVD Blend.

Usage:
    # Predict all missing scores (output CSV)
    python predict.py

    # Predict scores for a specific model
    python predict.py --model gpt-5.2

    # Predict scores on a specific benchmark
    python predict.py --benchmark aime_2025

    # Predict a single cell
    python predict.py --model gpt-5.2 --benchmark gpqa_diamond

    # Add a new model's known scores and predict the rest
    python predict.py --add-model my-model --scores "mmlu=85.2,gpqa_diamond=72.0,aime_2025=60.0"

    # Add a new benchmark's known scores and predict the rest
    python predict.py --add-benchmark my_bench --category Math --scores "gpt-5.2=90.0,o3-high=95.0"

    # Output as JSON instead of CSV
    python predict.py --model gpt-5.2 --format json

    # List all models or benchmarks
    python predict.py --list-models
    python predict.py --list-benchmarks
"""

import argparse
import csv
import json
import sys
import os
import io
import numpy as np

# ── Setup paths ──
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

# Suppress the matrix print from evaluation_harness
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH,
    MODEL_IDS, BENCH_IDS, MODEL_NAMES, BENCH_NAMES,
    MODEL_IDX, BENCH_IDX, MODEL_REASONING, MODEL_PROVIDERS,
)
from all_methods import predict_logit_svd_blend
sys.stdout = _old_stdout


def predict_all(M_input=None):
    """Run LogitSVD Blend on the matrix and return predictions."""
    if M_input is None:
        M_input = M_FULL.copy()
    return predict_logit_svd_blend(M_input)


def format_predictions(predictions, model_filter=None, bench_filter=None,
                       only_missing=True, fmt='csv'):
    """Format predictions as CSV or JSON rows."""
    rows = []
    obs = ~np.isnan(M_FULL) if only_missing else np.ones_like(OBSERVED)

    for i in range(predictions.shape[0]):
        mid = MODEL_IDS[i]
        if model_filter and mid != model_filter:
            continue
        for j in range(predictions.shape[1]):
            bid = BENCH_IDS[j]
            if bench_filter and bid != bench_filter:
                continue
            if only_missing and OBSERVED[i, j]:
                continue
            pred = predictions[i, j]
            actual = M_FULL[i, j] if OBSERVED[i, j] else None
            rows.append({
                'model': mid,
                'model_name': MODEL_NAMES[mid],
                'benchmark': bid,
                'benchmark_name': BENCH_NAMES[bid],
                'predicted': round(float(pred), 1) if np.isfinite(pred) else None,
                'actual': round(float(actual), 1) if actual is not None else None,
                'is_observed': bool(OBSERVED[i, j]),
            })

    if fmt == 'json':
        return json.dumps(rows, indent=2)
    else:
        if not rows:
            return "No predictions to show."
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()


def add_model_scores(model_name, scores_str, M_input):
    """Add a new model row to the matrix with known scores, return augmented matrix."""
    # Parse scores
    scores = {}
    for pair in scores_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            print(f"Warning: skipping malformed score '{pair}' (expected bench=score)")
            continue
        bench, val = pair.split('=', 1)
        bench = bench.strip()
        val = float(val.strip())
        if bench not in BENCH_IDX:
            print(f"Warning: benchmark '{bench}' not found. Available: {', '.join(BENCH_IDS[:10])}...")
            continue
        scores[bench] = val

    if not scores:
        print("Error: no valid scores provided.")
        sys.exit(1)

    # Add new row
    new_row = np.full((1, N_BENCH), np.nan)
    for bench, val in scores.items():
        new_row[0, BENCH_IDX[bench]] = val

    M_aug = np.vstack([M_input, new_row])
    return M_aug, scores


def add_benchmark_scores(bench_name, scores_str, M_input):
    """Add a new benchmark column to the matrix with known scores, return augmented matrix."""
    scores = {}
    for pair in scores_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue
        model, val = pair.split('=', 1)
        model = model.strip()
        val = float(val.strip())
        if model not in MODEL_IDX:
            print(f"Warning: model '{model}' not found. Available: {', '.join(MODEL_IDS[:10])}...")
            continue
        scores[model] = val

    if not scores:
        print("Error: no valid scores provided.")
        sys.exit(1)

    new_col = np.full((M_input.shape[0], 1), np.nan)
    for model, val in scores.items():
        new_col[MODEL_IDX[model], 0] = val

    M_aug = np.hstack([M_input, new_col])
    return M_aug, scores


def main():
    parser = argparse.ArgumentParser(
        description='Predict missing LLM benchmark scores using LogitSVD Blend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Filter predictions to a specific model ID')
    parser.add_argument('--benchmark', '-b', type=str, default=None,
                        help='Filter predictions to a specific benchmark ID')
    parser.add_argument('--format', '-f', type=str, default='csv', choices=['csv', 'json'],
                        help='Output format (default: csv)')
    parser.add_argument('--all', action='store_true',
                        help='Show all predictions (not just missing cells)')
    parser.add_argument('--list-models', action='store_true',
                        help='List all model IDs')
    parser.add_argument('--list-benchmarks', action='store_true',
                        help='List all benchmark IDs')
    parser.add_argument('--add-model', type=str, default=None,
                        help='Name of new model to add')
    parser.add_argument('--add-benchmark', type=str, default=None,
                        help='Name of new benchmark to add')
    parser.add_argument('--scores', '-s', type=str, default=None,
                        help='Known scores as "bench1=val1,bench2=val2" (for --add-model) '
                             'or "model1=val1,model2=val2" (for --add-benchmark)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: stdout)')

    args = parser.parse_args()

    # ── List modes ──
    if args.list_models:
        print(f"{'Model ID':<30s} {'Display Name':<35s} {'Provider':<15s} {'Reasoning':<10s} {'#Scores'}")
        print('-' * 100)
        for i, mid in enumerate(MODEL_IDS):
            n = int(OBSERVED[i].sum())
            print(f"{mid:<30s} {MODEL_NAMES[mid]:<35s} {MODEL_PROVIDERS[i]:<15s} "
                  f"{'Y' if MODEL_REASONING[i] else 'N':<10s} {n}")
        return

    if args.list_benchmarks:
        from evaluation_harness import BENCH_CATS
        print(f"{'Benchmark ID':<30s} {'Display Name':<35s} {'Category':<20s} {'#Models'}")
        print('-' * 95)
        for j, bid in enumerate(BENCH_IDS):
            n = int(OBSERVED[:, j].sum())
            print(f"{bid:<30s} {BENCH_NAMES[bid]:<35s} {BENCH_CATS[j]:<20s} {n}")
        return

    # ── Add-model mode ──
    if args.add_model:
        if not args.scores:
            print("Error: --add-model requires --scores")
            sys.exit(1)
        M_input = M_FULL.copy()
        M_aug, known = add_model_scores(args.add_model, args.scores, M_input)

        # We can't use predict_logit_svd_blend directly because the matrix shape changed.
        # Instead, inject the new model's known scores into an unused row (replace last row
        # temporarily) or use the standard predictor on the augmented matrix.
        # For simplicity, we'll use a workaround: find the most similar existing model
        # and use its predictions, adjusted by the known score differences.

        # Actually, the cleanest approach: modify the global N_MODELS/etc. is fragile.
        # Instead, we compute predictions from the original matrix, then use the known
        # scores to adjust via BenchReg-style local prediction.

        print(f"\nPredictions for new model: {args.add_model}")
        print(f"Known scores: {len(known)}")
        print(f"{'Benchmark':<30s} {'Predicted':>10s}  {'Known':>8s}")
        print('-' * 55)

        # Simple approach: find k nearest models by cosine similarity on known benchmarks
        known_indices = [BENCH_IDX[b] for b in known]
        known_vec = np.array([known[b] for b in known])

        # Compute similarity with all existing models
        sims = []
        for i in range(N_MODELS):
            existing = M_FULL[i, known_indices]
            mask = ~np.isnan(existing)
            if mask.sum() < 2:
                sims.append(-1)
                continue
            # Cosine similarity on shared known benchmarks
            a = known_vec[mask]
            b_vec = existing[mask]
            dot = np.dot(a, b_vec)
            norm = np.linalg.norm(a) * np.linalg.norm(b_vec)
            sims.append(dot / norm if norm > 0 else -1)

        sims = np.array(sims)
        top_k = np.argsort(sims)[-5:][::-1]  # top 5 most similar

        print(f"\nMost similar existing models:")
        for idx in top_k:
            print(f"  {MODEL_NAMES[MODEL_IDS[idx]]} (similarity: {sims[idx]:.3f})")

        # Weighted average of similar models' full predictions
        M_pred = predict_all()
        weights = np.maximum(sims[top_k], 0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(5) / 5

        new_predictions = np.zeros(N_BENCH)
        for j in range(N_BENCH):
            vals = M_pred[top_k, j]
            w = weights.copy()
            valid = np.isfinite(vals)
            if valid.sum() > 0:
                new_predictions[j] = np.average(vals[valid], weights=w[valid])
            else:
                new_predictions[j] = np.nanmean(M_FULL[:, j])

        # Override with known scores
        for bench, val in known.items():
            new_predictions[BENCH_IDX[bench]] = val

        for j in range(N_BENCH):
            bid = BENCH_IDS[j]
            is_known = bid in known
            print(f"  {BENCH_NAMES[bid]:<30s} {new_predictions[j]:>10.1f}  "
                  f"{'(' + str(round(known[bid], 1)) + ')' if is_known else ''}")
        return

    # ── Standard prediction mode ──
    if args.model and args.model not in MODEL_IDX:
        print(f"Error: model '{args.model}' not found.")
        print(f"Use --list-models to see available models.")
        sys.exit(1)
    if args.benchmark and args.benchmark not in BENCH_IDX:
        print(f"Error: benchmark '{args.benchmark}' not found.")
        print(f"Use --list-benchmarks to see available benchmarks.")
        sys.exit(1)

    # Single-cell prediction
    if args.model and args.benchmark:
        i = MODEL_IDX[args.model]
        j = BENCH_IDX[args.benchmark]
        if OBSERVED[i, j]:
            print(f"{MODEL_NAMES[args.model]} on {BENCH_NAMES[args.benchmark]}: "
                  f"{M_FULL[i, j]:.1f} (observed)")
        else:
            M_pred = predict_all()
            pred = M_pred[i, j]
            print(f"{MODEL_NAMES[args.model]} on {BENCH_NAMES[args.benchmark]}: "
                  f"{pred:.1f} (predicted)")
        return

    # Predict and output
    M_pred = predict_all()
    result = format_predictions(M_pred,
                                model_filter=args.model,
                                bench_filter=args.benchmark,
                                only_missing=not args.all,
                                fmt=args.format)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        n = result.count('\n') - 1 if args.format == 'csv' else result.count('"model"')
        print(f"Wrote {n} predictions to {args.output}")
    else:
        print(result)


if __name__ == '__main__':
    main()
