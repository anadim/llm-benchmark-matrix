#!/usr/bin/env python3
"""
Claude-as-Predictor: Use Claude API to predict missing benchmark scores.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 methods/claude_predictor.py [--model MODEL] [--holdout-only] [--full-only]

Runs two experiments:
  1. Holdout evaluation: hide 20% of known cells (seed=42), predict them, score
  2. Full prediction: predict all 2,684 missing cells
"""

import numpy as np
import sys, warnings, os, time, csv, json, re, argparse

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, BENCH_CATS,
    holdout_random_cells,
)
from all_methods import predict_B0, predict_logit_svd_blend, predict_benchreg

import anthropic

# ── Bimodal benchmarks ──
BIMODAL_IDS = ['arc_agi_1', 'arc_agi_2', 'imo_2025', 'usamo_2025', 'matharena_apex_2025']
BIMODAL_IDX = [BENCH_IDS.index(b) for b in BIMODAL_IDS if b in BENCH_IDS]
BIMODAL_THRESHOLD = 10.0

# ── Benchmark scale info ──
NON_PCT_BENCHMARKS = {
    'chatbot_arena_elo': 'Elo rating (~1000-1500)',
    'codeforces_rating': 'rating (~800-2200)',
    'aa_intelligence_index': 'index (0-100+)',
    'aa_lcr': 'index',
    'gdpval_aa': 'index',
}


def format_matrix_for_prompt(M, obs_mask, include_models=None):
    """Format the observed matrix as a compact CSV table for the system prompt.

    Returns the CSV string and a summary of the matrix structure.
    """
    bench_names_short = []
    for bid in BENCH_IDS:
        name = BENCH_NAMES[bid]
        # Shorten long names
        name = name.replace('(Humanity\'s Last Exam)', '(HLE)')
        bench_names_short.append(name)

    lines = []
    header = "model,reasoning," + ",".join(BENCH_IDS)
    lines.append(header)

    model_indices = range(N_MODELS) if include_models is None else include_models

    for i in model_indices:
        mid = MODEL_IDS[i]
        name = MODEL_NAMES[mid]
        reasoning = "Y" if MODEL_REASONING[i] else "N"
        vals = []
        for j in range(N_BENCH):
            if obs_mask[i, j]:
                v = M[i, j]
                # Format compactly
                if v == int(v):
                    vals.append(str(int(v)))
                else:
                    vals.append(f"{v:.1f}")
            else:
                vals.append("?")
        lines.append(f"{name},{reasoning},{','.join(vals)}")

    return "\n".join(lines)


def build_system_prompt(matrix_csv):
    """Build system prompt with full matrix context."""
    bench_info = []
    for j, bid in enumerate(BENCH_IDS):
        cat = BENCH_CATS[j]
        scale = NON_PCT_BENCHMARKS.get(bid, '0-100%')
        bench_info.append(f"  {bid}: {BENCH_NAMES[bid]} [{cat}] scale={scale}")

    return f"""You are a matrix completion system for LLM benchmark scores.

PURPOSE: We have an 83-model × 49-benchmark matrix of evaluation scores, but only 34% of cells are filled (each model is tested on a subset of benchmarks). Your job is to predict the missing scores as accurately as possible. You are being evaluated against statistical matrix completion methods (SVD, ridge regression) that achieve ~6% median error. Try to beat them by leveraging your knowledge of these models and benchmarks.

WHAT YOU KNOW: You likely have prior knowledge about many of these models (GPT-5, Claude Opus 4.6, Gemini 2.5 Pro, DeepSeek-R1, etc.) and benchmarks (GPQA, AIME, MMLU, etc.) from your training data. Use that knowledge AND the patterns in the matrix below.

KEY STRUCTURAL FACTS:
- The matrix is approximately rank-2: "general capability" and "frontier reasoning" explain ~51% of variance.
- Most benchmarks use 0-100% accuracy. Exceptions: chatbot_arena_elo (Elo ~1000-1500), codeforces_rating (~800-2200).
- Some benchmarks are bimodal (ARC-AGI-1/2, IMO 2025, USAMO 2025, MathArena Apex): models either score near 0% or well above 10%. Be decisive — predict 0-2% or 15%+ rather than ambiguous middle values.
- reasoning=Y models score much higher on hard math (AIME, HMMT, HLE) — the gap is +1.5 to +1.8 standard deviations.
- Saturated benchmarks: GSM8K and MATH-500 — most frontier models score 90%+.
- High correlations: GPQA↔LiveCodeBench (r=0.92), LiveCodeBench↔AIME2024 (r=0.95), GPQA↔AIME2024 (r=0.92).

Benchmark definitions:
{chr(10).join(bench_info)}

Full observed matrix ('?' = missing):
{matrix_csv}

RESPONSE FORMAT: Return ONLY valid JSON, no commentary, no markdown fences. Format:
{{"model_name": {{"benchmark_id": score, ...}}, ...}}

Use benchmark IDs (column headers) as keys. Use the exact model names from the matrix."""


def build_prediction_request(batch_models, obs_mask):
    """Build the user prompt for a batch of models."""
    requests = []
    for i in batch_models:
        name = MODEL_NAMES[MODEL_IDS[i]]
        missing_benches = [BENCH_IDS[j] for j in range(N_BENCH) if not obs_mask[i, j]]
        if missing_benches:
            known = []
            for j in range(N_BENCH):
                if obs_mask[i, j]:
                    v = M_FULL[i, j]
                    known.append(f"{BENCH_IDS[j]}={v:.1f}")
            requests.append(f"\n{name} (reasoning={'Y' if MODEL_REASONING[i] else 'N'}):\n"
                          f"  Known: {', '.join(known[:15])}" +
                          (f" + {len(known)-15} more" if len(known) > 15 else "") +
                          f"\n  Predict: {', '.join(missing_benches)}")

    return "Predict the missing benchmark scores for these models. Return valid JSON only.\n" + "\n".join(requests)


def parse_predictions(response_text, batch_models, obs_mask):
    """Parse Claude's JSON response into a predictions dict."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"    WARNING: Could not parse JSON response")
                return {}
        else:
            print(f"    WARNING: No JSON found in response")
            return {}

    # Map model names to indices
    name_to_idx = {}
    for i in batch_models:
        name = MODEL_NAMES[MODEL_IDS[i]]
        name_to_idx[name] = i
        # Also try with minor variations
        name_to_idx[name.strip()] = i

    predictions = {}  # (i, j) -> score

    for model_name, bench_scores in data.items():
        # Find the model index
        idx = name_to_idx.get(model_name)
        if idx is None:
            # Try fuzzy matching
            for known_name, known_idx in name_to_idx.items():
                if known_name.lower() in model_name.lower() or model_name.lower() in known_name.lower():
                    idx = known_idx
                    break
        if idx is None:
            print(f"    WARNING: Could not match model name '{model_name}'")
            continue

        if not isinstance(bench_scores, dict):
            continue

        for bench_id, score in bench_scores.items():
            j = BENCH_IDS.index(bench_id) if bench_id in BENCH_IDS else -1
            if j < 0:
                # Try fuzzy match
                for jj, bid in enumerate(BENCH_IDS):
                    if bid.replace('_', '') == bench_id.replace('_', ''):
                        j = jj
                        break
            if j < 0:
                continue

            try:
                score = float(score)
            except (ValueError, TypeError):
                continue

            predictions[(idx, j)] = score

    return predictions


def clamp_predictions(predictions):
    """Clamp predictions to valid ranges."""
    clamped = {}
    for (i, j), score in predictions.items():
        bid = BENCH_IDS[j]
        if bid == 'chatbot_arena_elo':
            score = np.clip(score, 800, 1600)
        elif bid == 'codeforces_rating':
            score = np.clip(score, 0, 2800)
        elif bid not in NON_PCT_BENCHMARKS:
            score = np.clip(score, 0, 100)
        clamped[(i, j)] = score
    return clamped


def compute_extended_metrics(actual_scores, predicted_scores, test_cells):
    """Compute full metrics matching run_final_eval.py format."""
    a = np.array(actual_scores, dtype=float)
    p = np.array(predicted_scores, dtype=float)

    valid = ~np.isnan(p) & ~np.isnan(a)
    a_v, p_v = a[valid], p[valid]
    cells_v = [c for c, v in zip(test_cells, valid) if v]

    if len(a_v) == 0:
        return {k: np.nan for k in ['medape', 'mae', 'within3', 'within5',
                                     'medape_hi', 'medape_lo', 'bimodal_acc',
                                     'bimodal_n', 'coverage']}

    abs_err = np.abs(p_v - a_v)
    mae = np.mean(abs_err)
    within3 = np.mean(abs_err <= 3.0) * 100
    within5 = np.mean(abs_err <= 5.0) * 100

    nonzero = np.abs(a_v) > 1e-6
    ape = abs_err[nonzero] / np.abs(a_v[nonzero]) * 100
    medape = np.median(ape) if len(ape) > 0 else np.nan

    hi = [abs_err[k] / abs(a_v[k]) * 100 for k in range(len(a_v)) if abs(a_v[k]) > 1e-6 and a_v[k] > 50]
    lo = [abs_err[k] / abs(a_v[k]) * 100 for k in range(len(a_v)) if abs(a_v[k]) > 1e-6 and a_v[k] <= 50]
    medape_hi = np.median(hi) if hi else np.nan
    medape_lo = np.median(lo) if lo else np.nan

    bimodal_correct, bimodal_total = 0, 0
    for idx, (i, j) in enumerate(cells_v):
        if j in BIMODAL_IDX:
            bimodal_total += 1
            if (a_v[idx] > BIMODAL_THRESHOLD) == (p_v[idx] > BIMODAL_THRESHOLD):
                bimodal_correct += 1
    bimodal_acc = (bimodal_correct / bimodal_total * 100) if bimodal_total > 0 else np.nan

    coverage = np.sum(valid) / len(a) * 100

    return {
        'medape': medape, 'mae': mae, 'within3': within3, 'within5': within5,
        'medape_hi': medape_hi, 'medape_lo': medape_lo,
        'bimodal_acc': bimodal_acc, 'bimodal_n': bimodal_total,
        'coverage': coverage, 'n': len(a_v),
    }


def run_claude_predictions(client, model, M_train, obs_mask, target_cells, batch_size=10):
    """Run Claude predictions on target cells, batched by model.

    Args:
        client: Anthropic client
        model: Model ID string
        M_train: Training matrix (with some cells as NaN)
        obs_mask: Boolean mask of observed cells in M_train
        target_cells: List of (i, j) cells to predict
        batch_size: Models per API call

    Returns:
        predictions: dict of (i, j) -> predicted_score
        token_usage: dict with input/output token counts
    """
    # Build system prompt with full observed matrix
    matrix_csv = format_matrix_for_prompt(M_train, obs_mask)
    system_prompt = build_system_prompt(matrix_csv)

    # Group target cells by model
    model_to_cells = {}
    for i, j in target_cells:
        model_to_cells.setdefault(i, []).append(j)

    model_indices = sorted(model_to_cells.keys())

    # Batch models
    batches = []
    for start in range(0, len(model_indices), batch_size):
        batches.append(model_indices[start:start + batch_size])

    all_predictions = {}
    total_input_tokens = 0
    total_output_tokens = 0

    use_thinking = 'opus' in model.lower()

    for batch_idx, batch in enumerate(batches):
        user_prompt = build_prediction_request(batch, obs_mask)

        print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} models, "
              f"{sum(len(model_to_cells[i]) for i in batch)} cells...")

        for attempt in range(2):  # retry once on failure
            try:
                kwargs = {
                    'model': model,
                    'max_tokens': 30000,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                }

                if use_thinking:
                    kwargs['thinking'] = {'type': 'adaptive'}

                # Use streaming to handle long-running requests (required for opus+thinking)
                text = ""
                input_tokens = 0
                output_tokens = 0
                with client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
                    for block in response.content:
                        if block.type == 'text':
                            text += block.text
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # Parse predictions
                preds = parse_predictions(text, batch, obs_mask)
                all_predictions.update(preds)

                n_expected = sum(len(model_to_cells[i]) for i in batch)
                print(f"    Parsed {len(preds)}/{n_expected} predictions "
                      f"(in={response.usage.input_tokens}, out={response.usage.output_tokens})")

                break  # success

            except Exception as e:
                if attempt == 0:
                    print(f"    Retry after error: {e}")
                    time.sleep(5)
                else:
                    print(f"    FAILED: {e}")

        # Rate limit spacing
        if batch_idx < len(batches) - 1:
            time.sleep(2)

    return clamp_predictions(all_predictions), {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description='Claude-as-Predictor for LLM benchmark matrix')
    parser.add_argument('--model', default='claude-opus-4-6',
                       help='Claude model to use')
    parser.add_argument('--holdout-only', action='store_true',
                       help='Only run holdout evaluation')
    parser.add_argument('--full-only', action='store_true',
                       help='Only run full prediction')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Models per API call')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 100)
    print(f"  CLAUDE-AS-PREDICTOR: {args.model}")
    print(f"  Matrix: {N_MODELS}×{N_BENCH}, observed: {OBSERVED.sum()}")
    print("=" * 100)

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 1: HOLDOUT EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    if not args.full_only:
        print("\n" + "=" * 100)
        print("  PART 1: HOLDOUT EVALUATION (random 20%, seed=42)")
        print("=" * 100)

        # Generate holdout — single fold, seed=42
        folds = holdout_random_cells(frac=0.2, n_folds=1, seed=42)
        M_train, test_set = folds[0]
        obs_mask = ~np.isnan(M_train)

        print(f"  Hidden {len(test_set)} cells for evaluation")

        # Run Claude predictions
        print(f"\n  Running Claude ({args.model})...")
        t0 = time.time()
        claude_preds, token_usage = run_claude_predictions(
            client, args.model, M_train, obs_mask, test_set, args.batch_size
        )
        claude_time = time.time() - t0

        # Collect actual vs predicted for test cells
        actual_scores = []
        predicted_scores = []
        test_cells_used = []
        for i, j in test_set:
            actual_scores.append(M_FULL[i, j])
            predicted_scores.append(claude_preds.get((i, j), np.nan))
            test_cells_used.append((i, j))

        claude_metrics = compute_extended_metrics(actual_scores, predicted_scores, test_cells_used)
        claude_metrics['time'] = claude_time
        claude_metrics['tokens'] = token_usage

        # Estimate cost
        if 'sonnet' in args.model.lower():
            cost_per_1m_in, cost_per_1m_out = 3.0, 15.0
        elif 'opus' in args.model.lower():
            cost_per_1m_in, cost_per_1m_out = 15.0, 75.0
        elif 'haiku' in args.model.lower():
            cost_per_1m_in, cost_per_1m_out = 0.25, 1.25
        else:
            cost_per_1m_in, cost_per_1m_out = 3.0, 15.0

        est_cost = (token_usage['input_tokens'] * cost_per_1m_in +
                    token_usage['output_tokens'] * cost_per_1m_out) / 1_000_000
        claude_metrics['est_cost'] = est_cost

        # Run baselines on same holdout for comparison
        print("\n  Running baselines on same holdout...")
        baselines = {
            'Benchmark Mean': predict_B0,
            'BenchReg': predict_benchreg,
            'LogitSVD Blend': predict_logit_svd_blend,
        }

        baseline_results = {}
        for name, fn in baselines.items():
            M_pred = fn(M_train)
            b_actual, b_predicted, b_cells = [], [], []
            for i, j in test_set:
                b_actual.append(M_FULL[i, j])
                b_predicted.append(M_pred[i, j])
                b_cells.append((i, j))
            baseline_results[name] = compute_extended_metrics(b_actual, b_predicted, b_cells)

        # Print comparison
        print("\n" + "=" * 120)
        print("  HOLDOUT EVALUATION RESULTS (random 20%, seed=42)")
        print("=" * 120)

        header = f"  {'Method':<30s} {'MedAPE':>8s} {'MAE':>7s} {'±3pts':>6s} {'±5pts':>6s} {'APE>50':>7s} {'APE≤50':>7s} {'BiAcc':>6s} {'Cov':>5s}"
        print(header)
        print("  " + "─" * 118)

        all_methods = {f'Claude ({args.model.split("-")[1]})': claude_metrics}
        all_methods.update(baseline_results)

        for name, m in sorted(all_methods.items(), key=lambda x: x[1].get('medape', 999)):
            print(f"  {name:<30s} {m['medape']:>7.2f}% {m['mae']:>7.2f} {m['within3']:>5.1f}% "
                  f"{m['within5']:>5.1f}% {m['medape_hi']:>6.2f}% {m['medape_lo']:>6.2f}% "
                  f"{m.get('bimodal_acc', float('nan')):>5.1f}% {m['coverage']:>4.1f}%")

        print(f"\n  Claude tokens: {token_usage['input_tokens']:,} in + {token_usage['output_tokens']:,} out")
        print(f"  Estimated cost: ${est_cost:.2f}")
        print(f"  Time: {claude_time:.1f}s")
        print(f"  Predictions returned: {sum(1 for v in predicted_scores if not np.isnan(v))}/{len(test_set)}")

        results['holdout'] = {
            'claude': claude_metrics,
            'baselines': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.floating) or np.isfinite(vv)}
                         for k, v in baseline_results.items()},
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 2: FULL MATRIX PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    if not args.holdout_only:
        print("\n\n" + "=" * 100)
        print("  PART 2: FULL MATRIX PREDICTION (all 2,684 missing cells)")
        print("=" * 100)

        # All missing cells
        missing_cells = [(i, j) for i in range(N_MODELS) for j in range(N_BENCH) if not OBSERVED[i, j]]
        print(f"  Predicting {len(missing_cells)} cells...")

        t0 = time.time()
        full_preds, full_tokens = run_claude_predictions(
            client, args.model, M_FULL, OBSERVED, missing_cells, args.batch_size
        )
        full_time = time.time() - t0

        # Save predictions CSV
        csv_path = os.path.join(REPO_ROOT, 'results', 'claude_predictions.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'benchmark', 'predicted_score', 'method'])
            for (i, j), score in sorted(full_preds.items()):
                writer.writerow([
                    MODEL_NAMES[MODEL_IDS[i]],
                    BENCH_NAMES[BENCH_IDS[j]],
                    f"{score:.1f}",
                    f'Claude({args.model.split("-")[1]})'
                ])

        n_predicted = len(full_preds)
        print(f"\n  Predicted {n_predicted}/{len(missing_cells)} cells ({n_predicted/len(missing_cells)*100:.1f}%)")
        print(f"  Tokens: {full_tokens['input_tokens']:,} in + {full_tokens['output_tokens']:,} out")

        if 'sonnet' in args.model.lower():
            cost_per_1m_in, cost_per_1m_out = 3.0, 15.0
        elif 'opus' in args.model.lower():
            cost_per_1m_in, cost_per_1m_out = 15.0, 75.0
        else:
            cost_per_1m_in, cost_per_1m_out = 3.0, 15.0
        full_cost = (full_tokens['input_tokens'] * cost_per_1m_in +
                     full_tokens['output_tokens'] * cost_per_1m_out) / 1_000_000

        print(f"  Estimated cost: ${full_cost:.2f}")
        print(f"  Time: {full_time:.1f}s")
        print(f"  Saved to {csv_path}")

        results['full_prediction'] = {
            'n_predicted': n_predicted,
            'n_missing': len(missing_cells),
            'coverage': n_predicted / len(missing_cells) * 100,
            'tokens': full_tokens,
            'est_cost': full_cost,
            'time': full_time,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE RESULTS JSON
    # ══════════════════════════════════════════════════════════════════════════
    json_path = os.path.join(REPO_ROOT, 'results', 'claude_eval.json')

    # Clean up numpy types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        return obj

    results['model'] = args.model
    with open(json_path, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\n  Results saved to {json_path}")

    print("\n" + "=" * 100)
    print("  ALL DONE")
    print("=" * 100)


if __name__ == '__main__':
    main()
