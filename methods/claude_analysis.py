#!/usr/bin/env python3
"""
Claude Predictor: Deep Analysis & Row-Only Ablation
====================================================
Three investigations:
  1. MAE breakdown: percentage vs non-percentage benchmarks
  2. Detailed cell-level comparison: Claude vs LogitSVD
  3. Row-only ablation: does Claude need the full matrix?

Usage:
    ANTHROPIC_API_KEY=sk-... python3 methods/claude_analysis.py
"""

import numpy as np
import sys, warnings, os, time, json, re

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, BENCH_CATS,
    holdout_random_cells,
)
from all_methods import predict_logit_svd_blend, predict_benchreg, predict_B0

import anthropic

# ── Constants ──
BIMODAL_IDS = ['arc_agi_1', 'arc_agi_2', 'imo_2025', 'usamo_2025', 'matharena_apex_2025']
BIMODAL_IDX = [BENCH_IDS.index(b) for b in BIMODAL_IDS if b in BENCH_IDS]
BIMODAL_THRESHOLD = 10.0

NON_PCT_BENCHMARKS = {
    'chatbot_arena_elo': 'Elo (~1000-1500)',
    'codeforces_rating': 'rating (~800-2200)',
    'aa_intelligence_index': 'index',
    'aa_lcr': 'index',
    'gdpval_aa': 'index',
}

# ── Identify percentage vs non-percentage benchmark indices ──
PCT_IDX = [j for j in range(N_BENCH) if BENCH_IDS[j] not in NON_PCT_BENCHMARKS]
NONPCT_IDX = [j for j in range(N_BENCH) if BENCH_IDS[j] in NON_PCT_BENCHMARKS]

MODEL = 'claude-sonnet-4-5-20250929'


def build_system_prompt_full(matrix_csv):
    """System prompt with full matrix context (same as original)."""
    bench_info = []
    for j, bid in enumerate(BENCH_IDS):
        cat = BENCH_CATS[j]
        scale = NON_PCT_BENCHMARKS.get(bid, '0-100%')
        bench_info.append(f"  {bid}: {BENCH_NAMES[bid]} [{cat}] scale={scale}")

    return f"""You are a matrix completion system for LLM benchmark scores.

PURPOSE: We have an 83-model x 49-benchmark matrix of evaluation scores, but only 34% of cells are filled. Your job is to predict the missing scores as accurately as possible. You are being evaluated against statistical matrix completion methods (SVD, ridge regression) that achieve ~6% median error. Try to beat them.

KEY STRUCTURAL FACTS:
- The matrix is approximately rank-2: "general capability" and "frontier reasoning" explain ~51% of variance.
- Most benchmarks use 0-100% accuracy. Exceptions: chatbot_arena_elo (Elo ~1000-1500), codeforces_rating (~800-2200).
- Some benchmarks are bimodal (ARC-AGI-1/2, IMO 2025, USAMO 2025, MathArena Apex): models either score near 0% or well above 10%. Be decisive.
- reasoning=Y models score much higher on hard math (AIME, HMMT, HLE).
- Saturated benchmarks: GSM8K and MATH-500 — most frontier models score 90%+.

Benchmark definitions:
{chr(10).join(bench_info)}

Full observed matrix ('?' = missing):
{matrix_csv}

RESPONSE FORMAT: Return ONLY valid JSON, no commentary, no markdown fences. Format:
{{"model_name": {{"benchmark_id": score, ...}}, ...}}

Use benchmark IDs (column headers) as keys. Use the exact model names from the matrix."""


def build_system_prompt_rowonly():
    """System prompt for row-only ablation (NO matrix context)."""
    bench_info = []
    for j, bid in enumerate(BENCH_IDS):
        cat = BENCH_CATS[j]
        scale = NON_PCT_BENCHMARKS.get(bid, '0-100%')
        bench_info.append(f"  {bid}: {BENCH_NAMES[bid]} [{cat}] scale={scale}")

    return f"""You are a benchmark score predictor for large language models.

PURPOSE: Given a model's known benchmark scores, predict its scores on other benchmarks. You should use your knowledge of these models and benchmarks to make predictions. You are being evaluated against statistical methods that achieve ~6% median error.

KEY FACTS ABOUT BENCHMARKS:
- Most benchmarks use 0-100% accuracy. Exceptions: chatbot_arena_elo (Elo ~1000-1500), codeforces_rating (~800-2200).
- Some benchmarks are bimodal (ARC-AGI-1/2, IMO 2025, USAMO 2025, MathArena Apex): models either score near 0% or well above 10%. Be decisive.
- reasoning=Y models score much higher on hard math (AIME, HMMT, HLE).
- Saturated benchmarks: GSM8K and MATH-500 — most frontier models score 90%+.

Benchmark definitions:
{chr(10).join(bench_info)}

IMPORTANT: You do NOT have access to other models' scores. Use your world knowledge of these models and the relationships between benchmarks to predict.

RESPONSE FORMAT: Return ONLY valid JSON, no commentary, no markdown fences. Format:
{{"model_name": {{"benchmark_id": score, ...}}, ...}}"""


def format_matrix_csv(M, obs_mask):
    """Format full matrix as CSV."""
    lines = []
    header = "model,reasoning," + ",".join(BENCH_IDS)
    lines.append(header)
    for i in range(N_MODELS):
        mid = MODEL_IDS[i]
        name = MODEL_NAMES[mid]
        reasoning = "Y" if MODEL_REASONING[i] else "N"
        vals = []
        for j in range(N_BENCH):
            if obs_mask[i, j]:
                v = M[i, j]
                if v == int(v):
                    vals.append(str(int(v)))
                else:
                    vals.append(f"{v:.1f}")
            else:
                vals.append("?")
        lines.append(f"{name},{reasoning},{','.join(vals)}")
    return "\n".join(lines)


def build_user_prompt_full(batch_models, obs_mask):
    """User prompt for full-matrix mode."""
    requests = []
    for i in batch_models:
        name = MODEL_NAMES[MODEL_IDS[i]]
        missing = [BENCH_IDS[j] for j in range(N_BENCH) if not obs_mask[i, j]]
        if missing:
            known = []
            for j in range(N_BENCH):
                if obs_mask[i, j]:
                    known.append(f"{BENCH_IDS[j]}={M_FULL[i,j]:.1f}")
            requests.append(f"\n{name} (reasoning={'Y' if MODEL_REASONING[i] else 'N'}):\n"
                          f"  Known: {', '.join(known[:15])}" +
                          (f" + {len(known)-15} more" if len(known) > 15 else "") +
                          f"\n  Predict: {', '.join(missing)}")
    return "Predict the missing benchmark scores for these models. Return valid JSON only.\n" + "\n".join(requests)


def build_user_prompt_rowonly(batch_models, obs_mask):
    """User prompt for row-only mode — each model sees ONLY its own known scores."""
    requests = []
    for i in batch_models:
        name = MODEL_NAMES[MODEL_IDS[i]]
        missing = [BENCH_IDS[j] for j in range(N_BENCH) if not obs_mask[i, j]]
        if missing:
            known = []
            for j in range(N_BENCH):
                if obs_mask[i, j]:
                    known.append(f"{BENCH_IDS[j]}={M_FULL[i,j]:.1f}")
            requests.append(
                f"\n{name} (reasoning={'Y' if MODEL_REASONING[i] else 'N'}):\n"
                f"  Known scores: {', '.join(known)}\n"
                f"  Predict: {', '.join(missing)}")
    return "For each model below, you are given ONLY that model's known benchmark scores. Using your knowledge of these models and benchmarks, predict the missing scores. Return valid JSON only.\n" + "\n".join(requests)


def parse_predictions(response_text, batch_models, obs_mask):
    """Parse Claude's JSON response into predictions dict."""
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    name_to_idx = {}
    for i in batch_models:
        name = MODEL_NAMES[MODEL_IDS[i]]
        name_to_idx[name] = i
        name_to_idx[name.strip()] = i

    predictions = {}
    for model_name, bench_scores in data.items():
        idx = name_to_idx.get(model_name)
        if idx is None:
            for known_name, known_idx in name_to_idx.items():
                if known_name.lower() in model_name.lower() or model_name.lower() in known_name.lower():
                    idx = known_idx
                    break
        if idx is None:
            continue
        if not isinstance(bench_scores, dict):
            continue
        for bench_id, score in bench_scores.items():
            j = BENCH_IDS.index(bench_id) if bench_id in BENCH_IDS else -1
            if j < 0:
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
            # Clamp
            bid = BENCH_IDS[j]
            if bid == 'chatbot_arena_elo':
                score = np.clip(score, 800, 1600)
            elif bid == 'codeforces_rating':
                score = np.clip(score, 0, 2800)
            elif bid not in NON_PCT_BENCHMARKS:
                score = np.clip(score, 0, 100)
            predictions[(idx, j)] = score
    return predictions


def run_predictions(client, system_prompt, user_prompt_fn, M_train, obs_mask, target_cells, label="", batch_size=10):
    """Run Claude predictions with given system/user prompt style."""
    model_to_cells = {}
    for i, j in target_cells:
        model_to_cells.setdefault(i, []).append(j)
    model_indices = sorted(model_to_cells.keys())

    batches = []
    for start in range(0, len(model_indices), batch_size):
        batches.append(model_indices[start:start + batch_size])

    all_preds = {}
    total_in, total_out = 0, 0

    for bi, batch in enumerate(batches):
        user_prompt = user_prompt_fn(batch, obs_mask)
        n_cells = sum(len(model_to_cells[i]) for i in batch)
        print(f"  [{label}] Batch {bi+1}/{len(batches)}: {len(batch)} models, {n_cells} cells...")

        for attempt in range(2):
            try:
                kwargs = {
                    'model': MODEL,
                    'max_tokens': 30000,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                }
                text = ""
                with client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
                    for block in response.content:
                        if block.type == 'text':
                            text += block.text
                    total_in += response.usage.input_tokens
                    total_out += response.usage.output_tokens

                preds = parse_predictions(text, batch, obs_mask)
                all_preds.update(preds)
                print(f"    Parsed {len(preds)}/{n_cells} (in={response.usage.input_tokens}, out={response.usage.output_tokens})")
                break
            except Exception as e:
                if attempt == 0:
                    print(f"    Retry: {e}")
                    time.sleep(5)
                else:
                    print(f"    FAILED: {e}")

        if bi < len(batches) - 1:
            time.sleep(2)

    return all_preds, {'input_tokens': total_in, 'output_tokens': total_out}


def compute_metrics(actual, predicted, cells, label="", filter_idx=None):
    """Compute metrics, optionally filtering to certain benchmark indices."""
    a_list, p_list = [], []
    for k, (i, j) in enumerate(cells):
        if filter_idx is not None and j not in filter_idx:
            continue
        if np.isnan(predicted[k]) or np.isnan(actual[k]):
            continue
        a_list.append(actual[k])
        p_list.append(predicted[k])

    if len(a_list) == 0:
        return {'n': 0, 'medape': np.nan, 'mae': np.nan, 'within3': np.nan, 'within5': np.nan}

    a = np.array(a_list)
    p = np.array(p_list)
    ae = np.abs(p - a)
    mae = np.mean(ae)
    within3 = np.mean(ae <= 3.0) * 100
    within5 = np.mean(ae <= 5.0) * 100

    nonzero = np.abs(a) > 1e-6
    ape = ae[nonzero] / np.abs(a[nonzero]) * 100
    medape = np.median(ape) if len(ape) > 0 else np.nan

    # Bimodal
    bimodal_ok, bimodal_n = 0, 0
    for k2 in range(len(a)):
        # find original cell
        pass  # handled below

    return {'n': len(a), 'medape': medape, 'mae': mae, 'within3': within3, 'within5': within5}


def compute_full_metrics(actual, predicted, cells):
    """Compute all metrics including bimodal."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    valid = ~np.isnan(p) & ~np.isnan(a)
    a_v, p_v = a[valid], p[valid]
    cells_v = [c for c, v in zip(cells, valid) if v]

    if len(a_v) == 0:
        return {k: np.nan for k in ['medape','mae','within3','within5','medape_hi','medape_lo','bimodal_acc','coverage','n']}

    ae = np.abs(p_v - a_v)
    mae = np.mean(ae)
    within3 = np.mean(ae <= 3.0) * 100
    within5 = np.mean(ae <= 5.0) * 100

    nonzero = np.abs(a_v) > 1e-6
    ape = ae[nonzero] / np.abs(a_v[nonzero]) * 100
    medape = np.median(ape) if len(ape) > 0 else np.nan

    hi = [ae[k]/abs(a_v[k])*100 for k in range(len(a_v)) if abs(a_v[k])>1e-6 and a_v[k]>50]
    lo = [ae[k]/abs(a_v[k])*100 for k in range(len(a_v)) if abs(a_v[k])>1e-6 and a_v[k]<=50]
    medape_hi = np.median(hi) if hi else np.nan
    medape_lo = np.median(lo) if lo else np.nan

    bimodal_ok, bimodal_n = 0, 0
    for idx, (i, j) in enumerate(cells_v):
        if j in BIMODAL_IDX:
            bimodal_n += 1
            if (a_v[idx] > BIMODAL_THRESHOLD) == (p_v[idx] > BIMODAL_THRESHOLD):
                bimodal_ok += 1
    bimodal_acc = (bimodal_ok / bimodal_n * 100) if bimodal_n > 0 else np.nan

    coverage = np.sum(valid) / len(a) * 100

    return {
        'medape': medape, 'mae': mae, 'within3': within3, 'within5': within5,
        'medape_hi': medape_hi, 'medape_lo': medape_lo,
        'bimodal_acc': bimodal_acc, 'bimodal_n': bimodal_n,
        'coverage': coverage, 'n': int(np.sum(valid)),
    }


def print_metrics_row(name, m, extra=""):
    """Print a single method's metrics row."""
    print(f"  {name:<28s} {m['medape']:>7.2f}% {m['mae']:>7.2f}  {m['within3']:>5.1f}% "
          f"{m['within5']:>5.1f}%  {m.get('bimodal_acc', float('nan')):>5.1f}%  "
          f"{m['coverage']:>5.1f}% {m['n']:>4d}  {extra}")


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Generate holdout (same seed as original) ──
    folds = holdout_random_cells(frac=0.2, n_folds=1, seed=42)
    M_train, test_set = folds[0]
    obs_mask = ~np.isnan(M_train)
    print(f"\nHoldout: {len(test_set)} test cells (random 20%, seed=42)")

    # Count pct vs non-pct test cells
    pct_cells = [(i,j) for i,j in test_set if j in PCT_IDX]
    nonpct_cells = [(i,j) for i,j in test_set if j in NONPCT_IDX]
    print(f"  Percentage-scale cells: {len(pct_cells)}")
    print(f"  Non-percentage cells:   {len(nonpct_cells)}")
    for i,j in nonpct_cells:
        print(f"    {MODEL_NAMES[MODEL_IDS[i]]:>35s}  {BENCH_NAMES[BENCH_IDS[j]]:<25s}  actual={M_FULL[i,j]:.1f}")

    # ══════════════════════════════════════════════════════════════════
    # RUN 1: Claude with FULL MATRIX context
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*100)
    print("  RUN 1: Claude Sonnet 4.5 — FULL MATRIX context")
    print("="*100)

    matrix_csv = format_matrix_csv(M_train, obs_mask)
    sys_full = build_system_prompt_full(matrix_csv)

    t0 = time.time()
    preds_full, tokens_full = run_predictions(
        client, sys_full, build_user_prompt_full, M_train, obs_mask, test_set,
        label="full-matrix", batch_size=10
    )
    time_full = time.time() - t0

    # Collect results
    actual_full, pred_full = [], []
    for i, j in test_set:
        actual_full.append(M_FULL[i, j])
        pred_full.append(preds_full.get((i, j), np.nan))

    # ══════════════════════════════════════════════════════════════════
    # RUN 2: Claude with ROW-ONLY context (no matrix)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*100)
    print("  RUN 2: Claude Sonnet 4.5 — ROW-ONLY context (no matrix)")
    print("="*100)

    sys_row = build_system_prompt_rowonly()

    t0 = time.time()
    preds_row, tokens_row = run_predictions(
        client, sys_row, build_user_prompt_rowonly, M_train, obs_mask, test_set,
        label="row-only", batch_size=10
    )
    time_row = time.time() - t0

    actual_row, pred_row = [], []
    for i, j in test_set:
        actual_row.append(M_FULL[i, j])
        pred_row.append(preds_row.get((i, j), np.nan))

    # ══════════════════════════════════════════════════════════════════
    # BASELINES
    # ══════════════════════════════════════════════════════════════════
    print("\n  Running baselines...")
    M_logitsvd = predict_logit_svd_blend(M_train)
    M_benchreg = predict_benchreg(M_train)
    M_bmean = predict_B0(M_train)

    actual_bl, pred_lsvd, pred_breg, pred_bmean = [], [], [], []
    for i, j in test_set:
        actual_bl.append(M_FULL[i, j])
        pred_lsvd.append(M_logitsvd[i, j])
        pred_breg.append(M_benchreg[i, j])
        pred_bmean.append(M_bmean[i, j])

    # ══════════════════════════════════════════════════════════════════
    # PART 1: MAE BREAKDOWN BY BENCHMARK SCALE
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("  PART 1: MAE BREAKDOWN BY BENCHMARK SCALE")
    print("="*100)

    methods = {
        'Claude (full-matrix)': (actual_full, pred_full),
        'Claude (row-only)':    (actual_row, pred_row),
        'LogitSVD Blend':       (actual_bl, pred_lsvd),
        'BenchReg':             (actual_bl, pred_breg),
        'Benchmark Mean':       (actual_bl, pred_bmean),
    }

    set_pct = set(PCT_IDX)
    set_nonpct = set(NONPCT_IDX)

    print(f"\n  {'Method':<28s} {'ALL':>10s}  {'Pct-only':>10s}  {'Non-pct':>10s}  | {'MedAPE-All':>10s} {'MedAPE-Pct':>10s}")
    print("  " + "─"*95)

    for name, (act, pred) in methods.items():
        # All
        m_all = compute_full_metrics(act, pred, test_set)
        # Pct only
        a_pct, p_pct, c_pct = [], [], []
        a_npct, p_npct, c_npct = [], [], []
        for k, (i, j) in enumerate(test_set):
            if j in set_pct:
                a_pct.append(act[k]); p_pct.append(pred[k]); c_pct.append((i,j))
            else:
                a_npct.append(act[k]); p_npct.append(pred[k]); c_npct.append((i,j))

        m_pct = compute_full_metrics(a_pct, p_pct, c_pct) if c_pct else {'mae': np.nan, 'medape': np.nan}
        m_npct = compute_full_metrics(a_npct, p_npct, c_npct) if c_npct else {'mae': np.nan, 'medape': np.nan}

        print(f"  {name:<28s} MAE={m_all['mae']:>6.2f}   MAE={m_pct['mae']:>6.2f}   MAE={m_npct['mae']:>6.2f}"
              f"   | MedAPE={m_all['medape']:>5.2f}%  MedAPE={m_pct['medape']:>5.2f}%")

    # Show non-pct predictions detail
    print(f"\n  Non-percentage benchmark predictions:")
    print(f"  {'Model':<35s} {'Benchmark':<22s} {'Actual':>7s} {'Claude-FM':>9s} {'Claude-RO':>9s} {'LogitSVD':>9s}")
    print("  " + "─"*95)
    for k, (i, j) in enumerate(test_set):
        if j not in set_nonpct:
            continue
        cf = pred_full[k] if not np.isnan(pred_full[k]) else -1
        cr = pred_row[k] if not np.isnan(pred_row[k]) else -1
        ls = pred_lsvd[k] if not np.isnan(pred_lsvd[k]) else -1
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<35s} {BENCH_NAMES[BENCH_IDS[j]]:<22s} "
              f"{M_FULL[i,j]:>7.1f} {cf:>9.1f} {cr:>9.1f} {ls:>9.1f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 1b: PERCENTAGE-ONLY COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n  PERCENTAGE-ONLY COMPARISON (n={len(pct_cells)} cells)")
    print(f"  {'Method':<28s} {'MedAPE':>8s} {'MAE':>7s} {'±3pts':>6s} {'±5pts':>6s} {'BiAcc':>6s} {'Cov':>5s} {'n':>4s}")
    print("  " + "─"*70)

    for name, (act, pred) in methods.items():
        a_pct, p_pct, c_pct = [], [], []
        for k, (i, j) in enumerate(test_set):
            if j in set_pct:
                a_pct.append(act[k]); p_pct.append(pred[k]); c_pct.append((i,j))
        m = compute_full_metrics(a_pct, p_pct, c_pct)
        print_metrics_row(name, m)

    # ══════════════════════════════════════════════════════════════════
    # PART 2: DETAILED CELL-LEVEL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("  PART 2: CELL-LEVEL COMPARISON (Claude full-matrix vs LogitSVD)")
    print("="*100)

    # Build per-cell comparison
    cell_data = []
    for k, (i, j) in enumerate(test_set):
        a_val = actual_full[k]
        c_val = pred_full[k]
        l_val = pred_lsvd[k]
        if np.isnan(c_val) or np.isnan(l_val) or np.isnan(a_val):
            continue
        c_err = abs(c_val - a_val)
        l_err = abs(l_val - a_val)
        cell_data.append({
            'i': i, 'j': j,
            'model': MODEL_NAMES[MODEL_IDS[i]],
            'bench': BENCH_NAMES[BENCH_IDS[j]],
            'bench_id': BENCH_IDS[j],
            'actual': a_val,
            'claude': c_val,
            'logitsvd': l_val,
            'claude_err': c_err,
            'logitsvd_err': l_err,
            'advantage': l_err - c_err,  # positive = Claude better
        })

    # Top 10 where Claude beats LogitSVD
    by_claude_better = sorted(cell_data, key=lambda x: -x['advantage'])
    print(f"\n  TOP 10: Claude beats LogitSVD by largest margin")
    print(f"  {'Model':<32s} {'Benchmark':<22s} {'Actual':>7s} {'Claude':>8s} {'LogSVD':>8s} {'C_err':>6s} {'L_err':>6s} {'Adv':>6s}")
    print("  " + "─"*95)
    for d in by_claude_better[:10]:
        print(f"  {d['model']:<32s} {d['bench']:<22s} {d['actual']:>7.1f} {d['claude']:>8.1f} {d['logitsvd']:>8.1f} "
              f"{d['claude_err']:>6.1f} {d['logitsvd_err']:>6.1f} {d['advantage']:>+6.1f}")

    # Top 10 where LogitSVD beats Claude
    by_logit_better = sorted(cell_data, key=lambda x: x['advantage'])
    print(f"\n  TOP 10: LogitSVD beats Claude by largest margin")
    print(f"  {'Model':<32s} {'Benchmark':<22s} {'Actual':>7s} {'Claude':>8s} {'LogSVD':>8s} {'C_err':>6s} {'L_err':>6s} {'Adv':>6s}")
    print("  " + "─"*95)
    for d in by_logit_better[:10]:
        print(f"  {d['model']:<32s} {d['bench']:<22s} {d['actual']:>7.1f} {d['claude']:>8.1f} {d['logitsvd']:>8.1f} "
              f"{d['claude_err']:>6.1f} {d['logitsvd_err']:>6.1f} {d['advantage']:>+6.1f}")

    # Bimodal benchmark predictions
    print(f"\n  BIMODAL BENCHMARK PREDICTIONS")
    bimodal_names = {BENCH_IDS[j]: BENCH_NAMES[BENCH_IDS[j]] for j in BIMODAL_IDX}
    print(f"  {'Model':<32s} {'Benchmark':<22s} {'Actual':>7s} {'Claude-FM':>9s} {'Claude-RO':>9s} {'LogSVD':>8s} {'Zone':>8s}")
    print("  " + "─"*100)
    for k, (i, j) in enumerate(test_set):
        if j not in BIMODAL_IDX:
            continue
        a_val = M_FULL[i, j]
        cf = pred_full[k]
        cr = pred_row[k]
        ls = pred_lsvd[k]
        zone = "HIGH" if a_val > BIMODAL_THRESHOLD else "LOW"
        cf_str = f"{cf:.1f}" if not np.isnan(cf) else "N/A"
        cr_str = f"{cr:.1f}" if not np.isnan(cr) else "N/A"
        ls_str = f"{ls:.1f}" if not np.isnan(ls) else "N/A"
        # Color: check if prediction is on right side of threshold
        cf_ok = "✓" if not np.isnan(cf) and ((cf > BIMODAL_THRESHOLD) == (a_val > BIMODAL_THRESHOLD)) else "✗"
        cr_ok = "✓" if not np.isnan(cr) and ((cr > BIMODAL_THRESHOLD) == (a_val > BIMODAL_THRESHOLD)) else "✗"
        ls_ok = "✓" if not np.isnan(ls) and ((ls > BIMODAL_THRESHOLD) == (a_val > BIMODAL_THRESHOLD)) else "✗"
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<32s} {BENCH_NAMES[BENCH_IDS[j]]:<22s} {a_val:>7.1f} "
              f"{cf_str:>7s} {cf_ok}  {cr_str:>7s} {cr_ok}  {ls_str:>6s} {ls_ok}  {zone:>5s}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: FULL-MATRIX vs ROW-ONLY COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("  PART 3: FULL-MATRIX vs ROW-ONLY ABLATION")
    print("="*100)

    m_full = compute_full_metrics(actual_full, pred_full, test_set)
    m_row = compute_full_metrics(actual_row, pred_row, test_set)
    m_lsvd = compute_full_metrics(actual_bl, pred_lsvd, test_set)
    m_breg = compute_full_metrics(actual_bl, pred_breg, test_set)
    m_bmean = compute_full_metrics(actual_bl, pred_bmean, test_set)

    print(f"\n  ALL BENCHMARKS (n={len(test_set)} cells)")
    print(f"  {'Method':<28s} {'MedAPE':>8s} {'MAE':>7s} {'±3pts':>6s} {'±5pts':>6s} {'BiAcc':>6s} {'Cov':>5s} {'n':>4s}")
    print("  " + "─"*70)
    for name, m in sorted([
        ('Claude (full-matrix)', m_full),
        ('Claude (row-only)', m_row),
        ('LogitSVD Blend', m_lsvd),
        ('BenchReg', m_breg),
        ('Benchmark Mean', m_bmean),
    ], key=lambda x: x[1].get('medape', 999)):
        print_metrics_row(name, m)

    # Pct-only version too
    print(f"\n  PERCENTAGE-ONLY BENCHMARKS (n={len(pct_cells)} cells)")
    print(f"  {'Method':<28s} {'MedAPE':>8s} {'MAE':>7s} {'±3pts':>6s} {'±5pts':>6s} {'BiAcc':>6s} {'Cov':>5s} {'n':>4s}")
    print("  " + "─"*70)
    for name, (act, pred) in [
        ('Claude (full-matrix)', (actual_full, pred_full)),
        ('Claude (row-only)', (actual_row, pred_row)),
        ('LogitSVD Blend', (actual_bl, pred_lsvd)),
        ('BenchReg', (actual_bl, pred_breg)),
        ('Benchmark Mean', (actual_bl, pred_bmean)),
    ]:
        a_pct, p_pct, c_pct = [], [], []
        for k, (i, j) in enumerate(test_set):
            if j in set_pct:
                a_pct.append(act[k]); p_pct.append(pred[k]); c_pct.append((i,j))
        m = compute_full_metrics(a_pct, p_pct, c_pct)
        print_metrics_row(name, m)

    # Token/cost summary
    cost_full = (tokens_full['input_tokens'] * 3.0 + tokens_full['output_tokens'] * 15.0) / 1_000_000
    cost_row = (tokens_row['input_tokens'] * 3.0 + tokens_row['output_tokens'] * 15.0) / 1_000_000

    print(f"\n  Token/Cost Summary:")
    print(f"  {'Mode':<20s} {'Input':>10s} {'Output':>10s} {'Cost':>8s} {'Time':>8s}")
    print(f"  {'Full-matrix':<20s} {tokens_full['input_tokens']:>10,} {tokens_full['output_tokens']:>10,} ${cost_full:>6.2f} {time_full:>6.1f}s")
    print(f"  {'Row-only':<20s} {tokens_row['input_tokens']:>10,} {tokens_row['output_tokens']:>10,} ${cost_row:>6.2f} {time_row:>6.1f}s")

    # ── Head-to-head: where does row-only differ most from full-matrix? ──
    print(f"\n  Per-cell: full-matrix vs row-only biggest differences:")
    diffs = []
    for k, (i, j) in enumerate(test_set):
        cf = pred_full[k]
        cr = pred_row[k]
        a_val = actual_full[k]
        if np.isnan(cf) or np.isnan(cr) or np.isnan(a_val):
            continue
        diff = abs(cf - cr)
        cf_err = abs(cf - a_val)
        cr_err = abs(cr - a_val)
        diffs.append({
            'model': MODEL_NAMES[MODEL_IDS[i]], 'bench': BENCH_NAMES[BENCH_IDS[j]],
            'actual': a_val, 'full': cf, 'row': cr, 'diff': diff,
            'full_err': cf_err, 'row_err': cr_err,
            'full_wins': cf_err < cr_err,
        })

    diffs.sort(key=lambda x: -x['diff'])
    print(f"  {'Model':<32s} {'Benchmark':<22s} {'Actual':>7s} {'Full':>7s} {'Row':>7s} {'F_err':>6s} {'R_err':>6s} {'Winner':>8s}")
    print("  " + "─"*95)
    for d in diffs[:15]:
        winner = "Full" if d['full_wins'] else "Row"
        print(f"  {d['model']:<32s} {d['bench']:<22s} {d['actual']:>7.1f} {d['full']:>7.1f} {d['row']:>7.1f} "
              f"{d['full_err']:>6.1f} {d['row_err']:>6.1f} {winner:>8s}")

    full_win_count = sum(1 for d in diffs if d['full_wins'])
    print(f"\n  Full-matrix wins {full_win_count}/{len(diffs)} cells ({full_win_count/len(diffs)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════
    results = {
        'model': MODEL,
        'holdout': {
            'seed': 42, 'frac': 0.2, 'n_cells': len(test_set),
            'n_pct': len(pct_cells), 'n_nonpct': len(nonpct_cells),
        },
        'full_matrix': {
            'all': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                    for k, v in m_full.items()},
            'tokens': tokens_full, 'cost': cost_full, 'time': time_full,
        },
        'row_only': {
            'all': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                    for k, v in m_row.items()},
            'tokens': tokens_row, 'cost': cost_row, 'time': time_row,
        },
        'baselines': {
            'LogitSVD Blend': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                               for k, v in m_lsvd.items()},
            'BenchReg': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                         for k, v in m_breg.items()},
            'Benchmark Mean': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                               for k, v in m_bmean.items()},
        },
    }

    # Clean numpy types
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
        return obj

    json_path = os.path.join(REPO_ROOT, 'results', 'claude_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(clean(results), f, indent=2)
    print(f"\n  Results saved to {json_path}")

    print("\n" + "="*100)
    print("  ALL DONE")
    print("="*100)


if __name__ == '__main__':
    main()
