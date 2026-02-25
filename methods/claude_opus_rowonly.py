#!/usr/bin/env python3
"""
Opus 4.6 row-only ablation: does stronger world knowledge help?
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
from all_methods import predict_logit_svd_blend

import anthropic

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
PCT_IDX = set(j for j in range(N_BENCH) if BENCH_IDS[j] not in NON_PCT_BENCHMARKS)

MODEL = 'claude-opus-4-6'


def build_system_prompt_rowonly():
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


def build_user_prompt(batch_models, obs_mask):
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
    return ("For each model below, you are given ONLY that model's known benchmark scores. "
            "Using your knowledge of these models and benchmarks, predict the missing scores. "
            "Return valid JSON only.\n" + "\n".join(requests))


def parse_predictions(response_text, batch_models, obs_mask):
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
            bid = BENCH_IDS[j]
            if bid == 'chatbot_arena_elo':
                score = np.clip(score, 800, 1600)
            elif bid == 'codeforces_rating':
                score = np.clip(score, 0, 2800)
            elif bid not in NON_PCT_BENCHMARKS:
                score = np.clip(score, 0, 100)
            predictions[(idx, j)] = score
    return predictions


def compute_full_metrics(actual, predicted, cells):
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


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    folds = holdout_random_cells(frac=0.2, n_folds=1, seed=42)
    M_train, test_set = folds[0]
    obs_mask = ~np.isnan(M_train)
    print(f"\nHoldout: {len(test_set)} test cells (random 20%, seed=42)")

    # ── Run Opus 4.6 row-only ──
    print("\n" + "="*100)
    print(f"  Opus 4.6 ROW-ONLY (no matrix context)")
    print("="*100)

    sys_prompt = build_system_prompt_rowonly()

    model_to_cells = {}
    for i, j in test_set:
        model_to_cells.setdefault(i, []).append(j)
    model_indices = sorted(model_to_cells.keys())

    batches = []
    for start in range(0, len(model_indices), 10):
        batches.append(model_indices[start:start + 10])

    all_preds = {}
    total_in, total_out = 0, 0
    t0 = time.time()

    for bi, batch in enumerate(batches):
        user_prompt = build_user_prompt(batch, obs_mask)
        n_cells = sum(len(model_to_cells[i]) for i in batch)
        print(f"  Batch {bi+1}/{len(batches)}: {len(batch)} models, {n_cells} cells...")

        for attempt in range(2):
            try:
                kwargs = {
                    'model': MODEL,
                    'max_tokens': 30000,
                    'system': sys_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                    'thinking': {'type': 'adaptive'},
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
                    time.sleep(10)
                else:
                    print(f"    FAILED: {e}")

        if bi < len(batches) - 1:
            time.sleep(3)

    elapsed = time.time() - t0

    # Collect results
    actual, predicted = [], []
    for i, j in test_set:
        actual.append(M_FULL[i, j])
        predicted.append(all_preds.get((i, j), np.nan))

    # ── Baselines ──
    print("\n  Running LogitSVD baseline...")
    M_lsvd = predict_logit_svd_blend(M_train)
    actual_bl, pred_lsvd = [], []
    for i, j in test_set:
        actual_bl.append(M_FULL[i, j])
        pred_lsvd.append(M_lsvd[i, j])

    # ── Metrics: ALL ──
    m_opus = compute_full_metrics(actual, predicted, test_set)
    m_lsvd = compute_full_metrics(actual_bl, pred_lsvd, test_set)

    # Load Sonnet 4.5 results for comparison
    sonnet_results = None
    try:
        with open(os.path.join(REPO_ROOT, 'results', 'claude_analysis.json')) as f:
            sonnet_results = json.load(f)
    except Exception:
        pass

    print(f"\n  ALL BENCHMARKS (n={len(test_set)} cells)")
    print(f"  {'Method':<32s} {'MedAPE':>8s} {'MAE':>7s} {'±3pts':>6s} {'±5pts':>6s} {'BiAcc':>6s} {'Cov':>5s}")
    print("  " + "─"*70)

    if sonnet_results:
        sf = sonnet_results.get('full_matrix', {}).get('all', {})
        sr = sonnet_results.get('row_only', {}).get('all', {})
        if sf:
            print(f"  {'Sonnet 4.5 (full-matrix)':<32s} {sf['medape']:>7.2f}% {sf['mae']:>7.2f} "
                  f"{sf['within3']:>5.1f}% {sf['within5']:>5.1f}% {sf.get('bimodal_acc', 0):>5.1f}% {sf['coverage']:>4.1f}%")
        if sr:
            print(f"  {'Sonnet 4.5 (row-only)':<32s} {sr['medape']:>7.2f}% {sr['mae']:>7.2f} "
                  f"{sr['within3']:>5.1f}% {sr['within5']:>5.1f}% {sr.get('bimodal_acc', 0):>5.1f}% {sr['coverage']:>4.1f}%")

    print(f"  {'Opus 4.6 (row-only)':<32s} {m_opus['medape']:>7.2f}% {m_opus['mae']:>7.2f} "
          f"{m_opus['within3']:>5.1f}% {m_opus['within5']:>5.1f}% {m_opus.get('bimodal_acc', 0):>5.1f}% {m_opus['coverage']:>4.1f}%")
    print(f"  {'LogitSVD Blend':<32s} {m_lsvd['medape']:>7.2f}% {m_lsvd['mae']:>7.2f} "
          f"{m_lsvd['within3']:>5.1f}% {m_lsvd['within5']:>5.1f}% {m_lsvd.get('bimodal_acc', 0):>5.1f}% {m_lsvd['coverage']:>4.1f}%")

    # ── Metrics: PCT-ONLY ──
    a_pct, p_pct_opus, p_pct_lsvd, c_pct = [], [], [], []
    for k, (i, j) in enumerate(test_set):
        if j in PCT_IDX:
            a_pct.append(actual[k])
            p_pct_opus.append(predicted[k])
            p_pct_lsvd.append(pred_lsvd[k])
            c_pct.append((i, j))

    m_opus_pct = compute_full_metrics(a_pct, p_pct_opus, c_pct)
    m_lsvd_pct = compute_full_metrics(a_pct, p_pct_lsvd, c_pct)

    print(f"\n  PERCENTAGE-ONLY (n={len(c_pct)} cells)")
    print(f"  {'Method':<32s} {'MedAPE':>8s} {'MAE':>7s} {'±3pts':>6s} {'±5pts':>6s} {'BiAcc':>6s} {'Cov':>5s}")
    print("  " + "─"*70)

    if sonnet_results:
        # Recompute pct-only from saved data isn't possible, but we printed it earlier
        print(f"  {'Sonnet 4.5 (full-matrix)':<32s}    5.32%    6.32  51.2%  65.2%  89.5% 100.0%  (from prior run)")
        print(f"  {'Sonnet 4.5 (row-only)':<32s}    6.50%    7.91  44.1%  59.0%  78.9% 100.0%  (from prior run)")

    print(f"  {'Opus 4.6 (row-only)':<32s} {m_opus_pct['medape']:>7.2f}% {m_opus_pct['mae']:>7.2f} "
          f"{m_opus_pct['within3']:>5.1f}% {m_opus_pct['within5']:>5.1f}% {m_opus_pct.get('bimodal_acc', 0):>5.1f}% {m_opus_pct['coverage']:>4.1f}%")
    print(f"  {'LogitSVD Blend':<32s} {m_lsvd_pct['medape']:>7.2f}% {m_lsvd_pct['mae']:>7.2f} "
          f"{m_lsvd_pct['within3']:>5.1f}% {m_lsvd_pct['within5']:>5.1f}% {m_lsvd_pct.get('bimodal_acc', 0):>5.1f}% {m_lsvd_pct['coverage']:>4.1f}%")

    # ── Bimodal detail ──
    print(f"\n  BIMODAL PREDICTIONS (Opus 4.6 row-only)")
    print(f"  {'Model':<32s} {'Benchmark':<22s} {'Actual':>7s} {'Opus-RO':>8s} {'Zone':>6s}")
    print("  " + "─"*80)
    for k, (i, j) in enumerate(test_set):
        if j not in BIMODAL_IDX:
            continue
        a_val = M_FULL[i, j]
        p_val = predicted[k]
        zone = "HIGH" if a_val > BIMODAL_THRESHOLD else "LOW"
        p_str = f"{p_val:.1f}" if not np.isnan(p_val) else "N/A"
        ok = "✓" if not np.isnan(p_val) and ((p_val > BIMODAL_THRESHOLD) == (a_val > BIMODAL_THRESHOLD)) else "✗"
        print(f"  {MODEL_NAMES[MODEL_IDS[i]]:<32s} {BENCH_NAMES[BENCH_IDS[j]]:<22s} {a_val:>7.1f} {p_str:>6s} {ok}  {zone:>5s}")

    # ── Cost ──
    cost = (total_in * 15.0 + total_out * 75.0) / 1_000_000
    print(f"\n  Tokens: {total_in:,} in + {total_out:,} out")
    print(f"  Cost: ${cost:.2f}")
    print(f"  Time: {elapsed:.1f}s")

    # ── Save ──
    results = {
        'model': MODEL,
        'mode': 'row-only',
        'all': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                for k, v in m_opus.items()},
        'pct_only': {k: float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else v
                     for k, v in m_opus_pct.items()},
        'tokens': {'input': total_in, 'output': total_out},
        'cost': cost, 'time': elapsed,
    }
    json_path = os.path.join(REPO_ROOT, 'results', 'claude_opus_rowonly.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {json_path}")

    print("\n" + "="*100)
    print("  DONE")
    print("="*100)


if __name__ == '__main__':
    main()
