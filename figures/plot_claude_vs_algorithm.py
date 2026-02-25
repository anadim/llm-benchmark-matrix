#!/usr/bin/env python3
"""
Head-to-head: Claude Sonnet 4.5 vs LogitSVD Blend prediction accuracy
as a function of revealed scores, for specific models.

For each k (number of revealed scores):
  - Hide all of target model's scores, reveal k
  - Run LogitSVD Blend → measure median absolute error on hidden scores
  - Run Claude Sonnet 4.5 with same training matrix → measure same
  - Plot both curves

Two panels: Claude Sonnet 4.6, Gemini 3.1 Pro
"""
import numpy as np
import sys, os, warnings, json, time, re

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, BENCH_CATS,
)
from all_methods import predict_logit_svd_blend

import anthropic

# ── Configuration ──
TARGET_MODELS = ['claude-sonnet-4.6', 'gemini-3.1-pro']
K_VALUES = [0, 1, 2, 3, 5, 7, 10, 15]
N_SEEDS = 3  # average over 3 random orderings
CLAUDE_MODEL = 'claude-sonnet-4-5-20250929'

DISPLAY_NAMES = {
    'claude-sonnet-4.6': 'Claude Sonnet 4.6',
    'gemini-3.1-pro': 'Gemini 3.1 Pro',
}

# ── Benchmark scale info ──
NON_PCT_BENCHMARKS = {
    'chatbot_arena_elo': 'Elo rating (~1000-1500)',
    'codeforces_rating': 'rating (~800-2200)',
    'aa_intelligence_index': 'index (0-100+)',
    'aa_lcr': 'index',
    'gdpval_aa': 'index',
}


def format_matrix_csv(M_train, obs_mask):
    """Format observed matrix as CSV for Claude's system prompt."""
    lines = []
    header = "model,reasoning," + ",".join(BENCH_IDS)
    lines.append(header)
    for i in range(N_MODELS):
        name = MODEL_NAMES[MODEL_IDS[i]]
        reasoning = "Y" if MODEL_REASONING[i] else "N"
        vals = []
        for j in range(N_BENCH):
            if obs_mask[i, j]:
                v = M_train[i, j]
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

PURPOSE: We have an 83-model × 49-benchmark matrix of evaluation scores, but only ~34% of cells are filled. Your job is to predict the missing scores as accurately as possible.

KEY STRUCTURAL FACTS:
- The matrix is approximately rank-2: "general capability" and "frontier reasoning" explain ~51% of variance.
- Most benchmarks use 0-100% accuracy. Exceptions: chatbot_arena_elo (Elo ~1000-1500), codeforces_rating (~800-2200).
- Some benchmarks are bimodal (ARC-AGI-1/2, IMO 2025, USAMO 2025, MathArena Apex): models either score near 0% or well above 10%.
- reasoning=Y models score much higher on hard math (AIME, HMMT, HLE).

Benchmark definitions:
{chr(10).join(bench_info)}

Full observed matrix ('?' = missing):
{matrix_csv}

RESPONSE FORMAT: Return ONLY valid JSON, no commentary, no markdown fences. Format:
{{"model_name": {{"benchmark_id": score, ...}}, ...}}

Use benchmark IDs (column headers) as keys. Use the exact model names from the matrix."""


def parse_claude_response(text, target_model_indices):
    """Parse Claude's JSON response."""
    text = text.strip()
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
            except:
                return {}
        else:
            return {}

    name_to_idx = {MODEL_NAMES[MODEL_IDS[i]]: i for i in target_model_indices}
    predictions = {}
    for model_name, bench_scores in data.items():
        idx = name_to_idx.get(model_name)
        if idx is None:
            for known_name, known_idx in name_to_idx.items():
                if known_name.lower() in model_name.lower() or model_name.lower() in known_name.lower():
                    idx = known_idx
                    break
        if idx is None or not isinstance(bench_scores, dict):
            continue
        for bench_id, score in bench_scores.items():
            j = BENCH_IDS.index(bench_id) if bench_id in BENCH_IDS else -1
            if j < 0:
                for jj, bid in enumerate(BENCH_IDS):
                    if bid.replace('_', '') == bench_id.replace('_', ''):
                        j = jj
                        break
            if j >= 0:
                try:
                    score = float(score)
                    bid = BENCH_IDS[j]
                    if bid == 'chatbot_arena_elo':
                        score = np.clip(score, 800, 1600)
                    elif bid == 'codeforces_rating':
                        score = np.clip(score, 0, 2800)
                    elif bid not in NON_PCT_BENCHMARKS:
                        score = np.clip(score, 0, 100)
                    predictions[(idx, j)] = score
                except:
                    pass
    return predictions


def run_claude_for_models(client, M_train, obs_mask, target_indices, hidden_benchmarks):
    """Run Claude to predict hidden benchmarks for target models."""
    matrix_csv = format_matrix_csv(M_train, obs_mask)
    system_prompt = build_system_prompt(matrix_csv)

    requests = []
    for mi in target_indices:
        name = MODEL_NAMES[MODEL_IDS[mi]]
        missing = [BENCH_IDS[j] for j in hidden_benchmarks[mi]]
        known = []
        for j in range(N_BENCH):
            if obs_mask[mi, j]:
                known.append(f"{BENCH_IDS[j]}={M_train[mi, j]:.1f}")
        requests.append(f"\n{name} (reasoning={'Y' if MODEL_REASONING[mi] else 'N'}):\n"
                       f"  Known: {', '.join(known[:20])}" +
                       (f" + {len(known)-20} more" if len(known) > 20 else "") +
                       f"\n  Predict: {', '.join(missing)}")

    user_prompt = "Predict the missing benchmark scores for these models. Return valid JSON only.\n" + "\n".join(requests)

    for attempt in range(2):
        try:
            text = ""
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=8000,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_prompt}],
            ) as stream:
                response = stream.get_final_message()
                for block in response.content:
                    if block.type == 'text':
                        text += block.text

            preds = parse_claude_response(text, target_indices)
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return preds, tokens
        except Exception as e:
            if attempt == 0:
                print(f"    Retry: {e}")
                time.sleep(5)
            else:
                print(f"    FAILED: {e}")
                return {}, 0

    return {}, 0


def compute_median_abs_error(actual_vals, pred_vals):
    """Median absolute error, ignoring NaN."""
    errors = []
    for a, p in zip(actual_vals, pred_vals):
        if np.isfinite(a) and np.isfinite(p):
            errors.append(abs(a - p))
    return np.median(errors) if errors else np.nan


# ── Main ──
api_key = os.environ.get('ANTHROPIC_API_KEY')
if not api_key:
    print("ERROR: Set ANTHROPIC_API_KEY")
    sys.exit(1)
client = anthropic.Anthropic(api_key=api_key)

# Resolve model indices
target_indices = []
for mid in TARGET_MODELS:
    if mid in MODEL_IDS:
        target_indices.append(MODEL_IDS.index(mid))
    else:
        print(f"WARNING: {mid} not found")

total_tokens = 0
all_results = {}  # model_id -> {k -> {'algo': [errors], 'claude': [errors]}}

for mi in target_indices:
    mid = MODEL_IDS[mi]
    obs_j = np.where(OBSERVED[mi])[0]
    n_obs = len(obs_j)
    max_k = min(max(K_VALUES), n_obs - 2)
    valid_ks = [k for k in K_VALUES if k <= max_k]

    print(f"\n{'='*60}")
    print(f"  {DISPLAY_NAMES.get(mid, mid)}: {n_obs} benchmarks, testing k={valid_ks}")
    print(f"{'='*60}")

    results = {k: {'algo': [], 'claude': []} for k in valid_ks}

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(42 + seed)
        order = obs_j.copy()
        rng.shuffle(order)

        for k in valid_ks:
            # Build training matrix
            M_train = M_FULL.copy()
            obs_mask = OBSERVED.copy()

            if k == 0:
                revealed = set()
            else:
                revealed = set(order[:k])
            hidden = [j for j in obs_j if j not in revealed]

            # Hide all of this model's scores
            for j in obs_j:
                M_train[mi, j] = np.nan
                obs_mask[mi, j] = False
            # Reveal k
            for j in revealed:
                M_train[mi, j] = M_FULL[mi, j]
                obs_mask[mi, j] = True

            # --- Algorithm ---
            M_pred = predict_logit_svd_blend(M_train)
            algo_actual, algo_pred = [], []
            for j in hidden:
                algo_actual.append(M_FULL[mi, j])
                algo_pred.append(M_pred[mi, j])
            algo_err = compute_median_abs_error(algo_actual, algo_pred)
            results[k]['algo'].append(algo_err)

            # --- Claude ---
            hidden_map = {mi: hidden}
            preds, tokens = run_claude_for_models(client, M_train, obs_mask, [mi], hidden_map)
            total_tokens += tokens

            claude_actual, claude_pred = [], []
            for j in hidden:
                claude_actual.append(M_FULL[mi, j])
                claude_pred.append(preds.get((mi, j), np.nan))
            claude_err = compute_median_abs_error(claude_actual, claude_pred)
            results[k]['claude'].append(claude_err)

            print(f"  seed={seed} k={k:>2d}: algo={algo_err:.1f}, claude={claude_err:.1f} ({tokens} tok)")

        time.sleep(1)  # rate limit between seeds

    all_results[mid] = results

# ── Save raw data ──
save_data = {}
for mid, results in all_results.items():
    save_data[mid] = {}
    for k, v in results.items():
        save_data[mid][str(k)] = {
            'algo_mean': float(np.mean(v['algo'])),
            'algo_std': float(np.std(v['algo'])),
            'claude_mean': float(np.mean(v['claude'])),
            'claude_std': float(np.std(v['claude'])),
        }
out_json = os.path.join(REPO_ROOT, 'results', 'claude_vs_algorithm_phase.json')
with open(out_json, 'w') as f:
    json.dump(save_data, f, indent=2)

# ── Plot ──
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

for idx, mi in enumerate(target_indices):
    mid = MODEL_IDS[mi]
    ax = axes[idx]
    results = all_results[mid]

    ks = sorted(results.keys())
    algo_means = [np.mean(results[k]['algo']) for k in ks]
    algo_stds = [np.std(results[k]['algo']) for k in ks]
    claude_means = [np.mean(results[k]['claude']) for k in ks]
    claude_stds = [np.std(results[k]['claude']) for k in ks]

    ax.plot(ks, algo_means, '-o', color='#2CA02C', linewidth=2.5, markersize=6,
            label='LogitSVD Blend', zorder=3)
    ax.fill_between(ks,
                    [m-s for m, s in zip(algo_means, algo_stds)],
                    [m+s for m, s in zip(algo_means, algo_stds)],
                    alpha=0.15, color='#2CA02C')

    ax.plot(ks, claude_means, '-s', color='#E07020', linewidth=2.5, markersize=6,
            label='Claude Sonnet 4.5', zorder=3)
    ax.fill_between(ks,
                    [m-s for m, s in zip(claude_means, claude_stds)],
                    [m+s for m, s in zip(claude_means, claude_stds)],
                    alpha=0.15, color='#E07020')

    ax.set_title(DISPLAY_NAMES.get(mid, mid), fontsize=16, fontweight='bold')
    ax.set_xlabel('Known scores', fontsize=14)
    if idx == 0:
        ax.set_ylabel('Median absolute error (points)', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.set_xlim(-0.5, max(ks) + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.4, linewidth=1)

fig.suptitle('Algorithm vs Claude: who predicts better with fewer scores?', fontsize=17, y=1.01)
plt.tight_layout()

out_path = os.path.join(REPO_ROOT, 'figures', 'claude_vs_algorithm.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to {out_path}")
print(f"Total API tokens: {total_tokens:,}")

# Estimate cost
est_cost = total_tokens * 3.0 / 1_000_000  # rough average $/token for sonnet
print(f"Estimated cost: ~${total_tokens * 5 / 1_000_000:.2f}")
plt.close()
