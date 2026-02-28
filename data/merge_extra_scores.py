#!/usr/bin/env python3
"""
Merge extra_scores_{1,2,3}.py into build_benchmark_matrix.py DATA list.
Uses comprehensive alias mapping and deduplication.

Key fix: computes 'existing' from BASE data only (excluding previous auto-merged
entries), so all valid extra_scores entries get re-added each time.
"""
import sys, re, importlib, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))

from build_benchmark_matrix import MODELS, BENCHMARKS

model_set = set(m[0] for m in MODELS)
bench_set = set(b[0] for b in BENCHMARKS)

# ── Step 1: Read file and find base DATA (excluding auto-merged section) ──
filepath = os.path.join(REPO_ROOT, 'data', 'build_benchmark_matrix.py')
with open(filepath, 'r') as f:
    content = f.read()

data_start_marker = 'DATA = ['
build_excel_marker = 'def build_excel():'
data_start_pos = content.index(data_start_marker)
build_excel_pos = content.index(build_excel_marker)

# Remove any previous auto-merged section to get base content
auto_marker = "    # ── Auto-merged from extra_scores"
prev_auto_start = content.find(auto_marker, data_start_pos)
if prev_auto_start != -1 and prev_auto_start < build_excel_pos:
    prev_auto_end = content.find('\n]', prev_auto_start)
    if prev_auto_end != -1:
        content = content[:prev_auto_start] + content[prev_auto_end:]

# Parse base existing entries from the stripped file content
# Extract just the DATA portion and exec it to get base entries
data_section_start = content.index('DATA = [')
# Find matching ]
bracket_depth = 0
data_section_end = None
for i in range(data_section_start + len('DATA = ['), len(content)):
    if content[i] == '[':
        bracket_depth += 1
    elif content[i] == ']':
        if bracket_depth == 0:
            data_section_end = i + 1
            break
        bracket_depth -= 1

data_text = content[data_section_start:data_section_end]
local_ns = {}
exec(data_text, {}, local_ns)
base_data = local_ns['DATA']
existing = set((d[0], d[1]) for d in base_data)
print(f"Base DATA entries: {len(base_data)} ({len(existing)} unique cells)")

# ── Step 2: Load all extra score files ──
scores_all = []
for fname in ['extra_scores_1', 'extra_scores_2', 'extra_scores_3', 'extra_scores_4', 'extra_scores_5', 'extra_scores_6']:
    try:
        mod = importlib.import_module(fname)
        importlib.reload(mod)  # ensure fresh load
        data = getattr(mod, 'EXTRA_SCORES')
        scores_all.extend(data)
        print(f'{fname}: {len(data)} entries')
    except Exception as e:
        print(f'{fname}: ERROR - {e}')

print(f'Total raw entries: {len(scores_all)}')

# ── Alias mappings ──
MODEL_ALIASES = {
    'o3': 'o3-high',
    'o3-high-compute': 'o3-high',
    'o4-mini': 'o4-mini-high',
    'o4-mini-high-compute': 'o4-mini-high',
    'seed-2.0-pro': 'doubao-seed-2.0-pro',
    'claude-opus-4-high': None,
    'qwen3-max': None,
    # These ARE correct IDs — reverse aliases for variant spellings
    'qwen3.5-397b-a17b': 'qwen3.5-397b',
    'falcon3-10b-instruct': 'falcon3-10b',
    'internlm3-8b-instruct': 'internlm3-8b',
    'gpt-5.3': 'gpt-5.3-codex',
    'deepseek-v3.2-special': 'deepseek-v3.2-speciale',
    'grok-3': 'grok-3-beta',
    'phi-4-reasoning': 'phi-4-reasoning-plus',
    # Models not in our matrix — skip
    'o3-pro': None,
    'grok-3-mini': None,
    'grok-3-think': None,
    'grok-4.1-thinking': None,
    'glm-4.5': None,
    'gpt-5.1-high': None,
    'gpt-5.2-high': None,
    'gpt-5.2-xhigh': None,
    'claude-opus-4.5-reasoning': None,
    'deepseek-v3.2-exp': None,
    'gemini-3-flash-reasoning': None,
    'gemini-3-pro-high': None,
    'gemini-3-pro-low': None,
    'codestral-25.01': None,
    'devstral-2': None,
}

BENCH_ALIASES = {
    'humanitys_last_exam': 'hle',
    'hle_humanity_last_exam': 'hle',
    'hmmt_feb': 'hmmt_2025',
    'hmmt_feb_2025': 'hmmt_2025',
    'math': 'math_500',
    'video_mme': 'video_mmu',
    'terminal_bench_2': 'terminal_bench',
    'arena_hard_auto': 'arena_hard',
    'chatbot_arena': 'chatbot_arena_elo',
    'chatbot_arena_elo_overall': 'chatbot_arena_elo',
    'arc_agi': 'arc_agi_1',
    'codeforces': 'codeforces_rating',
    'simpleqa_score': 'simpleqa',
    'artificial_analysis_index': 'aa_intelligence_index',
}

# ── Process entries ──
new_entries = []
seen = set()
skipped_models = set()
skipped_benches = set()

for mid, bid, score, url in scores_all:
    mid2 = MODEL_ALIASES.get(mid, mid)
    bid2 = BENCH_ALIASES.get(bid, bid)

    if mid2 is None:
        continue

    if mid2 not in model_set:
        skipped_models.add(mid)
        continue
    if bid2 not in bench_set:
        skipped_benches.add(bid)
        continue

    key = (mid2, bid2)
    if key in existing or key in seen:
        continue

    new_entries.append((mid2, bid2, score, url))
    seen.add(key)

print(f'\nEntries to merge: {len(new_entries)}')
print(f'Skipped models not in matrix: {sorted(skipped_models)[:10]}')
print(f'Skipped benchmarks not in matrix: {len(skipped_benches)}')

if len(new_entries) == 0:
    print("Nothing new to merge.")
    sys.exit(0)

# ── Find data_end in the stripped content ──
build_excel_pos = content.index(build_excel_marker)
data_end = None
for i in range(build_excel_pos - 1, data_start_pos, -1):
    if content[i] == ']':
        line_start = content.rfind('\n', 0, i) + 1
        line_content = content[line_start:i+1].strip()
        if line_content == ']':
            data_end = i
            break

if data_end is None:
    print("ERROR: Could not find DATA list closing bracket!")
    sys.exit(1)

# Build new entries text
lines = []
lines.append(f"    # ── Auto-merged from extra_scores (v3): {len(new_entries)} entries ──")
for mid, bid, score, url in sorted(new_entries, key=lambda x: (x[0], x[1])):
    if isinstance(score, float) and score == int(score) and abs(score) < 10000:
        score_str = f"{int(score)}"
    else:
        score_str = f"{score}"
    url_safe = url.replace('"', '\\"') if url else ''
    lines.append(f'    ("{mid}", "{bid}", {score_str}, "{url_safe}"),')

new_text = '\n'.join(lines) + '\n'

# Insert before the closing ]
new_content = content[:data_end] + new_text + content[data_end:]

with open(filepath, 'w') as f:
    f.write(new_content)

print(f'\nMerged {len(new_entries)} entries into build_benchmark_matrix.py')
print(f'Total cells: {len(existing) + len(new_entries)}')

# Verify syntax
import py_compile
try:
    py_compile.compile(filepath, doraise=True)
    print('Syntax check: OK')
except py_compile.PyCompileError as e:
    print(f'Syntax ERROR: {e}')
