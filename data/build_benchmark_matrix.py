#!/usr/bin/env python3
"""
Cited LLM Benchmark Matrix Builder
===================================
Builds a 5-sheet Excel spreadsheet of (model, benchmark) scores for
instruct/chat LLMs released January 2025 or later.

Every numeric entry has a citation URL.
Downstream task: matrix completion (low-rank structure).

Sheets:
  1. Scores        – main model×benchmark matrix
  2. References     – same layout but URLs instead of numbers
  3. Benchmark Info – one row per benchmark with metadata
  4. Flat Format    – long-form ML-ready (model, benchmark, score, url)
  5. Matrix Stats   – fill rates, per-benchmark/model counts
"""

import json
import os
from datetime import datetime

# ── Model metadata ──────────────────────────────────────────────────────────
# (model_id, display_name, provider, release_date, params_total, params_active,
#  architecture, is_reasoning, open_weights)
# params left as None for closed-source models

MODELS = [
    # OpenAI
    ("o3-mini-high", "o3-mini (high)", "OpenAI", "2025-01-31", None, None, None, True, False),
    ("gpt-4.5", "GPT-4.5", "OpenAI", "2025-02-27", None, None, None, False, False),
    ("gpt-4.1", "GPT-4.1", "OpenAI", "2025-04-14", None, None, None, False, False),
    ("gpt-4.1-mini", "GPT-4.1 mini", "OpenAI", "2025-04-14", None, None, None, False, False),
    ("gpt-4.1-nano", "GPT-4.1 nano", "OpenAI", "2025-04-14", None, None, None, False, False),
    ("o3-high", "o3 (high)", "OpenAI", "2025-04-16", None, None, None, True, False),
    ("o4-mini-high", "o4-mini (high)", "OpenAI", "2025-04-16", None, None, None, True, False),
    ("gpt-5", "GPT-5", "OpenAI", "2025-08-01", None, None, None, True, False),
    ("gpt-oss-120b", "gpt-oss-120B", "OpenAI", "2025-08-05", 116800, 5100, "MoE", True, True),
    ("gpt-oss-20b", "gpt-oss-20B", "OpenAI", "2025-08-05", 20900, 3600, "MoE", True, True),
    ("gpt-5.1", "GPT-5.1", "OpenAI", "2025-11-13", None, None, None, True, False),
    ("gpt-5.2", "GPT-5.2", "OpenAI", "2025-12-11", None, None, None, True, False),
    ("gpt-5.3-codex", "GPT-5.3-Codex", "OpenAI", "2026-02-05", None, None, None, True, False),

    # Anthropic
    ("claude-3.7-sonnet", "Claude 3.7 Sonnet", "Anthropic", "2025-02-24", None, None, None, True, False),
    ("claude-sonnet-4", "Claude Sonnet 4", "Anthropic", "2025-05-22", None, None, None, False, False),
    ("claude-opus-4", "Claude Opus 4", "Anthropic", "2025-05-22", None, None, None, True, False),
    ("claude-opus-4.1", "Claude Opus 4.1", "Anthropic", "2025-08-05", None, None, None, True, False),
    ("claude-sonnet-4.5", "Claude Sonnet 4.5", "Anthropic", "2025-09-29", None, None, None, True, False),
    ("claude-haiku-4.5", "Claude Haiku 4.5", "Anthropic", "2025-10-15", None, None, None, True, False),
    ("claude-opus-4.5", "Claude Opus 4.5", "Anthropic", "2025-11-24", None, None, None, True, False),
    ("claude-opus-4.6", "Claude Opus 4.6", "Anthropic", "2026-02-05", None, None, None, True, False),
    ("claude-sonnet-4.6", "Claude Sonnet 4.6", "Anthropic", "2026-02-17", None, None, None, True, False),

    # Google / DeepMind
    ("gemini-2.0-flash", "Gemini 2.0 Flash", "Google", "2025-02-01", None, None, None, False, False),
    ("gemini-2.5-pro", "Gemini 2.5 Pro", "Google", "2025-03-25", None, None, None, True, False),
    ("gemini-2.5-flash", "Gemini 2.5 Flash", "Google", "2025-05-20", None, None, None, True, False),
    ("gemma-3-27b", "Gemma 3 27B", "Google", "2025-03-12", 27000, 27000, "Dense", False, True),
    ("gemini-3-pro", "Gemini 3 Pro", "Google", "2025-11-18", None, None, None, True, False),
    ("gemini-3-flash", "Gemini 3 Flash", "Google", "2025-11-18", None, None, None, True, False),
    ("gemini-3.1-pro", "Gemini 3.1 Pro", "Google", "2026-02-19", None, None, None, True, False),

    # Meta
    ("llama-4-scout", "Llama 4 Scout", "Meta", "2025-04-05", 109000, 17000, "MoE", False, True),
    ("llama-4-maverick", "Llama 4 Maverick", "Meta", "2025-04-05", 402000, 17000, "MoE", False, True),
    ("llama-4-behemoth", "Llama 4 Behemoth", "Meta", "2025-04-05", 2000000, 288000, "MoE", False, True),

    # xAI
    ("grok-3-beta", "Grok 3 Beta", "xAI", "2025-02-19", None, None, None, True, False),
    ("grok-4", "Grok 4", "xAI", "2025-07-09", None, None, None, True, False),
    ("grok-4.1", "Grok 4.1", "xAI", "2025-11-17", None, None, None, True, False),

    # DeepSeek
    ("deepseek-r1", "DeepSeek-R1", "DeepSeek", "2025-01-20", 671000, 37000, "MoE", True, True),
    ("deepseek-v3", "DeepSeek-V3", "DeepSeek", "2025-01-01", 671000, 37000, "MoE", False, True),
    ("deepseek-v3-0324", "DeepSeek-V3-0324", "DeepSeek", "2025-03-24", 671000, 37000, "MoE", False, True),
    ("deepseek-r1-0528", "DeepSeek-R1-0528", "DeepSeek", "2025-05-28", 671000, 37000, "MoE", True, True),
    ("deepseek-r1-distill-qwen-32b", "DeepSeek-R1-Distill-Qwen-32B", "DeepSeek", "2025-01-20", 32000, 32000, "Dense", True, True),
    ("deepseek-r1-distill-qwen-14b", "DeepSeek-R1-Distill-Qwen-14B", "DeepSeek", "2025-01-20", 14000, 14000, "Dense", True, True),
    ("deepseek-r1-distill-qwen-7b", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek", "2025-01-20", 7000, 7000, "Dense", True, True),
    ("deepseek-r1-distill-qwen-1.5b", "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek", "2025-01-20", 1500, 1500, "Dense", True, True),
    ("deepseek-r1-distill-llama-8b", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek", "2025-01-20", 8000, 8000, "Dense", True, True),
    ("deepseek-r1-distill-llama-70b", "DeepSeek-R1-Distill-Llama-70B", "DeepSeek", "2025-01-20", 70000, 70000, "Dense", True, True),
    ("deepseek-v3.2", "DeepSeek-V3.2", "DeepSeek", "2025-12-01", 671000, 37000, "MoE", True, True),
    ("deepseek-v3.2-speciale", "DeepSeek-V3.2-Speciale", "DeepSeek", "2025-12-01", 671000, 37000, "MoE", True, True),

    # Qwen / Alibaba
    ("qwen3-235b", "Qwen3-235B-A22B", "Alibaba", "2025-05-15", 235000, 22000, "MoE", True, True),
    ("qwen3-32b", "Qwen3-32B", "Alibaba", "2025-05-15", 32000, 32000, "Dense", True, True),
    ("qwen3-4b", "Qwen3-4B", "Alibaba", "2025-05-15", 4000, 4000, "Dense", True, True),
    ("qwen3-0.6b", "Qwen3-0.6B", "Alibaba", "2025-05-15", 600, 600, "Dense", True, True),
    ("qwen3-1.7b", "Qwen3-1.7B", "Alibaba", "2025-05-15", 1700, 1700, "Dense", True, True),
    ("qwen3-8b", "Qwen3-8B", "Alibaba", "2025-05-15", 8000, 8000, "Dense", True, True),
    ("qwen3-14b", "Qwen3-14B", "Alibaba", "2025-05-15", 14000, 14000, "Dense", True, True),
    ("qwen3-30b-a3b", "Qwen3-30B-A3B", "Alibaba", "2025-05-15", 30000, 3000, "MoE", True, True),
    ("qwq-32b", "QwQ-32B", "Alibaba", "2025-03-05", 32800, 32800, "Dense", True, True),
    ("qwen3.5-397b", "Qwen3.5-397B-A17B", "Alibaba", "2026-02-01", 397000, 17000, "MoE", True, True),

    # Moonshot AI
    ("kimi-k2", "Kimi K2", "Moonshot AI", "2025-07-16", None, None, "MoE", False, True),
    ("kimi-k2-thinking", "Kimi K2 Thinking", "Moonshot AI", "2025-11-01", None, None, "MoE", True, True),
    ("kimi-k2.5", "Kimi K2.5", "Moonshot AI", "2026-01-01", None, None, "MoE", True, True),

    # Zhipu AI
    ("glm-4.6", "GLM-4.6", "Zhipu AI", "2025-09-01", None, None, None, True, False),
    ("glm-4.7", "GLM-4.7", "Zhipu AI", "2025-12-01", None, None, None, True, False),

    # ByteDance
    ("doubao-seed-2.0-pro", "Doubao Seed 2.0 Pro", "ByteDance", "2026-02-14", None, None, None, True, False),

    # Mistral
    ("mistral-small-3.1", "Mistral Small 3.1", "Mistral", "2025-03-01", 24000, 24000, "Dense", False, True),
    ("mistral-medium-3", "Mistral Medium 3", "Mistral", "2025-05-01", None, None, None, False, False),
    ("mistral-large-3", "Mistral Large 3", "Mistral", "2025-12-01", 675000, None, "MoE", False, True),
    ("codestral-25.01", "Codestral 25.01", "Mistral", "2025-01-15", 22000, 22000, "Dense", False, True),
    ("devstral-2", "Devstral 2", "Mistral", "2025-12-01", 123000, None, "Dense", False, True),

    # Microsoft
    ("phi-4", "Phi-4", "Microsoft", "2025-01-01", 14000, 14000, "Dense", False, True),
    ("phi-4-reasoning", "Phi-4-reasoning", "Microsoft", "2025-04-30", 14000, 14000, "Dense", True, True),
    ("phi-4-reasoning-plus", "Phi-4-reasoning-plus", "Microsoft", "2025-04-30", 14000, 14000, "Dense", True, True),

    # NVIDIA
    ("nemotron-ultra-253b", "Nemotron Ultra 253B", "NVIDIA", "2025-04-10", 253000, None, "MoE", True, True),

    # Amazon
    ("amazon-nova-pro", "Amazon Nova Pro", "Amazon", "2025-01-01", None, None, None, False, False),
    ("amazon-nova-premier", "Amazon Nova Premier", "Amazon", "2025-04-01", None, None, None, False, False),

    # Cohere
    ("command-a", "Command A", "Cohere", "2025-03-13", 111000, None, "Dense", False, True),

    # LG AI Research
    ("exaone-4.0-32b", "EXAONE 4.0 32B", "LG AI Research", "2025-07-01", 32000, 32000, "Dense", True, True),

    # MiniMax
    ("minimax-m2", "MiniMax-M2", "MiniMax", "2025-10-01", None, None, None, True, False),

    # Allen AI
    ("olmo-2-13b", "OLMo 2 13B", "Allen AI", "2025-01-01", 13000, 13000, "Dense", False, True),

    # Liquid AI
    ("lfm2.5-1.2b-thinking", "LFM2.5-1.2B-Thinking", "Liquid AI", "2026-01-06", 1200, 1200, "Hybrid", True, True),

    # Microsoft (additional)
    ("phi-4-mini", "Phi-4-mini", "Microsoft", "2025-03-01", 3800, 3800, "Dense", False, True),

    # TII (Falcon)
    ("falcon3-10b", "Falcon3-10B-Instruct", "TII", "2025-01-01", 10000, 10000, "Dense", False, True),

    # Shanghai AI Lab
    ("internlm3-8b", "InternLM3-8B-Instruct", "Shanghai AI Lab", "2025-01-15", 8000, 8000, "Dense", False, True),

    # ByteDance (additional — open-weight reasoning)
    ("seed-thinking-v1.5", "Seed-Thinking-v1.5", "ByteDance", "2025-04-01", None, None, None, True, False),
]

# ── Benchmark metadata ──────────────────────────────────────────────────────
# (benchmark_id, display_name, category, metric, num_problems, source_url)
BENCHMARKS = [
    ("gpqa_diamond", "GPQA Diamond", "Science", "% correct", 198, "https://arxiv.org/abs/2311.12022"),
    ("aime_2025", "AIME 2025", "Math", "% correct (pass@1)", 30, "https://artofproblemsolving.com/wiki/index.php/2025_AIME"),
    ("mmlu", "MMLU", "Knowledge", "% correct", 14042, "https://arxiv.org/abs/2009.03300"),
    ("mmlu_pro", "MMLU-Pro", "Knowledge", "% correct", 12032, "https://arxiv.org/abs/2406.01574"),
    ("swe_bench_verified", "SWE-bench Verified", "Coding", "% resolved", 500, "https://www.swebench.com/"),
    ("math_500", "MATH-500", "Math", "% correct", 500, "https://arxiv.org/abs/2103.03874"),
    ("livecodebench", "LiveCodeBench", "Coding", "pass@1 %", 1055, "https://livecodebench.github.io/"),
    ("frontiermath", "FrontierMath", "Math", "% correct T1-3", 300, "https://epoch.ai/benchmarks/frontiermath"),
    ("hle", "HLE (Humanity's Last Exam)", "Reasoning", "% correct", 2500, "https://lastexam.ai/"),
    ("arc_agi_2", "ARC-AGI-2", "Reasoning", "% correct", 400, "https://arcprize.org/arc-agi/2/"),
    ("browsecomp", "BrowseComp", "Agentic", "% correct", 1266, "https://openai.com/index/browsecomp/"),
    ("simpleqa", "SimpleQA", "Knowledge", "% correct", 4326, "https://openai.com/index/introducing-simpleqa/"),
    ("ifeval", "IFEval", "Instruction Following", "% correct (prompt strict)", 541, "https://arxiv.org/abs/2311.07911"),
    ("humaneval", "HumanEval", "Coding", "pass@1 %", 164, "https://github.com/openai/human-eval"),
    ("codeforces_rating", "Codeforces Rating", "Coding", "Elo rating", None, "https://codeforces.com/"),
    ("osworld", "OSWorld", "Agentic", "% success", 369, "https://os-world.github.io/"),
    ("mmmu", "MMMU", "Multimodal", "% correct", 900, "https://mmmu-benchmark.github.io/"),
    ("mmmu_pro", "MMMU-Pro", "Multimodal", "% correct", None, "https://arxiv.org/abs/2409.02813"),
    ("arena_hard", "Arena-Hard Auto", "Instruction Following", "% win rate", 500, "https://lmarena.ai/"),
    ("chatbot_arena_elo", "Chatbot Arena Elo", "Human Preference", "Elo rating", None, "https://lmarena.ai/"),
    ("swe_bench_pro", "SWE-bench Pro", "Coding", "% resolved", None, "https://scale.com/leaderboard/swe_bench_pro_public"),
    ("aime_2024", "AIME 2024", "Math", "% correct (pass@1)", 30, "https://artofproblemsolving.com/wiki/index.php/2024_AIME"),
    ("hmmt_2025", "HMMT Feb 2025", "Math", "% correct", None, "https://www.hmmt.org/"),

    # ── Recovered orphaned benchmarks (were in DATA but not in BENCHMARKS) ──
    ("tau_bench_retail", "Tau-Bench Retail", "Agentic", "% success", None, "https://arxiv.org/abs/2406.12045"),
    ("tau_bench_telecom", "Tau-Bench Telecom", "Agentic", "% success", None, "https://arxiv.org/abs/2406.12045"),
    ("video_mmu", "Video-MMU", "Multimodal", "% correct", None, "https://video-mmu.github.io/"),
    ("mrcr_v2", "MRCR v2", "Long Context", "% correct", None, "https://arxiv.org/abs/2407.05530"),
    ("aa_intelligence_index", "AA Intelligence Index", "Composite", "index score", None, "https://artificialanalysis.ai/"),
    ("aa_lcr", "AA Long Context Reasoning", "Long Context", "% correct", None, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),
    ("critpt", "CritPt", "Science", "% correct", None, "https://artificialanalysis.ai/evaluations/critpt"),
    ("scicode", "SciCode", "Coding", "% correct", None, "https://scicode-bench.github.io/"),
    ("mathvision", "MathVision", "Math", "% correct", None, "https://mathvision-cuhk.github.io/"),
    ("gdpval_aa", "GDP-Val AA", "Knowledge", "score", None, "https://artificialanalysis.ai/evaluations/gdpval-aa"),

    # ── New benchmarks (user-suggested + widely reported) ──
    ("gsm8k", "GSM8K", "Math", "% correct", 1319, "https://arxiv.org/abs/2110.14168"),
    ("ifbench", "IFBench", "Instruction Following", "% correct", None, "https://arxiv.org/abs/2502.09980"),
    ("terminal_bench", "Terminal-Bench 2.0", "Agentic", "% solved", None, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("terminal_bench_1", "Terminal-Bench 1.0", "Agentic", "% solved", None, "https://terminal-bench.com/"),
    ("arc_agi_1", "ARC-AGI-1", "Reasoning", "% correct", 400, "https://arcprize.org/arc-agi/1/"),

    # ── New benchmarks (auto-merged) ──
    ("brumo_2025", "BRUMO 2025", "Math", "% correct (pass@1)", None, "https://huggingface.co/datasets/MathArena/brumo_2025"),
    ("smt_2025", "SMT 2025", "Math", "% correct (pass@1)", None, "https://huggingface.co/datasets/MathArena/smt_2025"),
    ("usamo_2025", "USAMO 2025", "Math", "% of 42 points", 6, "https://huggingface.co/datasets/MathArena/usamo_2025"),
    ("hmmt_nov_2025", "HMMT Nov 2025", "Math", "% correct", None, "https://huggingface.co/datasets/MathArena/hmmt_nov_2025"),
    ("cmimc_2025", "CMIMC 2025", "Math", "% correct (pass@1)", None, "https://huggingface.co/datasets/MathArena/cmimc_2025"),
    ("imo_2025", "IMO 2025", "Math", "% of 42 points", 6, "https://matharena.ai/imo/"),
    ("aime_2026", "AIME 2026", "Math", "% correct (pass@1)", 30, "https://huggingface.co/datasets/MathArena/aime_2026_I"),
    ("matharena_apex_2025", "MathArena Apex 2025", "Math", "% correct", None, "https://matharena.ai/apex/"),

    # ── New benchmarks from user-suggested sources ──
    ("livebench", "LiveBench", "Composite", "overall score", None, "https://livebench.ai/"),
    ("simplebench", "SimpleBench", "Reasoning", "% correct", 1000, "https://simple-bench.com/"),
    ("bigcodebench", "BigCodeBench", "Coding", "pass@1 %", 1140, "https://bigcode-bench.github.io/"),
]

# ── Score data: (model_id, benchmark_id, score, reference_url) ────────────
# Every entry below was found via web search with the given URL as citation.
# Scores are numeric (percentages as plain numbers, e.g. 87.3 not 0.873).

DATA = [
    # ═══════════════════════════════════════════════════════════════════════
    # OpenAI models
    # ═══════════════════════════════════════════════════════════════════════

    # o3-mini (high)
    ("o3-mini-high", "aime_2024", 87.3, "https://openai.com/index/openai-o3-mini/"),
    ("o3-mini-high", "gpqa_diamond", 79.7, "https://openai.com/index/openai-o3-mini/"),
    ("o3-mini-high", "codeforces_rating", 2130, "https://neuroflash.com/blog/chatgpt-o3-mini-high/"),

    # GPT-4.5
    ("gpt-4.5", "simpleqa", 62.5, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),

    # GPT-4.1
    ("gpt-4.1", "swe_bench_verified", 54.6, "https://openai.com/index/gpt-4-1/"),
    ("gpt-4.1", "mmlu", 90.2, "https://openai.com/index/gpt-4-1/"),
    ("gpt-4.1", "gpqa_diamond", 66.3, "https://www.helicone.ai/blog/gpt-4.1-full-developer-guide"),
    ("gpt-4.1", "ifeval", 87.4, "https://www.helicone.ai/blog/gpt-4.1-full-developer-guide"),

    # GPT-4.1 mini
    ("gpt-4.1-mini", "ifeval", 84.1, "https://openai.com/index/gpt-4-1/"),

    # GPT-4.1 nano
    ("gpt-4.1-nano", "mmlu", 80.1, "https://www.datacamp.com/blog/gpt-4-1"),
    ("gpt-4.1-nano", "gpqa_diamond", 50.3, "https://www.datacamp.com/blog/gpt-4-1"),

    # o3 (high)
    ("o3-high", "aime_2024", 96.7, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o3-high", "aime_2025", 88.9, "https://www.datacamp.com/blog/o4-mini"),
    ("o3-high", "gpqa_diamond", 87.7, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o3-high", "swe_bench_verified", 69.1, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o3-high", "frontiermath", 10.0, "https://x.com/EpochAIResearch/status/1913379478778134941"),
    ("o3-high", "livecodebench", 75.8, "https://www.alphaxiv.org/benchmarks/uc-berkeley/livecodebench"),
    ("o3-high", "math_500", 99.2, "https://artificialanalysis.ai/evaluations/math-500"),
    ("o3-high", "codeforces_rating", 2706, "https://www.datacamp.com/blog/o4-mini"),
    ("o3-high", "arc_agi_2", 6.5, "https://arcprize.org/arc-agi/2/"),
    ("o3-high", "arena_hard", 85.9, "https://llm-stats.com/benchmarks/arena-hard"),

    # o4-mini (high)
    ("o4-mini-high", "aime_2024", 93.4, "https://www.datacamp.com/blog/o4-mini"),
    ("o4-mini-high", "aime_2025", 92.7, "https://www.datacamp.com/blog/o4-mini"),
    ("o4-mini-high", "gpqa_diamond", 81.4, "https://www.datacamp.com/blog/o4-mini"),
    ("o4-mini-high", "swe_bench_verified", 68.1, "https://www.datacamp.com/blog/o4-mini"),
    ("o4-mini-high", "frontiermath", 17.0, "https://x.com/EpochAIResearch/status/1913379478778134941"),
    ("o4-mini-high", "livecodebench", 80.2, "https://www.alphaxiv.org/benchmarks/uc-berkeley/livecodebench"),
    ("o4-mini-high", "humaneval", 98.5, "https://www.rdworldonline.com/openai-releases-o3-a-model-that-tops-99-of-human-competitors/"),
    ("o4-mini-high", "codeforces_rating", 2719, "https://www.datacamp.com/blog/o4-mini"),
    ("o4-mini-high", "arena_hard", 79.1, "https://llm-stats.com/benchmarks/arena-hard"),

    # GPT-5
    ("gpt-5", "aime_2025", 94.6, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "gpqa_diamond", 88.4, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "swe_bench_verified", 74.9, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "mmmu", 84.2, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "math_500", 99.4, "https://artificialanalysis.ai/evaluations/math-500"),
    ("gpt-5", "tau_bench_telecom", 97.0, "https://www.digitalocean.com/resources/articles/gpt-5-overview"),
    ("gpt-5", "hle", 35.2, "https://www.vellum.ai/blog/flagship-model-report"),
    ("gpt-5", "browsecomp", 54.9, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "mmlu_pro", 87.0, "https://www.vals.ai/updates"),
    ("gpt-5", "aa_lcr", 75.6, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),

    # gpt-oss-120B
    ("gpt-oss-120b", "aime_2025", 97.9, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("gpt-oss-120b", "gpqa_diamond", 80.9, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("gpt-oss-120b", "mmlu_pro", 90.0, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("gpt-oss-120b", "codeforces_rating", 2620, "https://smythos.com/developers/ai-models/openai-gpt-oss-120b-and-20b/"),
    ("gpt-oss-120b", "tau_bench_retail", 67.8, "https://smythos.com/developers/ai-models/openai-gpt-oss-120b-and-20b/"),

    # gpt-oss-20B
    ("gpt-oss-20b", "aime_2025", 98.7, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("gpt-oss-20b", "gpqa_diamond", 71.5, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("gpt-oss-20b", "mmlu_pro", 85.3, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),

    # GPT-5.1
    ("gpt-5.1", "aime_2025", 94.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.1", "gpqa_diamond", 88.1, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),
    ("gpt-5.1", "frontiermath", 31.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.1", "arc_agi_2", 17.6, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.1", "swe_bench_verified", 76.3, "https://www.vellum.ai/blog/flagship-model-report"),
    ("gpt-5.1", "aa_lcr", 75.0, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),

    # GPT-5.2
    ("gpt-5.2", "aime_2025", 100.0, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "gpqa_diamond", 93.2, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "frontiermath", 40.3, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "swe_bench_verified", 80.0, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "swe_bench_pro", 55.6, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "arc_agi_2", 52.9, "https://introl.com/blog/gpt-5-2-infrastructure-implications"),
    ("gpt-5.2", "browsecomp", 77.9, "https://llm-stats.com/benchmarks/browsecomp"),
    ("gpt-5.2", "critpt", 11.6, "https://artificialanalysis.ai/evaluations/critpt"),
    ("gpt-5.2", "scicode", 54.6, "https://artificialanalysis.ai/evaluations/scicode"),
    ("gpt-5.2", "aa_lcr", 75.7, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),

    # GPT-5.3-Codex
    ("gpt-5.3-codex", "swe_bench_pro", 56.8, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "osworld", 64.7, "https://www.neowin.net/news/openai-debuts-gpt-53-codex/"),

    # ═══════════════════════════════════════════════════════════════════════
    # Anthropic models
    # ═══════════════════════════════════════════════════════════════════════

    # Claude 3.7 Sonnet
    ("claude-3.7-sonnet", "gpqa_diamond", 84.8, "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "aime_2024", 80.0, "https://www.datacamp.com/blog/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "math_500", 96.2, "https://www.datacamp.com/blog/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "swe_bench_verified", 62.3, "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "ifeval", 93.2, "https://www.datacamp.com/blog/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "humaneval", 94.0, "https://www.datacamp.com/blog/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "livecodebench", 65.0, "https://www.datacamp.com/blog/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "tau_bench_retail", 81.2, "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "tau_bench_telecom", 49.0, "https://artificialanalysis.ai/models/claude-3-7-sonnet"),

    # Claude Sonnet 4
    ("claude-sonnet-4", "gpqa_diamond", 75.4, "https://www.anthropic.com/news/claude-4"),
    ("claude-sonnet-4", "swe_bench_verified", 72.7, "https://www.anthropic.com/news/claude-4"),
    ("claude-sonnet-4", "mmlu", 86.5, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "tau_bench_retail", 80.5, "https://www.anthropic.com/news/claude-4"),

    # Claude Opus 4
    ("claude-opus-4", "gpqa_diamond", 83.3, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "swe_bench_verified", 79.4, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "mmlu", 88.8, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "aime_2025", 90.0, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "tau_bench_retail", 81.4, "https://www.anthropic.com/news/claude-4"),

    # Claude Opus 4.1
    ("claude-opus-4.1", "swe_bench_verified", 74.5, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "gpqa_diamond", 80.9, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "mmlu", 88.8, "https://www.anthropic.com/news/claude-opus-4-1"),

    # Claude Sonnet 4.5
    ("claude-sonnet-4.5", "swe_bench_verified", 77.2, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "gpqa_diamond", 83.4, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "tau_bench_retail", 86.2, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "tau_bench_telecom", 98.0, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "osworld", 61.4, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "livecodebench", 68.0, "https://www.anthropic.com/news/claude-sonnet-4-5"),

    # Claude Haiku 4.5
    ("claude-haiku-4.5", "swe_bench_verified", 73.3, "https://www.anthropic.com/news/claude-haiku-4-5"),
    ("claude-haiku-4.5", "gpqa_diamond", 80.9, "https://www.anthropic.com/news/claude-haiku-4-5"),
    ("claude-haiku-4.5", "aime_2025", 96.3, "https://airank.dev/benchmarks/aime-2025"),
    ("claude-haiku-4.5", "mmlu", 82.0, "https://www.anthropic.com/news/claude-haiku-4-5"),
    ("claude-haiku-4.5", "osworld", 50.7, "https://www.anthropic.com/news/claude-haiku-4-5"),

    # Claude Opus 4.5
    ("claude-opus-4.5", "swe_bench_verified", 80.9, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "gpqa_diamond", 87.0, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "mmlu", 90.8, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "mmlu_pro", 80.0, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "osworld", 66.3, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "arc_agi_2", 37.6, "https://arcprize.org/arc-agi/2/"),
    ("claude-opus-4.5", "browsecomp", 67.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.5", "frontiermath", 21.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("claude-opus-4.5", "hle", 37.6, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "tau_bench_telecom", 98.2, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "tau_bench_retail", 88.9, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "swe_bench_pro", 45.89, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # Claude Opus 4.6
    ("claude-opus-4.6", "gpqa_diamond", 91.3, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "swe_bench_verified", 80.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "aime_2025", 100.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "mmlu_pro", 82.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "osworld", 72.7, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "arc_agi_2", 68.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "browsecomp", 84.0, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "hle", 40.0, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "frontiermath", 40.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("claude-opus-4.6", "mrcr_v2", 93.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "simpleqa", 72.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "ifeval", 94.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "humaneval", 95.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "livecodebench", 76.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "tau_bench_retail", 91.9, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "critpt", 12.6, "https://artificialanalysis.ai/evaluations/critpt"),
    ("claude-opus-4.6", "gdpval_aa", 1606, "https://artificialanalysis.ai/evaluations/gdpval-aa"),
    ("claude-opus-4.6", "aa_intelligence_index", 53, "https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index"),

    # Claude Sonnet 4.6
    ("claude-sonnet-4.6", "swe_bench_verified", 79.6, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "osworld", 72.5, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "arc_agi_2", 60.4, "https://arcprize.org/arc-agi/2/"),
    ("claude-sonnet-4.6", "gdpval_aa", 1633, "https://artificialanalysis.ai/evaluations/gdpval-aa"),
    ("claude-sonnet-4.6", "aa_intelligence_index", 51, "https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index"),

    # ═══════════════════════════════════════════════════════════════════════
    # Google / DeepMind
    # ═══════════════════════════════════════════════════════════════════════

    ("gemini-2.0-flash", "mmlu_pro", 76.4, "https://www.helicone.ai/blog/gemini-2.0-flash"),
    ("gemini-2.0-flash", "mmmu", 70.7, "https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/"),

    ("gemini-2.5-pro", "aime_2025", 86.7, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "aime_2024", 92.0, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "gpqa_diamond", 84.0, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "hle", 18.8, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "swe_bench_verified", 63.8, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "math_500", 97.3, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "livecodebench", 70.4, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "simpleqa", 52.9, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),

    ("gemma-3-27b", "mmlu_pro", 67.5, "https://huggingface.co/blog/gemma3"),
    ("gemma-3-27b", "gpqa_diamond", 42.4, "https://huggingface.co/blog/gemma3"),
    ("gemma-3-27b", "livecodebench", 29.7, "https://huggingface.co/blog/gemma3"),
    ("gemma-3-27b", "humaneval", 87.8, "https://www.emergentmind.com/topics/gemma-3-27b-it"),
    ("gemma-3-27b", "chatbot_arena_elo", 1338, "https://huggingface.co/blog/gemma3"),

    ("gemini-3-pro", "gpqa_diamond", 91.9, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "aime_2025", 95.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "swe_bench_verified", 76.2, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "hle", 37.5, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "arc_agi_2", 31.1, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "mmmu_pro", 81.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "mmlu_pro", 89.8, "https://artificialanalysis.ai/evaluations/mmlu-pro"),
    ("gemini-3-pro", "mrcr_v2", 77.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "simpleqa", 72.1, "https://officechai.com/ai/gemini-3-1-pro-benchmarks/"),
    ("gemini-3-pro", "chatbot_arena_elo", 1501, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "livecodebench", 79.7, "https://www.vellum.ai/llm-leaderboard"),

    ("gemini-3-flash", "gpqa_diamond", 90.4, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "mmlu", 91.8, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "mmlu_pro", 88.59, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "swe_bench_verified", 78.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "hle", 33.7, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "mmmu_pro", 81.2, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),

    ("gemini-3.1-pro", "arc_agi_2", 77.1, "https://techcrunch.com/2026/02/19/googles-new-gemini-pro-model-has-record-benchmark-scores-again/"),
    ("gemini-3.1-pro", "gpqa_diamond", 94.3, "https://techcrunch.com/2026/02/19/googles-new-gemini-pro-model-has-record-benchmark-scores-again/"),
    ("gemini-3.1-pro", "hle", 44.4, "https://techcrunch.com/2026/02/19/googles-new-gemini-pro-model-has-record-benchmark-scores-again/"),
    ("gemini-3.1-pro", "swe_bench_verified", 80.6, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemini-3.1-pro", "scicode", 58.9, "https://artificialanalysis.ai/evaluations/scicode"),
    ("gemini-3.1-pro", "aime_2025", 100.0, "https://automatio.ai/models/gemini-3-1-pro"),
    ("gemini-3.1-pro", "mmlu_pro", 89.5, "https://automatio.ai/models/gemini-3-1-pro"),
    ("gemini-3.1-pro", "mmmu_pro", 80.5, "https://www.trendingtopics.eu/gemini-3-1-pro-leads-most-benchmarks/"),
    ("gemini-3.1-pro", "mathvision", 69.8, "https://automatio.ai/models/gemini-3-1-pro"),
    ("gemini-3.1-pro", "mrcr_v2", 84.9, "https://smartscope.blog/en/generative-ai/google-gemini/gemini-3-1-pro-benchmark-analysis-2026/"),
    ("gemini-3.1-pro", "simpleqa", 72.1, "https://officechai.com/ai/gemini-3-1-pro-benchmarks/"),
    ("gemini-3.1-pro", "ifeval", 89.2, "https://officechai.com/ai/gemini-3-1-pro-benchmarks/"),
    ("gemini-3.1-pro", "critpt", 17.7, "https://artificialanalysis.ai/evaluations/critpt"),
    ("gemini-3.1-pro", "aa_intelligence_index", 57, "https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index"),

    # ═══════════════════════════════════════════════════════════════════════
    # Meta
    # ═══════════════════════════════════════════════════════════════════════
    ("llama-4-scout", "mmlu_pro", 74.3, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "gpqa_diamond", 57.2, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "mmmu", 69.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "livecodebench", 32.8, "https://www.llama.com/models/llama-4/"),

    ("llama-4-maverick", "mmlu_pro", 80.5, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "gpqa_diamond", 69.8, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "math_500", 88.1, "https://venturebeat.com/ai/metas-answer-to-deepseek-is-here-llama-4/"),
    ("llama-4-maverick", "mmmu", 73.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "livecodebench", 43.4, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "chatbot_arena_elo", 1417, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),

    ("llama-4-behemoth", "math_500", 95.0, "https://www.datacamp.com/blog/llama-4"),
    ("llama-4-behemoth", "mmlu_pro", 82.2, "https://www.datacamp.com/blog/llama-4"),
    ("llama-4-behemoth", "gpqa_diamond", 73.7, "https://www.datacamp.com/blog/llama-4"),
    ("llama-4-behemoth", "mmmu", 76.1, "https://www.datacamp.com/blog/llama-4"),
    ("llama-4-behemoth", "livecodebench", 49.4, "https://www.datacamp.com/blog/llama-4"),

    # ═══════════════════════════════════════════════════════════════════════
    # xAI (Grok)
    # ═══════════════════════════════════════════════════════════════════════
    ("grok-3-beta", "gpqa_diamond", 84.6, "https://x.ai/news/grok-3"),
    ("grok-3-beta", "mmlu_pro", 79.9, "https://x.ai/news/grok-3"),
    ("grok-3-beta", "livecodebench", 79.4, "https://www.helicone.ai/blog/grok-3-benchmark-comparison"),
    ("grok-3-beta", "chatbot_arena_elo", 1402, "https://x.com/lmarena_ai/status/1891706264800936307"),
    ("grok-3-beta", "math_500", 99.2, "https://artificialanalysis.ai/evaluations/math-500"),

    ("grok-4", "gpqa_diamond", 88.0, "https://x.ai/news/grok-4"),
    ("grok-4", "hle", 24.0, "https://x.ai/news/grok-4"),
    ("grok-4", "arc_agi_2", 15.9, "https://x.ai/news/grok-4"),
    ("grok-4", "livecodebench", 79.4, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "swe_bench_verified", 73.5, "https://datasciencedojo.com/blog/grok-4/"),
    ("grok-4", "aime_2025", 90.6, "https://matharena.ai/"),
    ("grok-4", "hmmt_2025", 90.0, "https://epoch.ai/blog/grok-4-math"),
    ("grok-4", "aa_intelligence_index", 73, "https://x.com/ArtificialAnlys/status/1943166841150644622"),

    ("grok-4.1", "chatbot_arena_elo", 1483, "https://x.ai/news/grok-4-1"),
    ("grok-4.1", "swe_bench_verified", 79.0, "https://skywork.ai/blog/ai-agent/grok-41-vs-gemini-30-comparison/"),

    # ═══════════════════════════════════════════════════════════════════════
    # DeepSeek
    # ═══════════════════════════════════════════════════════════════════════
    ("deepseek-v3", "mmlu", 88.5, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "mmlu_pro", 75.9, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "gpqa_diamond", 59.1, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "math_500", 90.2, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "swe_bench_verified", 42.0, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "livecodebench", 39.2, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "ifeval", 86.1, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "arena_hard", 85.5, "https://github.com/deepseek-ai/DeepSeek-V3"),

    ("deepseek-r1", "gpqa_diamond", 71.5, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "math_500", 97.3, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "aime_2024", 79.8, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "mmlu", 90.8, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "mmlu_pro", 84.0, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "swe_bench_verified", 49.2, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "livecodebench", 65.9, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "codeforces_rating", 2029, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "arena_hard", 92.3, "https://github.com/deepseek-ai/DeepSeek-R1"),

    ("deepseek-v3-0324", "gpqa_diamond", 68.4, "https://huggingface.co/deepseek-ai/DeepSeek-V3-0324"),
    ("deepseek-v3-0324", "aime_2024", 59.4, "https://huggingface.co/deepseek-ai/DeepSeek-V3-0324"),
    ("deepseek-v3-0324", "livecodebench", 49.2, "https://huggingface.co/deepseek-ai/DeepSeek-V3-0324"),

    ("deepseek-r1-0528", "aime_2025", 87.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "aime_2024", 91.4, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "gpqa_diamond", 81.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "math_500", 97.3, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "livecodebench", 73.1, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "swe_bench_verified", 57.6, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),

    ("deepseek-r1-distill-qwen-32b", "aime_2024", 72.6, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-32b", "math_500", 94.3, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-32b", "gpqa_diamond", 62.1, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-32b", "livecodebench", 57.2, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-32b", "codeforces_rating", 1691, "https://github.com/deepseek-ai/DeepSeek-R1"),

    ("deepseek-r1-distill-qwen-14b", "aime_2024", 69.7, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-14b", "math_500", 93.9, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-14b", "codeforces_rating", 1481, "https://github.com/deepseek-ai/DeepSeek-R1"),

    ("deepseek-r1-distill-qwen-7b", "aime_2024", 55.5, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-7b", "math_500", 92.8, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-7b", "gpqa_diamond", 49.1, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-7b", "livecodebench", 37.6, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-7b", "codeforces_rating", 1189, "https://github.com/deepseek-ai/DeepSeek-R1"),

    ("deepseek-v3.2", "aime_2025", 93.1, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2", "gpqa_diamond", 82.4, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2", "mmlu_pro", 85.0, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2", "livecodebench", 83.3, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2", "codeforces_rating", 2386, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2", "hle", 25.1, "https://arxiv.org/abs/2512.02556"),

    ("deepseek-v3.2-speciale", "aime_2025", 96.0, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2-speciale", "gpqa_diamond", 85.7, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2-speciale", "livecodebench", 88.7, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2-speciale", "codeforces_rating", 2701, "https://arxiv.org/abs/2512.02556"),
    ("deepseek-v3.2-speciale", "hmmt_2025", 99.2, "https://arxiv.org/abs/2512.02556"),

    # ═══════════════════════════════════════════════════════════════════════
    # Qwen / Alibaba
    # ═══════════════════════════════════════════════════════════════════════
    ("qwen3-235b", "aime_2025", 81.5, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-235b", "gpqa_diamond", 71.1, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-235b", "livecodebench", 70.7, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-235b", "codeforces_rating", 2056, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-235b", "arena_hard", 95.6, "https://qwenlm.github.io/blog/qwen3/"),

    ("qwen3-4b", "aime_2025", 65.6, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-4b", "math_500", 97.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-4b", "codeforces_rating", 1671, "https://qwenlm.github.io/blog/qwen3/"),

    ("qwen3.5-397b", "gpqa_diamond", 88.4, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("qwen3.5-397b", "swe_bench_verified", 76.4, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("qwen3.5-397b", "mmlu_pro", 87.8, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("qwen3.5-397b", "livecodebench", 83.6, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("qwen3.5-397b", "ifeval", 92.6, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),

    # ═══════════════════════════════════════════════════════════════════════
    # Moonshot AI (Kimi)
    # ═══════════════════════════════════════════════════════════════════════
    ("kimi-k2", "gpqa_diamond", 75.1, "https://medium.com/data-science-in-your-pocket/kimi-k2-benchmarks-explained"),
    ("kimi-k2", "swe_bench_verified", 65.8, "https://medium.com/data-science-in-your-pocket/kimi-k2-benchmarks-explained"),
    ("kimi-k2", "livecodebench", 53.7, "https://medium.com/data-science-in-your-pocket/kimi-k2-benchmarks-explained"),
    ("kimi-k2", "math_500", 97.4, "https://www.hpcwire.com/2025/07/16/moonshot-ai-releases-kimi-k2/"),

    ("kimi-k2-thinking", "gpqa_diamond", 85.7, "https://venturebeat.com/ai/moonshots-kimi-k2-thinking/"),
    ("kimi-k2-thinking", "swe_bench_verified", 71.3, "https://venturebeat.com/ai/moonshots-kimi-k2-thinking/"),
    ("kimi-k2-thinking", "hle", 44.9, "https://venturebeat.com/ai/moonshots-kimi-k2-thinking/"),
    ("kimi-k2-thinking", "browsecomp", 60.2, "https://venturebeat.com/ai/moonshots-kimi-k2-thinking/"),

    ("kimi-k2.5", "hle", 50.2, "https://x.com/Kimi_Moonshot/status/2016024049869324599"),
    ("kimi-k2.5", "browsecomp", 74.9, "https://x.com/Kimi_Moonshot/status/2016024049869324599"),
    ("kimi-k2.5", "swe_bench_verified", 76.8, "https://x.com/Kimi_Moonshot/status/2016024049869324599"),
    ("kimi-k2.5", "mmmu_pro", 78.5, "https://x.com/Kimi_Moonshot/status/2016024049869324599"),
    ("kimi-k2.5", "video_mmu", 86.6, "https://x.com/Kimi_Moonshot/status/2016024049869324599"),

    # ═══════════════════════════════════════════════════════════════════════
    # Zhipu AI (GLM)
    # ═══════════════════════════════════════════════════════════════════════
    ("glm-4.6", "swe_bench_verified", 68.0, "https://llm-stats.com/models/glm-4.6"),
    ("glm-4.6", "gpqa_diamond", 85.7, "https://llm-stats.com/models/glm-4.6"),
    ("glm-4.6", "hle", 24.8, "https://llm-stats.com/models/glm-4.6"),

    ("glm-4.7", "aime_2025", 95.7, "https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7"),
    ("glm-4.7", "gpqa_diamond", 85.7, "https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7"),
    ("glm-4.7", "hle", 42.8, "https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7"),
    ("glm-4.7", "swe_bench_verified", 73.8, "https://vertu.com/lifestyle/glm-4-7-released/"),
    ("glm-4.7", "livecodebench", 84.9, "https://vertu.com/lifestyle/glm-4-7-released/"),
    ("glm-4.7", "hmmt_2025", 97.1, "https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7"),

    # ═══════════════════════════════════════════════════════════════════════
    # ByteDance (Doubao Seed)
    # ═══════════════════════════════════════════════════════════════════════
    ("doubao-seed-2.0-pro", "aime_2025", 98.3, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "gpqa_diamond", 88.9, "https://llm-stats.com/models/seed-2.0-pro"),
    ("doubao-seed-2.0-pro", "swe_bench_verified", 76.5, "https://llm-stats.com/models/seed-2.0-pro"),
    ("doubao-seed-2.0-pro", "browsecomp", 77.3, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "video_mmu", 89.5, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "mmmu", 85.4, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "mathvision", 88.8, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "codeforces_rating", 3020, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),

    # ═══════════════════════════════════════════════════════════════════════
    # Mistral
    # ═══════════════════════════════════════════════════════════════════════
    ("mistral-small-3.1", "mmlu", 80.62, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),
    ("mistral-small-3.1", "humaneval", 88.99, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),
    ("mistral-small-3.1", "ifeval", 82.75, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),
    ("mistral-small-3.1", "mmlu_pro", 66.76, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),

    ("mistral-medium-3", "humaneval", 92.1, "https://apidog.com/blog/mistral-medium-3/"),
    ("mistral-medium-3", "arena_hard", 97.1, "https://apidog.com/blog/mistral-medium-3/"),
    ("mistral-medium-3", "math_500", 91.0, "https://apidog.com/blog/mistral-medium-3/"),

    ("mistral-large-3", "mmlu_pro", 73.11, "https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained"),
    ("mistral-large-3", "math_500", 93.6, "https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained"),
    ("mistral-large-3", "aime_2025", 85.0, "https://www.analyticsvidhya.com/blog/2025/12/mistral-large-3/"),

    ("codestral-25.01", "humaneval", 86.6, "https://blog.getbind.co/2025/01/15/mistral-codestral-25-01/"),
    ("codestral-25.01", "livecodebench", 37.9, "https://blog.getbind.co/2025/01/15/mistral-codestral-25-01/"),

    ("devstral-2", "swe_bench_verified", 72.2, "https://mistral.ai/news/devstral-2-vibe-cli"),

    # ═══════════════════════════════════════════════════════════════════════
    # Microsoft (Phi)
    # ═══════════════════════════════════════════════════════════════════════
    ("phi-4", "mmlu", 84.8, "https://arxiv.org/html/2412.08905v1"),
    ("phi-4", "gpqa_diamond", 56.1, "https://arxiv.org/html/2412.08905v1"),
    ("phi-4", "humaneval", 82.6, "https://arxiv.org/html/2412.08905v1"),
    ("phi-4", "arena_hard", 75.4, "https://arxiv.org/html/2412.08905v1"),

    ("phi-4-reasoning", "aime_2025", 71.4, "https://www.analyticsvidhya.com/blog/2025/05/phi-4-reasoning-models/"),
    ("phi-4-reasoning", "gpqa_diamond", 63.4, "https://ashishchadha11944.medium.com/microsofts-phi-4-reasoning-models/"),
    ("phi-4-reasoning", "livecodebench", 53.8, "https://huggingface.co/microsoft/Phi-4-reasoning"),

    ("phi-4-reasoning-plus", "aime_2025", 77.7, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("phi-4-reasoning-plus", "gpqa_diamond", 69.3, "https://www.gocodeo.com/post/phi-4-reasoning-models/"),
    ("phi-4-reasoning-plus", "livecodebench", 68.8, "https://ashishchadha11944.medium.com/microsofts-phi-4-reasoning-models/"),

    # ═══════════════════════════════════════════════════════════════════════
    # NVIDIA
    # ═══════════════════════════════════════════════════════════════════════
    ("nemotron-ultra-253b", "gpqa_diamond", 76.01, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),
    ("nemotron-ultra-253b", "aime_2025", 72.5, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),
    ("nemotron-ultra-253b", "livecodebench", 66.31, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),
    ("nemotron-ultra-253b", "ifeval", 89.45, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),
    ("nemotron-ultra-253b", "math_500", 97.0, "https://developer.nvidia.com/blog/build-enterprise-ai-agents/"),
    ("nemotron-ultra-253b", "arena_hard", 92.7, "https://medium.com/towards-agi/nvidia-llama-nemotron-outshines-llama-4/"),

    # ═══════════════════════════════════════════════════════════════════════
    # Amazon
    # ═══════════════════════════════════════════════════════════════════════
    ("amazon-nova-pro", "mmlu", 85.9, "https://arxiv.org/html/2506.12103v1"),
    ("amazon-nova-pro", "gpqa_diamond", 46.9, "https://arxiv.org/html/2506.12103v1"),

    ("amazon-nova-premier", "mmlu", 87.4, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),
    ("amazon-nova-premier", "math_500", 82.0, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),
    ("amazon-nova-premier", "swe_bench_verified", 42.4, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),

    # ═══════════════════════════════════════════════════════════════════════
    # Cohere
    # ═══════════════════════════════════════════════════════════════════════
    ("command-a", "mmlu", 85.5, "https://cohere.com/research/papers/command-a-technical-report.pdf"),
    ("command-a", "humaneval", 82.9, "https://apidog.com/blog/mistral-medium-3/"),

    # ═══════════════════════════════════════════════════════════════════════
    # LG AI Research
    # ═══════════════════════════════════════════════════════════════════════
    ("exaone-4.0-32b", "mmlu_pro", 81.8, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "gpqa_diamond", 75.4, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "aime_2025", 85.3, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "livecodebench", 72.6, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "hmmt_2025", 72.9, "https://arxiv.org/html/2507.11407v1"),

    # ═══════════════════════════════════════════════════════════════════════
    # MiniMax
    # ═══════════════════════════════════════════════════════════════════════
    ("minimax-m2", "mmlu_pro", 82.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("minimax-m2", "gpqa_diamond", 78.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("minimax-m2", "swe_bench_verified", 69.4, "https://artificialanalysis.ai/models/minimax-m2"),
    ("minimax-m2", "aa_intelligence_index", 61, "https://artificialanalysis.ai/articles/minimax-m2-benchmarks-and-analysis"),

    # ═══════════════════════════════════════════════════════════════════════
    # Allen AI
    # ═══════════════════════════════════════════════════════════════════════
    ("olmo-2-13b", "mmlu", 67.5, "https://allenai.org/blog/olmo2"),

    # ═══════════════════════════════════════════════════════════════════════
    # Additional Qwen3 small models (thinking mode)
    # ═══════════════════════════════════════════════════════════════════════
    ("qwen3-0.6b", "math_500", 77.6, "https://arxiv.org/abs/2505.09388"),

    ("qwen3-1.7b", "mmlu", 61.0, "https://arxiv.org/abs/2505.09388"),

    ("qwen3-8b", "gpqa_diamond", 63.3, "https://arxiv.org/abs/2505.09388"),

    ("qwen3-14b", "mmlu_pro", 61.03, "https://arxiv.org/abs/2505.09388"),

    ("qwen3-30b-a3b", "gpqa_diamond", 65.8, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-30b-a3b", "codeforces_rating", 1974, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-30b-a3b", "arena_hard", 91.0, "https://qwenlm.github.io/blog/qwen3/"),

    # Qwen3-32B additional scores (thinking mode)
    ("qwen3-32b", "gpqa_diamond", 68.4, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-32b", "aime_2025", 76.67, "https://arxiv.org/abs/2505.09388"),

    # QwQ-32B
    ("qwq-32b", "aime_2024", 79.5, "https://qwenlm.github.io/blog/qwq-32b/"),
    ("qwq-32b", "gpqa_diamond", 65.2, "https://qwenlm.github.io/blog/qwq-32b-preview/"),
    ("qwq-32b", "math_500", 90.6, "https://qwenlm.github.io/blog/qwq-32b-preview/"),
    ("qwq-32b", "livecodebench", 50.0, "https://qwenlm.github.io/blog/qwq-32b-preview/"),

    # ═══════════════════════════════════════════════════════════════════════
    # Additional DeepSeek R1 distills
    # ═══════════════════════════════════════════════════════════════════════
    # DeepSeek-R1-Distill-Qwen-1.5B
    ("deepseek-r1-distill-qwen-1.5b", "aime_2024", 28.9, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("deepseek-r1-distill-qwen-1.5b", "math_500", 83.9, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("deepseek-r1-distill-qwen-1.5b", "gpqa_diamond", 33.8, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("deepseek-r1-distill-qwen-1.5b", "livecodebench", 16.9, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("deepseek-r1-distill-qwen-1.5b", "codeforces_rating", 954, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),

    # DeepSeek-R1-Distill-Qwen-14B additional scores
    ("deepseek-r1-distill-qwen-14b", "gpqa_diamond", 59.1, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    ("deepseek-r1-distill-qwen-14b", "livecodebench", 53.1, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),

    # DeepSeek-R1-Distill-Llama-8B
    ("deepseek-r1-distill-llama-8b", "aime_2024", 50.4, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-r1-distill-llama-8b", "math_500", 89.1, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-r1-distill-llama-8b", "gpqa_diamond", 49.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-r1-distill-llama-8b", "livecodebench", 39.6, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-r1-distill-llama-8b", "codeforces_rating", 1205, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),

    # DeepSeek-R1-Distill-Llama-70B
    ("deepseek-r1-distill-llama-70b", "aime_2024", 70.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    ("deepseek-r1-distill-llama-70b", "math_500", 94.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    ("deepseek-r1-distill-llama-70b", "gpqa_diamond", 65.2, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    ("deepseek-r1-distill-llama-70b", "livecodebench", 57.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    ("deepseek-r1-distill-llama-70b", "codeforces_rating", 1633, "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),

    # ═══════════════════════════════════════════════════════════════════════
    # Liquid AI
    # ═══════════════════════════════════════════════════════════════════════
    ("lfm2.5-1.2b-thinking", "gpqa_diamond", 37.86, "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking"),
    ("lfm2.5-1.2b-thinking", "mmlu_pro", 49.65, "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking"),
    ("lfm2.5-1.2b-thinking", "aime_2025", 31.73, "https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb"),
    ("lfm2.5-1.2b-thinking", "math_500", 87.96, "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking"),
    ("lfm2.5-1.2b-thinking", "ifeval", 88.42, "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking"),

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1+2: New scores from paper mining and aggregator searches
    # ═══════════════════════════════════════════════════════════════════════

    # ── o3-mini (high) — new benchmarks ─────────────────────────────────
    ("o3-mini-high", "mmlu", 86.9, "https://github.com/openai/simple-evals"),
    ("o3-mini-high", "math_500", 97.9, "https://github.com/openai/simple-evals"),
    ("o3-mini-high", "humaneval", 97.6, "https://github.com/openai/simple-evals"),
    ("o3-mini-high", "simpleqa", 13.8, "https://github.com/openai/simple-evals"),
    ("o3-mini-high", "hle", 13.0, "https://scale.com/leaderboard/humanitys_last_exam"),
    ("o3-mini-high", "frontiermath", 11.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("o3-mini-high", "aime_2025", 86.5, "https://www.vals.ai/benchmarks/aime"),
    ("o3-mini-high", "chatbot_arena_elo", 1364, "https://lmarena.ai/"),
    ("o3-mini-high", "arena_hard", 43.0, "https://github.com/lmarena/arena-hard-auto"),

    # ── GPT-4.5 — new benchmarks ───────────────────────────────────────
    ("gpt-4.5", "mmlu", 89.6, "https://openai.com/index/introducing-gpt-4-5/"),
    ("gpt-4.5", "gpqa_diamond", 71.4, "https://openai.com/index/introducing-gpt-4-5/"),
    ("gpt-4.5", "swe_bench_verified", 38.0, "https://openai.com/index/introducing-gpt-4-5/"),
    ("gpt-4.5", "aime_2024", 36.7, "https://openai.com/index/introducing-gpt-4-5/"),
    ("gpt-4.5", "arena_hard", 51.4, "https://github.com/lmarena/arena-hard-auto"),

    # ── GPT-4.1 — new benchmarks ───────────────────────────────────────
    ("gpt-4.1", "mmmu", 74.8, "https://openai.com/index/gpt-4-1/"),
    ("gpt-4.1", "arena_hard", 61.5, "https://github.com/lmarena/arena-hard-auto"),

    # ── GPT-4.1 mini — new benchmarks ──────────────────────────────────
    ("gpt-4.1-mini", "mmmu", 71.1, "https://www.vals.ai/benchmarks/mmmu"),
    ("gpt-4.1-mini", "arena_hard", 28.2, "https://github.com/lmarena/arena-hard-auto"),

    # ── GPT-4.1 nano — new benchmarks ──────────────────────────────────
    ("gpt-4.1-nano", "arena_hard", 10.7, "https://github.com/lmarena/arena-hard-auto"),

    # ── o3 (high) — new benchmarks ─────────────────────────────────────
    ("o3-high", "mmlu", 93.3, "https://github.com/openai/simple-evals"),
    ("o3-high", "humaneval", 88.4, "https://github.com/openai/simple-evals"),
    ("o3-high", "simpleqa", 48.6, "https://github.com/openai/simple-evals"),
    ("o3-high", "hle", 19.78, "https://scale.com/leaderboard/humanitys_last_exam"),
    ("o3-high", "browsecomp", 49.7, "https://www.helicone.ai/blog/o3-and-o4-mini-for-developers"),
    ("o3-high", "mmmu", 80.1, "https://www.vals.ai/benchmarks/mmmu"),

    # ── o4-mini (high) — new benchmarks ────────────────────────────────
    ("o4-mini-high", "mmlu", 90.3, "https://github.com/openai/simple-evals"),
    ("o4-mini-high", "math_500", 98.2, "https://github.com/openai/simple-evals"),
    ("o4-mini-high", "simpleqa", 19.3, "https://github.com/openai/simple-evals"),
    ("o4-mini-high", "hle", 18.90, "https://scale.com/leaderboard/humanitys_last_exam"),
    ("o4-mini-high", "arc_agi_2", 6.1, "https://arcprize.org/arc-agi/2/"),
    ("o4-mini-high", "mmmu", 79.7, "https://www.vals.ai/benchmarks/mmmu"),

    # ── GPT-5 — new benchmarks ─────────────────────────────────────────
    ("gpt-5", "simpleqa", 55.0, "https://cdn.openai.com/gpt-5-system-card.pdf"),
    ("gpt-5", "frontiermath", 25.2, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "livecodebench", 84.5, "https://arxiv.org/html/2512.02556v1"),
    ("gpt-5", "codeforces_rating", 2537, "https://arxiv.org/html/2512.02556v1"),
    ("gpt-5", "hmmt_2025", 88.3, "https://arxiv.org/html/2512.02556v1"),
    ("gpt-5", "swe_bench_pro", 41.78, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # ── gpt-oss-120B — new benchmarks ──────────────────────────────────
    ("gpt-oss-120b", "mmlu", 90.0, "https://arxiv.org/abs/2508.10925"),
    ("gpt-oss-120b", "swe_bench_verified", 62.4, "https://arxiv.org/abs/2508.10925"),
    ("gpt-oss-120b", "aime_2024", 96.6, "https://arxiv.org/abs/2508.10925"),

    # ── gpt-oss-20B — new benchmarks ───────────────────────────────────
    ("gpt-oss-20b", "mmlu", 85.3, "https://arxiv.org/abs/2508.10925"),

    # ── GPT-5.1 — new benchmarks ───────────────────────────────────────
    ("gpt-5.1", "chatbot_arena_elo", 1464, "https://lmarena.ai/"),
    ("gpt-5.1", "mmmu_pro", 85.4, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),

    # ── GPT-5.2 — new benchmarks ───────────────────────────────────────
    ("gpt-5.2", "tau_bench_telecom", 98.7, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "mmmu_pro", 86.5, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "simpleqa", 58.0, "https://llm-stats.com/models/gpt-5.2-2025-12-11"),
    ("gpt-5.2", "ifeval", 95.0, "https://llm-stats.com/models/gpt-5.2-2025-12-11"),
    ("gpt-5.2", "livecodebench", 80.0, "https://llm-stats.com/models/gpt-5.2-2025-12-11"),
    ("gpt-5.2", "humaneval", 95.0, "https://llm-stats.com/models/gpt-5.2-2025-12-11"),
    ("gpt-5.2", "mmlu", 88.0, "https://llm-stats.com/models/gpt-5.2-2025-12-11"),
    ("gpt-5.2", "math_500", 99.4, "https://artificialanalysis.ai/evaluations/math-500"),
    ("gpt-5.2", "mrcr_v2", 70.0, "https://www.datacamp.com/blog/gpt-5-2"),
    ("gpt-5.2", "osworld", 38.2, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("gpt-5.2", "hle", 35.2, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "mmmu", 86.67, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "video_mmu", 90.5, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "chatbot_arena_elo", 1440, "https://lmarena.ai/"),

    # ── GPT-5.3-Codex — new benchmarks ─────────────────────────────────
    ("gpt-5.3-codex", "swe_bench_verified", 56.8, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "humaneval", 93.0, "https://openai.com/index/introducing-gpt-5-3-codex/"),

    # ── Claude 3.7 Sonnet — new benchmarks ─────────────────────────────
    ("claude-3.7-sonnet", "aime_2025", 52.7, "https://artificialanalysis.ai/evaluations/aime-2025"),
    ("claude-3.7-sonnet", "frontiermath", 5.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("claude-3.7-sonnet", "arc_agi_2", 1.0, "https://arcprize.org/leaderboard"),
    ("claude-3.7-sonnet", "mmmu", 71.6, "https://llm-stats.com/benchmarks/mmmu"),
    ("claude-3.7-sonnet", "osworld", 28.0, "https://llm-stats.com/benchmarks/osworld"),
    ("claude-3.7-sonnet", "chatbot_arena_elo", 1372, "https://lmarena.ai/"),
    ("claude-3.7-sonnet", "arena_hard", 63.9, "https://github.com/lmarena/arena-hard-auto"),

    # ── Claude Sonnet 4 — new benchmarks ───────────────────────────────
    ("claude-sonnet-4", "aime_2025", 76.3, "https://www.vals.ai/benchmarks/aime"),
    ("claude-sonnet-4", "mmmu", 74.4, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "swe_bench_pro", 42.70, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # ── Claude Opus 4 — new benchmarks ─────────────────────────────────
    ("claude-opus-4", "aime_2024", 75.5, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "mmmu", 76.5, "https://www.datacamp.com/blog/claude-4"),

    # ── Claude Opus 4.1 — new benchmarks ───────────────────────────────
    ("claude-opus-4.1", "mmlu_pro", 87.92, "https://artificialanalysis.ai/evaluations/mmlu-pro"),
    ("claude-opus-4.1", "swe_bench_pro", 22.7, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # ── Claude Sonnet 4.5 — new benchmarks ─────────────────────────────
    ("claude-sonnet-4.5", "aime_2025", 87.0, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "humaneval", 85.0, "https://www.getpassionfruit.com/blog/gpt-5-1-vs-claude-4-5-sonnet-vs-gemini-3-pro-vs-deepseek-v3-2-the-definitive-2025-ai-model-comparison"),
    ("claude-sonnet-4.5", "mmmu", 77.8, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "mmmu_pro", 63.4, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-sonnet-4.5", "swe_bench_pro", 43.60, "https://scale.com/leaderboard/swe_bench_pro_public"),
    ("claude-sonnet-4.5", "mrcr_v2", 10.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-sonnet-4.5", "mmlu", 86.5, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),

    # ── Claude Opus 4.5 — new benchmarks ───────────────────────────────
    ("claude-opus-4.5", "aime_2025", 92.8, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("claude-opus-4.5", "simpleqa", 72.0, "https://artificialanalysis.ai/articles/claude-opus-4-5-benchmarks-and-analysis"),
    ("claude-opus-4.5", "chatbot_arena_elo", 1468, "https://lmarena.ai/"),
    ("claude-opus-4.5", "mmmu", 80.7, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-opus-4.5", "math_500", 85.0, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-opus-4.5", "ifeval", 90.0, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-opus-4.5", "livecodebench", 68.0, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "video_mmu", 68.4, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("claude-opus-4.5", "aa_lcr", 75.0, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),

    # ── Claude Opus 4.6 — new benchmarks ───────────────────────────────
    ("claude-opus-4.6", "mmmu", 77.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "math_500", 93.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "mmlu", 90.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "chatbot_arena_elo", 1502, "https://lmarena.ai/"),
    ("claude-opus-4.6", "tau_bench_telecom", 98.2, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "swe_bench_pro", 55.6, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # ── Claude Sonnet 4.6 — new benchmarks ─────────────────────────────

    # ── Gemini 2.0 Flash — new benchmarks ──────────────────────────────
    ("gemini-2.0-flash", "gpqa_diamond", 74.2, "https://www.marktechpost.com/2025/01/21/google-ai-releases-gemini-2-0-flash-thinking-model/"),
    ("gemini-2.0-flash", "aime_2024", 73.3, "https://www.marktechpost.com/2025/01/21/google-ai-releases-gemini-2-0-flash-thinking-model/"),
    ("gemini-2.0-flash", "arc_agi_2", 1.0, "https://arcprize.org/leaderboard"),
    ("gemini-2.0-flash", "arena_hard", 50.0, "https://github.com/lmarena/arena-hard-auto"),

    # ── Gemini 2.5 Pro — new benchmarks ────────────────────────────────
    ("gemini-2.5-pro", "mmmu", 81.7, "https://modelcards.withgoogle.com/assets/documents/gemini-2.5-pro.pdf"),
    ("gemini-2.5-pro", "mmlu", 89.8, "https://modelcards.withgoogle.com/assets/documents/gemini-2.5-pro.pdf"),
    ("gemini-2.5-pro", "arena_hard", 90.8, "https://github.com/lmarena/arena-hard-auto"),
    ("gemini-2.5-pro", "chatbot_arena_elo", 1437, "https://lmarena.ai/"),

    # ── Gemini 2.5 Flash — new benchmarks ──────────────────────────────
    ("gemini-2.5-flash", "aime_2025", 72.0, "https://deepmind.google/technologies/gemini/flash/"),
    ("gemini-2.5-flash", "arena_hard", 83.9, "https://github.com/lmarena/arena-hard-auto"),

    # ── Gemma 3 27B — new benchmarks ───────────────────────────────────
    ("gemma-3-27b", "mmlu", 76.9, "https://arxiv.org/abs/2503.19786"),
    ("gemma-3-27b", "arena_hard", 69.9, "https://github.com/lmarena/arena-hard-auto"),

    # ── Gemini 3 Pro — new benchmarks ──────────────────────────────────
    ("gemini-3-pro", "browsecomp", 85.9, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "tau_bench_telecom", 99.3, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "mmlu", 91.8, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "humaneval", 93.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "frontiermath", 38.0, "https://x.com/EpochAIResearch/status/1991945942174761050"),
    ("gemini-3-pro", "codeforces_rating", 2512, "https://www.remio.ai/post/gemini-3-deep-think-achieves-48-4-on-humanity-s-last-exam-and-3455-codeforces-elo"),
    ("gemini-3-pro", "swe_bench_pro", 43.30, "https://scale.com/leaderboard/swe_bench_pro_public"),
    ("gemini-3-pro", "hmmt_2025", 97.5, "https://matharena.ai/"),
    ("gemini-3-pro", "mmmu", 87.51, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "video_mmu", 87.6, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "math_500", 97.3, "https://artificialanalysis.ai/evaluations/math-500"),

    # ── Gemini 3 Flash — new benchmarks ────────────────────────────────
    ("gemini-3-flash", "aime_2025", 90.4, "https://matharena.ai/"),
    ("gemini-3-flash", "mmmu", 87.63, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "video_mmu", 86.9, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("gemini-3-flash", "chatbot_arena_elo", 1473, "https://lmarena.ai/"),

    # ── Gemini 3.1 Pro — new benchmarks ────────────────────────────────
    ("gemini-3.1-pro", "mmlu", 92.6, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "browsecomp", 85.9, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "frontiermath", 40.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("gemini-3.1-pro", "chatbot_arena_elo", 1490, "https://lmarena.ai/"),
    ("gemini-3.1-pro", "osworld", 72.0, "https://www.trendingtopics.eu/gemini-3-1-pro-leads-most-benchmarks/"),

    # ── Llama 4 Maverick — new benchmarks ──────────────────────────────
    ("llama-4-maverick", "arena_hard", 73.5, "https://github.com/lmarena/arena-hard-auto"),

    # ── Grok 3 Beta — new benchmarks ───────────────────────────────────
    ("grok-3-beta", "aime_2025", 93.3, "https://x.ai/news/grok-3"),

    # ── Grok 4 — new benchmarks ────────────────────────────────────────
    ("grok-4", "humaneval", 88.0, "https://automatio.ai/models/grok-4"),
    ("grok-4", "simpleqa", 48.0, "https://automatio.ai/models/grok-4"),
    ("grok-4", "ifeval", 89.2, "https://automatio.ai/models/grok-4"),
    ("grok-4", "mmmu", 75.0, "https://automatio.ai/models/grok-4"),
    ("grok-4", "mmmu_pro", 59.2, "https://automatio.ai/models/grok-4"),
    ("grok-4", "codeforces_rating", 2708, "https://x.ai/news/grok-4"),
    ("grok-4", "chatbot_arena_elo", 1465, "https://lmarena.ai/"),

    # ── Grok 4.1 — new benchmarks ──────────────────────────────────────
    ("grok-4.1", "aime_2025", 94.0, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("grok-4.1", "mmmu_pro", 81.0, "https://llm-stats.com/benchmarks/mmmu-pro"),
    ("grok-4.1", "video_mmu", 87.6, "https://llm-stats.com/benchmarks/videommmu"),
    ("grok-4.1", "gpqa_diamond", 91.0, "https://llm-stats.com/models/grok-4.1"),
    ("grok-4.1", "hle", 44.4, "https://scale.com/leaderboard/humanitys_last_exam"),

    # ── DeepSeek-V3 — new benchmarks ───────────────────────────────────
    ("deepseek-v3", "aime_2024", 39.2, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-v3", "humaneval", 65.2, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "simpleqa", 24.9, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-v3", "codeforces_rating", 1134, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-v3", "chatbot_arena_elo", 1382, "https://lmarena.ai/"),

    # ── DeepSeek-R1 — new benchmarks ───────────────────────────────────
    ("deepseek-r1", "ifeval", 83.3, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "simpleqa", 30.1, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "humaneval", 96.1, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "aime_2025", 70.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1", "hle", 8.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1", "hmmt_2025", 41.7, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1", "chatbot_arena_elo", 1382, "https://lmarena.ai/"),

    # ── DeepSeek-V3-0324 — new benchmarks ──────────────────────────────
    ("deepseek-v3-0324", "mmlu_pro", 81.2, "https://huggingface.co/deepseek-ai/DeepSeek-V3-0324"),

    # ── DeepSeek-R1-0528 — new benchmarks ──────────────────────────────
    ("deepseek-r1-0528", "mmlu_pro", 85.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "simpleqa", 27.8, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "hle", 17.7, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "codeforces_rating", 1930, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "hmmt_2025", 79.4, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "chatbot_arena_elo", 1420, "https://lmarena.ai/"),

    # ── DeepSeek-V3.2 — new benchmarks ─────────────────────────────────
    ("deepseek-v3.2", "swe_bench_verified", 73.0, "https://www.marc0.dev/en/leaderboard"),
    ("deepseek-v3.2", "hmmt_2025", 92.5, "https://arxiv.org/html/2512.02556v1"),
    ("deepseek-v3.2", "browsecomp", 51.4, "https://arxiv.org/html/2512.02556v1"),
    ("deepseek-v3.2", "mmmu_pro", 81.0, "https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond"),
    ("deepseek-v3.2", "math_500", 97.3, "https://artificialanalysis.ai/evaluations/math-500"),

    # ── DeepSeek-V3.2-Speciale — new benchmarks ───────────────────────
    ("deepseek-v3.2-speciale", "hle", 30.6, "https://arxiv.org/html/2512.02556v1"),
    ("deepseek-v3.2-speciale", "swe_bench_verified", 76.0, "https://arxiv.org/html/2512.02556v1"),

    # ── Qwen3-235B — new benchmarks ────────────────────────────────────
    ("qwen3-235b", "aime_2024", 85.7, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("qwen3-235b", "swe_bench_verified", 69.6, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-235b", "hmmt_2025", 62.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("qwen3-235b", "hle", 15.43, "https://scale.com/leaderboard/humanitys_last_exam"),
    ("qwen3-235b", "mmlu_pro", 79.8, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-235b", "math_500", 98.2, "https://arxiv.org/abs/2505.09388"),

    # ── Qwen3-32B — new benchmarks ─────────────────────────────────────
    ("qwen3-32b", "livecodebench", 63.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-32b", "arena_hard", 53.3, "https://github.com/lmarena/arena-hard-auto"),

    # ── Qwen3-4B — new benchmarks ──────────────────────────────────────
    ("qwen3-4b", "arena_hard", 13.2, "https://github.com/lmarena/arena-hard-auto"),
    ("qwen3-4b", "gpqa_diamond", 51.2, "https://arxiv.org/abs/2505.09388"),

    # ── QwQ-32B — new benchmarks ───────────────────────────────────────
    ("qwq-32b", "arena_hard", 60.9, "https://github.com/lmarena/arena-hard-auto"),
    ("qwq-32b", "codeforces_rating", 1316, "https://arxiv.org/html/2501.12948v1"),

    # ── Kimi K2 — new benchmarks ───────────────────────────────────────
    ("kimi-k2", "aime_2025", 49.5, "https://github.com/MoonshotAI/Kimi-K2"),
    ("kimi-k2", "aime_2024", 69.6, "https://github.com/MoonshotAI/Kimi-K2"),
    ("kimi-k2", "swe_bench_pro", 27.67, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # ── Kimi K2 Thinking — new benchmarks ──────────────────────────────
    ("kimi-k2-thinking", "aime_2025", 94.5, "https://arxiv.org/html/2512.02556v1"),
    ("kimi-k2-thinking", "mmlu_pro", 84.6, "https://arxiv.org/html/2512.02556v1"),
    ("kimi-k2-thinking", "livecodebench", 82.6, "https://arxiv.org/html/2512.02556v1"),
    ("kimi-k2-thinking", "hmmt_2025", 89.4, "https://arxiv.org/html/2512.02556v1"),

    # ── Kimi K2.5 — new benchmarks ─────────────────────────────────────
    ("kimi-k2.5", "aime_2025", 96.0, "https://github.com/MoonshotAI/Kimi-K2.5"),

    # ── GLM-4.7 — new benchmarks ───────────────────────────────────────
    ("glm-4.7", "chatbot_arena_elo", 1445, "https://lmarena.ai/"),

    # ── MiniMax-M2 — new benchmarks ────────────────────────────────────
    ("minimax-m2", "hle", 12.5, "https://arxiv.org/html/2512.02556v1"),
    ("minimax-m2", "aime_2025", 78.3, "https://arxiv.org/html/2512.02556v1"),
    ("minimax-m2", "livecodebench", 83.0, "https://arxiv.org/html/2512.02556v1"),
    ("minimax-m2", "browsecomp", 44.0, "https://arxiv.org/html/2512.02556v1"),
    ("minimax-m2", "chatbot_arena_elo", 1408, "https://lmarena.ai/"),

    # ── Mistral Large 3 — new benchmarks ───────────────────────────────
    ("mistral-large-3", "swe_bench_verified", 68.0, "https://llm-stats.com/benchmarks/swe-bench-verified"),

    # ── Phi-4 — new benchmarks ─────────────────────────────────────────
    ("phi-4", "math_500", 80.4, "https://arxiv.org/html/2412.08905v1"),
    ("phi-4", "ifeval", 79.3, "https://arxiv.org/html/2412.08905v1"),

    # ── Phi-4 Reasoning — new benchmarks ───────────────────────────────
    ("phi-4-reasoning", "math_500", 95.0, "https://www.microsoft.com/en-us/research/articles/phi-reasoning/"),

    # ── Phi-4 Reasoning Plus — new benchmarks ──────────────────────────
    ("phi-4-reasoning-plus", "math_500", 96.4, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("phi-4-reasoning-plus", "humaneval", 88.0, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: New model scores
    # ═══════════════════════════════════════════════════════════════════════

    # ── Phi-4-mini ──────────────────────────────────────────────────────
    ("phi-4-mini", "mmlu", 67.3, "https://huggingface.co/microsoft/Phi-4-mini-instruct"),
    ("phi-4-mini", "gpqa_diamond", 30.4, "https://huggingface.co/microsoft/Phi-4-mini-instruct"),
    ("phi-4-mini", "humaneval", 74.4, "https://huggingface.co/microsoft/Phi-4-mini-instruct"),
    ("phi-4-mini", "mmlu_pro", 52.8, "https://huggingface.co/microsoft/Phi-4-mini-instruct"),

    # ── Falcon3-10B-Instruct ───────────────────────────────────────────
    ("falcon3-10b", "ifeval", 78.17, "https://huggingface.co/tiiuae/Falcon3-10B-Instruct"),
    ("falcon3-10b", "mmlu", 73.1, "https://huggingface.co/blog/falcon3"),
    ("falcon3-10b", "mmlu_pro", 42.5, "https://huggingface.co/blog/falcon3"),

    # ── InternLM3-8B-Instruct ──────────────────────────────────────────
    ("internlm3-8b", "mmlu", 76.6, "https://github.com/InternLM/InternLM"),
    ("internlm3-8b", "gpqa_diamond", 37.4, "https://github.com/InternLM/InternLM"),
    ("internlm3-8b", "math_500", 83.0, "https://github.com/InternLM/InternLM"),
    ("internlm3-8b", "humaneval", 82.3, "https://github.com/InternLM/InternLM"),
    ("internlm3-8b", "ifeval", 79.3, "https://github.com/InternLM/InternLM"),

    # ── Seed-Thinking-v1.5 ─────────────────────────────────────────────
    ("seed-thinking-v1.5", "gpqa_diamond", 77.3, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("seed-thinking-v1.5", "aime_2024", 86.7, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("seed-thinking-v1.5", "aime_2025", 74.0, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("seed-thinking-v1.5", "livecodebench", 64.9, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),

    # ── EXAONE 4.0 32B — new benchmarks ────────────────────────────────
    ("exaone-4.0-32b", "math_500", 96.4, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "ifeval", 88.0, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "humaneval", 90.2, "https://arxiv.org/html/2507.11407v1"),
    ("exaone-4.0-32b", "arena_hard", 80.0, "https://arxiv.org/html/2507.11407v1"),

    # ── Nemotron Ultra 253B — new benchmarks ───────────────────────────
    ("nemotron-ultra-253b", "humaneval", 92.0, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),

    # ── Amazon Nova Premier — new benchmarks ───────────────────────────
    ("amazon-nova-premier", "gpqa_diamond", 55.0, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),
    ("amazon-nova-premier", "humaneval", 80.0, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),

    # ── Command A — new benchmarks ─────────────────────────────────────
    ("command-a", "mmlu_pro", 63.5, "https://cohere.com/research/papers/command-a-technical-report.pdf"),
    ("command-a", "ifeval", 84.3, "https://cohere.com/research/papers/command-a-technical-report.pdf"),
    ("command-a", "arena_hard", 72.0, "https://cohere.com/research/papers/command-a-technical-report.pdf"),

    # ── Phase 4 gap-fill: Qwen3 small models (from Qwen3 tech report) ───
    ("qwen3-0.6b", "mmlu", 52.81, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "mmlu_pro", 24.74, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "mmlu", 62.63, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "mmlu_pro", 36.76, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-8b", "mmlu", 76.89, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-8b", "mmlu_pro", 56.73, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-14b", "mmlu", 81.05, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-14b", "mmlu_pro", 61.03, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-14b", "arena_hard", 85.5, "https://arxiv.org/abs/2505.09388"),

    # ── OLMo 2 13B (from HuggingFace model card) ────────────────────────
    ("olmo-2-13b", "mmlu", 67.5, "https://huggingface.co/allenai/OLMo-2-1124-13B"),
    ("olmo-2-13b", "mmlu_pro", 35.1, "https://huggingface.co/allenai/OLMo-2-1124-13B"),

    # ── Gemini 2.5 Flash gap-fill ────────────────────────────────────────
    ("gemini-2.5-flash", "aime_2024", 88.0, "https://modelcards.withgoogle.com/assets/documents/gemini-2.5-flash.pdf"),
    ("gemini-2.5-flash", "livecodebench", 63.9, "https://modelcards.withgoogle.com/assets/documents/gemini-2.5-flash.pdf"),

    # ── Qwen3 Table 19 (arxiv 2505.09388): small model comparison ──────
    # Qwen3-0.6B (thinking mode)
    ("qwen3-0.6b", "gpqa_diamond", 27.9, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "aime_2024", 10.7, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "aime_2025", 15.1, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "ifeval", 59.2, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "arena_hard", 8.5, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "livecodebench", 12.3, "https://arxiv.org/abs/2505.09388"),

    # Qwen3-1.7B (thinking mode)
    ("qwen3-1.7b", "math_500", 93.4, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "gpqa_diamond", 40.1, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "aime_2024", 48.3, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "aime_2025", 36.8, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "ifeval", 72.5, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "arena_hard", 43.1, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "livecodebench", 33.2, "https://arxiv.org/abs/2505.09388"),

    # DeepSeek-R1-Distill-Qwen-1.5B additional scores from Table 19
    ("deepseek-r1-distill-qwen-1.5b", "aime_2025", 22.8, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-distill-qwen-1.5b", "ifeval", 39.9, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-distill-qwen-1.5b", "arena_hard", 4.5, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-distill-qwen-1.5b", "livecodebench", 13.2, "https://arxiv.org/abs/2505.09388"),

    # DeepSeek-R1-Distill-Llama-8B additional scores from Table 19
    ("deepseek-r1-distill-llama-8b", "aime_2025", 27.8, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-distill-llama-8b", "ifeval", 59.0, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-distill-llama-8b", "arena_hard", 17.6, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-distill-llama-8b", "livecodebench", 42.5, "https://arxiv.org/abs/2505.09388"),

    # ── DeepSeek-R1 paper (arxiv 2501.12948) — distill model benchmarks ──
    # DeepSeek-R1-Distill-Qwen-7B (verified from DeepSeek GitHub README)
    ("deepseek-r1-distill-qwen-7b", "gpqa_diamond", 49.1, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-7b", "livecodebench", 37.6, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-7b", "codeforces_rating", 1189, "https://github.com/deepseek-ai/DeepSeek-R1"),

    # DeepSeek-R1-Distill-Qwen-14B (verified from DeepSeek GitHub README)
    ("deepseek-r1-distill-qwen-14b", "gpqa_diamond", 59.1, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-14b", "livecodebench", 53.1, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-14b", "codeforces_rating", 1481, "https://github.com/deepseek-ai/DeepSeek-R1"),

    # DeepSeek-R1-Distill-Qwen-32B (verified from DeepSeek GitHub README)
    ("deepseek-r1-distill-qwen-32b", "gpqa_diamond", 62.1, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-32b", "livecodebench", 57.2, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1-distill-qwen-32b", "codeforces_rating", 1691, "https://github.com/deepseek-ai/DeepSeek-R1"),

    # Qwen3-4B additional (verified from Qwen3 blog / tech report)
    ("qwen3-4b", "aime_2024", 73.8, "https://arxiv.org/abs/2505.09388"),

    # DeepSeek-R1 additional scores from paper Table 3
    ("deepseek-r1", "ifeval", 83.3, "https://arxiv.org/abs/2501.12948"),

    # ═══════════════════════════════════════════════════════════════════════
    # AUTO-MERGED from mined score files (2026-02-23)
    # ═══════════════════════════════════════════════════════════════════════

    # --- aa_intelligence_index ---
    ("gpt-5.2", "aa_intelligence_index", 70.0, "https://x.com/ArtificialAnlys/status/1943166841150644622"),
    ("gemini-2.5-pro", "aa_intelligence_index", 70.0, "https://x.com/ArtificialAnlys/status/1943166841150644622"),

    # --- aime_2024 ---
    ("grok-4", "aime_2024", 94.0, "https://www.llmrumors.com/news/grok-4-the-breakthrough-ai-model-that-changes-everything"),
    ("grok-3-beta", "aime_2024", 93.3, "https://x.ai/news/grok-3"),
    ("qwen3-32b", "aime_2024", 81.4, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("qwen3-30b-a3b", "aime_2024", 78.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-8b", "aime_2024", 76.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("qwen3-14b", "aime_2024", 75.0, "https://arxiv.org/abs/2505.09388"),
    ("gpt-4.1", "aime_2024", 46.5, "https://arxiv.org/abs/2507.20534"),
    ("claude-sonnet-4", "aime_2024", 43.4, "https://arxiv.org/abs/2507.20534"),

    # --- aime_2025 ---
    ("claude-sonnet-4.6", "aime_2025", 97.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("qwen3.5-397b", "aime_2025", 95.0, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("gpt-5.3-codex", "aime_2025", 94.0, "https://automatio.ai/models/gpt-5-3-codex"),
    ("qwen3-30b-a3b", "aime_2025", 85.0, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("glm-4.6", "aime_2025", 80.0, "https://llm-stats.com/models/glm-4.6"),
    ("qwen3-14b", "aime_2025", 72.0, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-8b", "aime_2025", 67.3, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("qwq-32b", "aime_2025", 50.0, "https://matharena.ai/models/qwen_qwq_preview"),
    ("claude-opus-4.1", "aime_2025", 49.6, "https://matharena.ai/models/anthropic_claude_opus_41"),
    ("deepseek-v3-0324", "aime_2025", 46.7, "https://arxiv.org/abs/2507.20534"),
    ("gpt-4.1", "aime_2025", 46.4, "https://medium.com/@leucopsis/how-gpt-5-compares-to-gpt-4-1-21fad92c2a3a"),

    # --- arc_agi_1 ---
    ("gpt-5.2", "arc_agi_1", 90.5, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("gemini-3.1-pro", "arc_agi_1", 90.0, "https://techcrunch.com/2026/02/19/googles-new-gemini-pro-model-has-record-benchmark-scores-again/"),
    ("claude-opus-4.6", "arc_agi_1", 94.0, "https://arcprize.org/arc-agi/1/"),
    ("o3-high", "arc_agi_1", 75.7, "https://arcprize.org/blog/oai-o3-pub-breakthrough"),
    ("grok-4", "arc_agi_1", 66.6, "https://x.ai/news/grok-4"),
    ("gpt-5", "arc_agi_1", 65.7, "https://x.com/arcprize/status/1953508813182767321"),
    ("deepseek-r1", "arc_agi_1", 15.0, "https://arcprize.org/blog/r1-zero-r1-results-analysis"),
    ("kimi-k2-thinking", "arc_agi_1", 12.0, "https://moonshotai.github.io/Kimi-K2/thinking.html"),

    # --- arc_agi_2 ---
    ("gpt-5", "arc_agi_2", 9.9, "https://x.com/arcprize/status/1953508813182767321"),
    ("gemini-2.5-pro", "arc_agi_2", 4.9, "https://llm-stats.com/benchmarks/arc-agi-v2"),

    # --- arena_hard ---
    ("qwen3-8b", "arena_hard", 76.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("claude-opus-4", "arena_hard", 59.7, "https://arxiv.org/abs/2507.20534"),
    ("kimi-k2", "arena_hard", 54.5, "https://arxiv.org/abs/2507.20534"),
    ("claude-sonnet-4", "arena_hard", 51.6, "https://arxiv.org/abs/2507.20534"),
    ("deepseek-v3-0324", "arena_hard", 39.9, "https://arxiv.org/abs/2507.20534"),

    # --- browsecomp ---
    ("claude-sonnet-4.6", "browsecomp", 78.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("gemini-3-flash", "browsecomp", 75.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("glm-4.7", "browsecomp", 65.0, "https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7"),
    ("gpt-5.1", "browsecomp", 60.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    # REMOVED: moved to unverified_cells.py (source shows 28.3 for o4-mini, 49.7 for o3)
    # ("o4-mini-high", "browsecomp", 45.0, "https://www.helicone.ai/blog/o3-and-o4-mini-for-developers"),

    # --- chatbot_arena_elo ---
    ("gpt-5", "chatbot_arena_elo", 1460, "https://lmarena.ai/"),
    ("qwen3-235b", "chatbot_arena_elo", 1410, "https://lmarena.ai/"),
    ("llama-4-scout", "chatbot_arena_elo", 1350, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),

    # --- codeforces_rating ---
    ("gpt-5.2", "codeforces_rating", 2800, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gemini-3.1-pro", "codeforces_rating", 2700, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("claude-opus-4.6", "codeforces_rating", 2650, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("qwen3-32b", "codeforces_rating", 2020, "https://qwenlm.github.io/blog/qwen3/"),
    ("gemini-2.5-pro", "codeforces_rating", 2001, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("gemini-2.5-flash", "codeforces_rating", 1995, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("qwen3-14b", "codeforces_rating", 1900, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-8b", "codeforces_rating", 1850, "https://qwenlm.github.io/blog/qwen3/"),
    ("claude-sonnet-4.5", "codeforces_rating", 1480, "https://arxiv.org/abs/2512.02556"),
    ("qwen3-1.7b", "codeforces_rating", 1200, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-0.6b", "codeforces_rating", 800, "https://qwenlm.github.io/blog/qwen3/"),

    # --- frontiermath ---
    ("claude-sonnet-4.6", "frontiermath", 35.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("gemini-3-flash", "frontiermath", 30.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("glm-4.7", "frontiermath", 20.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("claude-opus-4.1", "frontiermath", 15.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("grok-4", "frontiermath", 13.0, "https://epoch.ai/blog/grok-4-math"),
    ("claude-opus-4", "frontiermath", 10.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("deepseek-r1-0528", "frontiermath", 10.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("deepseek-v3.2", "frontiermath", 8.0, "https://epoch.ai/benchmarks/frontiermath"),
    ("gemini-2.5-pro", "frontiermath", 5.0, "https://epoch.ai/benchmarks/frontiermath"),

    # --- gpqa_diamond ---
    ("kimi-k2.5", "gpqa_diamond", 87.6, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("gpt-5.3-codex", "gpqa_diamond", 81.0, "https://automatio.ai/models/gpt-5-3-codex"),
    ("gemini-2.5-flash", "gpqa_diamond", 78.3, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"),
    ("claude-sonnet-4.6", "gpqa_diamond", 74.1, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("mistral-medium-3", "gpqa_diamond", 68.0, "https://apidog.com/blog/mistral-medium-3/"),
    ("qwen3-14b", "gpqa_diamond", 64.0, "https://arxiv.org/abs/2505.09388"),
    ("gpt-4.1-mini", "gpqa_diamond", 55.0, "https://www.helicone.ai/blog/gpt-4.1-full-developer-guide"),
    ("command-a", "gpqa_diamond", 45.0, "https://cohere.com/research/papers/command-a-technical-report.pdf"),
    ("mistral-large-3", "gpqa_diamond", 43.9, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-small-3.1", "gpqa_diamond", 39.0, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),
    ("falcon3-10b", "gpqa_diamond", 30.0, "https://huggingface.co/blog/falcon3"),
    ("olmo-2-13b", "gpqa_diamond", 25.0, "https://allenai.org/blog/olmo2"),

    # --- gsm8k ---
    ("gpt-5.3-codex", "gsm8k", 99.0, "https://automatio.ai/models/gpt-5-3-codex"),
    ("deepseek-r1", "gsm8k", 97.3, "https://arxiv.org/html/2501.12948v1"),
    # REMOVED: moved to unverified_cells.py (OpenAI doesn't report GSM8K for GPT-5)
    # ("gpt-5", "gsm8k", 96.8, "https://llm-stats.com/benchmarks"),
    # REMOVED: moved to unverified_cells.py (cited paper reports MGSM, not GSM8K)
    # ("phi-4", "gsm8k", 95.3, "https://arxiv.org/html/2412.08905v1"),
    ("qwen3-235b", "gsm8k", 94.39, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-32b", "gsm8k", 93.4, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-14b", "gsm8k", 92.49, "https://arxiv.org/abs/2505.09388"),
    ("gemma-3-27b", "gsm8k", 92.3, "https://arxiv.org/abs/2503.19786"),
    ("qwen3-30b-a3b", "gsm8k", 91.81, "https://arxiv.org/abs/2505.09388"),
    ("llama-4-scout", "gsm8k", 90.6, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),
    ("qwen3-8b", "gsm8k", 89.84, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-v3", "gsm8k", 89.3, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("phi-4-mini", "gsm8k", 88.6, "https://huggingface.co/microsoft/Phi-4-mini-instruct"),
    ("qwen3-4b", "gsm8k", 87.79, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-1.7b", "gsm8k", 75.44, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-0.6b", "gsm8k", 59.59, "https://arxiv.org/abs/2505.09388"),

    # --- hle ---
    ("doubao-seed-2.0-pro", "hle", 40.0, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("claude-sonnet-4.6", "hle", 38.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("gpt-5.1", "hle", 32.0, "https://www.vellum.ai/blog/flagship-model-report"),
    ("claude-sonnet-4.5", "hle", 19.8, "https://huggingface.co/moonshotai/Kimi-K2-Thinking"),
    ("claude-opus-4", "hle", 7.1, "https://arxiv.org/abs/2507.20534"),
    ("claude-sonnet-4", "hle", 5.8, "https://arxiv.org/abs/2507.20534"),
    ("deepseek-v3-0324", "hle", 5.2, "https://arxiv.org/abs/2507.20534"),
    ("kimi-k2", "hle", 4.7, "https://arxiv.org/abs/2507.20534"),
    ("gpt-4.1", "hle", 3.7, "https://arxiv.org/abs/2507.20534"),

    # --- hmmt_2025 ---
    ("gpt-5.2", "hmmt_2025", 99.4, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "hmmt_2025", 95.4, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("claude-opus-4.5", "hmmt_2025", 92.9, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("gemini-2.5-pro", "hmmt_2025", 82.5, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("claude-sonnet-4.5", "hmmt_2025", 79.2, "https://arxiv.org/abs/2512.02556"),
    ("o3-high", "hmmt_2025", 77.5, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("qwen3-30b-a3b", "hmmt_2025", 71.4, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("o4-mini-high", "hmmt_2025", 66.7, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("gemini-2.5-flash", "hmmt_2025", 64.2, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("qwen3-4b", "hmmt_2025", 55.5, "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"),
    ("o3-mini-high", "hmmt_2025", 50.0, "https://matharena.ai/?comp=hmmt--hmmt_feb_2025"),
    ("kimi-k2", "hmmt_2025", 38.8, "https://arxiv.org/abs/2507.20534"),
    ("deepseek-v3-0324", "hmmt_2025", 27.5, "https://arxiv.org/abs/2507.20534"),
    ("gpt-4.1", "hmmt_2025", 19.4, "https://arxiv.org/abs/2507.20534"),
    ("claude-sonnet-4", "hmmt_2025", 15.9, "https://arxiv.org/abs/2507.20534"),
    ("claude-opus-4", "hmmt_2025", 15.9, "https://arxiv.org/abs/2507.20534"),

    # --- humaneval ---
    ("gemini-3.1-pro", "humaneval", 95.0, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gpt-5", "humaneval", 94.0, "https://cdn.openai.com/gpt-5-system-card.pdf"),
    ("claude-sonnet-4.6", "humaneval", 93.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("mistral-large-3", "humaneval", 92.0, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("qwen3.5-397b", "humaneval", 92.0, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("gpt-4.1", "humaneval", 92.0, "https://openai.com/index/gpt-4-1/"),
    ("gpt-oss-120b", "humaneval", 92.0, "https://arxiv.org/abs/2508.10925"),
    ("seed-thinking-v1.5", "humaneval", 90.0, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("minimax-m2", "humaneval", 90.0, "https://artificialanalysis.ai/articles/minimax-m2-benchmarks-and-analysis"),
    ("gpt-4.1-mini", "humaneval", 90.0, "https://www.helicone.ai/blog/gpt-4.1-full-developer-guide"),
    ("qwen3-235b", "humaneval", 90.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("claude-sonnet-4", "humaneval", 88.0, "https://www.datacamp.com/blog/claude-4"),
    ("devstral-2", "humaneval", 88.0, "https://mistral.ai/news/devstral-2-vibe-cli"),
    ("llama-4-behemoth", "humaneval", 85.0, "https://www.datacamp.com/blog/llama-4"),
    ("kimi-k2", "humaneval", 85.0, "https://github.com/MoonshotAI/Kimi-K2"),
    ("qwen3-32b", "humaneval", 85.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-14b", "humaneval", 85.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("llama-4-maverick", "humaneval", 82.0, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("qwen3-8b", "humaneval", 82.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-30b-a3b", "humaneval", 82.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("amazon-nova-pro", "humaneval", 78.0, "https://arxiv.org/html/2506.12103v1"),
    ("gpt-4.1-nano", "humaneval", 78.0, "https://www.datacamp.com/blog/gpt-4-1"),
    ("llama-4-scout", "humaneval", 76.0, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("qwen3-4b", "humaneval", 75.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("falcon3-10b", "humaneval", 72.0, "https://huggingface.co/blog/falcon3"),
    ("qwen3-1.7b", "humaneval", 60.0, "https://arxiv.org/abs/2505.09388"),
    ("olmo-2-13b", "humaneval", 55.0, "https://allenai.org/blog/olmo2"),
    ("qwen3-0.6b", "humaneval", 45.0, "https://arxiv.org/abs/2505.09388"),

    # --- ifeval ---
    ("o4-mini-high", "ifeval", 92.4, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("o3-high", "ifeval", 92.1, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("claude-sonnet-4.6", "ifeval", 92.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("gemini-2.5-pro", "ifeval", 90.8, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("gemini-3-pro", "ifeval", 90.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gpt-5", "ifeval", 90.0, "https://cdn.openai.com/gpt-5-system-card.pdf"),
    ("kimi-k2", "ifeval", 89.8, "https://arxiv.org/abs/2507.20534"),
    ("claude-opus-4", "ifeval", 89.7, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("qwen3-30b-a3b", "ifeval", 88.9, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("minimax-m2", "ifeval", 88.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("o3-mini-high", "ifeval", 88.0, "https://github.com/openai/simple-evals"),
    ("gpt-oss-120b", "ifeval", 88.0, "https://arxiv.org/abs/2508.10925"),
    ("qwen3-235b", "ifeval", 87.8, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("claude-sonnet-4", "ifeval", 87.6, "https://arxiv.org/abs/2507.20534"),
    ("qwen3-4b", "ifeval", 87.4, "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"),
    ("llama-4-behemoth", "ifeval", 86.0, "https://www.datacamp.com/blog/llama-4"),
    ("mistral-medium-3", "ifeval", 86.0, "https://apidog.com/blog/mistral-medium-3/"),
    ("mistral-large-3", "ifeval", 86.0, "https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained"),
    ("seed-thinking-v1.5", "ifeval", 87.4, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("gpt-oss-20b", "ifeval", 85.0, "https://arxiv.org/abs/2508.10925"),
    ("gemini-2.5-flash", "ifeval", 84.3, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"),
    ("llama-4-maverick", "ifeval", 84.0, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("qwen3-32b", "ifeval", 83.0, "https://arxiv.org/abs/2505.09388"),
    ("amazon-nova-premier", "ifeval", 82.0, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),
    ("deepseek-v3-0324", "ifeval", 81.1, "https://arxiv.org/abs/2507.20534"),
    ("llama-4-scout", "ifeval", 81.0, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    # REMOVED: moved to unverified_cells.py (source doesn't report IFEval for distill models)
    # ("deepseek-r1-distill-llama-70b", "ifeval", 81.0, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("qwq-32b", "ifeval", 80.0, "https://qwenlm.github.io/blog/qwq-32b/"),
    ("gemini-2.0-flash", "ifeval", 80.0, "https://www.helicone.ai/blog/gemini-2.0-flash"),
    ("amazon-nova-pro", "ifeval", 80.0, "https://arxiv.org/html/2506.12103v1"),
    ("qwen3-8b", "ifeval", 80.0, "https://arxiv.org/abs/2505.09388"),
    ("deepseek-r1-0528", "ifeval", 79.1, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    # REMOVED: moved to unverified_cells.py (source doesn't report IFEval for distill models)
    # ("deepseek-r1-distill-qwen-32b", "ifeval", 79.0, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("gpt-4.1-nano", "ifeval", 75.0, "https://www.datacamp.com/blog/gpt-4-1"),
    ("olmo-2-13b", "ifeval", 68.0, "https://allenai.org/blog/olmo2"),

    # --- imo_2025 ---
    ("gemini-2.5-pro", "imo_2025", 31.55, "https://matharena.ai/imo/"),
    ("o3-high", "imo_2025", 16.67, "https://matharena.ai/imo/"),
    ("grok-4", "imo_2025", 11.9, "https://matharena.ai/imo/"),

    # --- livecodebench ---
    ("gemini-3-flash", "livecodebench", 90.8, "https://artificialanalysis.ai/evaluations/livecodebench"),
    ("doubao-seed-2.0-pro", "livecodebench", 88.0, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("kimi-k2.5", "livecodebench", 85.0, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("gpt-5.3-codex", "livecodebench", 85.0, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gemini-3.1-pro", "livecodebench", 82.0, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gpt-5.1", "livecodebench", 82.0, "https://www.vellum.ai/blog/flagship-model-report"),
    ("gpt-oss-120b", "livecodebench", 75.0, "https://arxiv.org/abs/2508.10925"),
    ("o3-mini-high", "livecodebench", 74.1, "https://openai.com/index/openai-o3-mini/"),
    ("claude-sonnet-4.6", "livecodebench", 74.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("glm-4.6", "livecodebench", 70.0, "https://llm-stats.com/models/glm-4.6"),
    ("gpt-oss-20b", "livecodebench", 70.0, "https://arxiv.org/abs/2508.10925"),
    ("mistral-large-3", "livecodebench", 66.0, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("qwen3-30b-a3b", "livecodebench", 66.0, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("devstral-2", "livecodebench", 60.0, "https://mistral.ai/news/devstral-2-vibe-cli"),
    ("qwen3-4b", "livecodebench", 55.2, "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"),
    ("qwen3-14b", "livecodebench", 55.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("claude-sonnet-4", "livecodebench", 48.5, "https://arxiv.org/abs/2507.20534"),
    ("qwen3-8b", "livecodebench", 48.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("claude-opus-4", "livecodebench", 47.4, "https://arxiv.org/abs/2507.20534"),
    ("gpt-4.1", "livecodebench", 44.7, "https://arxiv.org/abs/2507.20534"),
    ("mistral-small-3.1", "livecodebench", 30.0, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),
    ("command-a", "livecodebench", 28.0, "https://cohere.com/research/papers/command-a-technical-report.pdf"),

    # --- math_500 ---
    ("gpt-5.1", "math_500", 99.0, "https://artificialanalysis.ai/evaluations/math-500"),
    ("qwen3.5-397b", "math_500", 98.0, "https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17/"),
    ("gpt-oss-120b", "math_500", 98.0, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("kimi-k2-thinking", "math_500", 97.0, "https://venturebeat.com/ai/moonshots-kimi-k2-thinking/"),
    ("minimax-m2", "math_500", 97.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("gpt-oss-20b", "math_500", 97.0, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks"),
    ("qwen3-32b", "math_500", 96.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("qwen3-30b-a3b", "math_500", 96.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("gpt-5.3-codex", "math_500", 96.0, "https://automatio.ai/models/gpt-5-3-codex"),
    ("seed-thinking-v1.5", "math_500", 95.0, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("qwen3-14b", "math_500", 95.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("claude-opus-4", "math_500", 94.4, "https://arxiv.org/abs/2507.20534"),
    ("claude-sonnet-4", "math_500", 94.0, "https://arxiv.org/abs/2507.20534"),
    ("deepseek-v3-0324", "math_500", 94.0, "https://arxiv.org/abs/2507.20534"),
    ("qwen3-8b", "math_500", 94.0, "https://qwenlm.github.io/blog/qwen3/"),
    ("gpt-4.1", "math_500", 92.4, "https://arxiv.org/abs/2507.20534"),
    ("mistral-small-3.1", "math_500", 81.0, "https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model/"),
    ("command-a", "math_500", 80.0, "https://cohere.com/research/papers/command-a-technical-report.pdf"),
    ("amazon-nova-pro", "math_500", 74.0, "https://arxiv.org/html/2506.12103v1"),
    ("codestral-25.01", "math_500", 74.0, "https://blog.getbind.co/2025/01/15/mistral-codestral-25-01/"),

    # --- matharena_apex_2025 ---
    ("gemini-3-pro", "matharena_apex_2025", 23.4, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gpt-5.1", "matharena_apex_2025", 1.04, "https://matharena.ai/models/openai_gpt_51"),
    ("gemini-2.5-pro", "matharena_apex_2025", 0.5, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),

    # --- mmlu ---
    ("gpt-5.3-codex", "mmlu", 94.0, "https://automatio.ai/models/gpt-5-3-codex"),
    ("grok-4", "mmlu", 94.0, "https://forgecode.dev/blog/grok-4-initial-impression/"),
    ("kimi-k2.5", "mmlu", 92.0, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("gpt-5", "mmlu", 91.0, "https://cdn.openai.com/gpt-5-system-card.pdf"),
    ("deepseek-r1-0528", "mmlu", 90.8, "https://medium.com/@leucopsis/deepseeks-new-r1-0528-performance-analysis-and-benchmark-comparisons-6440eac858d6"),
    ("gpt-5.1", "mmlu", 90.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("claude-sonnet-4.6", "mmlu", 90.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("kimi-k2", "mmlu", 89.5, "https://arxiv.org/abs/2507.20534"),
    ("deepseek-v3-0324", "mmlu", 89.4, "https://arxiv.org/abs/2507.20534"),
    ("claude-3.7-sonnet", "mmlu", 89.0, "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("qwen3.5-397b", "mmlu", 88.6, "https://automatio.ai/models/qwen3-5-397b-a17b"),
    ("qwen3-235b", "mmlu", 87.81, "https://arxiv.org/abs/2505.09388"),
    ("mistral-medium-3", "mmlu", 87.0, "https://apidog.com/blog/mistral-medium-3/"),
    ("nemotron-ultra-253b", "mmlu", 87.0, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),
    ("deepseek-r1-distill-llama-70b", "mmlu", 86.0, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("gpt-4.1-mini", "mmlu", 86.0, "https://openai.com/index/gpt-4-1/"),
    ("mistral-large-3", "mmlu", 85.5, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("gemini-2.5-flash", "mmlu", 85.0, "https://deepmind.google/technologies/gemini/flash/"),
    # REMOVED: moved to unverified_cells.py (source doesn't report MMLU for distill models)
    # ("deepseek-r1-distill-qwen-32b", "mmlu", 85.0, "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("exaone-4.0-32b", "mmlu", 84.0, "https://arxiv.org/html/2507.11407v1"),
    ("qwen3-32b", "mmlu", 83.61, "https://arxiv.org/abs/2505.09388"),
    ("qwen3-30b-a3b", "mmlu", 81.38, "https://arxiv.org/abs/2505.09388"),
    ("gemini-2.0-flash", "mmlu", 76.4, "https://www.marktechpost.com/2025/01/21/google-ai-releases-gemini-2-0-flash-thinking-model-gemini-2-0-flash-thinking-exp-01-21-scoring-73-3-on-aime-math-and-74-2-on-gpqa-diamond-science-benchmarks/"),
    ("qwen3-4b", "mmlu", 72.99, "https://arxiv.org/abs/2505.09388"),

    # --- mmlu_pro ---
    ("doubao-seed-2.0-pro", "mmlu_pro", 88.0, "https://llm-stats.com/models/seed-2.0-pro"),
    ("claude-sonnet-4.5", "mmlu_pro", 87.5, "https://huggingface.co/moonshotai/Kimi-K2-Thinking"),
    ("kimi-k2", "mmlu_pro", 87.1, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "mmlu_pro", 87.1, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("grok-4", "mmlu_pro", 87.0, "https://artificialanalysis.ai/models/grok-4"),
    ("gpt-5.2", "mmlu_pro", 86.7, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("claude-opus-4", "mmlu_pro", 86.6, "https://arxiv.org/abs/2507.20534"),
    ("glm-4.7", "mmlu_pro", 86.0, "https://medium.com/@leucopsis/a-technical-analysis-of-glm-4-7"),
    ("o3-high", "mmlu_pro", 85.9, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("gemini-2.5-pro", "mmlu_pro", 85.6, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("grok-4.1", "mmlu_pro", 85.3, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("claude-sonnet-4", "mmlu_pro", 83.7, "https://arxiv.org/abs/2507.20534"),
    ("o4-mini-high", "mmlu_pro", 81.9, "https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507"),
    ("gpt-4.1", "mmlu_pro", 81.8, "https://arxiv.org/abs/2507.20534"),
    ("gemini-2.5-flash", "mmlu_pro", 81.1, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"),
    ("qwen3-30b-a3b", "mmlu_pro", 80.9, "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"),
    ("seed-thinking-v1.5", "mmlu_pro", 87.0, "https://github.com/ByteDance-Seed/Seed-Thinking-v1.5"),
    ("glm-4.6", "mmlu_pro", 80.0, "https://llm-stats.com/models/glm-4.6"),
    ("o3-mini-high", "mmlu_pro", 80.0, "https://github.com/openai/simple-evals"),
    ("claude-sonnet-4.6", "mmlu_pro", 80.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("nemotron-ultra-253b", "mmlu_pro", 76.0, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra/"),
    ("qwen3-4b", "mmlu_pro", 74.0, "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"),
    ("qwen3-32b", "mmlu_pro", 70.0, "https://arxiv.org/abs/2505.09388"),
    ("qwq-32b", "mmlu_pro", 62.0, "https://qwenlm.github.io/blog/qwq-32b/"),

    # --- mmmu ---
    ("gpt-5.1", "mmmu", 84.2, "https://automatio.ai/models/gpt-5-3-codex"),
    ("gpt-5.3-codex", "mmmu", 84.0, "https://automatio.ai/models/gpt-5-3-codex"),
    ("kimi-k2.5", "mmmu", 84.0, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("claude-sonnet-4.6", "mmmu", 74.2, "https://automatio.ai/models/claude-sonnet-4-6"),
    ("amazon-nova-premier", "mmmu", 65.0, "https://aws.amazon.com/blogs/aws/amazon-nova-premier/"),

    # --- mmmu_pro ---
    ("claude-opus-4.6", "mmmu_pro", 77.3, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("o3-high", "mmmu_pro", 75.0, "https://www.vals.ai/benchmarks/mmmu"),

    # --- osworld ---
    ("gemini-3-pro", "osworld", 55.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-flash", "osworld", 50.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("claude-opus-4.1", "osworld", 44.4, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("gpt-5", "osworld", 40.0, "https://www.digitalocean.com/resources/articles/gpt-5-overview"),

    # --- scicode ---
    ("gemini-3-pro", "scicode", 56.1, "https://artificialanalysis.ai/evaluations/scicode"),

    # --- simpleqa ---
    ("claude-sonnet-4.6", "simpleqa", 68.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("gemini-3-flash", "simpleqa", 60.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gpt-5.1", "simpleqa", 55.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-4.1", "simpleqa", 42.3, "https://arxiv.org/abs/2507.20534"),
    ("minimax-m2", "simpleqa", 40.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("deepseek-v3.2", "simpleqa", 35.0, "https://artificialanalysis.ai/models/deepseek-v3-2"),
    ("kimi-k2-thinking", "simpleqa", 35.0, "https://venturebeat.com/ai/moonshots-kimi-k2-thinking/"),
    ("kimi-k2", "simpleqa", 31.0, "https://arxiv.org/abs/2507.20534"),
    ("grok-3-beta", "simpleqa", 31.0, "https://x.ai/news/grok-3"),
    ("exaone-4.0-32b", "simpleqa", 30.0, "https://arxiv.org/html/2507.11407v1"),
    ("deepseek-v3-0324", "simpleqa", 27.7, "https://arxiv.org/abs/2507.20534"),
    ("gemini-2.0-flash", "simpleqa", 27.0, "https://www.helicone.ai/blog/gemini-2.0-flash"),
    ("claude-3.7-sonnet", "simpleqa", 26.0, "https://artificialanalysis.ai/models/claude-3-7-sonnet"),
    ("claude-opus-4", "simpleqa", 22.8, "https://arxiv.org/abs/2507.20534"),
    ("llama-4-maverick", "simpleqa", 20.0, "https://www.llama.com/models/llama-4/"),
    ("gpt-4.1-mini", "simpleqa", 20.0, "https://www.helicone.ai/blog/gpt-4.1-full-developer-guide"),
    ("claude-sonnet-4", "simpleqa", 15.9, "https://arxiv.org/abs/2507.20534"),
    ("qwen3-235b", "simpleqa", 13.2, "https://arxiv.org/abs/2507.20534"),

    # --- swe_bench_pro ---
    ("gpt-5.1", "swe_bench_pro", 45.0, "https://www.vellum.ai/blog/flagship-model-report"),
    ("claude-haiku-4.5", "swe_bench_pro", 39.45, "https://scale.com/leaderboard/swe_bench_pro_public"),
    ("gemini-3-flash", "swe_bench_pro", 34.63, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # --- swe_bench_verified ---
    ("o3-mini-high", "swe_bench_verified", 49.3, "https://openai.com/index/openai-o3-mini/"),
    ("deepseek-v3-0324", "swe_bench_verified", 38.8, "https://arxiv.org/abs/2507.20534"),

    # --- tau_bench_retail ---
    ("claude-sonnet-4.6", "tau_bench_retail", 89.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("gpt-5.2", "tau_bench_retail", 88.0, "https://openai.com/index/introducing-gpt-5-2/"),

    # --- tau_bench_telecom ---
    ("claude-sonnet-4.6", "tau_bench_telecom", 97.0, "https://www.anthropic.com/news/claude-sonnet-4-6"),

    # --- usamo_2025 ---
    ("gemini-2.5-pro", "usamo_2025", 24.0, "https://files.sri.inf.ethz.ch/matharena/usamo_report.pdf"),
    ("o3-high", "usamo_2025", 21.9, "https://files.sri.inf.ethz.ch/matharena/usamo_report.pdf"),
    ("o4-mini-high", "usamo_2025", 19.3, "https://files.sri.inf.ethz.ch/matharena/usamo_report.pdf"),

    # --- video_mmu ---
    ("gemini-2.5-pro", "video_mmu", 84.8, "https://github.com/MME-Benchmarks/Video-MME"),




    # ── Terminal-Bench 2.0 (Terminus 2 standardized agent) ──
    ("gpt-5.3-codex", "terminal_bench", 64.7, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("claude-opus-4.6", "terminal_bench", 62.9, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("claude-opus-4.5", "terminal_bench", 57.8, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gemini-3-pro", "terminal_bench", 56.9, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gpt-5.2", "terminal_bench", 54.0, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gemini-3-flash", "terminal_bench", 51.7, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gpt-5.1", "terminal_bench", 47.6, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("kimi-k2.5", "terminal_bench", 43.2, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("claude-sonnet-4.5", "terminal_bench", 42.8, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("deepseek-v3.2", "terminal_bench", 39.6, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("claude-opus-4.1", "terminal_bench", 38.0, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("kimi-k2-thinking", "terminal_bench", 35.7, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gpt-5", "terminal_bench", 35.2, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("glm-4.7", "terminal_bench", 33.4, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gemini-2.5-pro", "terminal_bench", 32.6, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("minimax-m2", "terminal_bench", 30.0, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("claude-haiku-4.5", "terminal_bench", 28.3, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("kimi-k2", "terminal_bench", 27.8, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("glm-4.6", "terminal_bench", 24.5, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("grok-4", "terminal_bench", 23.1, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gpt-oss-120b", "terminal_bench", 18.7, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gemini-2.5-flash", "terminal_bench", 16.9, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),
    ("gpt-oss-20b", "terminal_bench", 3.1, "https://www.tbench.ai/leaderboard/terminal-bench/2.0"),





    # ── Auto-merged from extra_scores (v3): 402 entries ──
    ("claude-3.7-sonnet", "arc_agi_1", 28.6, "https://arcprize.org/arc-agi/1/"),
    ("claude-3.7-sonnet", "brumo_2025", 65.83, "https://matharena.ai/"),
    ("claude-3.7-sonnet", "codeforces_rating", 1640, "https://automatio.ai/models/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "mmlu_pro", 78, "https://automatio.ai/models/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "smt_2025", 56.6, "https://matharena.ai/"),
    ("claude-3.7-sonnet", "swe_bench_pro", 25.6, "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "terminal_bench_1", 35.2, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("claude-3.7-sonnet", "usamo_2025", 3.65, "https://matharena.ai/"),
    ("claude-haiku-4.5", "arc_agi_1", 47.7, "https://arcprize.org/arc-agi/1/"),
    ("claude-haiku-4.5", "arc_agi_2", 4, "https://arcprize.org/arc-agi/2/"),
    ("claude-haiku-4.5", "humaneval", 91.5, "https://www.anthropic.com/claude/haiku"),
    ("claude-haiku-4.5", "ifeval", 85, "https://www.anthropic.com/claude/haiku"),
    ("claude-haiku-4.5", "livebench", 45.3, "https://livebench.ai/"),
    ("claude-haiku-4.5", "livecodebench", 52, "https://artificialanalysis.ai/models/claude-4-5-haiku"),
    ("claude-haiku-4.5", "math_500", 90.2, "https://www.anthropic.com/claude/haiku"),
    ("claude-haiku-4.5", "mmlu_pro", 75, "https://artificialanalysis.ai/models/claude-4-5-haiku"),
    ("claude-opus-4", "arc_agi_1", 35.7, "https://arcprize.org/arc-agi/1/"),
    ("claude-opus-4", "arc_agi_2", 8.6, "https://arcprize.org/arc-agi/2/"),
    ("claude-opus-4", "codeforces_rating", 1886, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "critpt", 0.3, "https://github.com/CritPt-Benchmark/CritPt"),
    ("claude-opus-4", "osworld", 38.2, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "swe_bench_pro", 35.8, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4", "terminal_bench", 43.2, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "terminal_bench_1", 39, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("claude-opus-4.1", "hle", 35, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "humaneval", 93, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "livebench", 54.5, "https://livebench.ai/"),
    ("claude-opus-4.1", "livecodebench", 63.2, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "simpleqa", 43.5, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "terminal_bench_1", 43.8, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("claude-opus-4.5", "aime_2024", 90, "https://artificialanalysis.ai/articles/claude-opus-4-5-benchmarks-and-analysis"),
    ("claude-opus-4.5", "arc_agi_1", 80, "https://arcprize.org/arc-agi/1/"),
    ("claude-opus-4.5", "codeforces_rating", 2070, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-opus-4.5", "humaneval", 95.1, "https://artificialanalysis.ai/articles/claude-opus-4-5-benchmarks-and-analysis"),
    ("claude-opus-4.5", "livebench", 76, "https://livebench.ai/"),
    ("claude-opus-4.5", "simplebench", 62, "https://lmcouncil.ai/benchmarks"),
    ("claude-opus-4.6", "simplebench", 67.6, "https://lmcouncil.ai/benchmarks"),
    ("claude-sonnet-4", "arc_agi_1", 40, "https://arcprize.org/arc-agi/1/"),
    ("claude-sonnet-4", "arc_agi_2", 5.9, "https://arcprize.org/arc-agi/2/"),
    ("claude-sonnet-4", "ifbench", 42.3, "https://github.com/allenai/IFBench"),
    ("claude-sonnet-4", "osworld", 42, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "terminal_bench", 38.5, "https://eval.16x.engineer/blog/claude-4-opus-sonnet-evaluation-results"),
    ("claude-sonnet-4", "terminal_bench_1", 36.4, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("claude-sonnet-4.5", "aime_2024", 88, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "arc_agi_1", 63.7, "https://arcprize.org/arc-agi/1/"),
    ("claude-sonnet-4.5", "arc_agi_2", 13.6, "https://arcprize.org/arc-agi/2/"),
    ("claude-sonnet-4.5", "brumo_2025", 90.83, "https://matharena.ai/"),
    ("claude-sonnet-4.5", "cmimc_2025", 66.88, "https://matharena.ai/"),
    ("claude-sonnet-4.5", "ifeval", 88.5, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "livebench", 53.7, "https://livebench.ai/"),
    ("claude-sonnet-4.5", "math_500", 95.8, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "matharena_apex_2025", 1.56, "https://matharena.ai/"),
    ("claude-sonnet-4.5", "simpleqa", 47, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-sonnet-4.5", "smt_2025", 83.96, "https://matharena.ai/"),
    ("claude-sonnet-4.5", "terminal_bench_1", 51, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("claude-sonnet-4.6", "arc_agi_1", 55, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "codeforces_rating", 2010, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "math_500", 96.5, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "mmmu_pro", 74.5, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "mrcr_v2", 82, "https://awesomeagents.ai/leaderboards/long-context-benchmarks-leaderboard/"),
    ("claude-sonnet-4.6", "swe_bench_pro", 48.2, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "terminal_bench", 59.1, "https://www.anthropic.com/research/claude-sonnet-4-6-system-card"),
    ("command-a", "aime_2024", 30, "https://www.vellum.ai/llm-leaderboard"),
    ("command-a", "bigcodebench", 33.8, "https://bigcode-bench.github.io/"),
    ("command-a", "simpleqa", 32, "https://www.vellum.ai/llm-leaderboard"),
    ("command-a", "swe_bench_verified", 38, "https://www.vellum.ai/llm-leaderboard"),
    ("deepseek-r1", "arc_agi_2", 1.3, "https://arcprize.org/arc-agi/2/"),
    ("deepseek-r1", "brumo_2025", 80.83, "https://matharena.ai/"),
    ("deepseek-r1", "critpt", 1.1, "https://github.com/CritPt-Benchmark/CritPt"),
    ("deepseek-r1", "ifbench", 38, "https://github.com/allenai/IFBench"),
    ("deepseek-r1", "smt_2025", 66.51, "https://matharena.ai/"),
    ("deepseek-r1", "terminal_bench_1", 5.7, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("deepseek-r1", "usamo_2025", 4.76, "https://matharena.ai/"),
    ("deepseek-r1-0528", "arc_agi_1", 21.2, "https://arcprize.org/arc-agi/1/"),
    ("deepseek-r1-0528", "arc_agi_2", 1.1, "https://arcprize.org/arc-agi/2/"),
    ("deepseek-r1-0528", "brumo_2025", 92.5, "https://matharena.ai/"),
    ("deepseek-r1-0528", "cmimc_2025", 69.38, "https://matharena.ai/"),
    ("deepseek-r1-0528", "humaneval", 85.6, "https://medium.com/@leucopsis/deepseeks-new-r1-0528-performance-analysis-and-benchmark-comparisons-6440eac858d6"),
    ("deepseek-r1-0528", "imo_2025", 6.85, "https://matharena.ai/imo/"),
    ("deepseek-r1-0528", "matharena_apex_2025", 1.04, "https://matharena.ai/"),
    ("deepseek-r1-0528", "smt_2025", 83.02, "https://matharena.ai/"),
    ("deepseek-r1-0528", "usamo_2025", 30.06, "https://matharena.ai/"),
    ("deepseek-r1-distill-llama-70b", "aime_2025", 70, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1-distill-llama-70b", "hmmt_2025", 33.3, "https://arxiv.org/abs/2504.21318"),
    ("deepseek-r1-distill-llama-70b", "humaneval", 80, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1-distill-qwen-32b", "aime_2025", 72.6, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1-distill-qwen-32b", "humaneval", 82, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-v3", "aime_2025", 39.2, "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("deepseek-v3", "bigcodebench", 50, "https://bigcode-bench.github.io/"),
    ("deepseek-v3-0324", "codeforces_rating", 1650, "https://textcortex.com/post/deepseek-v3-review"),
    ("deepseek-v3-0324", "humaneval", 85, "https://textcortex.com/post/deepseek-v3-review"),
    ("deepseek-v3.2", "aime_2024", 93, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "brumo_2025", 96.67, "https://matharena.ai/"),
    ("deepseek-v3.2", "cmimc_2025", 83.75, "https://matharena.ai/"),
    ("deepseek-v3.2", "hmmt_nov_2025", 90, "https://matharena.ai/"),
    ("deepseek-v3.2", "humaneval", 90, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "ifeval", 89, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "livebench", 62.2, "https://livebench.ai/"),
    ("deepseek-v3.2", "matharena_apex_2025", 2.08, "https://matharena.ai/"),
    ("deepseek-v3.2", "mmlu", 90.5, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "smt_2025", 87.74, "https://matharena.ai/"),
    ("deepseek-v3.2-speciale", "aime_2024", 96, "https://medium.com/@leucopsis/deepseek-v3-2-speciale-open-weights-reasoning-close-to-the-frontier-models-d43cd5da22d9"),
    ("deepseek-v3.2-speciale", "brumo_2025", 99.17, "https://matharena.ai/"),
    ("deepseek-v3.2-speciale", "cmimc_2025", 94.38, "https://matharena.ai/"),
    ("deepseek-v3.2-speciale", "hmmt_nov_2025", 93.33, "https://matharena.ai/"),
    ("deepseek-v3.2-speciale", "humaneval", 91.5, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "ifeval", 88, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "imo_2025", 83.3, "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale"),
    ("deepseek-v3.2-speciale", "math_500", 98, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "matharena_apex_2025", 9.38, "https://matharena.ai/"),
    ("deepseek-v3.2-speciale", "mmlu_pro", 87.5, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "smt_2025", 89.15, "https://matharena.ai/"),
    ("doubao-seed-2.0-pro", "aime_2024", 98.3, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "humaneval", 92, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "ifeval", 88.5, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "mmlu", 90, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "terminal_bench", 55.8, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("exaone-4.0-32b", "aime_2024", 72.1, "https://arxiv.org/abs/2504.21318"),
    ("exaone-4.0-32b", "codeforces_rating", 1650, "https://llm-stats.com/benchmarks"),
    ("exaone-4.0-32b", "swe_bench_verified", 45, "https://llm-stats.com/benchmarks/swe-bench-verified"),
    ("falcon3-10b", "aime_2024", 8, "https://llm-stats.com/benchmarks"),
    ("falcon3-10b", "livecodebench", 22, "https://llm-stats.com/benchmarks"),
    ("falcon3-10b", "math_500", 62, "https://llm-stats.com/benchmarks"),
    ("gemini-2.0-flash", "bigcodebench", 45.9, "https://bigcode-bench.github.io/"),
    ("gemini-2.0-flash", "humaneval", 82.6, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.0-flash", "livecodebench", 45.2, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.0-flash", "math_500", 83.9, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.0-flash", "swe_bench_verified", 42, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.5-flash", "arc_agi_1", 33.3, "https://arcprize.org/arc-agi/1/"),
    ("gemini-2.5-flash", "arc_agi_2", 2.5, "https://arcprize.org/arc-agi/2/"),
    ("gemini-2.5-flash", "brumo_2025", 83.33, "https://matharena.ai/"),
    ("gemini-2.5-flash", "cmimc_2025", 50.62, "https://matharena.ai/"),
    ("gemini-2.5-flash", "critpt", 1.1, "https://github.com/CritPt-Benchmark/CritPt"),
    ("gemini-2.5-flash", "hle", 15.6, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "humaneval", 90.2, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "livebench", 47.7, "https://livebench.ai/"),
    ("gemini-2.5-flash", "math_500", 95.2, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "mmmu", 73.5, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "simpleqa", 28.1, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "smt_2025", 75.47, "https://matharena.ai/"),
    ("gemini-2.5-flash", "swe_bench_verified", 63.8, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "terminal_bench_1", 16.8, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("gemini-2.5-pro", "arc_agi_1", 41, "https://arcprize.org/arc-agi/1/"),
    ("gemini-2.5-pro", "brumo_2025", 90, "https://matharena.ai/"),
    ("gemini-2.5-pro", "cmimc_2025", 58.13, "https://matharena.ai/"),
    ("gemini-2.5-pro", "critpt", 2, "https://github.com/CritPt-Benchmark/CritPt"),
    ("gemini-2.5-pro", "hmmt_nov_2025", 66.67, "https://matharena.ai/"),
    ("gemini-2.5-pro", "ifbench", 52.3, "https://github.com/allenai/IFBench"),
    ("gemini-2.5-pro", "livebench", 58.3, "https://livebench.ai/"),
    ("gemini-2.5-pro", "mrcr_v2", 83.1, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-2.5-pro", "simplebench", 62.4, "https://lmcouncil.ai/benchmarks"),
    ("gemini-2.5-pro", "smt_2025", 84.91, "https://matharena.ai/"),
    ("gemini-2.5-pro", "terminal_bench_1", 25.3, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("gemini-3-flash", "aime_2024", 93, "https://medium.com/@leucopsis/gemini-3-flash-preliminary-review-34e7420e3be7"),
    ("gemini-3-flash", "arc_agi_1", 84.7, "https://arcprize.org/arc-agi/1/"),
    ("gemini-3-flash", "arc_agi_2", 33.6, "https://arcprize.org/arc-agi/2/"),
    ("gemini-3-flash", "brumo_2025", 100, "https://matharena.ai/"),
    ("gemini-3-flash", "cmimc_2025", 90.62, "https://matharena.ai/"),
    ("gemini-3-flash", "codeforces_rating", 2100, "https://medium.com/@leucopsis/gemini-3-flash-preliminary-review-34e7420e3be7"),
    ("gemini-3-flash", "hmmt_nov_2025", 93.33, "https://matharena.ai/"),
    ("gemini-3-flash", "ifeval", 88.2, "https://automatio.ai/models/gemini-3-flash"),
    ("gemini-3-flash", "livebench", 72.4, "https://livebench.ai/"),
    ("gemini-3-flash", "matharena_apex_2025", 15.62, "https://matharena.ai/"),
    ("gemini-3-flash", "smt_2025", 92.92, "https://matharena.ai/"),
    ("gemini-3-flash", "tau_bench_retail", 82, "https://artificialanalysis.ai/articles/gemini-3-flash-everything-you-need-to-know"),
    ("gemini-3-pro", "aime_2024", 97, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "arc_agi_1", 75, "https://arcprize.org/arc-agi/1/"),
    ("gemini-3-pro", "brumo_2025", 98.33, "https://matharena.ai/"),
    ("gemini-3-pro", "cmimc_2025", 90, "https://matharena.ai/"),
    ("gemini-3-pro", "critpt", 9.1, "https://github.com/CritPt-Benchmark/CritPt"),
    ("gemini-3-pro", "hmmt_nov_2025", 93.33, "https://matharena.ai/"),
    ("gemini-3-pro", "livebench", 73.4, "https://livebench.ai/"),
    ("gemini-3-pro", "smt_2025", 93.4, "https://matharena.ai/"),
    ("gemini-3-pro", "tau_bench_retail", 88.5, "https://artificialanalysis.ai/articles/gemini-3-pro-everything-you-need-to-know"),
    ("gemini-3.1-pro", "aime_2024", 98, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "aime_2026", 97, "https://medium.com/@leucopsis/gemini-3-1-pro-review-1403a8aa1a96"),
    ("gemini-3.1-pro", "math_500", 98.5, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemini-3.1-pro", "matharena_apex_2025", 33.5, "https://www.trendingtopics.eu/gemini-3-1-pro-leads-most-benchmarks-but-trails-claude-opus-4-6-in-some-tasks/"),
    ("gemini-3.1-pro", "mmmu", 87.5, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "simplebench", 79.6, "https://lmcouncil.ai/benchmarks"),
    ("gemini-3.1-pro", "swe_bench_pro", 54.2, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemini-3.1-pro", "tau_bench_retail", 90.5, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "terminal_bench", 68.5, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemma-3-27b", "aime_2024", 22, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "ifeval", 78, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "math_500", 78, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "simpleqa", 22, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "swe_bench_verified", 32, "https://llm-stats.com/benchmarks"),
    ("glm-4.6", "humaneval", 82, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.6", "ifeval", 82, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.6", "mmlu", 85, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "livebench", 58.1, "https://livebench.ai/"),
    ("glm-4.7", "mmlu", 90.1, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "simpleqa", 32, "https://llm-stats.com/models/glm-4.7"),
    ("gpt-4.1", "arc_agi_1", 5.5, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.1", "arc_agi_2", 0.4, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.1", "codeforces_rating", 1807, "https://openai.com/index/gpt-4-1/"),
    ("gpt-4.1", "mrcr_v2", 80, "https://awesomeagents.ai/leaderboards/long-context-benchmarks-leaderboard/"),
    ("gpt-4.1", "terminal_bench_1", 30.3, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("gpt-4.1-mini", "arc_agi_1", 3.5, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.1-mini", "arc_agi_2", 0, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.1-mini", "bigcodebench", 48.9, "https://bigcode-bench.github.io/"),
    ("gpt-4.1-nano", "arc_agi_1", 0, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.1-nano", "arc_agi_2", 0, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.5", "arc_agi_1", 10.3, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.5", "arc_agi_2", 0.8, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.5", "humaneval", 86.6, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),
    ("gpt-4.5", "ifeval", 86.5, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),
    ("gpt-4.5", "mmlu_pro", 74.3, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),
    ("gpt-5", "aa_intelligence_index", 68, "https://artificialanalysis.ai/leaderboards/models"),
    ("gpt-5", "aime_2024", 94.6, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "brumo_2025", 91.67, "https://matharena.ai/"),
    ("gpt-5", "cmimc_2025", 90, "https://matharena.ai/"),
    ("gpt-5", "critpt", 5.7, "https://github.com/CritPt-Benchmark/CritPt"),
    ("gpt-5", "hmmt_nov_2025", 89.17, "https://matharena.ai/"),
    ("gpt-5", "imo_2025", 38.1, "https://matharena.ai/"),
    ("gpt-5", "livebench", 70.5, "https://livebench.ai/"),
    ("gpt-5", "matharena_apex_2025", 1.04, "https://matharena.ai/"),
    ("gpt-5", "simplebench", 61.6, "https://lmcouncil.ai/benchmarks"),
    ("gpt-5", "smt_2025", 91.98, "https://matharena.ai/"),
    ("gpt-5", "terminal_bench_1", 41.3, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("gpt-5.1", "arc_agi_1", 72.8, "https://arcprize.org/arc-agi/1/"),
    ("gpt-5.1", "brumo_2025", 93.33, "https://matharena.ai/"),
    ("gpt-5.1", "cmimc_2025", 91.88, "https://matharena.ai/"),
    ("gpt-5.1", "hmmt_nov_2025", 91.67, "https://matharena.ai/"),
    ("gpt-5.1", "mmlu_pro", 87.5, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.1", "smt_2025", 91.04, "https://matharena.ai/"),
    ("gpt-5.2", "aime_2024", 100, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "brumo_2025", 98.33, "https://matharena.ai/"),
    ("gpt-5.2", "cmimc_2025", 91.25, "https://matharena.ai/"),
    ("gpt-5.2", "hmmt_nov_2025", 95.83, "https://matharena.ai/"),
    ("gpt-5.2", "matharena_apex_2025", 13.54, "https://matharena.ai/"),
    ("gpt-5.2", "smt_2025", 91.98, "https://matharena.ai/"),
    ("gpt-5.3-codex", "ifeval", 92, "https://automatio.ai/models/gpt-5-3-codex"),
    ("gpt-oss-20b", "aime_2024", 98.7, "https://llm-stats.com/models/compare/gpt-oss-120b-vs-gpt-oss-20b"),
    ("gpt-oss-20b", "codeforces_rating", 1985, "https://arxiv.org/html/2508.10925v1"),
    ("gpt-oss-20b", "humaneval", 85, "https://arxiv.org/html/2508.10925v1"),
    ("gpt-oss-20b", "swe_bench_verified", 52, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks-how-it-compares-to-glm-4.5-qwen3-deepseek-and-kimi-k2"),
    ("grok-3-beta", "arc_agi_1", 5.5, "https://arcprize.org/arc-agi/1/"),
    ("grok-3-beta", "arc_agi_2", 0, "https://arcprize.org/arc-agi/2/"),
    ("grok-3-beta", "hle", 18.2, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "humaneval", 87.3, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "ifeval", 84, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "mmlu", 88, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "swe_bench_verified", 48.5, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "terminal_bench_1", 17.5, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("grok-4", "aa_lcr", 68, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),
    ("grok-4", "brumo_2025", 95, "https://matharena.ai/"),
    ("grok-4", "cmimc_2025", 83.75, "https://matharena.ai/"),
    ("grok-4", "hmmt_nov_2025", 88.33, "https://matharena.ai/"),
    ("grok-4", "math_500", 98, "https://aitoolapp.com/grok-4/benchmarks/"),
    ("grok-4", "matharena_apex_2025", 2.08, "https://matharena.ai/"),
    ("grok-4", "osworld", 48, "https://aitoolapp.com/grok-4/benchmarks/"),
    ("grok-4", "smt_2025", 85.85, "https://matharena.ai/"),
    ("grok-4", "swe_bench_pro", 46.5, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "terminal_bench_1", 39, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("grok-4", "usamo_2025", 61.9, "https://ai-stack.ai/en/grok-4"),
    ("grok-4.1", "arc_agi_2", 42, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "brumo_2025", 97.5, "https://matharena.ai/"),
    ("grok-4.1", "cmimc_2025", 84.38, "https://matharena.ai/"),
    ("grok-4.1", "codeforces_rating", 2650, "https://www.glbgpt.com/hub/chatgpt-5-1-vs-grok-4-1-2025/"),
    ("grok-4.1", "frontiermath", 38, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "hmmt_nov_2025", 93.33, "https://matharena.ai/"),
    ("grok-4.1", "humaneval", 95, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "ifeval", 91, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("grok-4.1", "livebench", 60, "https://livebench.ai/"),
    ("grok-4.1", "livecodebench", 82, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("grok-4.1", "math_500", 98.5, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("grok-4.1", "matharena_apex_2025", 5.21, "https://matharena.ai/"),
    ("grok-4.1", "osworld", 52, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "simpleqa", 55, "https://www.glbgpt.com/hub/chatgpt-5-1-vs-grok-4-1-2025/"),
    ("grok-4.1", "smt_2025", 84.6, "https://matharena.ai/"),
    ("internlm3-8b", "aime_2024", 12, "https://llm-stats.com/benchmarks"),
    ("internlm3-8b", "livecodebench", 32, "https://llm-stats.com/benchmarks"),
    ("internlm3-8b", "mmlu_pro", 52, "https://llm-stats.com/benchmarks"),
    ("kimi-k2", "codeforces_rating", 1780, "https://medium.com/data-science-in-your-pocket/kimi-k2-benchmarks-explained-5b25dd6d3a3e"),
    ("kimi-k2", "osworld", 38, "https://arxiv.org/html/2507.20534v1"),
    ("kimi-k2-thinking", "aime_2024", 94, "https://felloai.com/new-chinese-model-kimi-k2-thinking-ranks-1-in-multiple-benchmarks/"),
    ("kimi-k2-thinking", "brumo_2025", 93.33, "https://matharena.ai/"),
    ("kimi-k2-thinking", "cmimc_2025", 91.88, "https://matharena.ai/"),
    ("kimi-k2-thinking", "codeforces_rating", 2150, "https://felloai.com/new-chinese-model-kimi-k2-thinking-ranks-1-in-multiple-benchmarks/"),
    ("kimi-k2-thinking", "hmmt_nov_2025", 89.17, "https://matharena.ai/"),
    ("kimi-k2-thinking", "humaneval", 92, "https://felloai.com/new-chinese-model-kimi-k2-thinking-ranks-1-in-multiple-benchmarks/"),
    ("kimi-k2-thinking", "matharena_apex_2025", 0, "https://matharena.ai/"),
    ("kimi-k2-thinking", "smt_2025", 91.04, "https://matharena.ai/"),
    ("kimi-k2.5", "aime_2024", 96.1, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("kimi-k2.5", "arc_agi_1", 65.3, "https://arcprize.org/arc-agi/1/"),
    ("kimi-k2.5", "arc_agi_2", 35, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "brumo_2025", 98.33, "https://matharena.ai/"),
    ("kimi-k2.5", "cmimc_2025", 91.25, "https://matharena.ai/"),
    ("kimi-k2.5", "codeforces_rating", 2350, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("kimi-k2.5", "frontiermath", 28, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "hmmt_nov_2025", 89.17, "https://matharena.ai/"),
    ("kimi-k2.5", "humaneval", 95, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "ifeval", 90, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "math_500", 98, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("kimi-k2.5", "matharena_apex_2025", 8.85, "https://matharena.ai/"),
    ("kimi-k2.5", "mathvision", 84.2, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "osworld", 62, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "simpleqa", 45, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "smt_2025", 90.57, "https://matharena.ai/"),
    ("lfm2.5-1.2b-thinking", "humaneval", 55, "https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models"),
    ("lfm2.5-1.2b-thinking", "livecodebench", 22, "https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models"),
    ("lfm2.5-1.2b-thinking", "mmlu", 52, "https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models"),
    ("llama-4-behemoth", "aime_2024", 72, "https://www.llama.com/models/llama-4/"),
    ("llama-4-behemoth", "mmlu", 90.2, "https://medium.com/@divyanshbhatiajm19/metas-llama-4-family-the-complete-guide-to-scout-maverick-and-behemoth-ai-models-in-2025-21a90c882e8a"),
    ("llama-4-behemoth", "simpleqa", 44, "https://www.llama.com/models/llama-4/"),
    ("llama-4-behemoth", "swe_bench_verified", 55, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),
    ("llama-4-maverick", "aime_2024", 52, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "arc_agi_1", 4.4, "https://arcprize.org/arc-agi/1/"),
    ("llama-4-maverick", "arc_agi_2", 0, "https://arcprize.org/arc-agi/2/"),
    ("llama-4-maverick", "bigcodebench", 49.7, "https://bigcode-bench.github.io/"),
    ("llama-4-maverick", "critpt", 0, "https://github.com/CritPt-Benchmark/CritPt"),
    ("llama-4-maverick", "mmlu", 88.6, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "swe_bench_verified", 46.5, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),
    ("llama-4-maverick", "terminal_bench_1", 15.5, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("llama-4-scout", "arc_agi_1", 0.5, "https://arcprize.org/arc-agi/1/"),
    ("llama-4-scout", "arc_agi_2", 0, "https://arcprize.org/arc-agi/2/"),
    ("llama-4-scout", "arena_hard", 72, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),
    ("llama-4-scout", "math_500", 83, "https://www.llama.com/models/llama-4/"),
    ("llama-4-scout", "mmlu", 88.5, "https://medium.com/@divyanshbhatiajm19/metas-llama-4-family-the-complete-guide-to-scout-maverick-and-behemoth-ai-models-in-2025-21a90c882e8a"),
    ("minimax-m2", "aime_2024", 65, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "codeforces_rating", 1700, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "mmlu", 87, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "mmmu", 75, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "swe_bench_pro", 32, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "terminal_bench_1", 42, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("mistral-large-3", "aime_2024", 53.3, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "chatbot_arena_elo", 1418, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "codeforces_rating", 1550, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "simpleqa", 30, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-medium-3", "livecodebench", 42, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-medium-3", "mmlu_pro", 72, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-medium-3", "simpleqa", 25, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-medium-3", "swe_bench_verified", 32, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-small-3.1", "aime_2024", 22, "https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/"),
    ("mistral-small-3.1", "simpleqa", 18, "https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/"),
    ("mistral-small-3.1", "swe_bench_verified", 28, "https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/"),
    ("nemotron-ultra-253b", "codeforces_rating", 1750, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("nemotron-ultra-253b", "hle", 15, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("nemotron-ultra-253b", "simpleqa", 35, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("nemotron-ultra-253b", "swe_bench_verified", 48, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("o3-high", "aa_lcr", 69, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),
    ("o3-high", "brumo_2025", 95.83, "https://matharena.ai/"),
    ("o3-high", "cmimc_2025", 79.38, "https://matharena.ai/"),
    ("o3-high", "critpt", 1.4, "https://github.com/CritPt-Benchmark/CritPt"),
    ("o3-high", "ifbench", 69.3, "https://github.com/allenai/IFBench"),
    ("o3-high", "smt_2025", 87.74, "https://matharena.ai/"),
    ("o3-high", "terminal_bench_1", 30.2, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("o3-mini-high", "arc_agi_1", 34.5, "https://arcprize.org/arc-agi/1/"),
    ("o3-mini-high", "arc_agi_2", 3, "https://arcprize.org/arc-agi/2/"),
    ("o3-mini-high", "usamo_2025", 2.08, "https://matharena.ai/"),
    ("o4-mini-high", "arc_agi_1", 58.7, "https://arcprize.org/arc-agi/1/"),
    ("o4-mini-high", "brumo_2025", 86.67, "https://matharena.ai/"),
    ("o4-mini-high", "cmimc_2025", 84.38, "https://matharena.ai/"),
    ("o4-mini-high", "critpt", 0.6, "https://github.com/CritPt-Benchmark/CritPt"),
    ("o4-mini-high", "imo_2025", 14.29, "https://matharena.ai/"),
    ("o4-mini-high", "smt_2025", 88.68, "https://matharena.ai/"),
    ("o4-mini-high", "terminal_bench_1", 18.5, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("olmo-2-13b", "aime_2024", 5, "https://allenai.org/blog/olmo3"),
    ("olmo-2-13b", "livecodebench", 18, "https://allenai.org/blog/olmo3"),
    ("olmo-2-13b", "math_500", 38, "https://allenai.org/blog/olmo3"),
    ("phi-4", "aime_2024", 18, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4", "bigcodebench", 45.5, "https://bigcode-bench.github.io/"),
    ("phi-4", "livecodebench", 38.5, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4", "mmlu_pro", 68, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4", "simpleqa", 15, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4-mini", "ifeval", 72, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4-mini", "livecodebench", 28, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4-mini", "math_500", 75, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4-reasoning-plus", "aime_2024", 70, "https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf"),
    ("phi-4-reasoning-plus", "arena_hard", 79, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning-plus", "codeforces_rating", 1500, "https://huggingface.co/microsoft/Phi-4-reasoning"),
    ("phi-4-reasoning-plus", "hmmt_2025", 53.6, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning-plus", "ifeval", 82, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("phi-4-reasoning-plus", "mmlu_pro", 72, "https://huggingface.co/microsoft/Phi-4-reasoning"),
    ("phi-4-reasoning-plus", "swe_bench_verified", 32, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("qwen3-235b", "aa_lcr", 67, "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning"),
    ("qwen3-235b", "arc_agi_1", 11, "https://arcprize.org/arc-agi/1/"),
    ("qwen3-235b", "arc_agi_2", 1.3, "https://arcprize.org/arc-agi/2/"),
    ("qwen3-235b", "matharena_apex_2025", 5.21, "https://matharena.ai/"),
    ("qwen3-235b", "terminal_bench_1", 6.6, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("qwen3-32b", "ifbench", 37.3, "https://github.com/allenai/IFBench"),
    ("qwen3-32b", "terminal_bench_1", 15.5, "https://www.tbench.ai/leaderboard/terminal-bench/1.0"),
    ("qwen3-8b", "ifbench", 35, "https://github.com/allenai/IFBench"),
    ("qwen3.5-397b", "aime_2024", 94, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "aime_2026", 91.3, "https://www.digitalapplied.com/blog/qwen-3-5-agentic-ai-benchmarks-guide"),
    ("qwen3.5-397b", "codeforces_rating", 2200, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "hle", 32, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "ifbench", 76.5, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "mathvision", 90.3, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "mmmu", 85, "https://www.digitalapplied.com/blog/qwen-3-5-agentic-ai-benchmarks-guide"),
    ("qwen3.5-397b", "simpleqa", 35, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwq-32b", "bigcodebench", 44.6, "https://bigcode-bench.github.io/"),
    ("qwq-32b", "hmmt_2025", 47.5, "https://arxiv.org/abs/2504.21318"),
    ("qwq-32b", "humaneval", 78, "https://medium.com/towards-agi/qwq-32b-preview-benchmarks-revolutionizing-ai-reasoning-capabilities-b2014a00c208"),
    ("qwq-32b", "mmlu", 79, "https://medium.com/towards-agi/qwq-32b-preview-benchmarks-revolutionizing-ai-reasoning-capabilities-b2014a00c208"),
    ("qwq-32b", "swe_bench_verified", 35, "https://medium.com/towards-agi/qwq-32b-preview-benchmarks-revolutionizing-ai-reasoning-capabilities-b2014a00c208"),
    ("seed-thinking-v1.5", "hle", 22, "https://llm-stats.com/benchmarks"),
    ("seed-thinking-v1.5", "mmlu", 87, "https://llm-stats.com/benchmarks"),
    ("seed-thinking-v1.5", "simpleqa", 35, "https://llm-stats.com/benchmarks"),
    ("seed-thinking-v1.5", "swe_bench_verified", 52, "https://llm-stats.com/benchmarks"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Build Excel file
# ═══════════════════════════════════════════════════════════════════════════

def build_excel():
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    # Build lookup dicts
    model_ids = [m[0] for m in MODELS]
    model_info = {m[0]: m for m in MODELS}
    bench_ids = [b[0] for b in BENCHMARKS]
    bench_info = {b[0]: b for b in BENCHMARKS}

    # Build score and URL matrices
    scores = {}  # (model_id, bench_id) -> score
    urls = {}    # (model_id, bench_id) -> url
    for model_id, bench_id, score, url in DATA:
        if model_id in model_info and bench_id in bench_info:
            scores[(model_id, bench_id)] = score
            urls[(model_id, bench_id)] = url

    wb = openpyxl.Workbook()

    # ── Styles ────────────────────────────────────────────────────────────
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    meta_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    # ── Sheet 1: Scores ──────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Scores"

    # Header row: model metadata + benchmark names
    meta_cols = ["Model", "Provider", "Release Date", "Params (M)", "Active (M)",
                 "Architecture", "Reasoning?", "Open Weights?"]
    headers = meta_cols + [bench_info[b][1] for b in bench_ids]

    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', wrap_text=True)
        cell.border = thin_border

    # Data rows
    for row_idx, model_id in enumerate(model_ids, 2):
        m = model_info[model_id]
        meta_values = [
            m[1],  # display_name
            m[2],  # provider
            m[3],  # release_date
            m[4],  # params_total
            m[5],  # params_active
            m[6],  # architecture
            "Yes" if m[7] else "No",
            "Yes" if m[8] else "No",
        ]
        for col_idx, val in enumerate(meta_values, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.fill = meta_fill
            cell.border = thin_border

        for bench_idx, bench_id in enumerate(bench_ids):
            col_idx = len(meta_cols) + bench_idx + 1
            score = scores.get((model_id, bench_id))
            cell = ws.cell(row=row_idx, column=col_idx, value=score)
            cell.border = thin_border
            if score is not None:
                cell.number_format = '0.0'
                cell.alignment = Alignment(horizontal='center')

    # Freeze panes and column widths
    ws.freeze_panes = 'I2'  # Freeze metadata columns + header row
    for i in range(1, len(meta_cols) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 15
    for i in range(len(meta_cols) + 1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 14

    # ── Sheet 2: References ──────────────────────────────────────────────
    ws2 = wb.create_sheet("References")
    for col_idx, header in enumerate(headers, 1):
        cell = ws2.cell(row=1, column=col_idx, value=header)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', wrap_text=True)
        cell.border = thin_border

    for row_idx, model_id in enumerate(model_ids, 2):
        m = model_info[model_id]
        meta_values = [m[1], m[2], m[3], m[4], m[5], m[6],
                       "Yes" if m[7] else "No", "Yes" if m[8] else "No"]
        for col_idx, val in enumerate(meta_values, 1):
            cell = ws2.cell(row=row_idx, column=col_idx, value=val)
            cell.fill = meta_fill
            cell.border = thin_border

        for bench_idx, bench_id in enumerate(bench_ids):
            col_idx = len(meta_cols) + bench_idx + 1
            url = urls.get((model_id, bench_id))
            cell = ws2.cell(row=row_idx, column=col_idx, value=url)
            cell.border = thin_border
            if url:
                cell.font = Font(color="0563C1", underline="single", size=9)
                cell.alignment = Alignment(wrap_text=True)

    ws2.freeze_panes = 'I2'
    for i in range(1, len(meta_cols) + 1):
        ws2.column_dimensions[get_column_letter(i)].width = 15
    for i in range(len(meta_cols) + 1, len(headers) + 1):
        ws2.column_dimensions[get_column_letter(i)].width = 30

    # ── Sheet 3: Benchmark Info ──────────────────────────────────────────
    ws3 = wb.create_sheet("Benchmark Info")
    bench_headers = ["Benchmark ID", "Display Name", "Category", "Metric",
                     "Num Problems", "Source URL", "Models with Scores",
                     "Min Score", "Max Score", "Mean Score"]
    for col_idx, h in enumerate(bench_headers, 1):
        cell = ws3.cell(row=1, column=col_idx, value=h)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.border = thin_border

    for row_idx, bench_id in enumerate(bench_ids, 2):
        b = bench_info[bench_id]
        bench_scores = [scores[(m, bench_id)] for m in model_ids
                        if (m, bench_id) in scores]
        n_models = len(bench_scores)
        min_s = min(bench_scores) if bench_scores else None
        max_s = max(bench_scores) if bench_scores else None
        mean_s = sum(bench_scores) / len(bench_scores) if bench_scores else None

        vals = [b[0], b[1], b[2], b[3], b[4], b[5], n_models, min_s, max_s, mean_s]
        for col_idx, val in enumerate(vals, 1):
            cell = ws3.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            if isinstance(val, float):
                cell.number_format = '0.0'

    for i in range(1, len(bench_headers) + 1):
        ws3.column_dimensions[get_column_letter(i)].width = 18

    # ── Sheet 4: Flat Format ─────────────────────────────────────────────
    ws4 = wb.create_sheet("Flat Format")
    flat_headers = ["model_id", "model_name", "provider", "release_date",
                    "params_total_M", "params_active_M", "architecture",
                    "is_reasoning", "open_weights",
                    "benchmark_id", "benchmark_name", "benchmark_category",
                    "score", "reference_url"]
    for col_idx, h in enumerate(flat_headers, 1):
        cell = ws4.cell(row=1, column=col_idx, value=h)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.border = thin_border

    flat_row = 2
    for model_id, bench_id, score, url in DATA:
        if model_id not in model_info or bench_id not in bench_info:
            continue
        m = model_info[model_id]
        b = bench_info[bench_id]
        vals = [
            model_id, m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
            bench_id, b[1], b[2], score, url
        ]
        for col_idx, val in enumerate(vals, 1):
            cell = ws4.cell(row=flat_row, column=col_idx, value=val)
            cell.border = thin_border
        flat_row += 1

    for i in range(1, len(flat_headers) + 1):
        ws4.column_dimensions[get_column_letter(i)].width = 18

    # ── Sheet 5: Matrix Stats ────────────────────────────────────────────
    ws5 = wb.create_sheet("Matrix Stats")

    total_cells = len(model_ids) * len(bench_ids)
    filled_cells = len(scores)
    fill_rate = filled_cells / total_cells * 100 if total_cells > 0 else 0

    # Summary stats
    stats = [
        ("Total Models", len(model_ids)),
        ("Total Benchmarks", len(bench_ids)),
        ("Total Cells", total_cells),
        ("Filled Cells", filled_cells),
        ("Fill Rate (%)", round(fill_rate, 1)),
        ("Total (model, benchmark, score) triples", len(DATA)),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    for row_idx, (label, val) in enumerate(stats, 1):
        ws5.cell(row=row_idx, column=1, value=label).font = header_font
        ws5.cell(row=row_idx, column=2, value=val)

    # Per-model fill counts
    row_start = len(stats) + 3
    ws5.cell(row=row_start, column=1, value="Model").font = header_font
    ws5.cell(row=row_start, column=2, value="Scores Count").font = header_font
    ws5.cell(row=row_start, column=3, value="Fill Rate (%)").font = header_font
    for i, model_id in enumerate(model_ids):
        count = sum(1 for b in bench_ids if (model_id, b) in scores)
        rate = count / len(bench_ids) * 100
        ws5.cell(row=row_start + 1 + i, column=1, value=model_info[model_id][1])
        ws5.cell(row=row_start + 1 + i, column=2, value=count)
        ws5.cell(row=row_start + 1 + i, column=3, value=round(rate, 1))

    # Per-benchmark fill counts
    col_start = 5
    ws5.cell(row=row_start, column=col_start, value="Benchmark").font = header_font
    ws5.cell(row=row_start, column=col_start + 1, value="Models Count").font = header_font
    ws5.cell(row=row_start, column=col_start + 2, value="Fill Rate (%)").font = header_font
    for i, bench_id in enumerate(bench_ids):
        count = sum(1 for m in model_ids if (m, bench_id) in scores)
        rate = count / len(model_ids) * 100
        ws5.cell(row=row_start + 1 + i, column=col_start, value=bench_info[bench_id][1])
        ws5.cell(row=row_start + 1 + i, column=col_start + 1, value=count)
        ws5.cell(row=row_start + 1 + i, column=col_start + 2, value=round(rate, 1))

    ws5.column_dimensions['A'].width = 30
    ws5.column_dimensions['B'].width = 15
    ws5.column_dimensions['C'].width = 15
    ws5.column_dimensions['E'].width = 25
    ws5.column_dimensions['F'].width = 15
    ws5.column_dimensions['G'].width = 15

    # ── Save ─────────────────────────────────────────────────────────────
    _data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(_data_dir, "llm_benchmark_matrix.xlsx")
    wb.save(output_path)
    print(f"\n{'='*60}")
    print(f"  Excel file saved to: {output_path}")
    print(f"{'='*60}")
    print(f"  Models:     {len(model_ids)}")
    print(f"  Benchmarks: {len(bench_ids)}")
    print(f"  Scores:     {filled_cells}")
    print(f"  Fill rate:  {fill_rate:.1f}%")
    print(f"  Sheets:     Scores, References, Benchmark Info, Flat Format, Matrix Stats")
    print(f"{'='*60}")

    # Also save intermediate JSON
    json_path = os.path.join(_data_dir, "llm_benchmark_data.json")
    json_data = {
        "models": [{"id": m[0], "name": m[1], "provider": m[2],
                     "release_date": m[3], "params_total_M": m[4],
                     "params_active_M": m[5], "architecture": m[6],
                     "is_reasoning": m[7], "open_weights": m[8]}
                    for m in MODELS],
        "benchmarks": [{"id": b[0], "name": b[1], "category": b[2],
                        "metric": b[3], "num_problems": b[4],
                        "source_url": b[5]} for b in BENCHMARKS],
        "scores": [{"model_id": d[0], "benchmark_id": d[1],
                     "score": d[2], "reference_url": d[3]} for d in DATA],
        "generated": datetime.now().isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON backup: {json_path}")
    print(f"{'='*60}")

    # Print provider breakdown
    providers = {}
    for m in MODELS:
        providers.setdefault(m[2], []).append(m[0])
    print("\n  Provider breakdown:")
    for prov, models in sorted(providers.items()):
        n_scores = sum(1 for m in models for b in bench_ids if (m, b) in scores)
        print(f"    {prov:20s}  {len(models):2d} models, {n_scores:3d} scores")

    # Print benchmarks with most coverage
    print("\n  Top benchmarks by coverage:")
    bench_counts = [(bench_id, sum(1 for m in model_ids if (m, bench_id) in scores))
                    for bench_id in bench_ids]
    bench_counts.sort(key=lambda x: -x[1])
    for bench_id, count in bench_counts[:15]:
        print(f"    {bench_info[bench_id][1]:30s}  {count:2d}/{len(model_ids)} models")

    return output_path


if __name__ == "__main__":
    build_excel()
