#!/usr/bin/env python3
"""
Extra scores batch 5: ARC-AGI-1 and ARC-AGI-2 from official ARC Prize leaderboard.
Source: https://arcprize.org/arc-agi/1/ and https://arcprize.org/arc-agi/2/
Date: Feb 2026
Scores use "High" thinking config (or best standard CoT) for each model.
"""

EXTRA_SCORES = [
    # ── ARC-AGI-1 (new entries from official ARC Prize leaderboard) ──
    # Using CoT "High" or standard config for each model
    ("claude-opus-4.5", "arc_agi_1", 80.0, "https://arcprize.org/arc-agi/1/"),
    ("gemini-3-pro", "arc_agi_1", 75.0, "https://arcprize.org/arc-agi/1/"),
    ("gemini-3-flash", "arc_agi_1", 84.7, "https://arcprize.org/arc-agi/1/"),
    ("gpt-5.1", "arc_agi_1", 72.8, "https://arcprize.org/arc-agi/1/"),
    ("kimi-k2.5", "arc_agi_1", 65.3, "https://arcprize.org/arc-agi/1/"),
    ("claude-sonnet-4.5", "arc_agi_1", 63.7, "https://arcprize.org/arc-agi/1/"),
    ("o4-mini-high", "arc_agi_1", 58.7, "https://arcprize.org/arc-agi/1/"),
    ("claude-haiku-4.5", "arc_agi_1", 47.7, "https://arcprize.org/arc-agi/1/"),
    ("gemini-2.5-pro", "arc_agi_1", 41.0, "https://arcprize.org/arc-agi/1/"),
    ("claude-sonnet-4", "arc_agi_1", 40.0, "https://arcprize.org/arc-agi/1/"),
    ("claude-opus-4", "arc_agi_1", 35.7, "https://arcprize.org/arc-agi/1/"),
    ("o3-mini-high", "arc_agi_1", 34.5, "https://arcprize.org/arc-agi/1/"),
    ("gemini-2.5-flash", "arc_agi_1", 33.3, "https://arcprize.org/arc-agi/1/"),
    ("claude-3.7-sonnet", "arc_agi_1", 28.6, "https://arcprize.org/arc-agi/1/"),
    ("deepseek-r1-0528", "arc_agi_1", 21.2, "https://arcprize.org/arc-agi/1/"),
    ("qwen3-235b", "arc_agi_1", 11.0, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.5", "arc_agi_1", 10.3, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.1", "arc_agi_1", 5.5, "https://arcprize.org/arc-agi/1/"),
    ("grok-3-beta", "arc_agi_1", 5.5, "https://arcprize.org/arc-agi/1/"),
    ("llama-4-maverick", "arc_agi_1", 4.4, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.1-mini", "arc_agi_1", 3.5, "https://arcprize.org/arc-agi/1/"),
    ("llama-4-scout", "arc_agi_1", 0.5, "https://arcprize.org/arc-agi/1/"),
    ("gpt-4.1-nano", "arc_agi_1", 0.0, "https://arcprize.org/arc-agi/1/"),

    # ── ARC-AGI-2 (new entries from official ARC Prize leaderboard) ──
    ("gemini-3-flash", "arc_agi_2", 33.6, "https://arcprize.org/arc-agi/2/"),
    ("claude-sonnet-4.5", "arc_agi_2", 13.6, "https://arcprize.org/arc-agi/2/"),
    ("claude-opus-4", "arc_agi_2", 8.6, "https://arcprize.org/arc-agi/2/"),
    ("claude-sonnet-4", "arc_agi_2", 5.9, "https://arcprize.org/arc-agi/2/"),
    ("claude-haiku-4.5", "arc_agi_2", 4.0, "https://arcprize.org/arc-agi/2/"),
    ("o3-mini-high", "arc_agi_2", 3.0, "https://arcprize.org/arc-agi/2/"),
    ("gemini-2.5-flash", "arc_agi_2", 2.5, "https://arcprize.org/arc-agi/2/"),
    ("deepseek-r1", "arc_agi_2", 1.3, "https://arcprize.org/arc-agi/2/"),
    ("qwen3-235b", "arc_agi_2", 1.3, "https://arcprize.org/arc-agi/2/"),
    ("deepseek-r1-0528", "arc_agi_2", 1.1, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.5", "arc_agi_2", 0.8, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.1", "arc_agi_2", 0.4, "https://arcprize.org/arc-agi/2/"),
    ("llama-4-maverick", "arc_agi_2", 0.0, "https://arcprize.org/arc-agi/2/"),
    ("grok-3-beta", "arc_agi_2", 0.0, "https://arcprize.org/arc-agi/2/"),
    ("llama-4-scout", "arc_agi_2", 0.0, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.1-mini", "arc_agi_2", 0.0, "https://arcprize.org/arc-agi/2/"),
    ("gpt-4.1-nano", "arc_agi_2", 0.0, "https://arcprize.org/arc-agi/2/"),

    # ── From phi-4-reasoning-plus technical report (arXiv:2504.21318) ──
    # Only adding scores for benchmarks already in our matrix, where model has no existing entry
    ("phi-4-reasoning-plus", "hmmt_2025", 53.6, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning-plus", "arena_hard", 79.0, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "aime_2024", 74.6, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "hmmt_2025", 43.8, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "codeforces_rating", 1736, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "ifeval", 83.4, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "arena_hard", 73.3, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "humaneval", 92.9, "https://arxiv.org/abs/2504.21318"),
    ("phi-4-reasoning", "mmlu_pro", 74.3, "https://arxiv.org/abs/2504.21318"),
    ("qwq-32b", "hmmt_2025", 47.5, "https://arxiv.org/abs/2504.21318"),
    ("exaone-4.0-32b", "aime_2024", 72.1, "https://arxiv.org/abs/2504.21318"),
    ("deepseek-r1-distill-llama-70b", "hmmt_2025", 33.3, "https://arxiv.org/abs/2504.21318"),
]
