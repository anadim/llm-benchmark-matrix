#!/usr/bin/env python3
"""
Extra scores batch 6: Imported from benchmark-stitching project CSV data
and augmented with live leaderboard fetches.
Source: ../benchmark-stitching/data/external_benchmark_*.csv + live leaderboards
Date: Feb 2026

Only includes models already in our MODELS list.
"""

EXTRA_SCORES = [
    # ══════════════════════════════════════════════════════════════════════
    # Aider Polyglot (NEW benchmark)
    # Source: https://aider.chat/docs/leaderboards/#polyglot-leaderboard
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "aider_polyglot", 60.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("claude-opus-4", "aider_polyglot", 70.7, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("claude-sonnet-4", "aider_polyglot", 56.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("deepseek-r1", "aider_polyglot", 56.9, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("deepseek-r1-0528", "aider_polyglot", 71.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("deepseek-v3", "aider_polyglot", 48.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("deepseek-v3-0324", "aider_polyglot", 55.1, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gemini-2.5-flash", "aider_polyglot", 47.1, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gemini-2.5-pro", "aider_polyglot", 76.9, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gemma-3-27b", "aider_polyglot", 4.9, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gpt-4.1", "aider_polyglot", 52.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gpt-4.1-mini", "aider_polyglot", 32.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gpt-4.1-nano", "aider_polyglot", 8.9, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gpt-4.5", "aider_polyglot", 44.9, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gpt-5", "aider_polyglot", 88.0, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("grok-3-beta", "aider_polyglot", 53.3, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("grok-4", "aider_polyglot", 79.6, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("o3-high", "aider_polyglot", 81.3, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("qwen3-235b", "aider_polyglot", 49.8, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("qwen3-32b", "aider_polyglot", 40.0, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("qwq-32b", "aider_polyglot", 20.9, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    # Additional models from live leaderboard
    ("deepseek-v3.2", "aider_polyglot", 74.2, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("o4-mini-high", "aider_polyglot", 72.0, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("o3-mini-high", "aider_polyglot", 60.4, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("kimi-k2", "aider_polyglot", 59.1, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("gpt-oss-120b", "aider_polyglot", 41.8, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("llama-4-maverick", "aider_polyglot", 15.6, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("command-a", "aider_polyglot", 12.0, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),
    ("codestral-25.01", "aider_polyglot", 11.1, "https://aider.chat/docs/leaderboards/#polyglot-leaderboard"),

    # ══════════════════════════════════════════════════════════════════════
    # BALROG (NEW benchmark)
    # Source: https://balrogai.com/
    # ══════════════════════════════════════════════════════════════════════
    ("deepseek-r1", "balrog", 34.9, "https://balrogai.com/"),
    ("deepseek-r1-distill-qwen-32b", "balrog", 19.5, "https://balrogai.com/"),
    ("gemini-2.5-flash", "balrog", 33.5, "https://balrogai.com/"),
    ("gemini-2.5-pro", "balrog", 43.3, "https://balrogai.com/"),
    ("grok-3-beta", "balrog", 29.5, "https://balrogai.com/"),
    ("grok-4", "balrog", 43.6, "https://balrogai.com/"),
    ("phi-4", "balrog", 11.6, "https://balrogai.com/"),
    ("gemini-3-flash", "balrog", 48.1, "https://balrogai.com/"),

    # ══════════════════════════════════════════════════════════════════════
    # CAD-Eval (NEW benchmark)
    # Source: https://willpatrick.xyz/cadevalresults_20250422_095709/
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "cadeval", 54.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    ("gemini-2.0-flash", "cadeval", 30.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    ("gpt-4.1", "cadeval", 42.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    ("gpt-4.1-mini", "cadeval", 16.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    ("o3-high", "cadeval", 74.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    ("o4-mini-high", "cadeval", 62.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    ("gemini-2.5-pro", "cadeval", 64.0, "https://willpatrick.xyz/cadevalresults_20250422_095709/"),

    # ══════════════════════════════════════════════════════════════════════
    # CyBench (NEW benchmark)
    # Source: https://cybench.github.io/
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "cybench", 20.0, "https://cybench.github.io/"),
    ("gpt-4.5", "cybench", 17.5, "https://cybench.github.io/"),
    # Additional models from live leaderboard
    ("claude-opus-4.6", "cybench", 93.0, "https://cybench.github.io/"),
    ("claude-opus-4.5", "cybench", 82.0, "https://cybench.github.io/"),
    ("claude-sonnet-4.5", "cybench", 60.0, "https://cybench.github.io/"),
    ("grok-4", "cybench", 43.0, "https://cybench.github.io/"),
    ("claude-opus-4.1", "cybench", 42.0, "https://cybench.github.io/"),
    ("grok-4.1", "cybench", 39.0, "https://cybench.github.io/"),
    ("claude-opus-4", "cybench", 38.0, "https://cybench.github.io/"),
    ("claude-sonnet-4", "cybench", 35.0, "https://cybench.github.io/"),
    ("o3-mini-high", "cybench", 22.5, "https://cybench.github.io/"),

    # ══════════════════════════════════════════════════════════════════════
    # Deep Research Benchmark (NEW benchmark)
    # Source: https://drb.futuresearch.ai
    # ══════════════════════════════════════════════════════════════════════
    ("gpt-5", "deepresearch", 49.6, "https://drb.futuresearch.ai"),
    ("grok-4", "deepresearch", 47.3, "https://drb.futuresearch.ai"),
    ("claude-opus-4.6", "deepresearch", 55.0, "https://drb.futuresearch.ai"),
    ("claude-sonnet-4.6", "deepresearch", 54.9, "https://drb.futuresearch.ai"),
    ("claude-opus-4.5", "deepresearch", 54.9, "https://drb.futuresearch.ai"),
    ("gemini-3-flash", "deepresearch", 50.4, "https://drb.futuresearch.ai"),
    ("gemini-3.1-pro", "deepresearch", 47.8, "https://drb.futuresearch.ai"),

    # ══════════════════════════════════════════════════════════════════════
    # GSO-Bench (NEW benchmark)
    # Source: https://gso-bench.github.io/leaderboard.html
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "gso_bench", 3.8, "https://gso-bench.github.io/leaderboard.html"),
    ("claude-opus-4", "gso_bench", 6.9, "https://gso-bench.github.io/leaderboard.html"),
    ("claude-sonnet-4", "gso_bench", 4.9, "https://gso-bench.github.io/leaderboard.html"),
    ("gemini-2.5-pro", "gso_bench", 3.9, "https://gso-bench.github.io/leaderboard.html"),
    ("gpt-5", "gso_bench", 6.9, "https://gso-bench.github.io/leaderboard.html"),
    ("o3-high", "gso_bench", 8.8, "https://gso-bench.github.io/leaderboard.html"),
    # Additional models from live leaderboard
    ("claude-opus-4.6", "gso_bench", 33.33, "https://gso-bench.github.io/leaderboard.html"),
    ("gpt-5.2", "gso_bench", 27.45, "https://gso-bench.github.io/leaderboard.html"),
    ("claude-opus-4.5", "gso_bench", 26.47, "https://gso-bench.github.io/leaderboard.html"),
    ("gemini-3-pro", "gso_bench", 18.63, "https://gso-bench.github.io/leaderboard.html"),
    ("claude-sonnet-4.5", "gso_bench", 14.71, "https://gso-bench.github.io/leaderboard.html"),
    ("gpt-5.1", "gso_bench", 13.73, "https://gso-bench.github.io/leaderboard.html"),
    ("gemini-3-flash", "gso_bench", 9.8, "https://gso-bench.github.io/leaderboard.html"),
    ("kimi-k2", "gso_bench", 4.9, "https://gso-bench.github.io/leaderboard.html"),
    ("o4-mini-high", "gso_bench", 3.6, "https://gso-bench.github.io/leaderboard.html"),
    ("o3-mini-high", "gso_bench", 1.3, "https://gso-bench.github.io/leaderboard.html"),

    # ══════════════════════════════════════════════════════════════════════
    # Lech Mazur Creative Writing (NEW benchmark)
    # Source: https://github.com/lechmazur/Writing
    # ══════════════════════════════════════════════════════════════════════
    ("claude-opus-4.6", "lech_mazur_writing", 8.56, "https://github.com/lechmazur/Writing"),
    ("gpt-5.2", "lech_mazur_writing", 8.51, "https://github.com/lechmazur/Writing"),
    ("gpt-5", "lech_mazur_writing", 8.47, "https://github.com/lechmazur/Writing"),
    ("gpt-5.1", "lech_mazur_writing", 8.44, "https://github.com/lechmazur/Writing"),
    ("kimi-k2", "lech_mazur_writing", 8.33, "https://github.com/lechmazur/Writing"),
    ("gemini-3-pro", "lech_mazur_writing", 8.22, "https://github.com/lechmazur/Writing"),
    ("claude-opus-4.5", "lech_mazur_writing", 8.20, "https://github.com/lechmazur/Writing"),
    ("claude-sonnet-4.5", "lech_mazur_writing", 8.17, "https://github.com/lechmazur/Writing"),
    ("kimi-k2.5", "lech_mazur_writing", 8.07, "https://github.com/lechmazur/Writing"),
    ("claude-opus-4.1", "lech_mazur_writing", 8.07, "https://github.com/lechmazur/Writing"),
    ("kimi-k2-thinking", "lech_mazur_writing", 7.69, "https://github.com/lechmazur/Writing"),
    ("deepseek-v3.2", "lech_mazur_writing", 7.60, "https://github.com/lechmazur/Writing"),
    ("mistral-large-3", "lech_mazur_writing", 7.60, "https://github.com/lechmazur/Writing"),
    ("grok-4.1", "lech_mazur_writing", 7.57, "https://github.com/lechmazur/Writing"),
    ("glm-4.6", "lech_mazur_writing", 7.45, "https://github.com/lechmazur/Writing"),
    ("deepseek-v3.2-speciale", "lech_mazur_writing", 7.16, "https://github.com/lechmazur/Writing"),
    ("command-a", "lech_mazur_writing", 6.79, "https://github.com/lechmazur/Writing"),
    ("llama-4-maverick", "lech_mazur_writing", 5.78, "https://github.com/lechmazur/Writing"),

    # ══════════════════════════════════════════════════════════════════════
    # METR (NEW benchmark)
    # Source: https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
    # Success rate @ 50%
    # ══════════════════════════════════════════════════════════════════════
    ("claude-opus-4.6", "metr", 0.789, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("gpt-5.3-codex", "metr", 0.745, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("claude-opus-4.5", "metr", 0.730, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("gpt-5.2", "metr", 0.753, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("gpt-5.1", "metr", 0.708, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("gemini-3-pro", "metr", 0.71, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("gpt-5", "metr", 0.694, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("claude-opus-4.1", "metr", 0.616, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),
    ("claude-opus-4", "metr", 0.615, "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"),

    # ══════════════════════════════════════════════════════════════════════
    # VPCT (NEW benchmark)
    # Source: https://cbrower.dev/vpct
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "vpct", 39.0, "https://cbrower.dev/vpct"),
    ("claude-opus-4", "vpct", 33.0, "https://cbrower.dev/vpct"),
    ("claude-sonnet-4", "vpct", 30.0, "https://cbrower.dev/vpct"),
    ("gemini-2.5-flash", "vpct", 38.0, "https://cbrower.dev/vpct"),
    ("gemini-2.5-pro", "vpct", 40.5, "https://cbrower.dev/vpct"),
    ("gpt-4.5", "vpct", 45.0, "https://cbrower.dev/vpct"),
    ("gpt-5", "vpct", 66.0, "https://cbrower.dev/vpct"),
    ("gpt-5.1", "vpct", 53.3, "https://cbrower.dev/vpct"),
    # Additional models from live leaderboard
    ("gemini-3-pro", "vpct", 91.0, "https://cbrower.dev/vpct"),
    ("gemini-3-flash", "vpct", 72.6, "https://cbrower.dev/vpct"),
    ("gpt-5.2", "vpct", 67.0, "https://cbrower.dev/vpct"),
    ("o4-mini-high", "vpct", 57.5, "https://cbrower.dev/vpct"),
    ("o3-high", "vpct", 52.0, "https://cbrower.dev/vpct"),
    ("claude-opus-4.5", "vpct", 40.0, "https://cbrower.dev/vpct"),
    ("claude-sonnet-4.5", "vpct", 38.0, "https://cbrower.dev/vpct"),
    ("claude-opus-4.1", "vpct", 35.0, "https://cbrower.dev/vpct"),
    ("gpt-4.1", "vpct", 33.0, "https://cbrower.dev/vpct"),

    # ══════════════════════════════════════════════════════════════════════
    # WeirdML (NEW benchmark)
    # Source: https://htihle.github.io/weirdml.html
    # ══════════════════════════════════════════════════════════════════════
    ("claude-sonnet-4", "weirdml", 43.0, "https://htihle.github.io/weirdml.html"),
    ("deepseek-r1", "weirdml", 35.6, "https://htihle.github.io/weirdml.html"),
    ("deepseek-r1-0528", "weirdml", 40.9, "https://htihle.github.io/weirdml.html"),
    ("deepseek-v3-0324", "weirdml", 35.1, "https://htihle.github.io/weirdml.html"),
    ("gemini-2.0-flash", "weirdml", 25.2, "https://htihle.github.io/weirdml.html"),
    ("gpt-4.1", "weirdml", 37.9, "https://htihle.github.io/weirdml.html"),
    ("gpt-4.1-mini", "weirdml", 37.2, "https://htihle.github.io/weirdml.html"),
    ("gpt-4.1-nano", "weirdml", 19.0, "https://htihle.github.io/weirdml.html"),
    ("gpt-4.5", "weirdml", 37.6, "https://htihle.github.io/weirdml.html"),
    ("gpt-5", "weirdml", 56.3, "https://htihle.github.io/weirdml.html"),
    ("grok-3-beta", "weirdml", 36.4, "https://htihle.github.io/weirdml.html"),
    ("grok-4", "weirdml", 42.6, "https://htihle.github.io/weirdml.html"),
    ("kimi-k2-thinking", "weirdml", 42.1, "https://htihle.github.io/weirdml.html"),
    ("o3-high", "weirdml", 49.8, "https://htihle.github.io/weirdml.html"),
    ("qwen3-235b", "weirdml", 36.2, "https://htihle.github.io/weirdml.html"),
    ("qwen3-30b-a3b", "weirdml", 29.7, "https://htihle.github.io/weirdml.html"),
    # Additional models from live leaderboard (raw fractions converted to %)
    ("claude-opus-4.6", "weirdml", 77.9, "https://htihle.github.io/weirdml.html"),
    ("claude-sonnet-4.5", "weirdml", 75.6, "https://htihle.github.io/weirdml.html"),
    ("gemini-2.5-pro", "weirdml", 73.1, "https://htihle.github.io/weirdml.html"),
    ("gpt-5.2", "weirdml", 72.2, "https://htihle.github.io/weirdml.html"),
    ("gemini-3.1-pro", "weirdml", 72.1, "https://htihle.github.io/weirdml.html"),
    ("gemini-3-pro", "weirdml", 69.9, "https://htihle.github.io/weirdml.html"),
    ("gpt-5.1", "weirdml", 68.3, "https://htihle.github.io/weirdml.html"),
    ("kimi-k2", "weirdml", 55.6, "https://htihle.github.io/weirdml.html"),
    ("gemini-3-flash", "weirdml", 52.3, "https://htihle.github.io/weirdml.html"),
    ("deepseek-v3.2", "weirdml", 50.5, "https://htihle.github.io/weirdml.html"),
    ("o3-mini-high", "weirdml", 44.9, "https://htihle.github.io/weirdml.html"),
    ("claude-opus-4", "weirdml", 44.8, "https://htihle.github.io/weirdml.html"),
    ("claude-sonnet-4.6", "weirdml", 42.7, "https://htihle.github.io/weirdml.html"),
    ("o4-mini-high", "weirdml", 41.9, "https://htihle.github.io/weirdml.html"),
    ("claude-opus-4.5", "weirdml", 39.6, "https://htihle.github.io/weirdml.html"),
    ("gpt-oss-20b", "weirdml", 37.1, "https://htihle.github.io/weirdml.html"),
    ("claude-opus-4.1", "weirdml", 28.8, "https://htihle.github.io/weirdml.html"),
    ("deepseek-v3.2-speciale", "weirdml", 19.9, "https://htihle.github.io/weirdml.html"),
    ("gpt-oss-120b", "weirdml", 19.0, "https://htihle.github.io/weirdml.html"),
    ("llama-4-maverick", "weirdml", 14.0, "https://htihle.github.io/weirdml.html"),
    ("kimi-k2.5", "weirdml", 8.0, "https://htihle.github.io/weirdml.html"),

    # ══════════════════════════════════════════════════════════════════════
    # The Agent Company (NEW benchmark)
    # Source: TheAgentCompany leaderboard
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "the_agent_company", 30.9, "https://the-agent-company.com/#/leaderboard"),
    ("claude-sonnet-4", "the_agent_company", 33.1, "https://the-agent-company.com/#/leaderboard"),
    ("gemini-2.0-flash", "the_agent_company", 11.4, "https://github.com/TheAgentCompany/experiments/tree/main"),
    ("gemini-2.5-pro", "the_agent_company", 30.3, "https://the-agent-company.com/#/leaderboard"),

    # ══════════════════════════════════════════════════════════════════════
    # OS-Universe (NEW benchmark)
    # Source: https://arxiv.org/pdf/2505.03570
    # ══════════════════════════════════════════════════════════════════════
    ("gemini-2.0-flash", "osuniverse", 8.26, "https://arxiv.org/pdf/2505.03570"),
    ("gemini-2.5-pro", "osuniverse", 9.59, "https://arxiv.org/pdf/2505.03570"),
    ("claude-3.5-sonnet", "osuniverse", 28.36, "https://arxiv.org/pdf/2505.03570"),

    # ══════════════════════════════════════════════════════════════════════
    # SimpleBench (additional entries for EXISTING benchmark)
    # Source: SimpleBench Leaderboard (https://simple-bench.com/)
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "simplebench", 44.9, "https://simple-bench.com/"),
    ("deepseek-r1", "simplebench", 30.9, "https://simple-bench.com/"),
    ("deepseek-r1-0528", "simplebench", 40.8, "https://simple-bench.com/"),
    ("deepseek-v3", "simplebench", 18.9, "https://simple-bench.com/"),
    ("deepseek-v3-0324", "simplebench", 27.2, "https://simple-bench.com/"),
    ("gemini-2.5-flash", "simplebench", 41.2, "https://simple-bench.com/"),
    ("gpt-4.1", "simplebench", 27.0, "https://simple-bench.com/"),
    ("gpt-4.5", "simplebench", 34.5, "https://simple-bench.com/"),
    ("gpt-oss-120b", "simplebench", 22.1, "https://simple-bench.com/"),
    ("grok-3-beta", "simplebench", 36.1, "https://simple-bench.com/"),
    ("grok-4", "simplebench", 60.5, "https://simple-bench.com/"),
    ("o3-high", "simplebench", 53.1, "https://simple-bench.com/"),
    ("qwen3-235b", "simplebench", 31.0, "https://simple-bench.com/"),
    # Additional models from live leaderboard
    ("gemini-3.1-pro", "simplebench", 79.6, "https://simple-bench.com/"),
    ("gemini-3-pro", "simplebench", 76.4, "https://simple-bench.com/"),
    ("gemini-3-flash", "simplebench", 61.1, "https://simple-bench.com/"),
    ("claude-opus-4.1", "simplebench", 60.0, "https://simple-bench.com/"),
    ("claude-opus-4", "simplebench", 58.8, "https://simple-bench.com/"),
    ("deepseek-v3.2-speciale", "simplebench", 52.6, "https://simple-bench.com/"),
    ("glm-4.7", "simplebench", 47.7, "https://simple-bench.com/"),
    ("kimi-k2.5", "simplebench", 46.8, "https://simple-bench.com/"),
    ("claude-sonnet-4", "simplebench", 45.5, "https://simple-bench.com/"),
    ("kimi-k2-thinking", "simplebench", 39.6, "https://simple-bench.com/"),
    ("gemini-2.0-flash", "simplebench", 30.7, "https://simple-bench.com/"),
    ("llama-4-maverick", "simplebench", 27.7, "https://simple-bench.com/"),
    ("kimi-k2", "simplebench", 26.3, "https://simple-bench.com/"),
    ("minimax-m2", "simplebench", 25.0, "https://simple-bench.com/"),
    ("o3-mini-high", "simplebench", 22.8, "https://simple-bench.com/"),
    ("mistral-large-3", "simplebench", 20.4, "https://simple-bench.com/"),

    # ══════════════════════════════════════════════════════════════════════
    # LiveBench (additional entries for EXISTING benchmark)
    # Source: https://livebench.ai/
    # ══════════════════════════════════════════════════════════════════════
    # Additional models from live leaderboard (LiveBench 2026-01-08)
    ("o3-high", "livebench", 80.71, "https://livebench.ai/#/"),
    ("gemini-3.1-pro", "livebench", 79.93, "https://livebench.ai/#/"),
    ("gpt-5", "livebench", 78.85, "https://livebench.ai/#/"),
    ("o4-mini-high", "livebench", 78.72, "https://livebench.ai/#/"),
    ("claude-opus-4.6", "livebench", 76.33, "https://livebench.ai/#/"),
    ("claude-opus-4.5", "livebench", 75.96, "https://livebench.ai/#/"),
    ("claude-sonnet-4.6", "livebench", 75.32, "https://livebench.ai/#/"),
    ("gpt-5.2", "livebench", 74.84, "https://livebench.ai/#/"),
    ("gemini-3-pro", "livebench", 73.39, "https://livebench.ai/#/"),
    ("gpt-5.3-codex", "livebench", 72.76, "https://livebench.ai/#/"),
    ("gemini-3-flash", "livebench", 72.40, "https://livebench.ai/#/"),
    ("gemini-2.5-flash", "livebench", 71.98, "https://livebench.ai/#/"),
    ("claude-opus-4", "livebench", 71.52, "https://livebench.ai/#/"),
    ("deepseek-r1-0528", "livebench", 69.41, "https://livebench.ai/#/"),
    ("kimi-k2.5", "livebench", 69.07, "https://livebench.ai/#/"),
    ("minimax-m2", "livebench", 64.26, "https://livebench.ai/#/"),
    ("deepseek-v3.2-speciale", "livebench", 64.05, "https://livebench.ai/#/"),
    ("qwen3-235b", "livebench", 63.42, "https://livebench.ai/#/"),
    ("grok-3-beta", "livebench", 63.17, "https://livebench.ai/#/"),
    ("gpt-4.1", "livebench", 62.99, "https://livebench.ai/#/"),
    ("grok-4", "livebench", 62.02, "https://livebench.ai/#/"),
    ("kimi-k2-thinking", "livebench", 61.59, "https://livebench.ai/#/"),
    ("grok-4.1", "livebench", 59.99, "https://livebench.ai/#/"),
    ("gpt-4.1-mini", "livebench", 59.05, "https://livebench.ai/#/"),
    ("glm-4.7", "livebench", 58.09, "https://livebench.ai/#/"),
    ("phi-4-reasoning-plus", "livebench", 56.64, "https://livebench.ai/#/"),
    ("glm-4.6", "livebench", 55.19, "https://livebench.ai/#/"),
    ("llama-4-maverick", "livebench", 54.38, "https://livebench.ai/#/"),
    ("claude-opus-4.1", "livebench", 54.45, "https://livebench.ai/#/"),
    ("claude-sonnet-4.5", "livebench", 53.69, "https://livebench.ai/#/"),
    ("deepseek-v3.2", "livebench", 51.84, "https://livebench.ai/#/"),
    ("claude-sonnet-4", "livebench", 50.98, "https://livebench.ai/#/"),
    ("mistral-medium-3", "livebench", 50.29, "https://livebench.ai/#/"),
    ("mistral-large-3", "livebench", 50.25, "https://livebench.ai/#/"),
    ("kimi-k2", "livebench", 48.10, "https://livebench.ai/#/"),
    ("gpt-4.1-nano", "livebench", 46.58, "https://livebench.ai/#/"),
    ("gpt-oss-120b", "livebench", 46.09, "https://livebench.ai/#/"),
    ("claude-haiku-4.5", "livebench", 45.33, "https://livebench.ai/#/"),
    ("amazon-nova-pro", "livebench", 44.33, "https://livebench.ai/#/"),
    ("command-a", "livebench", 44.09, "https://livebench.ai/#/"),
    ("qwen3-32b", "livebench", 43.56, "https://livebench.ai/#/"),
    ("o3-mini-high", "livebench", 75.88, "https://livebench.ai/#/"),
    ("olmo-2-13b", "livebench", 22.12, "https://livebench.ai/#/"),

    # ══════════════════════════════════════════════════════════════════════
    # Terminal-Bench 2.0 (additional entries for EXISTING benchmark)
    # Source: https://www.tbench.ai/leaderboard
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.7-sonnet", "terminal_bench", 35.2, "https://www.tbench.ai/leaderboard"),
    ("deepseek-r1", "terminal_bench", 5.7, "https://www.tbench.ai/leaderboard"),
    ("gpt-4.1", "terminal_bench", 30.3, "https://www.tbench.ai/leaderboard"),
    ("grok-3-beta", "terminal_bench", 17.5, "https://www.tbench.ai/leaderboard"),
    ("qwen3-235b", "terminal_bench", 6.6, "https://www.tbench.ai/leaderboard"),
    ("qwen3-32b", "terminal_bench", 15.5, "https://www.tbench.ai/leaderboard"),
    # Additional models from live leaderboard (best agent per model)
    ("gpt-5.3-codex", "terminal_bench", 77.3, "https://www.tbench.ai/leaderboard"),
    ("gemini-3.1-pro", "terminal_bench", 74.8, "https://www.tbench.ai/leaderboard"),
    ("claude-opus-4.6", "terminal_bench", 74.7, "https://www.tbench.ai/leaderboard"),
    ("gemini-3-pro", "terminal_bench", 65.2, "https://www.tbench.ai/leaderboard"),
    ("gemini-3-flash", "terminal_bench", 64.3, "https://www.tbench.ai/leaderboard"),
    ("kimi-k2.5", "terminal_bench", 43.2, "https://www.tbench.ai/leaderboard"),
    ("deepseek-v3.2", "terminal_bench", 39.6, "https://www.tbench.ai/leaderboard"),
    ("claude-opus-4.1", "terminal_bench", 38.0, "https://www.tbench.ai/leaderboard"),
    ("kimi-k2-thinking", "terminal_bench", 35.7, "https://www.tbench.ai/leaderboard"),
    ("glm-4.7", "terminal_bench", 33.4, "https://www.tbench.ai/leaderboard"),
    ("claude-haiku-4.5", "terminal_bench", 29.8, "https://www.tbench.ai/leaderboard"),
    ("kimi-k2", "terminal_bench", 27.8, "https://www.tbench.ai/leaderboard"),
    ("glm-4.6", "terminal_bench", 24.5, "https://www.tbench.ai/leaderboard"),
    ("gpt-oss-120b", "terminal_bench", 18.7, "https://www.tbench.ai/leaderboard"),
    ("gemini-2.5-flash", "terminal_bench", 17.1, "https://www.tbench.ai/leaderboard"),
    ("gpt-oss-20b", "terminal_bench", 3.4, "https://www.tbench.ai/leaderboard"),

    # ══════════════════════════════════════════════════════════════════════
    # Factorio Learning Environment (NEW benchmark)
    # Source: https://jackhopkins.github.io/factorio-learning-environment/leaderboard/
    # Metric: Lab Success %
    # ══════════════════════════════════════════════════════════════════════
    ("claude-3.5-sonnet", "factorio", 29.1, "https://jackhopkins.github.io/factorio-learning-environment/leaderboard/"),
    ("deepseek-v3", "factorio", 15.1, "https://jackhopkins.github.io/factorio-learning-environment/leaderboard/"),
    ("gemini-2.0-flash", "factorio", 13.0, "https://jackhopkins.github.io/factorio-learning-environment/leaderboard/"),

    # ══════════════════════════════════════════════════════════════════════
    # FictionLiveBench (NEW benchmark)
    # Source: https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home
    # Metric: 16k token score
    # ══════════════════════════════════════════════════════════════════════
    ("gpt-5.2", "fictionlivebench", 100., "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("claude-opus-4.6", "fictionlivebench", 100., "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("claude-opus-4.5", "fictionlivebench", 94.4, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("claude-sonnet-4.5", "fictionlivebench", 86.1, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("deepseek-v3.2", "fictionlivebench", 86.1, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("gemini-3-flash", "fictionlivebench", 100., "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("gemini-3-pro", "fictionlivebench", 96.6, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("glm-4.7", "fictionlivebench", 83.3, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("grok-4", "fictionlivebench", 94.4, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),
    ("kimi-k2.5", "fictionlivebench", 86.1, "https://fiction.live/stories/Fiction-liveBench-Feb-12-2026/oQdzQvKHw8JyXbN87/home"),

    # ══════════════════════════════════════════════════════════════════════
    # GeoBench (NEW benchmark)
    # Source: https://geobench.org/
    # Metric: acw-02-25-25 Country % (All Categories Worldwide, country-level accuracy)
    # ══════════════════════════════════════════════════════════════════════
    ("gemini-3.0-flash", "geobench", 88, "https://geobench.org/"),
    ("gemini-2.5-pro", "geobench", 81, "https://geobench.org/"),
    ("gemini-3.0-pro", "geobench", 84, "https://geobench.org/"),
    ("gpt-5", "geobench", 81, "https://geobench.org/"),
    ("gemini-2.0-flash", "geobench", 77, "https://geobench.org/"),
    ("gemini-2.5-flash", "geobench", 76, "https://geobench.org/"),
    ("claude-opus-4.5", "geobench", 75, "https://geobench.org/"),
    ("gpt-4.1", "geobench", 72, "https://geobench.org/"),
    ("claude-3.7-sonnet", "geobench", 68, "https://geobench.org/"),
    ("o4-mini-high", "geobench", 64, "https://geobench.org/"),
    ("claude-3.5-sonnet", "geobench", 62, "https://geobench.org/"),
    ("o3-high", "geobench", 60, "https://geobench.org/"),
    ("claude-sonnet-4", "geobench", 37, "https://geobench.org/"),
    ("gemma-3-27b", "geobench", 52, "https://geobench.org/"),
    ("grok-4", "geobench", 45, "https://geobench.org/"),
    ("claude-4-sonnet", "geobench", 37, "https://geobench.org/"),
    ("llama-4-maverick", "geobench", 52, "https://geobench.org/"),
]
