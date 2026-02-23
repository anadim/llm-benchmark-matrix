"""
Extra benchmark scores collected from third-party evaluation sites.
Sources: Artificial Analysis, LMSys Chatbot Arena, LiveBench, CanAICode,
Berkeley Function Calling Leaderboard (BFCL), Aider polyglot benchmark,
WebDev Arena, Papers With Code, Scale SEAL, and various review sites.

Collected: 2026-02-23
Models: Jan 2025+ releases only (with a few late-2024 baselines for context).

Format: ("model_id", "benchmark_id", score, "source_url")
  - model_id: lowercase-hyphenated
  - benchmark_id: lowercase_underscore
  - score: float (percentages as 0-100, Elo as raw number)
"""

EXTRA_SCORES = [
    # =========================================================================
    # GPT-5 (released ~June 2025)
    # =========================================================================
    ("gpt-5", "aime_2025", 94.6, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "gpqa_diamond", 87.3, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),
    ("gpt-5", "gpqa_diamond_pro", 88.4, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "swe_bench_verified", 74.9, "https://www.vellum.ai/blog/gpt-5-benchmarks"),
    ("gpt-5", "aider_polyglot", 88.0, "https://aider.chat/docs/leaderboards/"),
    ("gpt-5", "mmmu", 84.2, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "bfcl", 59.22, "https://gorilla.cs.berkeley.edu/leaderboard.html"),
    ("gpt-5", "chatbot_arena_elo", 1464, "https://lmarena.ai/leaderboard"),  # as GPT-5.1-high

    # =========================================================================
    # GPT-5.1 (released ~Sep 2025)
    # =========================================================================
    ("gpt-5.1", "gpqa_diamond", 88.1, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),
    ("gpt-5.1", "aime_2025", 94.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.1", "frontiermath_t1_t3", 31.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.1", "chatbot_arena_elo", 1464, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),

    # =========================================================================
    # GPT-5.2 (released ~Dec 2025)
    # =========================================================================
    ("gpt-5.2", "aime_2025", 100.0, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "gpqa_diamond", 92.4, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),
    ("gpt-5.2", "gpqa_diamond_pro", 93.2, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "frontiermath_t1_t3", 40.3, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "swe_bench_verified", 80.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),
    ("gpt-5.2", "swe_bench_pro", 55.6, "https://vertu.com/lifestyle/gpt-5-2-review-benchmark-results-real-world-testing-and-competitive-analysis/"),
    ("gpt-5.2", "humanitys_last_exam", 31.64, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),

    # =========================================================================
    # Claude Opus 4 (released May 2025)
    # =========================================================================
    ("claude-opus-4", "swe_bench_verified", 72.5, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "swe_bench_verified_high_compute", 79.4, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "gpqa_diamond", 79.6, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "gpqa_diamond_high_compute", 83.3, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "mmlu", 88.8, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "aime_2025", 75.5, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "aime_2025_high_compute", 90.0, "https://www.datacamp.com/blog/claude-4"),

    # =========================================================================
    # Claude Opus 4.1 (released ~Aug 2025)
    # =========================================================================
    ("claude-opus-4.1", "swe_bench_verified", 74.5, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "mmlu_pro", 87.92, "https://www.datastudios.org/post/claude-4-1-vs-grok-4-full-report-and-comparison-august-2025-updated"),
    ("claude-opus-4.1", "bfcl", 70.36, "https://gorilla.cs.berkeley.edu/leaderboard.html"),

    # =========================================================================
    # Claude Opus 4.5 (released ~Nov 2025)
    # =========================================================================
    ("claude-opus-4.5", "swe_bench_verified", 80.9, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "gpqa_diamond", 87.0, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-opus-4.5", "aider_polyglot", 89.4, "https://aider.chat/docs/leaderboards/"),
    ("claude-opus-4.5", "aider_polyglot_no_thinking", 70.7, "https://www.getpassionfruit.com/blog/gpt-5-1-vs-claude-4-5-sonnet-vs-gemini-3-pro-vs-deepseek-v3-2-the-definitive-2025-ai-model-comparison"),
    ("claude-opus-4.5", "mmlu_pro", 89.5, "https://artificialanalysis.ai/evaluations/mmlu-pro"),
    ("claude-opus-4.5", "chatbot_arena_coding_elo", 1510, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-coding-leaderboard-2026.html"),
    ("claude-opus-4.5", "swe_bench_pro", 45.89, "https://scale.com/leaderboard/swe_bench_pro_public"),
    ("claude-opus-4.5", "humanitys_last_exam", 36.7, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),

    # =========================================================================
    # Claude Opus 4.6 (released ~Feb 2026)
    # =========================================================================
    ("claude-opus-4.6", "gpqa_diamond", 91.3, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "mmlu", 91.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "aime_2025", 100.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "swe_bench_verified", 80.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-opus-4.6", "humanitys_last_exam", 36.7, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),

    # =========================================================================
    # Claude Sonnet 4 (released May 2025)
    # =========================================================================
    ("claude-sonnet-4", "swe_bench_verified", 72.7, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "gpqa_diamond", 75.4, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "mmlu", 86.5, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "aime_2025", 70.5, "https://www.datacamp.com/blog/claude-4"),
    ("claude-sonnet-4", "bfcl", 70.29, "https://gorilla.cs.berkeley.edu/leaderboard.html"),
    ("claude-sonnet-4", "canaicode", 71.0, "https://huggingface.co/spaces/mike-ravkine/can-ai-code-results"),
    ("claude-sonnet-4", "swe_bench_pro", 42.70, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # =========================================================================
    # Claude Sonnet 4.5 (released ~Sep 2025)
    # =========================================================================
    ("claude-sonnet-4.5", "swe_bench_verified", 77.2, "https://caylent.com/blog/claude-sonnet-4-5-highest-scoring-claude-model-yet-on-swe-bench"),
    ("claude-sonnet-4.5", "swe_bench_verified_parallel", 82.0, "https://caylent.com/blog/claude-sonnet-4-5-highest-scoring-claude-model-yet-on-swe-bench"),
    ("claude-sonnet-4.5", "gpqa_diamond", 83.4, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "aime_2025_with_python", 100.0, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "aime_2025", 87.0, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "mmmlu", 89.1, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "mmmu", 77.8, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "osworld", 61.4, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "aider_polyglot", 82.4, "https://www.getpassionfruit.com/blog/gpt-5-1-vs-claude-4-5-sonnet-vs-gemini-3-pro-vs-deepseek-v3-2-the-definitive-2025-ai-model-comparison"),
    ("claude-sonnet-4.5", "swe_bench_pro", 43.60, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # =========================================================================
    # Claude Sonnet 4.6 (released ~Feb 2026)
    # =========================================================================
    ("claude-sonnet-4.6", "swe_bench_verified", 79.6, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "gpqa_diamond", 74.1, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "math_benchmark", 89.0, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "osworld_verified", 72.5, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),

    # =========================================================================
    # Gemini 2.5 Pro (released ~Mar 2025)
    # =========================================================================
    ("gemini-2.5-pro", "gpqa_diamond", 84.0, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "aime_2025", 86.7, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "livecodebench_v5", 70.4, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "global_mmlu_lite", 89.8, "https://www.helicone.ai/blog/gemini-2.5-full-developer-guide"),
    ("gemini-2.5-pro", "aider_polyglot", 72.9, "https://aider.chat/docs/leaderboards/"),  # without thinking
    ("gemini-2.5-pro", "aider_polyglot_thinking", 82.2, "https://aider.chat/docs/leaderboards/"),
    ("gemini-2.5-pro", "canaicode", 77.0, "https://huggingface.co/spaces/mike-ravkine/can-ai-code-results"),

    # =========================================================================
    # Gemini 2.5 Flash (released ~May 2025)
    # =========================================================================
    ("gemini-2.5-flash", "aime_2025", 72.0, "https://artificialanalysis.ai/models/gemini-2-5-flash"),
    ("gemini-2.5-flash", "livecodebench", 69.0, "https://artificialanalysis.ai/models/gemini-2-5-flash"),

    # =========================================================================
    # Gemini 3 Pro (released ~Nov 2025)
    # =========================================================================
    ("gemini-3-pro", "gpqa_diamond", 91.9, "https://venturebeat.com/ai/google-unveils-gemini-3-claiming-the-lead-in-math-science-multimodal-and"),
    ("gemini-3-pro", "aime_2025", 95.0, "https://venturebeat.com/ai/google-unveils-gemini-3-claiming-the-lead-in-math-science-multimodal-and"),
    ("gemini-3-pro", "aime_2025_with_code", 100.0, "https://venturebeat.com/ai/google-unveils-gemini-3-claiming-the-lead-in-math-science-multimodal-and"),
    ("gemini-3-pro", "mmlu_pro", 89.8, "https://artificialanalysis.ai/evaluations/mmlu-pro"),
    ("gemini-3-pro", "livecodebench", 91.7, "https://artificialanalysis.ai/evaluations/livecodebench"),
    ("gemini-3-pro", "frontiermath_t1_t3", 38.0, "https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5"),
    ("gemini-3-pro", "swe_bench_verified", 76.8, "https://live-swe-agent.github.io/"),
    ("gemini-3-pro", "humanitys_last_exam", 37.2, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),
    ("gemini-3-pro", "chatbot_arena_elo", 1492, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),
    ("gemini-3-pro", "swe_bench_pro", 43.30, "https://scale.com/leaderboard/swe_bench_pro_public"),
    ("gemini-3-pro", "gpqa_diamond_deep_think", 93.8, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),

    # =========================================================================
    # Gemini 3 Flash (released ~Nov 2025)
    # =========================================================================
    ("gemini-3-flash", "livecodebench", 90.8, "https://artificialanalysis.ai/evaluations/livecodebench"),
    ("gemini-3-flash", "aime_2025", 100.0, "https://artificialanalysis.ai/evaluations/aime-2025"),

    # =========================================================================
    # Gemini 3.1 Pro (released Feb 2026)
    # =========================================================================
    ("gemini-3.1-pro", "gpqa_diamond", 94.3, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "mmlu_pro", 89.5, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "humanitys_last_exam", 44.7, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),
    ("gemini-3.1-pro", "arc_agi_2", 77.1, "https://lmcouncil.ai/benchmarks"),

    # =========================================================================
    # DeepSeek R1 (released Jan 2025)
    # =========================================================================
    ("deepseek-r1", "mmlu", 90.8, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "mmlu_pro", 84.0, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "gpqa_diamond", 71.5, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "math_500", 97.3, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1", "humaneval", 96.1, "https://medium.com/@leucopsis/deepseeks-new-r1-0528-performance-analysis-and-benchmark-comparisons-6440eac858d6"),
    ("deepseek-r1", "swe_bench_verified", 49.2, "https://www.swebench.com/"),
    ("deepseek-r1", "aime_2025", 79.2, "https://intuitionlabs.ai/articles/aime-2025-ai-benchmark-explained"),

    # =========================================================================
    # DeepSeek R1-0528 (released May 2025)
    # =========================================================================
    ("deepseek-r1-0528", "aime_2025", 87.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "gpqa_diamond", 81.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "livecodebench_v6", 73.3, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "mmlu_pro", 85.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),

    # =========================================================================
    # DeepSeek V3 (released Dec 2024 / Jan 2025)
    # =========================================================================
    ("deepseek-v3", "mmlu", 88.5, "https://arxiv.org/pdf/2412.19437"),
    ("deepseek-v3", "mmlu_pro", 75.9, "https://arxiv.org/pdf/2412.19437"),
    ("deepseek-v3", "gpqa_diamond", 59.1, "https://arxiv.org/pdf/2412.19437"),
    ("deepseek-v3", "humaneval", 85.0, "https://medium.com/data-science-in-your-pocket/deepseek-v3-2-vs-gemini-3-0-vs-claude-4-5-vs-gpt-5-55a7d865debc"),

    # =========================================================================
    # DeepSeek V3.2 Speciale (released ~Dec 2025)
    # =========================================================================
    ("deepseek-v3.2-speciale", "mmlu_pro", 85.0, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "gpqa_diamond", 82.4, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "aime_2025", 93.1, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "livecodebench", 89.6, "https://artificialanalysis.ai/evaluations/livecodebench"),
    ("deepseek-v3.2-speciale", "livecodebench_cot", 88.7, "https://medium.com/@leucopsis/deepseek-v3-2-speciale-open-weights-reasoning-close-to-the-frontier-models-d43cd5da22d9"),
    ("deepseek-v3.2-speciale", "codeforces_rating", 2386, "https://medium.com/@leucopsis/deepseek-v3-2-speciale-open-weights-reasoning-close-to-the-frontier-models-d43cd5da22d9"),

    # =========================================================================
    # DeepSeek V3.2 Exp
    # =========================================================================
    ("deepseek-v3.2-exp", "aider_polyglot", 74.2, "https://llm-stats.com/benchmarks/aider-polyglot"),

    # =========================================================================
    # Grok 3 (released Feb 2025)
    # =========================================================================
    ("grok-3", "gpqa_diamond", 84.6, "https://x.ai/news/grok-3"),
    ("grok-3", "aime_2025", 93.3, "https://x.ai/news/grok-3"),
    ("grok-3", "livecodebench", 79.4, "https://x.ai/news/grok-3"),
    ("grok-3", "aider_polyglot", 53.3, "https://aider.chat/docs/leaderboards/"),

    # =========================================================================
    # Grok 4 (released ~Jul 2025)
    # =========================================================================
    ("grok-4", "gpqa_diamond", 87.5, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),
    ("grok-4", "aime_2025", 94.0, "https://www.baytechconsulting.com/blog/grok-4-vs-gpt-4o-claude-gemini-the-ultimate-b2b-ai-showdown-2025"),
    ("grok-4", "swe_bench_verified", 73.5, "https://www.baytechconsulting.com/blog/grok-4-vs-gpt-4o-claude-gemini-the-ultimate-b2b-ai-showdown-2025"),
    ("grok-4", "mmlu_pro", 85.3, "https://www.llmrumors.com/news/grok-4-the-breakthrough-ai-model-that-changes-everything"),
    ("grok-4", "humanitys_last_exam", 35.0, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),

    # =========================================================================
    # Grok 4.1 (released ~Oct 2025)
    # =========================================================================
    ("grok-4.1", "chatbot_arena_elo", 1482, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),
    ("grok-4.1", "swe_bench_verified", 79.0, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),

    # =========================================================================
    # Llama 4 Scout (released Apr 2025)
    # =========================================================================
    ("llama-4-scout", "mmlu_pro", 74.3, "https://www.llama.com/models/llama-4/"),
    ("llama-4-scout", "gpqa_diamond", 57.2, "https://www.llama.com/models/llama-4/"),
    ("llama-4-scout", "livecodebench", 32.8, "https://www.llama.com/models/llama-4/"),

    # =========================================================================
    # Llama 4 Maverick (released Apr 2025)
    # =========================================================================
    ("llama-4-maverick", "mmlu_pro", 80.5, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "gpqa_diamond", 69.8, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "livecodebench", 43.4, "https://www.llama.com/models/llama-4/"),

    # =========================================================================
    # Llama 4 Behemoth (released ~mid 2025)
    # =========================================================================
    ("llama-4-behemoth", "aime_2024", 73.7, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),

    # =========================================================================
    # Kimi K2.5 (released ~Dec 2025)
    # =========================================================================
    ("kimi-k2.5", "mmlu", 92.0, "https://artificialanalysis.ai/models/kimi-k2-5"),
    ("kimi-k2.5", "mmlu_pro", 87.1, "https://artificialanalysis.ai/models/kimi-k2-5"),
    ("kimi-k2.5", "gpqa_diamond", 87.6, "https://artificialanalysis.ai/models/kimi-k2-5"),
    ("kimi-k2.5", "aime_2025", 96.1, "https://artificialanalysis.ai/models/kimi-k2-5"),
    ("kimi-k2.5", "swe_bench_verified", 76.8, "https://artificialanalysis.ai/models/kimi-k2-5"),

    # =========================================================================
    # Qwen3-235B-A22B (released May 2025)
    # =========================================================================
    ("qwen3-235b", "mmlu_redux", 89.2, "https://arxiv.org/html/2505.09388v1"),
    ("qwen3-235b", "gpqa_diamond", 62.9, "https://arxiv.org/html/2505.09388v1"),
    ("qwen3-235b", "aime_2024", 85.7, "https://arxiv.org/html/2505.09388v1"),
    ("qwen3-235b", "aime_2025", 81.5, "https://arxiv.org/html/2505.09388v1"),
    ("qwen3-235b", "livecodebench_v5", 70.7, "https://arxiv.org/html/2505.09388v1"),
    ("qwen3-235b", "codeforces_rating", 2056, "https://arxiv.org/html/2505.09388v1"),

    # =========================================================================
    # Qwen3-32B (released May 2025)
    # =========================================================================
    ("qwen3-32b", "mmlu_redux", 85.7, "https://arxiv.org/html/2505.09388v1"),
    ("qwen3-32b", "gpqa_diamond", 54.6, "https://arxiv.org/html/2505.09388v1"),

    # =========================================================================
    # Qwen3-Max (Alibaba hosted, ~2025)
    # =========================================================================
    ("qwen3-max", "gpqa_diamond", 85.4, "https://dev.to/czmilo/qwen3-max-2025-complete-release-analysis-in-depth-review-of-alibabas-most-powerful-ai-model-3j7l"),
    ("qwen3-max", "swe_bench_verified", 69.6, "https://dev.to/czmilo/qwen3-max-2025-complete-release-analysis-in-depth-review-of-alibabas-most-powerful-ai-model-3j7l"),
    ("qwen3-max", "swe_bench_verified_max", 88.3, "https://artificialanalysis.ai/models/kimi-k2-5"),  # reported in comparison

    # =========================================================================
    # Mistral Large 3 (released ~Dec 2025)
    # =========================================================================
    ("mistral-large-3", "mmlu", 85.5, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "gpqa_diamond", 43.9, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "humaneval", 92.0, "https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained"),
    ("mistral-large-3", "chatbot_arena_elo", 1418, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),

    # =========================================================================
    # GLM-4.5 (released Jul 2025, Zhipu AI)
    # =========================================================================
    ("glm-4.5", "mmlu_pro", 84.6, "https://llm-stats.com/models/glm-4.5"),
    ("glm-4.5", "bfcl", 70.85, "https://gorilla.cs.berkeley.edu/leaderboard.html"),

    # =========================================================================
    # GLM-4.7 (released Dec 2025, Zhipu AI)
    # =========================================================================
    ("glm-4.7", "aime_2025", 95.7, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "gpqa_diamond", 85.7, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "livecodebench_v6", 84.9, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "chatbot_arena_elo", 1445, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),

    # =========================================================================
    # MiniMax-M2 (released Oct 2025)
    # =========================================================================
    ("minimax-m2", "artificial_analysis_index", 61.0, "https://artificialanalysis.ai/models/minimax-m2"),

    # =========================================================================
    # WebDev Arena Elo scores (LMArena, Feb 2026)
    # =========================================================================
    ("claude-opus-4.5", "webdev_arena_elo", 1510, "https://lmarena.ai/leaderboard/code"),
    ("deepseek-v3.2", "webdev_arena_elo", 1373, "https://felloai.com/best-ai-of-january-2026/"),

    # =========================================================================
    # Chatbot Arena — overall Elo (Feb 2026 snapshot)
    # =========================================================================
    ("gemini-3-pro", "chatbot_arena_elo_overall", 1492, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),
    ("grok-4.1-thinking", "chatbot_arena_elo_overall", 1482, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),
    ("gpt-5.1-high", "chatbot_arena_elo_overall", 1464, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),
    ("glm-4.7", "chatbot_arena_elo_overall", 1445, "https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html"),
    ("mistral-large-3", "chatbot_arena_elo_overall", 1418, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),

    # =========================================================================
    # WebArena (web agent benchmark)
    # =========================================================================
    ("gemini-2.5-pro", "webarena", 54.8, "https://webarena.dev/"),

    # =========================================================================
    # Aider polyglot — additional entries
    # =========================================================================
    ("o3", "aider_polyglot", 81.3, "https://aider.chat/docs/leaderboards/"),
    ("o3-pro", "aider_polyglot", 84.9, "https://www.getpassionfruit.com/blog/gpt-5-1-vs-claude-4-5-sonnet-vs-gemini-3-pro-vs-deepseek-v3-2-the-definitive-2025-ai-model-comparison"),
    ("claude-3.7-sonnet", "aider_polyglot", 64.9, "https://aider.chat/docs/leaderboards/"),
    ("grok-3-mini", "aider_polyglot", 49.3, "https://aider.chat/docs/leaderboards/"),

    # =========================================================================
    # CanAICode (coding interview benchmark)
    # =========================================================================
    ("claude-opus-4", "canaicode", 72.0, "https://huggingface.co/spaces/mike-ravkine/can-ai-code-results"),

    # =========================================================================
    # BFCL v4 (Berkeley Function Calling Leaderboard)
    # =========================================================================
    ("glm-4.5", "bfcl_v4", 70.85, "https://gorilla.cs.berkeley.edu/leaderboard.html"),
    ("claude-opus-4.1", "bfcl_v4", 70.36, "https://gorilla.cs.berkeley.edu/leaderboard.html"),
    ("claude-sonnet-4", "bfcl_v4", 70.29, "https://gorilla.cs.berkeley.edu/leaderboard.html"),
    ("gpt-5", "bfcl_v4", 59.22, "https://gorilla.cs.berkeley.edu/leaderboard.html"),

    # =========================================================================
    # Artificial Analysis Intelligence Index v4.0 (composite)
    # =========================================================================
    ("gemini-3-pro-high", "aa_intelligence_index", 68.0, "https://artificialanalysis.ai/leaderboards/models"),
    ("gpt-5", "aa_intelligence_index", 68.0, "https://artificialanalysis.ai/leaderboards/models"),

    # =========================================================================
    # SWE-Bench Pro (Scale SEAL, Jan 2026)
    # =========================================================================
    ("gpt-5.2-high", "swe_bench_pro", 41.78, "https://scale.com/leaderboard/swe_bench_pro_public"),

    # =========================================================================
    # Humanity's Last Exam (HLE) — additional reasoning variants
    # =========================================================================
    ("gemini-3-pro-high", "humanitys_last_exam", 37.2, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),
    ("claude-opus-4.6", "humanitys_last_exam_adaptive", 36.7, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),
    ("gpt-5.2-xhigh", "humanitys_last_exam", 31.64, "https://artificialanalysis.ai/evaluations/humanitys-last-exam"),

    # =========================================================================
    # GPQA Diamond — high-compute / reasoning variants (Artificial Analysis)
    # =========================================================================
    ("gemini-3-pro-high", "gpqa_diamond", 90.8, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),
    ("gpt-5.2-xhigh", "gpqa_diamond", 90.3, "https://artificialanalysis.ai/evaluations/gpqa-diamond"),

    # =========================================================================
    # LiveCodeBench — reasoning-variant top performers (Artificial Analysis)
    # =========================================================================
    ("gemini-3-pro-high", "livecodebench", 91.7, "https://artificialanalysis.ai/evaluations/livecodebench"),
    ("gemini-3-flash-reasoning", "livecodebench", 90.8, "https://artificialanalysis.ai/evaluations/livecodebench"),

    # =========================================================================
    # MMLU-Pro — reasoning variants (Artificial Analysis)
    # =========================================================================
    ("gemini-3-pro-high", "mmlu_pro", 89.8, "https://artificialanalysis.ai/evaluations/mmlu-pro"),
    ("gemini-3-pro-low", "mmlu_pro", 89.5, "https://artificialanalysis.ai/evaluations/mmlu-pro"),
    ("claude-opus-4.5-reasoning", "mmlu_pro", 89.5, "https://artificialanalysis.ai/evaluations/mmlu-pro"),

]


# ---- Helper: quick summary ----
def summarize():
    """Print a summary of the scores collected."""
    from collections import Counter
    models = Counter(m for m, _, _, _ in EXTRA_SCORES)
    benchmarks = Counter(b for _, b, _, _ in EXTRA_SCORES)
    print(f"Total entries: {len(EXTRA_SCORES)}")
    print(f"Unique models: {len(models)}")
    print(f"Unique benchmarks: {len(benchmarks)}")
    print(f"\nTop models by entry count:")
    for m, c in models.most_common(15):
        print(f"  {m}: {c}")
    print(f"\nTop benchmarks by entry count:")
    for b, c in benchmarks.most_common(15):
        print(f"  {b}: {c}")


if __name__ == "__main__":
    summarize()
