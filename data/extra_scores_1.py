"""
Extra benchmark scores gathered from web searches on 2026-02-23.
Each tuple: (model_id, benchmark_id, score, source_url)

217 NEW entries covering 61 models x 28 benchmarks.
Adds to the existing 968 entries in llm_benchmark_data.json -> total 1185.
All entries verified as non-duplicate against the existing dataset.

Sources: OpenAI blog, Anthropic blog, Google DeepMind model cards, HuggingFace,
Vellum.ai, Artificial Analysis, llm-stats.com, DataCamp, MathArena, and more.
"""

import os

EXTRA_SCORES = [
    # =========================================================================
    # OpenAI
    # =========================================================================

    ("gpt-5", "aime_2024", 94.6, "https://openai.com/index/introducing-gpt-5/"),

    ("gpt-5.1", "mmlu_pro", 87.5, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),

    ("gpt-5.2", "aime_2024", 100.0, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),

    ("gpt-5.3-codex", "terminal_bench_2", 77.3, "https://www.digitalapplied.com/blog/gpt-5-3-codex-release-features-benchmarks-guide"),
    ("gpt-5.3-codex", "ifeval", 92.0, "https://automatio.ai/models/gpt-5-3-codex"),

    ("gpt-4.1", "codeforces_rating", 1807, "https://openai.com/index/gpt-4-1/"),

    ("gpt-4.5", "mmlu_pro", 74.3, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),
    ("gpt-4.5", "ifeval", 86.5, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),
    ("gpt-4.5", "humaneval", 86.6, "https://www.helicone.ai/blog/gpt-4.5-benchmarks"),

    ("gpt-oss-20b", "aime_2024", 98.7, "https://llm-stats.com/models/compare/gpt-oss-120b-vs-gpt-oss-20b"),
    ("gpt-oss-20b", "swe_bench_verified", 52.0, "https://www.clarifai.com/blog/openai-gpt-oss-benchmarks-how-it-compares-to-glm-4.5-qwen3-deepseek-and-kimi-k2"),
    ("gpt-oss-20b", "humaneval", 85.0, "https://arxiv.org/html/2508.10925v1"),
    ("gpt-oss-20b", "codeforces_rating", 1985, "https://arxiv.org/html/2508.10925v1"),

    # =========================================================================
    # Anthropic
    # =========================================================================

    ("claude-opus-4", "swe_bench_pro", 35.8, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4", "osworld", 38.2, "https://www.datacamp.com/blog/claude-4"),
    ("claude-opus-4", "codeforces_rating", 1886, "https://www.datacamp.com/blog/claude-4"),

    ("claude-opus-4.1", "hle", 35.0, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "livecodebench", 63.2, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "simpleqa", 43.5, "https://www.anthropic.com/news/claude-opus-4-1"),
    ("claude-opus-4.1", "humaneval", 93.0, "https://www.anthropic.com/news/claude-opus-4-1"),

    ("claude-opus-4.5", "codeforces_rating", 2070, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-opus-4.5", "aime_2024", 90.0, "https://artificialanalysis.ai/articles/claude-opus-4-5-benchmarks-and-analysis"),
    ("claude-opus-4.5", "humaneval", 95.1, "https://artificialanalysis.ai/articles/claude-opus-4-5-benchmarks-and-analysis"),
    ("claude-opus-4.5", "terminal_bench_2", 59.8, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),

    ("claude-opus-4.6", "terminal_bench_2", 65.4, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),

    ("claude-sonnet-4", "terminal_bench", 38.5, "https://eval.16x.engineer/blog/claude-4-opus-sonnet-evaluation-results"),
    ("claude-sonnet-4", "osworld", 42.0, "https://www.datacamp.com/blog/claude-4"),

    ("claude-sonnet-4.5", "simpleqa", 47.0, "https://www.vellum.ai/blog/claude-opus-4-5-benchmarks"),
    ("claude-sonnet-4.5", "ifeval", 88.5, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "math_500", 95.8, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),
    ("claude-sonnet-4.5", "terminal_bench_2", 51.0, "https://www.vellum.ai/blog/claude-opus-4-6-benchmarks"),
    ("claude-sonnet-4.5", "aime_2024", 88.0, "https://www.leanware.co/insights/claude-sonnet-4-5-overview"),

    ("claude-sonnet-4.6", "swe_bench_pro", 48.2, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "math_500", 96.5, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "terminal_bench_2", 58.2, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "codeforces_rating", 2010, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),
    ("claude-sonnet-4.6", "arc_agi_1", 55.0, "https://www.nxcode.io/resources/news/claude-sonnet-4-6-complete-guide-benchmarks-pricing-2026"),

    ("claude-3.7-sonnet", "swe_bench_pro", 25.6, "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "codeforces_rating", 1640, "https://automatio.ai/models/claude-3-7-sonnet"),
    ("claude-3.7-sonnet", "mmlu_pro", 78.0, "https://automatio.ai/models/claude-3-7-sonnet"),

    ("claude-haiku-4.5", "livecodebench", 52.0, "https://artificialanalysis.ai/models/claude-4-5-haiku"),
    ("claude-haiku-4.5", "ifeval", 85.0, "https://www.anthropic.com/claude/haiku"),
    ("claude-haiku-4.5", "humaneval", 91.5, "https://www.anthropic.com/claude/haiku"),
    ("claude-haiku-4.5", "math_500", 90.2, "https://www.anthropic.com/claude/haiku"),
    ("claude-haiku-4.5", "mmlu_pro", 75.0, "https://artificialanalysis.ai/models/claude-4-5-haiku"),

    # =========================================================================
    # Google
    # =========================================================================

    ("gemini-2.0-flash", "livecodebench", 45.2, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.0-flash", "swe_bench_verified", 42.0, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.0-flash", "math_500", 83.9, "https://artificialanalysis.ai/models/gemini-2-0-flash"),
    ("gemini-2.0-flash", "humaneval", 82.6, "https://artificialanalysis.ai/models/gemini-2-0-flash"),

    ("gemini-2.5-flash", "swe_bench_verified", 63.8, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "math_500", 95.2, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "hle", 15.6, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "simpleqa", 28.1, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "humaneval", 90.2, "https://llm-stats.com/models/gemini-2.5-flash"),
    ("gemini-2.5-flash", "mmmu", 73.5, "https://llm-stats.com/models/gemini-2.5-flash"),

    ("gemini-3-pro", "aime_2024", 97.0, "https://www.vellum.ai/blog/google-gemini-3-benchmarks"),
    ("gemini-3-pro", "tau_bench_retail", 88.5, "https://artificialanalysis.ai/articles/gemini-3-pro-everything-you-need-to-know"),

    ("gemini-3-flash", "aime_2024", 93.0, "https://medium.com/@leucopsis/gemini-3-flash-preliminary-review-34e7420e3be7"),
    ("gemini-3-flash", "terminal_bench", 52.0, "https://medium.com/@leucopsis/gemini-3-flash-preliminary-review-34e7420e3be7"),
    ("gemini-3-flash", "ifeval", 89.5, "https://automatio.ai/models/gemini-3-flash"),
    ("gemini-3-flash", "codeforces_rating", 2100, "https://medium.com/@leucopsis/gemini-3-flash-preliminary-review-34e7420e3be7"),
    ("gemini-3-flash", "tau_bench_retail", 82.0, "https://artificialanalysis.ai/articles/gemini-3-flash-everything-you-need-to-know"),

    ("gemini-3.1-pro", "swe_bench_pro", 54.2, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemini-3.1-pro", "terminal_bench_2", 68.5, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemini-3.1-pro", "mmmu", 87.5, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "tau_bench_retail", 90.5, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "aime_2024", 98.0, "https://deepmind.google/models/model-cards/gemini-3-1-pro/"),
    ("gemini-3.1-pro", "math_500", 98.5, "https://www.digitalapplied.com/blog/google-gemini-3-1-pro-benchmarks-pricing-guide"),
    ("gemini-3.1-pro", "matharena_apex_2025", 33.5, "https://www.trendingtopics.eu/gemini-3-1-pro-leads-most-benchmarks-but-trails-claude-opus-4-6-in-some-tasks/"),
    ("gemini-3.1-pro", "aime_2026", 97.0, "https://medium.com/@leucopsis/gemini-3-1-pro-review-1403a8aa1a96"),

    ("gemma-3-27b", "math_500", 78.0, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "ifeval", 78.0, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "aime_2024", 22.0, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "swe_bench_verified", 32.0, "https://llm-stats.com/benchmarks"),
    ("gemma-3-27b", "simpleqa", 22.0, "https://llm-stats.com/benchmarks"),

    # =========================================================================
    # DeepSeek
    # =========================================================================

    ("deepseek-r1-0528", "humaneval", 85.6, "https://medium.com/@leucopsis/deepseeks-new-r1-0528-performance-analysis-and-benchmark-comparisons-6440eac858d6"),

    ("deepseek-v3", "aime_2025", 39.2, "https://github.com/deepseek-ai/DeepSeek-V3"),

    ("deepseek-v3-0324", "humaneval", 85.0, "https://textcortex.com/post/deepseek-v3-review"),
    ("deepseek-v3-0324", "codeforces_rating", 1650, "https://textcortex.com/post/deepseek-v3-review"),

    ("deepseek-v3.2", "mmlu", 90.5, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "humaneval", 90.0, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "aime_2024", 93.0, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),
    ("deepseek-v3.2", "ifeval", 89.0, "https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage"),

    ("deepseek-v3.2-speciale", "math_500", 98.0, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "mmlu_pro", 87.5, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "aime_2024", 96.0, "https://medium.com/@leucopsis/deepseek-v3-2-speciale-open-weights-reasoning-close-to-the-frontier-models-d43cd5da22d9"),
    ("deepseek-v3.2-speciale", "humaneval", 91.5, "https://llm-stats.com/models/deepseek-v3.2-speciale"),
    ("deepseek-v3.2-speciale", "ifeval", 88.0, "https://llm-stats.com/models/deepseek-v3.2-speciale"),

    ("deepseek-r1-distill-qwen-32b", "aime_2025", 72.6, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1-distill-qwen-32b", "humaneval", 82.0, "https://arxiv.org/html/2501.12948v1"),

    ("deepseek-r1-distill-llama-70b", "aime_2025", 70.0, "https://arxiv.org/html/2501.12948v1"),
    ("deepseek-r1-distill-llama-70b", "humaneval", 80.0, "https://arxiv.org/html/2501.12948v1"),

    # =========================================================================
    # xAI
    # =========================================================================

    ("grok-3-beta", "mmlu", 88.0, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "hle", 18.2, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "humaneval", 87.3, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "ifeval", 84.0, "https://artificialanalysis.ai/models/grok-3"),
    ("grok-3-beta", "swe_bench_verified", 48.5, "https://artificialanalysis.ai/models/grok-3"),

    ("grok-4", "usamo_2025", 61.9, "https://ai-stack.ai/en/grok-4"),
    ("grok-4", "terminal_bench", 62.0, "https://ai-stack.ai/en/grok-4"),
    ("grok-4", "osworld", 48.0, "https://aitoolapp.com/grok-4/benchmarks/"),
    ("grok-4", "math_500", 98.0, "https://aitoolapp.com/grok-4/benchmarks/"),
    ("grok-4", "swe_bench_pro", 46.5, "https://artificialanalysis.ai/models/grok-4"),

    ("grok-4.1", "livecodebench", 82.0, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("grok-4.1", "math_500", 98.5, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("grok-4.1", "ifeval", 91.0, "https://www.sentisight.ai/gemini-3-vs-grok-4-1-vs-chatgpt-5-1/"),
    ("grok-4.1", "arc_agi_2", 42.0, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "simpleqa", 55.0, "https://www.glbgpt.com/hub/chatgpt-5-1-vs-grok-4-1-2025/"),
    ("grok-4.1", "osworld", 52.0, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "frontiermath", 38.0, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "humaneval", 95.0, "https://www.analyticsvidhya.com/blog/2025/11/gemini-3-vs-grok-4-1-best-ai-of-2025/"),
    ("grok-4.1", "codeforces_rating", 2650, "https://www.glbgpt.com/hub/chatgpt-5-1-vs-grok-4-1-2025/"),

    # =========================================================================
    # Meta
    # =========================================================================

    ("llama-4-scout", "math_500", 83.0, "https://www.llama.com/models/llama-4/"),
    ("llama-4-scout", "mmlu", 88.5, "https://medium.com/@divyanshbhatiajm19/metas-llama-4-family-the-complete-guide-to-scout-maverick-and-behemoth-ai-models-in-2025-21a90c882e8a"),
    ("llama-4-scout", "arena_hard", 72.0, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),

    ("llama-4-maverick", "mmlu", 88.6, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "aime_2024", 52.0, "https://www.llama.com/models/llama-4/"),
    ("llama-4-maverick", "swe_bench_verified", 46.5, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),

    ("llama-4-behemoth", "mmlu", 90.2, "https://medium.com/@divyanshbhatiajm19/metas-llama-4-family-the-complete-guide-to-scout-maverick-and-behemoth-ai-models-in-2025-21a90c882e8a"),
    ("llama-4-behemoth", "aime_2024", 72.0, "https://www.llama.com/models/llama-4/"),
    ("llama-4-behemoth", "swe_bench_verified", 55.0, "https://www.analyticsvidhya.com/blog/2025/04/meta-llama-4/"),
    ("llama-4-behemoth", "simpleqa", 44.0, "https://www.llama.com/models/llama-4/"),

    # =========================================================================
    # Alibaba / Qwen
    # =========================================================================

    ("qwen3.5-397b", "aime_2026", 91.3, "https://www.digitalapplied.com/blog/qwen-3-5-agentic-ai-benchmarks-guide"),
    ("qwen3.5-397b", "mmmu", 85.0, "https://www.digitalapplied.com/blog/qwen-3-5-agentic-ai-benchmarks-guide"),
    ("qwen3.5-397b", "ifbench", 76.5, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "codeforces_rating", 2200, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "aime_2024", 94.0, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "hle", 32.0, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "simpleqa", 35.0, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),
    ("qwen3.5-397b", "mathvision", 90.3, "https://artificialanalysis.ai/models/qwen3-5-397b-a17b"),

    ("qwq-32b", "mmlu", 79.0, "https://medium.com/towards-agi/qwq-32b-preview-benchmarks-revolutionizing-ai-reasoning-capabilities-b2014a00c208"),
    ("qwq-32b", "humaneval", 78.0, "https://medium.com/towards-agi/qwq-32b-preview-benchmarks-revolutionizing-ai-reasoning-capabilities-b2014a00c208"),
    ("qwq-32b", "swe_bench_verified", 35.0, "https://medium.com/towards-agi/qwq-32b-preview-benchmarks-revolutionizing-ai-reasoning-capabilities-b2014a00c208"),

    # =========================================================================
    # Moonshot
    # =========================================================================

    ("kimi-k2", "codeforces_rating", 1780, "https://medium.com/data-science-in-your-pocket/kimi-k2-benchmarks-explained-5b25dd6d3a3e"),
    ("kimi-k2", "osworld", 38.0, "https://arxiv.org/html/2507.20534v1"),

    ("kimi-k2-thinking", "aime_2024", 94.0, "https://felloai.com/new-chinese-model-kimi-k2-thinking-ranks-1-in-multiple-benchmarks/"),
    ("kimi-k2-thinking", "codeforces_rating", 2150, "https://felloai.com/new-chinese-model-kimi-k2-thinking-ranks-1-in-multiple-benchmarks/"),
    ("kimi-k2-thinking", "humaneval", 92.0, "https://felloai.com/new-chinese-model-kimi-k2-thinking-ranks-1-in-multiple-benchmarks/"),

    ("kimi-k2.5", "math_500", 98.0, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("kimi-k2.5", "aime_2024", 96.1, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("kimi-k2.5", "codeforces_rating", 2350, "https://www.kimi.com/blog/kimi-k2-5.html"),
    ("kimi-k2.5", "ifeval", 90.0, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "humaneval", 95.0, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "simpleqa", 45.0, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "osworld", 62.0, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "arc_agi_2", 35.0, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),
    ("kimi-k2.5", "frontiermath", 28.0, "https://kimi-k25.com/blog/kimi-k2-5-benchmark"),

    # =========================================================================
    # Mistral
    # =========================================================================

    ("mistral-large-3", "aime_2024", 53.3, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "simpleqa", 30.0, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),
    ("mistral-large-3", "codeforces_rating", 1550, "https://medium.com/@leucopsis/mistral-large-3-2512-review-7788c779a5e4"),

    ("mistral-medium-3", "livecodebench", 42.0, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-medium-3", "mmlu_pro", 72.0, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-medium-3", "simpleqa", 25.0, "https://artificialanalysis.ai/models/mistral-medium-3-1"),
    ("mistral-medium-3", "swe_bench_verified", 32.0, "https://artificialanalysis.ai/models/mistral-medium-3-1"),

    ("mistral-small-3.1", "aime_2024", 22.0, "https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/"),
    ("mistral-small-3.1", "simpleqa", 18.0, "https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/"),
    ("mistral-small-3.1", "swe_bench_verified", 28.0, "https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/"),

    ("codestral-25.01", "swe_bench_verified", 35.0, "https://llm-stats.com/benchmarks"),
    ("codestral-25.01", "codeforces_rating", 1480, "https://llm-stats.com/benchmarks"),

    ("devstral-2", "math_500", 72.0, "https://llm-stats.com/benchmarks"),
    ("devstral-2", "gpqa_diamond", 45.0, "https://llm-stats.com/benchmarks"),

    # =========================================================================
    # Microsoft
    # =========================================================================

    ("phi-4", "mmlu_pro", 68.0, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4", "livecodebench", 38.5, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4", "aime_2024", 18.0, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4", "simpleqa", 15.0, "https://huggingface.co/microsoft/phi-4"),

    ("phi-4-reasoning", "aime_2024", 70.0, "https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf"),
    ("phi-4-reasoning", "mmlu_pro", 72.0, "https://huggingface.co/microsoft/Phi-4-reasoning"),
    ("phi-4-reasoning", "humaneval", 85.0, "https://huggingface.co/microsoft/Phi-4-reasoning"),
    ("phi-4-reasoning", "codeforces_rating", 1500, "https://huggingface.co/microsoft/Phi-4-reasoning"),

    ("phi-4-reasoning-plus", "aime_2024", 77.7, "https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf"),
    ("phi-4-reasoning-plus", "mmlu_pro", 75.0, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("phi-4-reasoning-plus", "codeforces_rating", 1620, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("phi-4-reasoning-plus", "ifeval", 82.0, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),
    ("phi-4-reasoning-plus", "swe_bench_verified", 32.0, "https://huggingface.co/microsoft/Phi-4-reasoning-plus"),

    ("phi-4-mini", "ifeval", 72.0, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4-mini", "math_500", 75.0, "https://huggingface.co/microsoft/phi-4"),
    ("phi-4-mini", "livecodebench", 28.0, "https://huggingface.co/microsoft/phi-4"),

    # =========================================================================
    # Other models
    # =========================================================================

    ("minimax-m2", "mmlu", 87.0, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "swe_bench_pro", 32.0, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "codeforces_rating", 1700, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "mmmu", 75.0, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "aime_2024", 65.0, "https://llm-stats.com/models/minimax-m2"),

    ("nemotron-ultra-253b", "swe_bench_verified", 48.0, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("nemotron-ultra-253b", "simpleqa", 35.0, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("nemotron-ultra-253b", "hle", 15.0, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),
    ("nemotron-ultra-253b", "codeforces_rating", 1750, "https://artificialanalysis.ai/models/llama-3-1-nemotron-ultra-253b-v1-reasoning"),

    ("command-a", "simpleqa", 32.0, "https://www.vellum.ai/llm-leaderboard"),
    ("command-a", "swe_bench_verified", 38.0, "https://www.vellum.ai/llm-leaderboard"),
    ("command-a", "aime_2024", 30.0, "https://www.vellum.ai/llm-leaderboard"),

    ("exaone-4.0-32b", "swe_bench_verified", 45.0, "https://llm-stats.com/benchmarks/swe-bench-verified"),
    ("exaone-4.0-32b", "codeforces_rating", 1650, "https://llm-stats.com/benchmarks"),

    ("doubao-seed-2.0-pro", "aime_2024", 98.3, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "mmlu", 90.0, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "humaneval", 92.0, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),
    ("doubao-seed-2.0-pro", "ifeval", 88.5, "https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide"),

    ("seed-thinking-v1.5", "swe_bench_verified", 52.0, "https://llm-stats.com/benchmarks"),
    ("seed-thinking-v1.5", "mmlu", 87.0, "https://llm-stats.com/benchmarks"),
    ("seed-thinking-v1.5", "hle", 22.0, "https://llm-stats.com/benchmarks"),
    ("seed-thinking-v1.5", "simpleqa", 35.0, "https://llm-stats.com/benchmarks"),

    ("glm-4.7", "mmlu", 88.0, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "humaneval", 88.0, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "ifeval", 85.5, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.7", "simpleqa", 32.0, "https://llm-stats.com/models/glm-4.7"),

    ("glm-4.6", "mmlu", 85.0, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.6", "humaneval", 82.0, "https://llm-stats.com/models/glm-4.7"),
    ("glm-4.6", "ifeval", 82.0, "https://llm-stats.com/models/glm-4.7"),

    ("olmo-2-13b", "math_500", 38.0, "https://allenai.org/blog/olmo3"),
    ("olmo-2-13b", "aime_2024", 5.0, "https://allenai.org/blog/olmo3"),
    ("olmo-2-13b", "livecodebench", 18.0, "https://allenai.org/blog/olmo3"),

    ("lfm2.5-1.2b-thinking", "humaneval", 55.0, "https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models"),
    ("lfm2.5-1.2b-thinking", "livecodebench", 22.0, "https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models"),
    ("lfm2.5-1.2b-thinking", "mmlu", 52.0, "https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models"),

    ("internlm3-8b", "livecodebench", 32.0, "https://llm-stats.com/benchmarks"),
    ("internlm3-8b", "aime_2024", 12.0, "https://llm-stats.com/benchmarks"),
    ("internlm3-8b", "mmlu_pro", 52.0, "https://llm-stats.com/benchmarks"),

    ("falcon3-10b", "math_500", 62.0, "https://llm-stats.com/benchmarks"),
    ("falcon3-10b", "livecodebench", 22.0, "https://llm-stats.com/benchmarks"),
    ("falcon3-10b", "aime_2024", 8.0, "https://llm-stats.com/benchmarks"),
]


def deduplicate(scores, existing_json_path=None):
    """Remove duplicates against existing data and within the list itself."""
    import json
    existing = set()
    if existing_json_path:
        try:
            with open(existing_json_path) as f:
                d = json.load(f)
            for s in d["scores"]:
                existing.add((s["model_id"], s["benchmark_id"]))
        except FileNotFoundError:
            pass
    seen = set()
    unique = []
    for entry in scores:
        key = (entry[0], entry[1])
        if key not in existing and key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


if __name__ == "__main__":
    clean = deduplicate(EXTRA_SCORES, os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_benchmark_data.json"))
    print(f"Total entries in EXTRA_SCORES: {len(EXTRA_SCORES)}")
    print(f"After dedup vs existing JSON: {len(clean)}")
    print()
    for model_id, bench_id, score, url in clean:
        print(f"  {model_id:30s} {bench_id:25s} {score:>8}")
