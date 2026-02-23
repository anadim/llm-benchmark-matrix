"""
Extra benchmark scores scraped from official model technical reports and blog posts.
Each entry: (model_id, benchmark_id, score, source_url)

Collected: 2026-02-23
"""

EXTRA_SCORES = [
    # =========================================================================
    # OpenAI GPT-4.1 (April 2025)
    # =========================================================================
    ("gpt-4.1", "mmlu", 90.2, "https://openai.com/index/gpt-4-1/"),
    ("gpt-4.1", "gpqa_diamond", 66.3, "https://openai.com/index/gpt-4-1/"),
    ("gpt-4.1", "swe_bench_verified", 54.6, "https://openai.com/index/gpt-4-1/"),

    # =========================================================================
    # OpenAI o3 (April 2025)
    # =========================================================================
    ("o3", "gpqa_diamond", 87.7, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o3", "swe_bench_verified", 69.1, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o3", "mmmu", 82.9, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o3", "aime_2025", 96.7, "https://openai.com/index/introducing-o3-and-o4-mini/"),

    # =========================================================================
    # OpenAI o4-mini (April 2025)
    # =========================================================================
    ("o4-mini", "gpqa_diamond", 81.4, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o4-mini", "swe_bench_verified", 68.1, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o4-mini", "mmmu", 81.6, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o4-mini", "aime_2024", 93.4, "https://openai.com/index/introducing-o3-and-o4-mini/"),
    ("o4-mini", "aime_2025", 92.7, "https://openai.com/index/introducing-o3-and-o4-mini/"),

    # =========================================================================
    # OpenAI GPT-5 (August 2025)
    # =========================================================================
    ("gpt-5", "gpqa_diamond", 88.4, "https://openai.com/index/introducing-gpt-5/"),  # GPT-5 pro
    ("gpt-5", "swe_bench_verified", 74.9, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "aime_2025", 94.6, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "mmmu", 84.2, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "aider_polyglot", 88.0, "https://openai.com/index/introducing-gpt-5/"),
    ("gpt-5", "healthbench_hard", 46.2, "https://openai.com/index/introducing-gpt-5/"),

    # =========================================================================
    # OpenAI GPT-5.1 (November 2025)
    # =========================================================================
    ("gpt-5.1", "gpqa_diamond", 88.1, "https://openai.com/index/introducing-gpt-5-2/"),  # referenced in 5.2 comparison
    ("gpt-5.1", "swe_bench_verified", 76.3, "https://claude5.com/news/gpt-5-1-swe-bench-score-benchmark-analysis"),
    ("gpt-5.1", "aime_2025", 94.0, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.1", "frontiermath", 31.0, "https://openai.com/index/introducing-gpt-5-2/"),

    # =========================================================================
    # OpenAI GPT-5.2 (December 2025)
    # =========================================================================
    ("gpt-5.2", "gpqa_diamond", 92.4, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "swe_bench_verified", 80.0, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "aime_2025", 100.0, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "frontiermath", 40.3, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "arc_agi_1", 90.5, "https://openai.com/index/introducing-gpt-5-2/"),  # ARC-AGI-1 verified
    ("gpt-5.2", "arc_agi_2", 54.2, "https://openai.com/index/introducing-gpt-5-2/"),  # pass@2
    ("gpt-5.2", "swe_bench_pro", 55.6, "https://openai.com/index/introducing-gpt-5-2/"),
    ("gpt-5.2", "mmmu_pro", 86.5, "https://www.vellum.ai/blog/gpt-5-2-benchmarks"),

    # =========================================================================
    # OpenAI GPT-5.3-Codex (February 2026)
    # =========================================================================
    ("gpt-5.3-codex", "swe_bench_pro", 56.8, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "terminal_bench_2", 77.3, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "gpqa_diamond", 81.0, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "mmlu", 93.0, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "math", 96.0, "https://openai.com/index/introducing-gpt-5-3-codex/"),
    ("gpt-5.3-codex", "gsm8k", 99.0, "https://openai.com/index/introducing-gpt-5-3-codex/"),

    # =========================================================================
    # Anthropic Claude Opus 4 (May 2025)
    # =========================================================================
    ("claude-opus-4", "swe_bench_verified", 72.5, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "gpqa_diamond", 79.6, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "aime_2025", 75.5, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "mmlu", 88.8, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4", "terminal_bench", 43.2, "https://www.anthropic.com/news/claude-4"),
    # High-compute variants
    ("claude-opus-4-high", "swe_bench_verified", 79.4, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4-high", "gpqa_diamond", 83.3, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4-high", "aime_2025", 90.0, "https://www.anthropic.com/news/claude-4"),
    ("claude-opus-4-high", "terminal_bench", 50.0, "https://www.anthropic.com/news/claude-4"),

    # =========================================================================
    # Anthropic Claude Sonnet 4 (May 2025)
    # =========================================================================
    ("claude-sonnet-4", "swe_bench_verified", 72.7, "https://www.anthropic.com/news/claude-4"),
    ("claude-sonnet-4", "gpqa_diamond", 75.4, "https://www.anthropic.com/news/claude-4"),
    ("claude-sonnet-4", "aime_2025", 70.5, "https://www.anthropic.com/news/claude-4"),
    ("claude-sonnet-4", "mmlu", 86.5, "https://www.anthropic.com/news/claude-4"),

    # =========================================================================
    # Anthropic Claude Opus 4.1 (August 2025)
    # =========================================================================
    ("claude-opus-4.1", "swe_bench_verified", 74.5, "https://www.anthropic.com/news/claude-opus-4-1"),

    # =========================================================================
    # Anthropic Claude Sonnet 4.5 (September 2025)
    # =========================================================================
    ("claude-sonnet-4.5", "swe_bench_verified", 77.2, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "gpqa_diamond", 83.4, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "aime_2025", 87.0, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "terminal_bench", 50.0, "https://www.anthropic.com/news/claude-sonnet-4-5"),
    ("claude-sonnet-4.5", "osworld", 61.4, "https://www.anthropic.com/news/claude-sonnet-4-5"),

    # =========================================================================
    # Anthropic Claude Opus 4.5 (November 2025)
    # =========================================================================
    ("claude-opus-4.5", "swe_bench_verified", 80.9, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "gpqa_diamond", 87.0, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "mmmu", 80.7, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "terminal_bench", 59.3, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "osworld", 66.3, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "arc_agi_2", 37.6, "https://www.anthropic.com/news/claude-opus-4-5"),
    ("claude-opus-4.5", "humanitys_last_exam", 43.2, "https://www.anthropic.com/news/claude-opus-4-5"),  # with web search

    # =========================================================================
    # Anthropic Claude Opus 4.6 (February 2026)
    # =========================================================================
    ("claude-opus-4.6", "gpqa_diamond", 91.3, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "swe_bench_verified", 80.8, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "aime_2025", 100.0, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "arc_agi_2", 68.8, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "terminal_bench_2", 65.4, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "osworld", 72.7, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "mmmu_pro", 73.9, "https://www.anthropic.com/news/claude-opus-4-6"),  # without tools
    ("claude-opus-4.6", "mmmu_pro_tools", 77.3, "https://www.anthropic.com/news/claude-opus-4-6"),  # with tools
    ("claude-opus-4.6", "humanitys_last_exam", 40.0, "https://www.anthropic.com/news/claude-opus-4-6"),  # without tools
    ("claude-opus-4.6", "humanitys_last_exam_tools", 53.1, "https://www.anthropic.com/news/claude-opus-4-6"),  # with tools
    ("claude-opus-4.6", "browsecomp", 84.0, "https://www.anthropic.com/news/claude-opus-4-6"),
    ("claude-opus-4.6", "mrcr_v2_1m", 76.0, "https://www.anthropic.com/news/claude-opus-4-6"),  # 1M token context

    # =========================================================================
    # Anthropic Claude Sonnet 4.6 (February 2026)
    # =========================================================================
    ("claude-sonnet-4.6", "swe_bench_verified", 79.6, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "osworld", 72.5, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "gpqa_diamond", 74.1, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "arc_agi_1", 86.5, "https://www.anthropic.com/news/claude-sonnet-4-6"),
    ("claude-sonnet-4.6", "arc_agi_2", 60.4, "https://www.anthropic.com/news/claude-sonnet-4-6"),  # high effort
    ("claude-sonnet-4.6", "mmmu_pro", 74.5, "https://www.anthropic.com/news/claude-sonnet-4-6"),

    # =========================================================================
    # Google Gemini 2.5 Pro (March 2025)
    # =========================================================================
    ("gemini-2.5-pro", "gpqa_diamond", 84.0, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "aime_2025", 86.7, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "aime_2024", 92.0, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "swe_bench_verified", 63.8, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "mmmu", 81.7, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "humanitys_last_exam", 18.8, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "livecodebench_v5", 70.4, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),
    ("gemini-2.5-pro", "global_mmlu_lite", 89.8, "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"),

    # =========================================================================
    # Google Gemini 3 Pro (November 2025)
    # =========================================================================
    ("gemini-3-pro", "gpqa_diamond", 91.9, "https://blog.google/products/gemini/gemini-3/"),
    ("gemini-3-pro", "swe_bench_verified", 76.2, "https://blog.google/products/gemini/gemini-3/"),
    ("gemini-3-pro", "aime_2025", 95.0, "https://blog.google/products/gemini/gemini-3/"),  # tools off
    ("gemini-3-pro", "aime_2025_tools", 100.0, "https://blog.google/products/gemini/gemini-3/"),  # with code execution
    ("gemini-3-pro", "mmmu_pro", 81.0, "https://blog.google/products/gemini/gemini-3/"),
    ("gemini-3-pro", "arc_agi_2", 31.1, "https://blog.google/products/gemini/gemini-3/"),
    ("gemini-3-pro", "humanitys_last_exam", 37.5, "https://blog.google/products/gemini/gemini-3/"),
    ("gemini-3-pro", "terminal_bench_2", 54.2, "https://blog.google/products/gemini/gemini-3/"),

    # =========================================================================
    # Google Gemini 3 Flash (December 2025)
    # =========================================================================
    ("gemini-3-flash", "gpqa_diamond", 90.4, "https://blog.google/products/gemini/gemini-3-flash/"),
    ("gemini-3-flash", "swe_bench_verified", 78.0, "https://blog.google/products/gemini/gemini-3-flash/"),
    ("gemini-3-flash", "aime_2025", 95.2, "https://blog.google/products/gemini/gemini-3-flash/"),  # without tools
    ("gemini-3-flash", "aime_2025_tools", 99.7, "https://blog.google/products/gemini/gemini-3-flash/"),  # with code execution
    ("gemini-3-flash", "humanitys_last_exam", 33.7, "https://blog.google/products/gemini/gemini-3-flash/"),
    ("gemini-3-flash", "mmmu_pro", 81.2, "https://blog.google/products/gemini/gemini-3-flash/"),

    # =========================================================================
    # Google Gemini 3.1 Pro (February 2026)
    # =========================================================================
    ("gemini-3.1-pro", "gpqa_diamond", 94.3, "https://deepmind.google/models/gemini/pro/"),
    ("gemini-3.1-pro", "swe_bench_verified", 80.6, "https://deepmind.google/models/gemini/pro/"),
    ("gemini-3.1-pro", "arc_agi_2", 77.1, "https://deepmind.google/models/gemini/pro/"),
    ("gemini-3.1-pro", "humanitys_last_exam", 44.4, "https://deepmind.google/models/gemini/pro/"),
    ("gemini-3.1-pro", "scicode", 59.0, "https://deepmind.google/models/gemini/pro/"),
    ("gemini-3.1-pro", "terminal_bench_2", 68.5, "https://deepmind.google/models/gemini/pro/"),

    # =========================================================================
    # Meta Llama 4 Scout (April 2025)
    # =========================================================================
    ("llama-4-scout", "mmmu", 69.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "mathvista", 70.7, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "chartqa", 88.8, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "docvqa", 94.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "livecodebench", 32.8, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-scout", "mmlu_pro", 74.3, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),

    # =========================================================================
    # Meta Llama 4 Maverick (April 2025)
    # =========================================================================
    ("llama-4-maverick", "mmmu", 73.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "mathvista", 73.7, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "chartqa", 90.0, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "docvqa", 94.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "livecodebench", 43.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "mmlu_pro", 80.5, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-maverick", "multilingual_mmlu", 84.6, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),

    # =========================================================================
    # Meta Llama 4 Behemoth (preview / still training as of April 2025)
    # =========================================================================
    ("llama-4-behemoth", "math_500", 95.0, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-behemoth", "gpqa_diamond", 73.7, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-behemoth", "mmmu", 76.1, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-behemoth", "livecodebench", 49.4, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-behemoth", "mmlu_pro", 82.2, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),
    ("llama-4-behemoth", "multilingual_mmlu", 85.8, "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"),

    # =========================================================================
    # xAI Grok 4 (July 2025)
    # =========================================================================
    ("grok-4", "gpqa_diamond", 88.0, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "aime_2025", 100.0, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "aime_2024", 94.0, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "mmlu_pro", 87.0, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "humanitys_last_exam", 24.0, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "hmmt_2025", 96.7, "https://artificialanalysis.ai/models/grok-4"),
    ("grok-4", "livecodebench", 82.0, "https://artificialanalysis.ai/models/grok-4"),

    # =========================================================================
    # xAI Grok 4.1 (November 2025)
    # =========================================================================
    ("grok-4.1", "swe_bench_verified", 79.0, "https://llm-stats.com/models/compare/grok-4-vs-grok-4.1-2025-11-17"),

    # =========================================================================
    # Moonshot AI Kimi K2.5 (January 2026)
    # =========================================================================
    ("kimi-k2.5", "gpqa_diamond", 88.0, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "swe_bench_verified", 76.8, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "aime_2025", 96.0, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "mmmu_pro", 78.5, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "livecodebench", 85.0, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "mathvision", 84.2, "https://huggingface.co/moonshotai/Kimi-K2.5"),
    ("kimi-k2.5", "humanitys_last_exam_tools", 50.2, "https://huggingface.co/moonshotai/Kimi-K2.5"),  # with tools
    ("kimi-k2.5", "browsecomp", 74.9, "https://huggingface.co/moonshotai/Kimi-K2.5"),  # without swarm
    ("kimi-k2.5", "browsecomp_swarm", 78.4, "https://huggingface.co/moonshotai/Kimi-K2.5"),  # with swarm

    # =========================================================================
    # MiniMax M2 (October 2025)
    # =========================================================================
    ("minimax-m2", "mmlu_pro", 82.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("minimax-m2", "gpqa_diamond", 78.0, "https://artificialanalysis.ai/models/minimax-m2"),
    ("minimax-m2", "swe_bench_verified", 69.4, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "aime_2025", 78.0, "https://llm-stats.com/models/minimax-m2"),
    ("minimax-m2", "livecodebench", 83.0, "https://llm-stats.com/models/minimax-m2"),

    # =========================================================================
    # NVIDIA Llama Nemotron Ultra 253B (April 2025)
    # =========================================================================
    ("nemotron-ultra-253b", "gpqa_diamond", 76.0, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/"),
    ("nemotron-ultra-253b", "aime_2025", 72.5, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/"),
    ("nemotron-ultra-253b", "math_500", 97.0, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/"),
    ("nemotron-ultra-253b", "ifeval", 89.5, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/"),
    ("nemotron-ultra-253b", "livecodebench", 66.3, "https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/"),

    # =========================================================================
    # Zhipu AI GLM-4.7 (December 2025)
    # =========================================================================
    ("glm-4.7", "gpqa_diamond", 85.7, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "swe_bench_verified", 73.8, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "aime_2025", 95.7, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "livecodebench", 84.9, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "mmlu", 90.1, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "hmmt_2025", 97.1, "https://huggingface.co/zai-org/GLM-4.7"),
    ("glm-4.7", "humanitys_last_exam_tools", 42.8, "https://huggingface.co/zai-org/GLM-4.7"),  # with tools

    # =========================================================================
    # ByteDance Doubao Seed 2.0 Pro (February 2026)
    # =========================================================================
    ("seed-2.0-pro", "gpqa_diamond", 88.9, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "aime_2025", 98.3, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "swe_bench_verified", 76.5, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "livecodebench_v6", 87.8, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "mmmu", 85.4, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "mathvision", 88.8, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "video_mme", 89.5, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "terminal_bench", 55.8, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),
    ("seed-2.0-pro", "browsecomp", 77.3, "https://seed.bytedance.com/en/blog/seed-2-0-official-launch"),

    # =========================================================================
    # DeepSeek R1 (January 2025)
    # =========================================================================
    ("deepseek-r1", "gpqa_diamond", 71.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "aime_2024", 79.8, "https://huggingface.co/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "swe_bench_verified", 49.2, "https://huggingface.co/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "math_500", 97.3, "https://huggingface.co/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "ifeval", 83.3, "https://huggingface.co/deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "livecodebench", 65.9, "https://huggingface.co/deepseek-ai/DeepSeek-R1"),

    # =========================================================================
    # DeepSeek R1-0528 (May 2025)
    # =========================================================================
    ("deepseek-r1-0528", "gpqa_diamond", 81.0, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),
    ("deepseek-r1-0528", "aime_2025", 87.5, "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"),

    # =========================================================================
    # DeepSeek V3.2 (December 2025)
    # =========================================================================
    ("deepseek-v3.2", "aime_2025", 96.0, "https://arxiv.org/pdf/2512.02556"),
    ("deepseek-v3.2", "swe_bench_verified", 73.1, "https://arxiv.org/pdf/2512.02556"),
    ("deepseek-v3.2", "gpqa_diamond", 79.9, "https://arxiv.org/pdf/2512.02556"),  # V3.2-Exp

    # =========================================================================
    # Alibaba Qwen3-Max (May 2025)
    # =========================================================================
    ("qwen3-max", "swe_bench_verified", 69.6, "https://arxiv.org/pdf/2505.09388"),
    ("qwen3-max", "gpqa_diamond", 85.4, "https://arxiv.org/pdf/2505.09388"),  # Qwen3-Max-Thinking
    ("qwen3-max", "aime_2025", 81.5, "https://arxiv.org/pdf/2505.09388"),
]


# =========================================================================
# Summary statistics
# =========================================================================
if __name__ == "__main__":
    models = sorted(set(s[0] for s in EXTRA_SCORES))
    benchmarks = sorted(set(s[1] for s in EXTRA_SCORES))
    print(f"Total entries: {len(EXTRA_SCORES)}")
    print(f"Unique models: {len(models)}")
    print(f"Unique benchmarks: {len(benchmarks)}")
    print(f"\nModels: {models}")
    print(f"\nBenchmarks: {benchmarks}")
