"""
Unverified cells removed from the canonical matrix during audit (Feb 2026).

These entries were flagged because:
  - The cited source URL does not contain the claimed value, AND
  - No authoritative alternative source was found to verify the value.

They are preserved here with reasons and candidate replacement sources
for future re-verification. Do NOT use these for training/eval until re-verified.

To restore a cell: verify the value against an authoritative source, update the
source URL, and move the entry back to build_benchmark_matrix.py or extra_scores.
"""

UNVERIFIED_CELLS = [
    # (model, benchmark, removed_value, original_source_url, reason, candidate_sources)

    ("deepseek-r1-distill-qwen-32b", "ifeval", 79.0,
     "https://github.com/deepseek-ai/DeepSeek-R1",
     "DeepSeek-R1 GitHub does not report IFEval for distilled models. "
     "HuggingFace Open LLM Leaderboard shows ~40-42 (different eval setup). "
     "IFEval scores are highly sensitive to chat template and prompting strategy.",
     ["https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]),

    ("deepseek-r1-distill-llama-70b", "ifeval", 81.0,
     "https://github.com/deepseek-ai/DeepSeek-R1",
     "DeepSeek-R1 GitHub does not report IFEval for distilled models. "
     "Third-party evaluations suggest ~43. Same eval-setup sensitivity as above.",
     ["https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"]),

    ("deepseek-r1-distill-qwen-32b", "mmlu", 85.0,
     "https://github.com/deepseek-ai/DeepSeek-R1",
     "DeepSeek-R1 GitHub does not include MMLU in distilled model benchmarks. "
     "Value cannot be traced to any authoritative source.",
     []),

    ("o4-mini-high", "browsecomp", 45.0,
     "https://www.helicone.ai/blog/o3-and-o4-mini-for-developers",
     "Cited source shows o3 at 49.7% and o4-mini at 28.3% (basic tools). "
     "The 45.0 value matches neither. BrowseComp scores vary dramatically "
     "by tool configuration (no tools: 3.3%, basic: 28.3%, agentic: 45-51%).",
     ["https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf"]),

    ("glm-4.7", "mmlu", 88.0,
     "https://llm-stats.com/models/glm-4.7",
     "llm-stats.com uses JS-rendered tables that cannot be verified by scraping. "
     "Official GLM-4.7 docs report MMLU-Pro (84.3) but not standard MMLU. "
     "RESOLVED: HuggingFace model card shows 90.1 — that entry (extra_scores_2.py) "
     "is now the canonical value since the 88.0 entry was removed.",
     ["https://huggingface.co/zai-org/GLM-4.7"]),

    ("glm-4.7", "humaneval", 88.0,
     "https://llm-stats.com/models/glm-4.7",
     "llm-stats.com JS-rendered table, unverifiable. "
     "One web source suggests 94.2, which would make 88.0 incorrect. "
     "Official docs do not report HumanEval.",
     ["https://docs.z.ai/guides/llm/glm-4.7"]),

    ("glm-4.7", "ifeval", 85.5,
     "https://llm-stats.com/models/glm-4.7",
     "llm-stats.com JS-rendered table, unverifiable. "
     "No alternative source found for this value.",
     []),

    ("gpt-5", "gsm8k", 96.8,
     "https://llm-stats.com/benchmarks",
     "OpenAI does not report GSM8K for GPT-5 (saturated benchmark). "
     "llm-stats.com may aggregate third-party evaluations. "
     "Value is plausible (GPT-5.2 scores 99% per automatio.ai) but unverifiable.",
     []),

    ("phi-4", "gsm8k", 95.3,
     "https://arxiv.org/html/2412.08905v1",
     "Phi-4 technical report does not include GSM8K. "
     "Paper reports MGSM (Multilingual GSM) at 80.6, not GSM8K at 95.3. "
     "Value is plausible but source citation is wrong.",
     ["https://huggingface.co/microsoft/phi-4"]),
]
