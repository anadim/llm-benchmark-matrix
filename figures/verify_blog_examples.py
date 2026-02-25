#!/usr/bin/env python3
"""
Verify specific prediction examples claimed in the blog post.

Each prediction is done in proper holdout mode: the cell being predicted
is hidden (set to NaN) before running the LogitSVD Blend predictor.
This ensures we are measuring true out-of-sample prediction, not just
recovering observed values.
"""

import numpy as np
import sys, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))

from evaluation_harness import (
    M_FULL, OBSERVED, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_IDX, BENCH_IDX,
    N_MODELS, N_BENCH,
)
from methods.all_methods import predict_logit_svd_blend


def holdout_predict(model_id, bench_id):
    """Hide one cell from M_FULL, predict it with LogitSVD Blend, return (actual, predicted)."""
    i = MODEL_IDX[model_id]
    j = BENCH_IDX[bench_id]

    actual = M_FULL[i, j]
    if np.isnan(actual):
        return actual, np.nan, False  # cell is already missing

    # Create training matrix with this cell hidden
    M_train = M_FULL.copy()
    M_train[i, j] = np.nan

    # Run predictor
    M_pred = predict_logit_svd_blend(M_train)
    predicted = M_pred[i, j]

    return actual, predicted, True


# ════════════════════════════════════════════════════════════════════════
#  Blog claim #1-5: Specific prediction examples
# ════════════════════════════════════════════════════════════════════════

examples = [
    # (description, model_id, bench_id, blog_actual, blog_predicted)
    ("Gemini 3 Pro on Terminal-Bench 2.0",    "gemini-3-pro",     "terminal_bench",       56.9,  56.2),
    ("Claude Opus 4.6 on AIME 2025",          "claude-opus-4.6",  "aime_2025",           100.0,  98.1),
    ("GPT-5.2 on SWE-bench Verified",         "gpt-5.2",         "swe_bench_verified",    80.0,  80.6),
    ("Qwen3-14B on AIME 2025",                "qwen3-14b",       "aime_2025",             72.0,  73.6),
    ("Claude Sonnet 4.6 on ARC-AGI-2",        "claude-sonnet-4.6","arc_agi_2",            60.4,  15.3),
]

print("=" * 90)
print("  BLOG CLAIM VERIFICATION: LogitSVD Blend Predictions (holdout mode)")
print("=" * 90)
print()
print(f"  {'Example':<40s} {'Actual':>8s} {'BlogPred':>9s} {'OurPred':>9s} {'BlogMatch':>10s}")
print(f"  {'─'*40} {'─'*8} {'─'*9} {'─'*9} {'─'*10}")

for desc, mid, bid, blog_actual, blog_predicted in examples:
    actual, predicted, was_observed = holdout_predict(mid, bid)
    if not was_observed:
        print(f"  {desc:<40s} {'MISSING':>8s}    —         —         —")
        continue

    actual_match = abs(actual - blog_actual) < 0.05
    pred_match = abs(predicted - blog_predicted) < 0.15  # allow small float tolerance
    status = "OK" if (actual_match and pred_match) else "MISMATCH"

    print(f"  {desc:<40s} {actual:8.1f} {blog_predicted:9.1f} {predicted:9.1f} {status:>10s}")
    if not actual_match:
        print(f"    ** ACTUAL mismatch: data has {actual}, blog claims {blog_actual}")
    if not pred_match:
        print(f"    ** PREDICTED mismatch: we get {predicted:.1f}, blog claims {blog_predicted}")

# ════════════════════════════════════════════════════════════════════════
#  Blog claim #6: Intro table values
# ════════════════════════════════════════════════════════════════════════

print()
print("=" * 90)
print("  BLOG CLAIM #6: Intro table actual values")
print("=" * 90)

intro_checks = [
    ("Claude Opus 4.6 AIME 2025 = 100", "claude-opus-4.6", "aime_2025", 100.0),
    ("GPT-5.2 GPQA Diamond = ? (blog says answer is 93)", "gpt-5.2", "gpqa_diamond", 93.0),
]

for desc, mid, bid, blog_val in intro_checks:
    i = MODEL_IDX[mid]
    j = BENCH_IDX[bid]
    actual = M_FULL[i, j]
    is_observed = OBSERVED[i, j]

    if is_observed:
        match = abs(actual - blog_val) < 0.5
        print(f"  {desc}")
        print(f"    Data value: {actual:.1f}  (observed={is_observed})  Blog: {blog_val}  Match: {match}")
        if not match:
            print(f"    ** MISMATCH: data has {actual:.1f}, blog claims {blog_val}")
    else:
        # Cell is missing in data - predict it
        M_train = M_FULL.copy()
        M_pred = predict_logit_svd_blend(M_train)
        predicted = M_pred[i, j]
        print(f"  {desc}")
        print(f"    Cell is MISSING in data. Predicted: {predicted:.1f}  Blog claims: {blog_val}")


# ════════════════════════════════════════════════════════════════════════
#  Blog claim #7: Gemini 3 Pro scores
# ════════════════════════════════════════════════════════════════════════

print()
print("=" * 90)
print("  BLOG CLAIM #7: Gemini 3 Pro actual scores")
print("=" * 90)

gemini_checks = [
    ("GPQA Diamond = 92", "gemini-3-pro", "gpqa_diamond", 92.0),
    ("AIME 2025 = 95",    "gemini-3-pro", "aime_2025",    95.0),
    ("SWE-bench = 76",    "gemini-3-pro", "swe_bench_verified", 76.0),
]

for desc, mid, bid, blog_val in gemini_checks:
    i = MODEL_IDX[mid]
    j = BENCH_IDX[bid]
    actual = M_FULL[i, j]
    is_observed = OBSERVED[i, j]
    match = abs(actual - blog_val) < 0.5 if is_observed else False

    print(f"  Gemini 3 Pro {desc}")
    if is_observed:
        print(f"    Data value: {actual:.1f}  Blog: {blog_val}  Match: {match}")
        if not match:
            print(f"    ** MISMATCH: data has {actual:.1f}, blog says {blog_val}")
    else:
        print(f"    Cell is MISSING in data")

print()
print("=" * 90)
print("  SUMMARY")
print("=" * 90)
print()
print("  Done. Check above for any MISMATCH flags.")
