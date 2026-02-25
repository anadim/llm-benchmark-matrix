#!/usr/bin/env python3
"""
Generate publication-quality figures for the LLM Benchmark Matrix report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os, json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, MODEL_PROVIDERS, BENCH_CATS,
)

FIG_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

COLORS = {
    'LogitSVD': '#2563EB',
    'BenchReg': '#7C3AED',
    'Blend': '#059669',
    'SVD': '#D97706',
    'Mean': '#9CA3AF',
    'Claude': '#E11D48',
}


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 1: Method Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_method_comparison():
    """Horizontal bar chart of all methods sorted by PM-MedAPE."""
    # Load eval data
    eval_path = os.path.join(REPO_ROOT, 'results', 'logit_svd_eval.json')
    with open(eval_path) as f:
        data = json.load(f)

    methods = []
    for name, vals in data.items():
        methods.append((name, vals['pm_medape'], vals.get('pm_coverage', 100)))

    # Add Claude Sonnet 4 result if available
    claude_path = os.path.join(REPO_ROOT, 'results', 'claude_eval.json')
    if os.path.exists(claude_path):
        with open(claude_path) as f:
            cdata = json.load(f)
        if 'holdout' in cdata and 'claude' in cdata['holdout']:
            cm = cdata['holdout']['claude']
            if cm.get('medape') is not None:
                methods.append((f"Claude ({cdata.get('model', 'API').split('-')[1]})", cm['medape'], cm.get('coverage', 100)))

    methods.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [m[0] for m in methods]
    medapes = [m[1] for m in methods]

    colors = []
    for name in names:
        if 'LogitSVD' in name:
            colors.append(COLORS['LogitSVD'])
        elif 'LogitBenchReg' in name:
            colors.append('#3B82F6')
        elif 'SVD-Logit' in name:
            colors.append(COLORS['SVD'])
        elif 'BenchReg' in name and 'KNN' not in name:
            colors.append(COLORS['BenchReg'])
        elif 'Blend' in name or 'KNN' in name:
            colors.append(COLORS['Blend'])
        elif 'SVD' in name:
            colors.append(COLORS['SVD'])
        elif 'Claude' in name:
            colors.append(COLORS['Claude'])
        else:
            colors.append(COLORS['Mean'])

    bars = ax.barh(range(len(names)), medapes, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Per-Model MedAPE (%)', fontsize=12)
    ax.set_title('Prediction Accuracy by Method', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, medapes):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    ax.set_xlim(0, max(medapes) * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'method_comparison.png'))
    plt.close()
    print("  Saved method_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 2: Phase Transition Curve
# ═══════════════════════════════════════════════════════════════════════════
def fig_phase_transition():
    """MedAPE and ±3pts vs number of known scores."""
    import csv
    pt_path = os.path.join(REPO_ROOT, 'results', 'phase_transition.csv')
    rows = []
    with open(pt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    n_known = [r['n_known'] for r in rows]
    medape = [r['mean_medape'] for r in rows]
    within3 = [r['mean_within3'] for r in rows]
    within5 = [r['mean_within5'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: MedAPE
    ax1.plot(n_known, medape, 'o-', color=COLORS['LogitSVD'], linewidth=2, markersize=5)
    ax1.axhline(y=6.74, color=COLORS['LogitSVD'], linestyle='--', alpha=0.4, label='Full-matrix LogitSVD (6.74%)')
    ax1.axhline(y=7.86, color=COLORS['Blend'], linestyle='--', alpha=0.4, label='Full-matrix old Blend (7.86%)')
    ax1.set_xlabel('Number of Known Benchmark Scores', fontsize=12)
    ax1.set_ylabel('Median APE (%)', fontsize=12)
    ax1.set_title('Prediction Error vs Known Scores', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(0.5, 20.5)
    ax1.set_ylim(0, 18)

    # Right: ±3 and ±5 pts
    ax2.plot(n_known, within3, 's-', color='#2563EB', linewidth=2, markersize=5, label='Within ±3 pts')
    ax2.plot(n_known, within5, 'D-', color='#059669', linewidth=2, markersize=5, label='Within ±5 pts')
    ax2.set_xlabel('Number of Known Benchmark Scores', fontsize=12)
    ax2.set_ylabel('Fraction of Predictions (%)', fontsize=12)
    ax2.set_title('Prediction Precision vs Known Scores', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.5, 20.5)
    ax2.set_ylim(15, 70)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'phase_transition.png'))
    plt.close()
    print("  Saved phase_transition.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 3: Matrix Sparsity Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_sparsity_heatmap():
    """Heatmap showing observed vs missing cells."""
    # Sort models by coverage (most covered first)
    model_coverage = OBSERVED.sum(axis=1)
    model_order = np.argsort(-model_coverage)

    # Sort benchmarks by coverage
    bench_coverage = OBSERVED.sum(axis=0)
    bench_order = np.argsort(-bench_coverage)

    obs_sorted = OBSERVED[model_order][:, bench_order]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap: observed=blue, missing=light gray
    cmap = plt.cm.colors.ListedColormap(['#F3F4F6', '#2563EB'])
    ax.imshow(obs_sorted, aspect='auto', cmap=cmap, interpolation='nearest')

    # Labels (show every 5th)
    y_labels = [MODEL_NAMES[MODEL_IDS[i]] for i in model_order]
    x_labels = [BENCH_IDS[j] for j in bench_order]

    ax.set_yticks(range(0, len(y_labels), 3))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), 3)], fontsize=6)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=6, rotation=90)

    ax.set_title(f'Matrix Sparsity Pattern: {OBSERVED.sum()} observed / {N_MODELS*N_BENCH} total ({OBSERVED.sum()/(N_MODELS*N_BENCH)*100:.0f}% fill)',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Benchmarks (sorted by coverage)', fontsize=11)
    ax.set_ylabel('Models (sorted by coverage)', fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2563EB', label='Observed'),
        mpatches.Patch(facecolor='#F3F4F6', edgecolor='#D1D5DB', label='Missing'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'sparsity_heatmap.png'))
    plt.close()
    print("  Saved sparsity_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 4: Logit Transform Effect
# ═══════════════════════════════════════════════════════════════════════════
def fig_logit_effect():
    """Show why logit transform helps: comparison of raw vs logit space."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: The logit function
    ax = axes[0]
    x = np.linspace(0.5, 99.5, 200)
    y = np.log((x/100) / (1 - x/100))
    ax.plot(x, y, color=COLORS['LogitSVD'], linewidth=2.5)
    ax.set_xlabel('Score (%)', fontsize=11)
    ax.set_ylabel('logit(score)', fontsize=11)
    ax.set_title('(A) The Logit Transform', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=50, color='gray', linewidth=0.5, linestyle='--')
    # Annotate compression at extremes
    ax.annotate('Compressed\nnear 0%', xy=(5, -3), fontsize=9, color='#6B7280', ha='center')
    ax.annotate('Compressed\nnear 100%', xy=(95, 3), fontsize=9, color='#6B7280', ha='center')
    ax.annotate('Linear\naround 50%', xy=(50, 0.5), fontsize=9, color='#6B7280', ha='center')

    # Panel B: Method comparison (MedAPE)
    ax = axes[1]
    methods = ['BenchReg', 'BenchReg\n(logit)', 'SVD(r=2)', 'SVD-Logit\n(r=2)', 'Old Blend', 'LogitSVD\nBlend']
    values = [7.45, 6.61, 8.62, 7.52, 7.86, 6.74]
    colors_bar = [COLORS['BenchReg'], '#3B82F6', COLORS['SVD'], '#F59E0B', COLORS['Blend'], COLORS['LogitSVD']]

    bars = ax.bar(range(len(methods)), values, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('PM-MedAPE (%)', fontsize=11)
    ax.set_title('(B) Logit Improves Every Method', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 10)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add improvement arrows
    for idx_from, idx_to, label in [(0, 1, '-11%'), (2, 3, '-13%'), (4, 5, '-14%')]:
        ax.annotate('', xy=(idx_to, values[idx_to]-0.3), xytext=(idx_from, values[idx_from]-0.3),
                   arrowprops=dict(arrowstyle='->', color='#DC2626', linewidth=1.5))

    # Panel C: Bimodal accuracy
    ax = axes[2]
    methods_bi = ['Benchmark\nMean', 'BenchReg', 'Old\nBlend', 'SVD(r=2)', 'SVD-Logit\n(r=2)', 'LogitSVD\nBlend']
    biacc = [80.6, 81.4, 84.9, 87.5, 89.0, 89.9]
    colors_bi = [COLORS['Mean'], COLORS['BenchReg'], COLORS['Blend'], COLORS['SVD'], '#F59E0B', COLORS['LogitSVD']]

    bars = ax.bar(range(len(methods_bi)), biacc, color=colors_bi, edgecolor='white')
    ax.set_xticks(range(len(methods_bi)))
    ax.set_xticklabels(methods_bi, fontsize=9)
    ax.set_ylabel('Bimodal Classification Accuracy (%)', fontsize=10)
    ax.set_title('(C) Logit Handles Bimodal Benchmarks', fontsize=12, fontweight='bold')
    ax.set_ylim(75, 95)

    for bar, val in zip(bars, biacc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'logit_effect.png'))
    plt.close()
    print("  Saved logit_effect.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 5: SVD Factor Scatter (Factor 1 vs Factor 2)
# ═══════════════════════════════════════════════════════════════════════════
def fig_latent_factors():
    """Scatter plot of models in the 2D latent space."""
    import csv

    # Load factor loadings to interpret axes
    # We need to compute model scores in the latent space
    # Use SVD on z-scored matrix
    M_z = M_FULL.copy()
    col_mean = np.nanmean(M_z, axis=0)
    col_std = np.nanstd(M_z, axis=0)
    col_std[col_std < 1e-6] = 1
    for j in range(N_BENCH):
        M_z[:, j] = (M_z[:, j] - col_mean[j]) / col_std[j]

    # Impute missing with 0 (mean in z-score space)
    M_imp = np.where(np.isnan(M_z), 0, M_z)

    U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)

    # Model scores on factor 1 and factor 2
    f1 = U[:, 0] * s[0]
    f2 = U[:, 1] * s[1]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by provider
    provider_colors = {
        'OpenAI': '#10B981', 'Anthropic': '#8B5CF6', 'Google': '#3B82F6',
        'DeepSeek': '#EF4444', 'Alibaba': '#F59E0B', 'Meta': '#6366F1',
        'xAI': '#EC4899', 'Mistral': '#14B8A6', 'Microsoft': '#06B6D4',
        'Moonshot AI': '#F97316', 'ByteDance': '#84CC16',
    }

    # Plot each model
    for i in range(N_MODELS):
        provider = MODEL_PROVIDERS[i]
        color = provider_colors.get(provider, '#9CA3AF')
        marker = '^' if MODEL_REASONING[i] else 'o'
        size = 40 if MODEL_REASONING[i] else 25

        ax.scatter(f1[i], f2[i], c=color, marker=marker, s=size, alpha=0.8,
                  edgecolors='white', linewidth=0.3, zorder=3)

    # Label notable models
    notable = ['gpt-5', 'o3-high', 'claude-opus-4.6', 'gemini-2.5-pro', 'deepseek-r1',
               'gpt-4.5', 'grok-4', 'qwen3-235b', 'llama-4-maverick', 'gemini-3.1-pro',
               'gpt-5.2', 'kimi-k2.5']
    for mid in notable:
        if mid in [MODEL_IDS[k] for k in range(N_MODELS)]:
            i = MODEL_IDS.index(mid)
            ax.annotate(MODEL_NAMES[mid], (f1[i], f2[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8)

    ax.set_xlabel('Factor 1: General Capability (37% variance)', fontsize=12)
    ax.set_ylabel('Factor 2: Frontier Reasoning (14% variance)', fontsize=12)
    ax.set_title('83 LLMs in 2D Latent Space', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.3, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.3, linestyle='--')

    # Provider legend
    handles = []
    for prov, color in sorted(provider_colors.items()):
        if prov in MODEL_PROVIDERS:
            handles.append(mpatches.Patch(color=color, label=prov))
    handles.append(plt.Line2D([0], [0], marker='^', color='gray', label='Reasoning', markersize=8, linestyle=''))
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', label='Non-reasoning', markersize=6, linestyle=''))
    ax.legend(handles=handles, loc='lower left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'latent_factors.png'))
    plt.close()
    print("  Saved latent_factors.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 6: Reasoning Gap by Benchmark
# ═══════════════════════════════════════════════════════════════════════════
def fig_reasoning_gap():
    """Bar chart of reasoning vs non-reasoning z-score gap per benchmark."""
    # Compute z-scores
    M_z = M_FULL.copy()
    col_mean = np.nanmean(M_z, axis=0)
    col_std = np.nanstd(M_z, axis=0)
    col_std[col_std < 1e-6] = 1

    gaps = []
    for j in range(N_BENCH):
        r_scores = [M_z[i, j] for i in range(N_MODELS) if MODEL_REASONING[i] and not np.isnan(M_z[i, j])]
        nr_scores = [M_z[i, j] for i in range(N_MODELS) if not MODEL_REASONING[i] and not np.isnan(M_z[i, j])]

        if len(r_scores) >= 3 and len(nr_scores) >= 3:
            r_z = (np.mean(r_scores) - col_mean[j]) / col_std[j]
            nr_z = (np.mean(nr_scores) - col_mean[j]) / col_std[j]
            gaps.append((BENCH_IDS[j], r_z - nr_z, BENCH_CATS[j]))

    gaps.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    names = [g[0] for g in gaps]
    values = [g[1] for g in gaps]
    cats = [g[2] for g in gaps]

    cat_colors = {
        'Math': '#EF4444', 'Coding': '#3B82F6', 'Reasoning': '#8B5CF6',
        'Knowledge': '#10B981', 'Agentic': '#F59E0B', 'Multimodal': '#EC4899',
        'Instruction Following': '#14B8A6', 'Science': '#6366F1',
        'Long Context': '#84CC16', 'Composite': '#F97316', 'Human Preference': '#9CA3AF',
    }

    colors = [cat_colors.get(c, '#9CA3AF') for c in cats]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='white', linewidth=0.3)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Reasoning Advantage (z-score gap)', fontsize=11)
    ax.set_title('Reasoning vs Non-Reasoning Model Gap by Benchmark', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend for categories
    handles = [mpatches.Patch(color=c, label=cat) for cat, c in cat_colors.items() if cat in cats]
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'reasoning_gap.png'))
    plt.close()
    print("  Saved reasoning_gap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures...")
    fig_method_comparison()
    fig_phase_transition()
    fig_sparsity_heatmap()
    fig_logit_effect()
    fig_latent_factors()
    fig_reasoning_gap()
    print("\nAll figures saved to figures/")
