"""
Plot SVD spectrum of the largest fully-observed sub-matrix of the LLM benchmark matrix.
Mean-centers columns before SVD (= PCA) so the first component isn't just the mean.
Filters to percentage-scale benchmarks only (excludes Elo, Codeforces rating, etc.).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Load matrix ──────────────────────────────────────────────────────────
df = pd.read_excel('/Users/anadim/llm-benchmark-matrix/data/llm_benchmark_matrix.xlsx',
                   index_col=0)

# Drop metadata columns
META_COLS = ['Provider', 'Release Date', 'Params (M)', 'Active (M)',
             'Architecture', 'Reasoning?', 'Open Weights?']
df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')

# Drop non-percentage-scale columns (they dominate SVD due to scale difference)
NON_PCT = ['chatbot_arena_elo', 'codeforces_rating', 'aa_intelligence_index',
           'aa_lcr', 'gdpval_aa']
df = df.drop(columns=[c for c in NON_PCT if c in df.columns], errors='ignore')

# ── Find largest complete sub-matrix with both dims >= 10 ────────────────
best_area = 0
best_sub = None

for n_bench in range(7, min(len(df.columns) + 1, 20)):
    fill_counts = df.notna().sum(axis=0).sort_values(ascending=False)
    top_benches = list(fill_counts.index[:n_bench])
    sub = df[top_benches].dropna()
    if len(sub) >= 10 and len(top_benches) >= 10:
        area = len(sub) * len(top_benches)
        if area > best_area:
            best_area = area
            best_sub = sub

n_models, n_bench = best_sub.shape
print(f"Complete sub-matrix: {n_models} models x {n_bench} benchmarks")
print(f"Models: {list(best_sub.index)}")
print(f"Benchmarks: {list(best_sub.columns)}")

M = best_sub.values.astype(float)

# ── Mean-center columns (so SVD = PCA, first component isn't just the mean) ──
col_means = M.mean(axis=0)
M_centered = M - col_means

# ── SVD on centered matrix ───────────────────────────────────────────────
U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
n_comp = min(10, len(S))
S = S[:n_comp]

eigs = S**2
total = eigs.sum()
frac = eigs / total              # fraction of total spectrum per component
cumul = np.cumsum(frac)           # cumulative fraction captured
remaining = 1.0 - cumul           # fraction of spectrum still remaining

# Normalize bars: sigma_i^2 / sigma_1^2
norm_eigs = eigs / eigs[0]

idx = np.arange(1, n_comp + 1)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 7.5))

# Blue bars: normalized eigenvalues
bars = ax1.bar(idx, norm_eigs, color='#4A90D9', alpha=0.85, width=0.6, zorder=3)
ax1.set_xlabel('Component index', fontsize=18, fontweight='bold')
ax1.set_ylabel(r'$\sigma_i^2\,/\,\sigma_1^2$', fontsize=18, color='#4A90D9',
               fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#4A90D9', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.set_xticks(idx)
ax1.set_ylim(0, 1.18)

# Red curve: remaining spectrum (right axis)
ax2 = ax1.twinx()
ax2.plot(idx, remaining, 'o-', color='#D94A4A', linewidth=2.5, markersize=9, zorder=4)
ax2.set_ylabel('Fraction of spectrum remaining', fontsize=18, color='#D94A4A',
               fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#D94A4A', labelsize=15)
# Set right y-axis to fit the data nicely
max_rem = remaining[0] * 1.35
ax2.set_ylim(-0.02, max_rem)

# ── Annotate blue bars (% of spectrum) ───────────────────────────────────
# Place blue annotations ABOVE each bar
for i in range(min(5, n_comp)):
    pct = frac[i] * 100
    if pct >= 1:  # only annotate if >= 1%
        ax1.text(idx[i], norm_eigs[i] + 0.04, f'{pct:.0f}%',
                 ha='center', va='bottom', fontsize=15, fontweight='bold',
                 color='#4A90D9')

# ── Annotate red curve (remaining spectrum %) ────────────────────────────
# Place red annotations to the LEFT of each dot to avoid overlapping with blue
red_indices = [0, 1, 2, 4]  # which points to annotate
for i in red_indices:
    if i < n_comp:
        pct = remaining[i] * 100
        # Offset labels to the left and slightly above to avoid blue bar labels
        ax2.annotate(f'{pct:.0f}%',
                     xy=(idx[i], remaining[i]),
                     xytext=(-35, 10), textcoords='offset points',
                     fontsize=14, fontweight='bold', color='#D94A4A',
                     arrowprops=dict(arrowstyle='-', color='#D94A4A', lw=0.8),
                     ha='center')

# Title
ax1.set_title(
    f'Singular value spectrum of a fully-observed {n_models} x {n_bench} sub-matrix\n'
    f'({n_models} models, {n_bench} percentage-scale benchmarks, no missing data, mean-centered)',
    fontsize=17, fontweight='bold', pad=18
)

# Legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#4A90D9', alpha=0.85,
                   label=r'Normalized eigenvalue  $\sigma_i^2/\sigma_1^2$'),
    Line2D([0], [0], color='#D94A4A', marker='o', linewidth=2.5, markersize=9,
           label='Fraction of spectrum remaining')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.9)

# Grid
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

# Caption
caption = (
    f"Figure: Singular value spectrum of the largest fully-observed sub-matrix "
    f"({n_models} models x {n_bench} benchmarks), mean-centered.\n"
    f"Blue bars: each squared singular value normalized by the largest. "
    f"Red curve: fraction of total spectrum not yet captured by the top i components.\n"
    f"The first component captures {frac[0]*100:.0f}% of the spectrum; "
    f"two components capture {cumul[1]*100:.0f}%. By component 5, only {remaining[4]*100:.0f}% remains.\n"
    f"This steep decay — from raw data with no imputation — confirms the approximate low-rank structure "
    f"of LLM benchmark scores."
)
fig.text(0.5, -0.02, caption, ha='center', va='top', fontsize=13,
         fontstyle='italic', color='#555555', wrap=True)

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)
plt.savefig('/Users/anadim/llm-benchmark-matrix/figures/svd_complete_submatrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nSaved to figures/svd_complete_submatrix.png")

# Print the numbers
print(f"\nTop {n_comp} singular values: {np.round(S, 2)}")
print(f"Spectrum fractions: {np.round(frac*100, 1)}%")
print(f"Cumulative: {np.round(cumul*100, 1)}%")
print(f"Remaining: {np.round(remaining*100, 1)}%")
