# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/robustness/Rev3C_SoC_Trajectory_Plots_Colab.ipynb
# Notebook SHA256: c3586d977c2c4fe4ab193c83b0547e6f2107ec83d2f4a7b184976b5d703be5ce
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] SoC trajectory figures — manuscript-ready preview

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
REV3_OUT = ROOT / 'Revizyon3' / 'outputs'

R2C_PRED = REV2_OUT / 'R2C_corrected_sparse_prior_curve' / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet'
R2D_PRED = REV2_OUT / 'R2D_corrected_cross_seed_aggregation' / 'R2D_CROSS_SEED_AGGREGATION_FIX_SEED_AVERAGED_PREDICTIONS.parquet'
R3A_PRED = REV2_OUT / 'R3A_residual_ensemble' / 'R3A_RESIDUAL_ENSEMBLE_SEED_AVERAGED_PREDICTIONS.parquet'

OUT = REV3_OUT / 'Rev3C_SoC_Trajectory_Plots'
FIG = OUT / 'paper_ready_preview'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

for p in [R2C_PRED, R2D_PRED, R3A_PRED]:
    assert p.exists(), f'Missing prediction file: {p}'
    print(p.name, 'OK')
print('Preview output:', FIG)

# %% Code cell 2 (notebook cell 3)
def prepare_pred(df, source_name):
    z = df.copy()

    # Frozen files use two naming conventions:
    # R2C -> model; R2D/R3A -> method.
    if 'model' not in z.columns and 'method' in z.columns:
        z = z.rename(columns={'method': 'model'})

    required = {
        'domain', 'fraction', 'model', 'vehicle_id', 'segment_id',
        'time_value', 'target', 'prediction'
    }
    missing = sorted(required - set(z.columns))
    if missing:
        raise RuntimeError(
            f'{source_name}: missing columns {missing}. '
            f'Available columns: {list(z.columns)}'
        )

    z['domain'] = z['domain'].astype(str)
    z['model'] = z['model'].astype(str)
    z['vehicle_id'] = z['vehicle_id'].astype(str)
    z['segment_id'] = z['segment_id'].astype(str)
    z['fraction'] = z['fraction'].astype(float)

    if 'compound_segment' not in z.columns:
        z['compound_segment'] = z['vehicle_id'] + '||' + z['segment_id'].astype(str)
    else:
        z['compound_segment'] = z['compound_segment'].astype(str)

    if 'elapsed_s' not in z.columns:
        group_cols = ['domain', 'fraction', 'vehicle_id', 'segment_id', 'model']
        z['elapsed_s'] = z.groupby(group_cols)['time_value'].transform(
            lambda x: x.astype(float) - float(x.astype(float).min())
        )
    z['elapsed_s'] = z['elapsed_s'].astype(float)

    return z.sort_values(
        ['domain', 'fraction', 'vehicle_id', 'segment_id', 'model', 'elapsed_s', 'time_value'],
        kind='mergesort'
    ).reset_index(drop=True)

def require_models(df, required, source_name):
    available = set(df['model'].unique())
    missing = sorted(set(required) - available)
    if missing:
        raise RuntimeError(
            f'{source_name}: expected names not found: {missing}. '
            f'Available: {sorted(available)}'
        )

r2c = prepare_pred(pd.read_parquet(R2C_PRED), 'factorization predictions')
r2d = prepare_pred(pd.read_parquet(R2D_PRED), 'literature benchmark predictions')
r3a = prepare_pred(pd.read_parquet(R3A_PRED), 'ensemble predictions')

DIRECT = 'Direct+Prior'
PR = 'PriorResidual+Prior'
OURS = 'PRSoC_CommonInfo_ET_Residual'
HOU = 'Hou2024_Adapted_XGB_Residual'
CHE = 'Che2026_Adapted_RF_Compensation'
GUO = 'Guo2026_PRKL_Inspired_GRU_Residual'
ET = 'ET_Residual'
ENSEMBLE = 'DisagreementGated_ResidualEnsemble'

require_models(r2c, [DIRECT, PR], 'factorization predictions')
require_models(r2d, [OURS, HOU, CHE, GUO], 'literature benchmark predictions')
require_models(r3a, [ET, ENSEMBLE], 'ensemble predictions')

print('Factorization models:', sorted(r2c['model'].unique()))
print('Benchmark methods:', sorted(r2d['model'].unique()))
print('Ensemble methods:', sorted(r3a['model'].unique()))

# %% [markdown] Outcome-independent representative-segment selection

# %% Code cell 3 (notebook cell 5)
DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
MIN_ROWS = 30

def physical_segment_table(df, domain, fraction):
    # Use one row per physical time point so model duplication cannot affect
    # duration or row-count selection.
    z = df[
        (df['domain'] == domain)
        & np.isclose(df['fraction'], float(fraction))
    ].copy()

    if z.empty:
        raise RuntimeError(f'No rows for {domain}, fraction={fraction}')

    phys = (
        z[['vehicle_id','segment_id','compound_segment','elapsed_s','time_value']]
        .drop_duplicates()
    )

    seg = (
        phys.groupby(['vehicle_id','segment_id','compound_segment'], as_index=False)
        .agg(
            n_rows=('time_value', 'size'),
            duration_s=('elapsed_s', 'max'),
        )
    )
    seg = seg[np.isfinite(seg['duration_s'])].copy()
    if seg.empty:
        raise RuntimeError(f'No valid segments for {domain}, fraction={fraction}')
    return seg

def select_median_duration_segment(df, domain, fraction, min_rows=MIN_ROWS):
    seg = physical_segment_table(df, domain, fraction)
    eligible = seg[seg['n_rows'] >= min_rows].copy()
    if eligible.empty:
        eligible = seg.copy()

    median_duration = float(eligible['duration_s'].median())
    eligible['distance_to_median_s'] = np.abs(
        eligible['duration_s'] - median_duration
    )
    chosen = (
        eligible.sort_values(
            ['distance_to_median_s', 'compound_segment'],
            kind='mergesort'
        )
        .iloc[0]
    )
    return chosen, median_duration

# Use the factorization file as the physical-row reference because all
# five domains and both primary supervision conditions are present.
selection_rows = []
for frac in [0.10, 1.00]:
    for domain in DOMAINS:
        chosen, med = select_median_duration_segment(r2c, domain, frac)
        selection_rows.append({
            'domain': domain,
            'fraction': frac,
            'vehicle_id': chosen['vehicle_id'],
            'segment_id': chosen['segment_id'],
            'compound_segment': chosen['compound_segment'],
            'n_rows': int(chosen['n_rows']),
            'duration_min': float(chosen['duration_s']) / 60.0,
            'domain_median_duration_min': med / 60.0,
            'selection_rule': 'closest_to_domain_median_duration; no prediction/error information used',
        })

selection = pd.DataFrame(selection_rows)
selection.to_csv(OUT / 'Trajectory_Figure_Selection_Audit.csv', index=False)
display(selection)

# %% Code cell 4 (notebook cell 6)
def wide_for_segment(df, domain, fraction, compound_segment, models):
    z = df[
        (df['domain'] == domain)
        & np.isclose(df['fraction'], float(fraction))
        & (df['compound_segment'] == compound_segment)
        & df['model'].isin(models)
    ].copy()

    idx = ['elapsed_s', 'time_value', 'target']
    w = (
        z.pivot_table(
            index=idx,
            columns='model',
            values='prediction',
            aggfunc='first'
        )
        .reset_index()
        .sort_values('elapsed_s')
    )
    w = w.dropna(subset=models)
    if w.empty:
        raise RuntimeError(
            f'No common rows for {domain}, fraction={fraction}, '
            f'segment={compound_segment}, models={models}'
        )
    return w

def selected_segment(domain, fraction):
    row = selection[
        (selection['domain'] == domain)
        & np.isclose(selection['fraction'], float(fraction))
    ].iloc[0]
    return row['compound_segment']

def save_figure(fig, stem):
    png = FIG / f'{stem}.png'
    pdf = FIG / f'{stem}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print('Saved:', png)
    print('Saved:', pdf)
    return png, pdf

# %% Code cell 5 (notebook cell 7)
def plot_factorization_all_domains(fraction):
    fig, axes = plt.subplots(
        1, 5,
        figsize=(18, 4.2),
        constrained_layout=True
    )

    for ax, domain in zip(axes, DOMAINS):
        seg = selected_segment(domain, fraction)
        d = wide_for_segment(r2c, domain, fraction, seg, [DIRECT, PR])

        x = d['elapsed_s'].to_numpy(float) / 60.0
        ax.plot(x, d['target'], linewidth=2.1, label='BMS-reported SoC')
        ax.plot(x, d[DIRECT], linewidth=1.5, label='Direct target model')
        ax.plot(x, d[PR], linewidth=1.5, label='Prior–Residual model')

        ax.set_title(domain)
        ax.set_xlabel('Elapsed time (min)')
        ax.grid(alpha=0.20)

    axes[0].set_ylabel('SoC (%)')
    axes[-1].legend(frameon=False, fontsize=8, loc='best')

    if np.isclose(fraction, 0.10):
        stem = 'Trajectory_Factorization_10pct_AllDomains'
        fig.suptitle(
            'Representative test trajectories — 10% training-target supervision',
            fontsize=12
        )
    else:
        stem = 'Trajectory_Factorization_Full_AllDomains'
        fig.suptitle(
            'Representative test trajectories — full training-target supervision',
            fontsize=12
        )

    save_figure(fig, stem)
    plt.show()
    plt.close(fig)

plot_factorization_all_domains(0.10)
plot_factorization_all_domains(1.00)

# %% Code cell 6 (notebook cell 8)
# Literature-adapted benchmark at 10% supervision.
# We use the same outcome-independent representative physical segment selected above.
BENCH_MODELS = [OURS, HOU, CHE, GUO]
BENCH_LABELS = {
    OURS: 'Proposed ExtraTrees residual',
    HOU: 'Hou et al.-adapted XGBoost residual',
    CHE: 'Che et al.-adapted RF compensation',
    GUO: 'Guo et al.-inspired GRU residual',
}

fig, axes = plt.subplots(
    1, 5,
    figsize=(19, 4.3),
    constrained_layout=True
)

for ax, domain in zip(axes, DOMAINS):
    seg = selected_segment(domain, 0.10)
    d = wide_for_segment(r2d, domain, 0.10, seg, BENCH_MODELS)

    x = d['elapsed_s'].to_numpy(float) / 60.0
    ax.plot(x, d['target'], linewidth=2.1, label='BMS-reported SoC')
    for m in BENCH_MODELS:
        ax.plot(x, d[m], linewidth=1.25, label=BENCH_LABELS[m])

    ax.set_title(domain)
    ax.set_xlabel('Elapsed time (min)')
    ax.grid(alpha=0.20)

axes[0].set_ylabel('SoC (%)')
axes[-1].legend(frameon=False, fontsize=7, loc='best')
fig.suptitle(
    'Representative test trajectories — literature-adapted residual benchmark (10% supervision)',
    fontsize=12
)

save_figure(fig, 'Trajectory_LiteratureBenchmark_10pct_AllDomains')
plt.show()
plt.close(fig)

# %% Code cell 7 (notebook cell 9)
# Residual ensemble comparison at 10% supervision.
ENSEMBLE_MODELS = [ET, ENSEMBLE]
ENSEMBLE_LABELS = {
    ET: 'ExtraTrees residual',
    ENSEMBLE: 'Heterogeneous residual ensemble',
}

fig, axes = plt.subplots(
    1, 5,
    figsize=(18, 4.2),
    constrained_layout=True
)

for ax, domain in zip(axes, DOMAINS):
    seg = selected_segment(domain, 0.10)
    d = wide_for_segment(r3a, domain, 0.10, seg, ENSEMBLE_MODELS)

    x = d['elapsed_s'].to_numpy(float) / 60.0
    ax.plot(x, d['target'], linewidth=2.1, label='BMS-reported SoC')
    for m in ENSEMBLE_MODELS:
        ax.plot(x, d[m], linewidth=1.5, label=ENSEMBLE_LABELS[m])

    ax.set_title(domain)
    ax.set_xlabel('Elapsed time (min)')
    ax.grid(alpha=0.20)

axes[0].set_ylabel('SoC (%)')
axes[-1].legend(frameon=False, fontsize=8, loc='best')
fig.suptitle(
    'Representative test trajectories — residual ensemble (10% supervision)',
    fontsize=12
)

save_figure(fig, 'Trajectory_ResidualEnsemble_10pct_AllDomains')
plt.show()
plt.close(fig)

# %% Code cell 8 (notebook cell 10)
# Compact metrics for the selected segments — for inspection only.
# This table is not used to choose the segments.
metric_rows = []

for frac in [0.10, 1.00]:
    for domain in DOMAINS:
        seg = selected_segment(domain, frac)
        d = wide_for_segment(r2c, domain, frac, seg, [DIRECT, PR])

        metric_rows.append({
            'figure_family': 'factorization',
            'domain': domain,
            'fraction': frac,
            'compound_segment': seg,
            'n_rows': len(d),
            'direct_MAE': float(np.mean(np.abs(d[DIRECT] - d['target']))),
            'prior_residual_MAE': float(np.mean(np.abs(d[PR] - d['target']))),
        })

selected_metrics = pd.DataFrame(metric_rows)
selected_metrics.to_csv(OUT / 'Trajectory_Selected_Segment_Metrics_After_Selection.csv', index=False)
display(selected_metrics)

# %% [markdown] Suggested manuscript use
