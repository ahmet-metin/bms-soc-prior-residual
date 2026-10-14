# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/robustness/Rev3D_SoC_ChangeTracking_Colab.ipynb
# Notebook SHA256: 492bf0e53c55d5b160fa9c67652724487402bcf91f9ed1c72bbb6d3e8e0033b1
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] SoC change-tracking audit — manuscript figure preview

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

PRED = (
    REV2_OUT
    / 'R2C_corrected_sparse_prior_curve'
    / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet'
)

OUT = REV3_OUT / 'Rev3D_SoC_ChangeTracking'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

assert PRED.exists(), f'Missing prediction file: {PRED}'
print('Prediction source:', PRED)
print('Output:', OUT)

DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
FRACTIONS = [0.10, 1.00]
MODELS = ['Direct+Prior', 'PriorResidual+Prior']

HORIZONS_S = {
    '5min': 5 * 60,
    '10min': 10 * 60,
    '20min': 20 * 60,
}

# Realized elapsed interval must stay close to the requested horizon.
TOLERANCE_S = {
    '5min': 60,       # 4–6 min
    '10min': 120,     # 8–12 min
    '20min': 240,     # 16–24 min
}

DIRECTION_THRESHOLD_PP = 0.25
MAX_SCATTER_POINTS_PER_DOMAIN = 5000
N_BINS = 12

# %% Code cell 2 (notebook cell 3)
# ---------- Load and normalize the frozen prediction file ----------

z = pd.read_parquet(PRED).copy()

required = {
    'domain', 'fraction', 'model',
    'vehicle_id', 'segment_id',
    'time_value', 'target', 'prediction'
}
missing = sorted(required - set(z.columns))
if missing:
    raise RuntimeError(
        f'Missing required columns: {missing}. Available: {list(z.columns)}'
    )

z['domain'] = z['domain'].astype(str)
z['model'] = z['model'].astype(str)
z['vehicle_id'] = z['vehicle_id'].astype(str)
z['segment_id'] = z['segment_id'].astype(str)
z['fraction'] = z['fraction'].astype(float)

if 'compound_segment' not in z.columns:
    z['compound_segment'] = z['vehicle_id'] + '||' + z['segment_id']
else:
    z['compound_segment'] = z['compound_segment'].astype(str)

if 'elapsed_s' not in z.columns:
    # Reconstruct elapsed time independently within each physical segment.
    phys_min = (
        z.groupby(
            ['domain','fraction','vehicle_id','segment_id','compound_segment']
        )['time_value']
        .transform('min')
    )
    z['elapsed_s'] = z['time_value'].astype(float) - phys_min.astype(float)

z['elapsed_s'] = z['elapsed_s'].astype(float)

available_models = set(z['model'].unique())
missing_models = sorted(set(MODELS) - available_models)
assert not missing_models, (
    f'Expected models missing: {missing_models}. '
    f'Available: {sorted(available_models)}'
)

z = z[
    z['domain'].isin(DOMAINS)
    & z['model'].isin(MODELS)
    & z['fraction'].apply(lambda x: any(np.isclose(x, f) for f in FRACTIONS))
].copy()

print('Rows:', len(z))
print('Domains:', sorted(z['domain'].unique()))
print('Fractions:', sorted(z['fraction'].unique()))
print('Models:', sorted(z['model'].unique()))

# %% Code cell 3 (notebook cell 4)
# ---------- Build one wide row per physical test observation ----------

key = [
    'domain', 'fraction', 'vehicle_id', 'segment_id',
    'compound_segment', 'elapsed_s', 'time_value'
]

target = (
    z[key + ['target']]
    .drop_duplicates(key)
)

pred = (
    z.pivot_table(
        index=key,
        columns='model',
        values='prediction',
        aggfunc='first'
    )
    .reset_index()
)

wide = target.merge(pred, on=key, how='inner', validate='one_to_one')
wide = wide.dropna(subset=['target'] + MODELS).copy()

wide = wide.sort_values(
    ['domain','fraction','vehicle_id','segment_id','elapsed_s','time_value'],
    kind='mergesort'
).reset_index(drop=True)

print('Physical rows:', len(wide))
display(wide.head())

# %% [markdown] Fixed-horizon physical-row matching

# %% Code cell 4 (notebook cell 6)
def build_horizon_pairs(df, horizon_s, tolerance_s):
    records = []
    n_candidate_starts = 0

    group_cols = [
        'domain', 'fraction', 'vehicle_id',
        'segment_id', 'compound_segment'
    ]

    for keys, g in df.groupby(group_cols, sort=False):
        g = g.sort_values('elapsed_s', kind='mergesort').reset_index(drop=True)
        t = g['elapsed_s'].to_numpy(float)

        if len(g) < 2:
            continue

        for i in range(len(g) - 1):
            desired = t[i] + float(horizon_s)
            n_candidate_starts += 1

            j = int(np.searchsorted(t, desired, side='left'))

            candidates = []
            if j < len(t):
                candidates.append(j)
            if j - 1 > i:
                candidates.append(j - 1)

            if not candidates:
                continue

            # nearest future observation to requested horizon
            jbest = min(
                candidates,
                key=lambda jj: (abs(t[jj] - desired), t[jj])
            )

            realized = float(t[jbest] - t[i])
            if realized <= 0:
                continue
            if abs(realized - float(horizon_s)) > float(tolerance_s):
                continue

            start = g.iloc[i]
            end = g.iloc[jbest]

            rec = {
                'domain': keys[0],
                'fraction': float(keys[1]),
                'vehicle_id': keys[2],
                'segment_id': keys[3],
                'compound_segment': keys[4],
                'start_elapsed_s': float(start['elapsed_s']),
                'end_elapsed_s': float(end['elapsed_s']),
                'realized_horizon_s': realized,
                'target_start': float(start['target']),
                'target_end': float(end['target']),
                'delta_true': float(end['target'] - start['target']),
            }

            for model in MODELS:
                rec[f'{model}_start'] = float(start[model])
                rec[f'{model}_end'] = float(end[model])
                rec[f'delta_{model}'] = float(end[model] - start[model])

            records.append(rec)

    pairs = pd.DataFrame(records)
    return pairs, int(n_candidate_starts)


all_pairs = {}
coverage_rows = []

for hname, hs in HORIZONS_S.items():
    pairs, nstarts = build_horizon_pairs(
        wide,
        horizon_s=hs,
        tolerance_s=TOLERANCE_S[hname]
    )
    all_pairs[hname] = pairs

    for domain in DOMAINS:
        for frac in FRACTIONS:
            sub = pairs[
                (pairs['domain'] == domain)
                & np.isclose(pairs['fraction'], frac)
            ].copy()

            # denominator: possible starts in that domain/fraction
            base = wide[
                (wide['domain'] == domain)
                & np.isclose(wide['fraction'], frac)
            ]
            denom = 0
            for _, gg in base.groupby('compound_segment'):
                denom += max(0, len(gg) - 1)

            coverage_rows.append({
                'horizon': hname,
                'domain': domain,
                'fraction': frac,
                'accepted_pairs': int(len(sub)),
                'candidate_starts': int(denom),
                'pair_coverage_pct': (
                    100.0 * len(sub) / denom if denom else np.nan
                ),
                'n_segments_with_pairs': int(sub['compound_segment'].nunique()) if len(sub) else 0,
                'realized_horizon_median_min': (
                    float(sub['realized_horizon_s'].median()) / 60.0 if len(sub) else np.nan
                ),
                'realized_horizon_q25_min': (
                    float(sub['realized_horizon_s'].quantile(0.25)) / 60.0 if len(sub) else np.nan
                ),
                'realized_horizon_q75_min': (
                    float(sub['realized_horizon_s'].quantile(0.75)) / 60.0 if len(sub) else np.nan
                ),
            })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(OUT / 'SoC_ChangeTracking_CoverageAudit.csv', index=False)

display(coverage)

# %% Code cell 5 (notebook cell 7)
# ---------- Descriptive change-tracking metrics ----------

def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def linear_calibration(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.std(x) <= 1e-12:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)

def segment_balanced_mae(df, pred_col):
    q = df.copy()
    q['ae'] = np.abs(q[pred_col] - q['delta_true'])
    by_seg = q.groupby('compound_segment')['ae'].mean()
    return float(by_seg.mean()) if len(by_seg) else np.nan

metrics_rows = []

for hname, pairs in all_pairs.items():
    for domain in DOMAINS:
        for frac in FRACTIONS:
            sub = pairs[
                (pairs['domain'] == domain)
                & np.isclose(pairs['fraction'], frac)
            ].copy()

            if sub.empty:
                continue

            for model in MODELS:
                pcol = f'delta_{model}'
                err = sub[pcol] - sub['delta_true']

                slope, intercept = linear_calibration(
                    sub['delta_true'], sub[pcol]
                )

                nz = np.abs(sub['delta_true']) >= DIRECTION_THRESHOLD_PP
                if nz.any():
                    direction_agreement = float(
                        np.mean(
                            np.sign(sub.loc[nz, pcol])
                            == np.sign(sub.loc[nz, 'delta_true'])
                        )
                    )
                else:
                    direction_agreement = np.nan

                metrics_rows.append({
                    'horizon': hname,
                    'domain': domain,
                    'fraction': frac,
                    'model': model,
                    'n_pairs': int(len(sub)),
                    'n_segments': int(sub['compound_segment'].nunique()),
                    'delta_MAE_pp': float(np.mean(np.abs(err))),
                    'segment_balanced_delta_MAE_pp': segment_balanced_mae(sub, pcol),
                    'delta_bias_pp': float(np.mean(err)),
                    'pearson_r': safe_corr(sub['delta_true'], sub[pcol]),
                    'calibration_slope': slope,
                    'calibration_intercept_pp': intercept,
                    'direction_threshold_pp': DIRECTION_THRESHOLD_PP,
                    'direction_evaluable_pairs': int(nz.sum()),
                    'direction_agreement_pct': (
                        100.0 * direction_agreement
                        if np.isfinite(direction_agreement)
                        else np.nan
                    ),
                    'realized_horizon_median_min': (
                        float(sub['realized_horizon_s'].median()) / 60.0
                    ),
                })

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv(OUT / 'SoC_ChangeTracking_Metrics.csv', index=False)

display(metrics)

# %% [markdown] Manuscript-ready change-tracking figures

# %% Code cell 6 (notebook cell 9)
def deterministic_subsample(df, max_n):
    if len(df) <= max_n:
        return df.copy()
    # deterministic, evenly spaced rows after stable sorting
    q = df.sort_values(
        ['compound_segment','start_elapsed_s','end_elapsed_s'],
        kind='mergesort'
    ).reset_index(drop=True)
    idx = np.linspace(0, len(q)-1, max_n).round().astype(int)
    return q.iloc[np.unique(idx)].copy()

def binned_median(df, pred_col, n_bins=N_BINS):
    q = df[['delta_true', pred_col]].dropna().copy()
    if len(q) < 20:
        return pd.DataFrame(columns=['x','y','n'])

    # quantile bins avoid empty tails and show calibration over the observed range
    try:
        q['bin'] = pd.qcut(
            q['delta_true'],
            q=min(n_bins, max(3, q['delta_true'].nunique())),
            duplicates='drop'
        )
    except Exception:
        return pd.DataFrame(columns=['x','y','n'])

    out = (
        q.groupby('bin', observed=True)
        .agg(
            x=('delta_true','median'),
            y=(pred_col,'median'),
            n=('delta_true','size')
        )
        .reset_index(drop=True)
    )
    return out

def make_change_tracking_figure(hname, fraction):
    pairs = all_pairs[hname]
    minutes = int(round(HORIZONS_S[hname] / 60))

    fig, axes = plt.subplots(
        1, 5,
        figsize=(18.5, 4.3),
        constrained_layout=True
    )

    for ax, domain in zip(axes, DOMAINS):
        sub = pairs[
            (pairs['domain'] == domain)
            & np.isclose(pairs['fraction'], fraction)
        ].copy()

        if sub.empty:
            ax.text(
                0.5, 0.5, 'Insufficient matched pairs',
                ha='center', va='center', transform=ax.transAxes
            )
            ax.set_title(domain)
            continue

        show = deterministic_subsample(
            sub,
            MAX_SCATTER_POINTS_PER_DOMAIN
        )

        # Symmetric axis range using robust 99th percentile across all displayed quantities.
        vals = np.concatenate([
            np.abs(sub['delta_true'].to_numpy(float)),
            np.abs(sub[f'delta_{MODELS[0]}'].to_numpy(float)),
            np.abs(sub[f'delta_{MODELS[1]}'].to_numpy(float)),
        ])
        lim = float(np.nanquantile(vals[np.isfinite(vals)], 0.99))
        lim = max(0.75, lim * 1.08)

        ax.scatter(
            show['delta_true'],
            show[f'delta_{MODELS[0]}'],
            s=8, alpha=0.12,
            label='Direct target model'
        )
        ax.scatter(
            show['delta_true'],
            show[f'delta_{MODELS[1]}'],
            s=8, alpha=0.12,
            label='Prior–Residual model'
        )

        b0 = binned_median(sub, f'delta_{MODELS[0]}')
        b1 = binned_median(sub, f'delta_{MODELS[1]}')

        if len(b0):
            ax.plot(
                b0['x'], b0['y'],
                marker='o', linewidth=1.8,
                label='Direct binned median'
            )
        if len(b1):
            ax.plot(
                b1['x'], b1['y'],
                marker='o', linewidth=1.8,
                label='Prior–Residual binned median'
            )

        ax.plot(
            [-lim, lim], [-lim, lim],
            linestyle='--', linewidth=1.2,
            label='Identity'
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(domain)
        ax.set_xlabel(f'Observed {minutes}-min SoC change (pp)')
        ax.grid(alpha=0.20)

    axes[0].set_ylabel(f'Reconstructed {minutes}-min SoC change (pp)')
    axes[-1].legend(frameon=False, fontsize=7, loc='best')

    supervision = '10% training-target supervision' if np.isclose(fraction, 0.10) else 'full training-target supervision'
    fig.suptitle(
        f'{minutes}-minute BMS SoC change tracking — {supervision}',
        fontsize=12
    )

    stem = (
        f'SoC_ChangeTracking_{minutes}min_'
        + ('10pct' if np.isclose(fraction, 0.10) else 'Full')
        + '_AllDomains'
    )

    png = FIG / f'{stem}.png'
    pdf = FIG / f'{stem}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print('Saved:', png)
    print('Saved:', pdf)

# Primary manuscript candidates
make_change_tracking_figure('5min', 0.10)
make_change_tracking_figure('5min', 1.00)

# %% Code cell 7 (notebook cell 10)
# ---------- Optional robustness figures at 10 and 20 min ----------

for hname in ['10min', '20min']:
    make_change_tracking_figure(hname, 0.10)

# %% Code cell 8 (notebook cell 11)
# ---------- Compact comparison table for the 5-min primary figure ----------

m5 = metrics[
    (metrics['horizon'] == '5min')
    & np.isclose(metrics['fraction'], 0.10)
].copy()

cols = [
    'domain','model','n_pairs','n_segments',
    'segment_balanced_delta_MAE_pp',
    'pearson_r',
    'calibration_slope',
    'calibration_intercept_pp',
    'direction_agreement_pct',
    'realized_horizon_median_min'
]

primary_table = m5[cols].sort_values(['domain','model'])
primary_table.to_csv(
    OUT / 'SoC_ChangeTracking_5min_10pct_PrimaryTable.csv',
    index=False
)

display(primary_table)

# %% [markdown] Suggested manuscript use
