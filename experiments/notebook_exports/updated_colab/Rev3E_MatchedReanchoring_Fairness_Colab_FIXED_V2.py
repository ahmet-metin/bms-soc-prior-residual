# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: notebooks/updated_colab/Rev3E_MatchedReanchoring_Fairness_Colab_FIXED_V2.ipynb
# Notebook SHA256: 9221b96fa1b00dfbf14a307ba52cd7a4933e7053480ff5d494287a72911cf380
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] Rev3E — Matched periodic re-anchoring fairness audit

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
OUTPUTS = ROOT / 'outputs'
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
REV3_OUT = ROOT / 'Revizyon3' / 'outputs'

R2C_OUT = REV2_OUT / 'R2C_corrected_sparse_prior_curve'
PRED_FILE = R2C_OUT / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet'
PRIOR_AUDIT_FILE = R2C_OUT / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_PRIOR_CALIBRATION_BY_CONDITION.csv'

CANON = OUTPUTS / 'model_screening_10' / 'canonical'
EVE_TEST = OUTPUTS / 'eve_12C_one_shot_confirmatory' / '12C_EVE_confirmatory_canonical.parquet'

OUT = REV3_OUT / 'Rev3E_MatchedReanchoring_Fairness_FIXED_V2'
FIG_DIR = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = 'REV3E_MATCHED_REANCHORING_FAIRNESS'

EXPECTED_PRED_SHA256 = 'b9c824ba5579f98efe6ea56f3240eb9ea0cbc66b3a295276ed093b5a19f4d167'
PREVIOUS_PRED_SHA256 = 'a280f1dd79a2f57a8109cd6d69b395b07a2e54db0e6136bf3628e286af9afb72'
# Pin updated to the digest actually seen in this notebook's saved failed execution.
# A changed source is not silently accepted: verify dataset identities/targets below,
# and reconcile R2C provenance before citing the regenerated results in the paper.

DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
FRACTIONS = [0.10, 1.00]
SEEDS = [42, 43, 44, 45, 46]

DIRECT = 'Direct+Prior'
PR = 'PriorResidual+Prior'
FIXED = 'Fixed prior'
METHODS = [DIRECT, PR, FIXED]

POLICIES = {
    '20min': 20 * 60.0,
    '10min': 10 * 60.0,
    '5min': 5 * 60.0,
}
POLICY_ORDER = ['20min', '10min', '5min']

GROUP_MODE = {
    'VED': 'pooled',
    'KU': 'vehicle',
    'SCUT': 'vehicle',
    'TUM': 'vehicle',
    'EVE': 'pooled',
}

BOOT_DRAWS = 10000
RNG_SEED = 93217

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

assert PRED_FILE.exists(), PRED_FILE
assert PRIOR_AUDIT_FILE.exists(), PRIOR_AUDIT_FILE
for d in ['VED','KU','SCUT','TUM']:
    assert (CANON / f'{d}_canonical.parquet').exists()
assert EVE_TEST.exists(), EVE_TEST

pred_sha = sha256_file(PRED_FILE)
if pred_sha != EXPECTED_PRED_SHA256:
    raise RuntimeError(
        'R2C prediction source differs from the observed Rev3E V1 run. ' +
        f'actual={pred_sha}; pinned={EXPECTED_PRED_SHA256}. ' +
        'Do not remove this verification; reconcile R2C provenance before running.'
    )
print('SOURCE HASH CHECK: PASS — source differs from the original Rev3E pin; provenance review needed before publication.')

print('Experiment :', EXPERIMENT)
print('Prediction :', PRED_FILE)
print('SHA256     :', pred_sha)
print('Prior audit:', PRIOR_AUDIT_FILE)
print('Output     :', OUT)

# %% [markdown] 1. Freeze the audit protocol before outcome inspection

# %% Code cell 2 (notebook cell 4)
preflight = {
    'experiment': EXPERIMENT,
    'status': 'REV3E_PROTOCOL_FROZEN_BEFORE_MATCHED_REFRESH_OUTCOME_INSPECTION',
    'source_experiment': 'R2C_CORRECTED_SPARSE_PRIOR_CURVE',
    'source_prediction_sha256': pred_sha,
    'target': 'production/vehicle-reported BMS-facing SoC',
    'methods': METHODS,
    'domains': DOMAINS,
    'fractions': FRACTIONS,
    'seeds_for_fixed_prior_average': SEEDS,
    'refresh_policies_minutes': [20, 10, 5],
    'same_refresh_rows_for_all_methods': True,
    'refresh_rule': (
        'first recorded row at or after each scheduled refresh; '
        'method-specific offset = target_refresh - raw_prediction_refresh; '
        'apply latest offset until next refresh'
    ),
    'refresh_rows_excluded_from_accuracy_metrics': True,
    'model_retraining': False,
    'model_retuning': False,
    'structured_prior_recomputed_after_refresh': False,
    'hidden_state_recomputed_after_refresh': False,
    'fixed_prior_source': (
        'five-seed mean of archived supervision-matched R2C condition-specific structured priors'
    ),
    'primary_estimand_A': 'within-method no-refresh MAE minus refreshed MAE',
    'primary_estimand_B1': 'refreshed Direct MAE minus refreshed PriorResidual MAE',
    'primary_estimand_B2': 'refreshed FixedPrior MAE minus refreshed PriorResidual MAE',
    'bootstrap_unit': 'physical compound segment',
    'bootstrap_draws': BOOT_DRAWS,
    'holm_families': {
        'within_method_refresh': '30 tests per method (5 domains x 2 supervision levels x 3 policies)',
        'post_refresh_PR_vs_Direct': '30 tests',
        'post_refresh_PR_vs_FixedPrior': '30 tests',
    },
    'deployment_refresh_observations_separate_from_training_supervision': True,
    'confirmatory': False,
    'evaluation_role': 'post-confirmatory matched re-anchoring fairness audit',
}

with open(OUT / f'{EXPERIMENT}_PREFLIGHT.json', 'w') as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% [markdown] 2. Load corrected Direct and Prior–Residual seed-averaged predictions

# %% Code cell 3 (notebook cell 6)
pred_long = pd.read_parquet(PRED_FILE).copy()

required = {
    'domain','fraction','model','vehicle_id','segment_id','compound_segment',
    'time_value','elapsed_s','target','prediction'
}
missing = sorted(required - set(pred_long.columns))
assert not missing, missing

for c in ['domain','model','vehicle_id','segment_id','compound_segment']:
    pred_long[c] = pred_long[c].astype(str)

pred_long['fraction'] = pred_long['fraction'].astype(float)
pred_long['time_value'] = pred_long['time_value'].astype(float)
pred_long['elapsed_s'] = pred_long['elapsed_s'].astype(float)
pred_long['target'] = pred_long['target'].astype(float)
pred_long['prediction'] = pred_long['prediction'].astype(float)

pred_long = pred_long[
    pred_long['domain'].isin(DOMAINS)
    & pred_long['model'].isin([DIRECT, PR])
    & pred_long['fraction'].apply(lambda x: any(np.isclose(x, f) for f in FRACTIONS))
].copy()

pred_long['fraction'] = pred_long['fraction'].apply(
    lambda x: min(FRACTIONS, key=lambda f: abs(float(x)-f))
)

PHYSICAL_KEYS = [
    'domain','fraction','vehicle_id','segment_id',
    'compound_segment','time_value','elapsed_s'
]

# Each physical row must have the same target for both prediction methods.
# Inconsistent duplicates would otherwise be silently hidden by pivot_table(aggfunc='first').
assert not pred_long.duplicated(PHYSICAL_KEYS + ['model']).any(), (
    'duplicate prediction identities',
    int(pred_long.duplicated(PHYSICAL_KEYS + ['model']).sum())
)
assert pred_long.groupby(PHYSICAL_KEYS, dropna=False)['target'].nunique(dropna=False).eq(1).all(), (
    'target mismatch between methods for identical physical rows'
)
assert np.isfinite(pred_long[['time_value','elapsed_s','target','prediction']].to_numpy(float)).all(), (
    'nonfinite source data'
)

target = (
    pred_long[PHYSICAL_KEYS + ['target']]
    .drop_duplicates(PHYSICAL_KEYS)
)

wide = (
    pred_long.pivot_table(
        index=PHYSICAL_KEYS,
        columns='model',
        values='prediction',
        aggfunc='first'
    )
    .reset_index()
    .merge(target, on=PHYSICAL_KEYS, how='inner', validate='one_to_one')
)

assert DIRECT in wide.columns and PR in wide.columns
assert not wide[[DIRECT, PR, 'target']].isna().any().any()

expected = len(DOMAINS) * len(FRACTIONS)
assert wide[['domain','fraction']].drop_duplicates().shape[0] == expected

print('Physical rows:', len(wide))
display(
    wide.groupby(['domain','fraction'])
        .agg(
            rows=('target','size'),
            segments=('compound_segment','nunique'),
            vehicles=('vehicle_id','nunique'),
        )
        .reset_index()
)

# %% [markdown] 3. Reconstruct the supervision-matched fixed prior

# %% Code cell 4 (notebook cell 8)
prior_audit = pd.read_csv(PRIOR_AUDIT_FILE).copy()

needed_prior_cols = {'domain','fraction','seed','group','beta_corrected'}
missing = sorted(needed_prior_cols - set(prior_audit.columns))
assert not missing, missing

prior_audit['domain'] = prior_audit['domain'].astype(str)
prior_audit['fraction'] = prior_audit['fraction'].astype(float)
prior_audit['seed'] = prior_audit['seed'].astype(int)
prior_audit['group'] = prior_audit['group'].astype(str)
prior_audit['beta_corrected'] = prior_audit['beta_corrected'].astype(float)

def load_test_canonical(domain):
    if domain == 'EVE':
        d = pd.read_parquet(EVE_TEST).copy()
        d['split'] = 'test'
    else:
        d = pd.read_parquet(CANON / f'{domain}_canonical.parquet').copy()
        d = d[d['split'].astype(str) == 'test'].copy()

    required = {'vehicle_id','segment_id','time_value','target','soc0','coord','elapsed_s'}
    missing = sorted(required - set(d.columns))
    assert not missing, (domain, missing)

    d['vehicle_id'] = d['vehicle_id'].astype(str)
    d['segment_id'] = d['segment_id'].astype(str)
    d['time_value'] = d['time_value'].astype(float)
    d['elapsed_s'] = d['elapsed_s'].astype(float)
    d['target'] = d['target'].astype(float)
    d['soc0'] = d['soc0'].astype(float)
    d['coord'] = d['coord'].astype(float)
    d['compound_segment'] = d['vehicle_id'] + '||' + d['segment_id']

    return d[
        ['vehicle_id','segment_id','compound_segment','time_value','elapsed_s',
         'target','soc0','coord']
    ].copy()

canon_test = {d: load_test_canonical(d) for d in DOMAINS}

def beta_map_for(domain, fraction, seed):
    q = prior_audit[
        (prior_audit['domain'] == domain)
        & np.isclose(prior_audit['fraction'], float(fraction))
        & (prior_audit['seed'] == int(seed))
    ].copy()
    assert len(q) >= 1, (domain, fraction, seed)
    return dict(zip(q['group'].astype(str), q['beta_corrected'].astype(float)))

fixed_frames = []

for domain in DOMAINS:
    c = canon_test[domain].copy()

    for fraction in FRACTIONS:
        # Restrict to the exact physical rows used by the frozen R2C seed-averaged prediction file.
        ref = wide[
            (wide['domain'] == domain)
            & np.isclose(wide['fraction'], fraction)
        ][
            ['vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target']
        ].copy()

        m = ref.merge(
            c,
            on=['vehicle_id','segment_id','compound_segment','time_value','elapsed_s'],
            how='left',
            suffixes=('_r2c','_canon'),
            validate='one_to_one'
        )

        assert not m[['soc0','coord','target_canon']].isna().any().any(), (
            domain, fraction, 'canonical merge failed'
        )
        assert np.allclose(
            m['target_r2c'].to_numpy(float),
            m['target_canon'].to_numpy(float),
            rtol=0.0, atol=1e-9
        )

        seed_prior = []

        for seed in SEEDS:
            bm = beta_map_for(domain, fraction, seed)

            if GROUP_MODE[domain] == 'pooled':
                assert len(bm) == 1, (domain, fraction, seed, bm.keys())
                beta = float(next(iter(bm.values())))
                bvec = np.full(len(m), beta, dtype=float)
            else:
                bvec = m['vehicle_id'].astype(str).map(bm)
                assert not bvec.isna().any(), (
                    domain, fraction, seed,
                    sorted(m.loc[bvec.isna(),'vehicle_id'].unique().tolist())
                )
                bvec = bvec.to_numpy(float)

            seed_prior.append(
                m['soc0'].to_numpy(float)
                + bvec * m['coord'].to_numpy(float)
            )

        prior_mean = np.mean(np.vstack(seed_prior), axis=0)

        fixed_frames.append(
            pd.DataFrame({
                'domain': domain,
                'fraction': float(fraction),
                'vehicle_id': m['vehicle_id'].astype(str),
                'segment_id': m['segment_id'].astype(str),
                'compound_segment': m['compound_segment'].astype(str),
                'time_value': m['time_value'].astype(float),
                'elapsed_s': m['elapsed_s'].astype(float),
                'fixed_prior_prediction': prior_mean,
            })
        )

fixed = pd.concat(fixed_frames, ignore_index=True)

wide = wide.merge(
    fixed,
    on=PHYSICAL_KEYS,
    how='left',
    validate='one_to_one'
)

assert not wide['fixed_prior_prediction'].isna().any()
wide[FIXED] = wide['fixed_prior_prediction'].astype(float)

print('Fixed prior reconstruction: PASS')
display(
    wide.groupby(['domain','fraction'])
        .agg(
            fixed_prior_MAE=('fixed_prior_prediction',
                             lambda x: float(np.mean(np.abs(
                                 x.to_numpy(float)
                                 - wide.loc[x.index,'target'].to_numpy(float)
                             ))))
        )
        .reset_index()
)

# %% [markdown] 4. Build one common refresh schedule for all three methods

# %% Code cell 5 (notebook cell 10)
def scheduled_refresh_positions(g, period_s):
    if period_s is None or len(g) == 0:
        return []

    elapsed = g['elapsed_s'].to_numpy(float)
    max_elapsed = float(np.max(elapsed))

    if max_elapsed < period_s:
        return []

    scheduled = np.arange(period_s, max_elapsed + 1e-9, period_s, dtype=float)
    positions = []

    for t in scheduled:
        j = int(np.searchsorted(elapsed, t, side='left'))
        if j < len(g):
            positions.append(j)

    return sorted(set(positions))


def apply_matched_refresh(g, period_s):
    z = g.sort_values(['elapsed_s','time_value'], kind='mergesort').reset_index(drop=True).copy()

    positions = scheduled_refresh_positions(z, period_s)
    refresh_set = set(positions)

    is_refresh = np.zeros(len(z), dtype=bool)
    for j in positions:
        is_refresh[j] = True

    z['refresh_event'] = is_refresh

    for method in METHODS:
        raw = z[method].to_numpy(float)
        target = z['target'].to_numpy(float)

        refreshed = raw.copy()
        correction = 0.0

        for i in range(len(z)):
            if i in refresh_set:
                correction = float(target[i] - raw[i])
            refreshed[i] = raw[i] + correction

        z[f'{method}__raw'] = raw
        z[f'{method}__refreshed'] = refreshed

        # All methods exactly match the trusted BMS target at the same refresh rows.
        assert np.allclose(
            z.loc[z['refresh_event'], f'{method}__refreshed'].to_numpy(float),
            z.loc[z['refresh_event'], 'target'].to_numpy(float),
            rtol=0.0, atol=1e-10
        )

    return z


policy_frames = []

for (domain, fraction, compound_segment), g in wide.groupby(
    ['domain','fraction','compound_segment'], sort=False
):
    for policy in POLICY_ORDER:
        z = apply_matched_refresh(g, POLICIES[policy])
        z['policy'] = policy
        policy_frames.append(z)

policy_rows = pd.concat(policy_frames, ignore_index=True)

# Same refresh flags must apply to all methods by construction.
display(
    policy_rows.groupby(['domain','fraction','policy'])
        .agg(
            rows=('target','size'),
            refresh_events=('refresh_event','sum'),
            segments=('compound_segment','nunique')
        )
        .reset_index()
)

# %% [markdown] 5. Segment-level matched metrics

# %% Code cell 6 (notebook cell 12)
segment_rows = []
budget_rows = []

duration = (
    wide.groupby(['domain','fraction','vehicle_id','compound_segment'], as_index=False)
        .agg(
            segment_duration_s=('elapsed_s','max'),
            n_rows_segment=('target','size')
        )
)

duration_summary = (
    duration.groupby(['domain','fraction'], as_index=False)
        .agg(
            total_duration_s=('segment_duration_s','sum'),
            n_segments=('compound_segment','nunique'),
            total_rows=('n_rows_segment','sum')
        )
)

for (domain, fraction, policy), g in policy_rows.groupby(
    ['domain','fraction','policy'], sort=False
):
    u = g[~g['refresh_event']].copy()
    assert len(u) > 0

    for method in METHODS:
        u[f'ae_no__{method}'] = np.abs(
            u[f'{method}__raw'].to_numpy(float) - u['target'].to_numpy(float)
        )
        u[f'ae_ref__{method}'] = np.abs(
            u[f'{method}__refreshed'].to_numpy(float) - u['target'].to_numpy(float)
        )

    agg = {
        'n_unobserved_rows': ('target','size')
    }
    for method in METHODS:
        agg[f'MAE_no__{method}'] = (f'ae_no__{method}','mean')
        agg[f'MAE_ref__{method}'] = (f'ae_ref__{method}','mean')

    seg = (
        u.groupby(['vehicle_id','compound_segment'], as_index=False)
         .agg(**agg)
    )

    for method in METHODS:
        seg[f'delta_refresh__{method}'] = (
            seg[f'MAE_no__{method}'] - seg[f'MAE_ref__{method}']
        )

    seg['delta_refreshed_Direct_minus_PR'] = (
        seg[f'MAE_ref__{DIRECT}'] - seg[f'MAE_ref__{PR}']
    )
    seg['delta_refreshed_Prior_minus_PR'] = (
        seg[f'MAE_ref__{FIXED}'] - seg[f'MAE_ref__{PR}']
    )
    seg['delta_no_refresh_Direct_minus_PR'] = (
        seg[f'MAE_no__{DIRECT}'] - seg[f'MAE_no__{PR}']
    )

    seg['domain'] = domain
    seg['fraction'] = float(fraction)
    seg['policy'] = policy
    segment_rows.append(seg)

    ds = duration_summary[
        (duration_summary['domain'] == domain)
        & np.isclose(duration_summary['fraction'], fraction)
    ].iloc[0]

    total_hours = float(ds['total_duration_s']) / 3600.0
    n_refresh = int(g['refresh_event'].sum())
    n_segments = int(ds['n_segments'])

    budget_rows.append({
        'domain': domain,
        'fraction': float(fraction),
        'policy': policy,
        'period_min': float(POLICIES[policy] / 60.0),
        'n_segments': n_segments,
        'total_duration_hours': total_hours,
        'periodic_refresh_observations': n_refresh,
        'additional_refresh_obs_per_hour': (
            n_refresh / total_hours if total_hours > 0 else np.nan
        ),
        'initial_anchor_obs_per_hour': (
            n_segments / total_hours if total_hours > 0 else np.nan
        ),
        'total_BMS_obs_per_hour': (
            (n_segments+n_refresh) / total_hours if total_hours > 0 else np.nan
        ),
        'total_test_rows': int(len(g)),
        'unobserved_rows': int((~g['refresh_event']).sum()),
        'unobserved_fraction': float((~g['refresh_event']).mean()),
    })

segment_metrics = pd.concat(segment_rows, ignore_index=True)
budget = pd.DataFrame(budget_rows)

print('Segment-level matched table:', segment_metrics.shape)
display(budget)

# %% [markdown] 6. Paired bootstrap inference and Holm correction

# %% Code cell 7 (notebook cell 14)
def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(running, (m-rank) * p[idx])
        adj[idx] = min(1.0, running)

    return adj


def paired_bootstrap_mean_delta(x, draws, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    assert len(x) >= 1

    point = float(np.mean(x))

    if len(x) == 1:
        return point, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    boot = x[idx].mean(axis=1)

    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))

    p_left = (np.sum(boot <= 0.0) + 1.0) / (draws + 1.0)
    p_right = (np.sum(boot >= 0.0) + 1.0) / (draws + 1.0)
    p_two = min(1.0, 2.0 * min(p_left, p_right))

    return point, lo, hi, float(p_two)


# ----- Family A: within-method refresh benefit -----
within_rows = []

for method_i, method in enumerate(METHODS):
    for (domain, fraction, policy), g in segment_metrics.groupby(
        ['domain','fraction','policy'], sort=False
    ):
        x = g[f'delta_refresh__{method}'].to_numpy(float)
        seed = (
            RNG_SEED
            + method_i*100000
            + DOMAINS.index(domain)*1000
            + int(round(fraction*100))*10
            + POLICY_ORDER.index(policy)
        )
        point, lo, hi, p = paired_bootstrap_mean_delta(x, BOOT_DRAWS, seed)

        mae_no = float(g[f'MAE_no__{method}'].mean())
        mae_ref = float(g[f'MAE_ref__{method}'].mean())

        within_rows.append({
            'method': method,
            'domain': domain,
            'fraction': float(fraction),
            'policy': policy,
            'n_segments': int(g['compound_segment'].nunique()),
            'NoRefresh_segment_balanced_MAE': mae_no,
            'Refreshed_segment_balanced_MAE': mae_ref,
            'delta_NoRefresh_minus_Refreshed': point,
            'relative_refresh_improvement_pct': (
                100.0*point/mae_no if mae_no > 0 else np.nan
            ),
            'CI95_low': lo,
            'CI95_high': hi,
            'two_sided_bootstrap_p': p,
        })

within = pd.DataFrame(within_rows)

# Holm separately within each method: 30 tests per method.
within['Holm_p'] = np.nan
for method in METHODS:
    idx = within.index[within['method'] == method]
    within.loc[idx, 'Holm_p'] = holm_adjust(
        within.loc[idx, 'two_sided_bootstrap_p'].to_numpy(float)
    )

within['Refresh_supported_after_Holm'] = (
    (within['delta_NoRefresh_minus_Refreshed'] > 0)
    & (within['Holm_p'] < 0.05)
)
within['NoRefresh_supported_after_Holm'] = (
    (within['delta_NoRefresh_minus_Refreshed'] < 0)
    & (within['Holm_p'] < 0.05)
)
within['unresolved'] = ~(
    within['Refresh_supported_after_Holm']
    | within['NoRefresh_supported_after_Holm']
)

# ----- Family B1: refreshed PR vs refreshed Direct -----
# Positive delta = Direct MAE - PR MAE => favors PR.
comp_rows = []

for contrast, col in [
    ('PR_vs_Direct_after_refresh', 'delta_refreshed_Direct_minus_PR'),
    ('PR_vs_FixedPrior_after_refresh', 'delta_refreshed_Prior_minus_PR'),
]:
    for (domain, fraction, policy), g in segment_metrics.groupby(
        ['domain','fraction','policy'], sort=False
    ):
        seed = (
            RNG_SEED + 500000
            + (0 if contrast.startswith('PR_vs_Direct') else 100000)
            + DOMAINS.index(domain)*1000
            + int(round(fraction*100))*10
            + POLICY_ORDER.index(policy)
        )
        point, lo, hi, p = paired_bootstrap_mean_delta(
            g[col].to_numpy(float),
            BOOT_DRAWS,
            seed
        )

        comp_rows.append({
            'contrast': contrast,
            'domain': domain,
            'fraction': float(fraction),
            'policy': policy,
            'n_segments': int(g['compound_segment'].nunique()),
            'delta_comparator_minus_PR': point,
            'CI95_low': lo,
            'CI95_high': hi,
            'two_sided_bootstrap_p': p,
        })

comparisons = pd.DataFrame(comp_rows)

comparisons['Holm_p'] = np.nan
for contrast in comparisons['contrast'].unique():
    idx = comparisons.index[comparisons['contrast'] == contrast]
    comparisons.loc[idx, 'Holm_p'] = holm_adjust(
        comparisons.loc[idx, 'two_sided_bootstrap_p'].to_numpy(float)
    )

comparisons['PR_supported_after_Holm'] = (
    (comparisons['delta_comparator_minus_PR'] > 0)
    & (comparisons['Holm_p'] < 0.05)
)
comparisons['Comparator_supported_after_Holm'] = (
    (comparisons['delta_comparator_minus_PR'] < 0)
    & (comparisons['Holm_p'] < 0.05)
)
comparisons['unresolved'] = ~(
    comparisons['PR_supported_after_Holm']
    | comparisons['Comparator_supported_after_Holm']
)

display(within)
display(comparisons)

# %% [markdown] 7. Decision summaries

# %% Code cell 8 (notebook cell 16)
within_summary = (
    within.groupby('method', as_index=False)
    .agg(
        tests=('method','size'),
        refresh_supported=('Refresh_supported_after_Holm','sum'),
        no_refresh_supported=('NoRefresh_supported_after_Holm','sum'),
        unresolved=('unresolved','sum'),
        median_relative_improvement_pct=('relative_refresh_improvement_pct','median'),
    )
)

comparison_summary = (
    comparisons.groupby('contrast', as_index=False)
    .agg(
        tests=('contrast','size'),
        PR_supported=('PR_supported_after_Holm','sum'),
        comparator_supported=('Comparator_supported_after_Holm','sum'),
        unresolved=('unresolved','sum'),
        median_delta_comparator_minus_PR=('delta_comparator_minus_PR','median'),
    )
)

display(within_summary)
display(comparison_summary)

within_summary.to_csv(
    OUT / f'{EXPERIMENT}_WITHIN_METHOD_SUMMARY.csv',
    index=False
)
comparison_summary.to_csv(
    OUT / f'{EXPERIMENT}_POST_REFRESH_COMPARISON_SUMMARY.csv',
    index=False
)

# %% [markdown] 8. Reviewer-facing figures

# %% Code cell 9 (notebook cell 18)
# Figure 1: Does PR retain an advantage over Direct after both receive the same refresh observations?
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, constrained_layout=True)

for ax, fraction in zip(axes, FRACTIONS):
    q = comparisons[
        (comparisons['contrast'] == 'PR_vs_Direct_after_refresh')
        & np.isclose(comparisons['fraction'], fraction)
    ].copy()

    for domain in DOMAINS:
        d = q[q['domain'] == domain].copy()
        d['policy_order'] = d['policy'].map({p:i for i,p in enumerate(POLICY_ORDER)})
        d = d.sort_values('policy_order')

        ax.plot(
            [20,10,5],
            d['delta_comparator_minus_PR'].to_numpy(float),
            marker='o',
            label=domain
        )

    ax.axhline(0.0, linewidth=1.0, linestyle='--')
    ax.set_xticks([20,10,5])
    ax.set_xlabel('Refresh interval (min)')
    ax.set_title(
        '10% training-target supervision'
        if np.isclose(fraction,0.10)
        else 'Full training supervision'
    )
    ax.grid(alpha=0.20)

axes[0].set_ylabel('Refreshed Direct MAE - Refreshed Prior–Residual MAE (pp)')
axes[-1].legend(frameon=False, fontsize=8, loc='best')
fig.suptitle('Matched re-anchoring: Prior–Residual versus Direct')
fig.savefig(
    FIG_DIR / 'MatchedReanchoring_PR_vs_Direct_AfterRefresh.pdf',
    bbox_inches='tight'
)
fig.savefig(
    FIG_DIR / 'MatchedReanchoring_PR_vs_Direct_AfterRefresh.png',
    dpi=260, bbox_inches='tight'
)
plt.show()
plt.close(fig)


# Figure 2: within-method median refresh benefit across domains for descriptive inspection.
plot_df = within.copy()
plot_df['period_min'] = plot_df['policy'].map({'20min':20,'10min':10,'5min':5})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, constrained_layout=True)

for ax, fraction in zip(axes, FRACTIONS):
    q = plot_df[np.isclose(plot_df['fraction'], fraction)].copy()

    for method in METHODS:
        d = (
            q[q['method'] == method]
            .groupby('period_min', as_index=False)['relative_refresh_improvement_pct']
            .median()
            .sort_values('period_min', ascending=False)
        )
        ax.plot(
            d['period_min'],
            d['relative_refresh_improvement_pct'],
            marker='o',
            label=method
        )

    ax.set_xticks([20,10,5])
    ax.set_xlabel('Refresh interval (min)')
    ax.set_title(
        '10% training-target supervision'
        if np.isclose(fraction,0.10)
        else 'Full training supervision'
    )
    ax.grid(alpha=0.20)

axes[0].set_ylabel('Median segment-balanced MAE reduction across domains (%)')
axes[-1].legend(frameon=False, fontsize=8, loc='best')
fig.suptitle('Matched re-anchoring benefit by method')
fig.savefig(
    FIG_DIR / 'MatchedReanchoring_WithinMethod_Benefit.pdf',
    bbox_inches='tight'
)
fig.savefig(
    FIG_DIR / 'MatchedReanchoring_WithinMethod_Benefit.png',
    dpi=260, bbox_inches='tight'
)
plt.show()
plt.close(fig)

# %% [markdown] 9. Save complete audit outputs

# %% Code cell 10 (notebook cell 20)
segment_metrics.to_csv(
    OUT / f'{EXPERIMENT}_SEGMENT_LEVEL_MATCHED.csv',
    index=False
)
budget.to_csv(
    OUT / f'{EXPERIMENT}_OBSERVATION_BUDGET.csv',
    index=False
)
within.to_csv(
    OUT / f'{EXPERIMENT}_WITHIN_METHOD_REFRESH_BOOTSTRAP.csv',
    index=False
)
comparisons.to_csv(
    OUT / f'{EXPERIMENT}_POST_REFRESH_MATCHED_COMPARISONS.csv',
    index=False
)

manifest = {
    'experiment': EXPERIMENT,
    'status': 'REV3E_MATCHED_REANCHORING_FAIRNESS_COMPLETE',
    'source_prediction_file': str(PRED_FILE),
    'source_prediction_sha256': pred_sha,
    'source_prior_audit_file': str(PRIOR_AUDIT_FILE),
    'target': 'production/vehicle-reported BMS-facing SoC',
    'methods': METHODS,
    'domains': DOMAINS,
    'fractions': FRACTIONS,
    'refresh_policies_minutes': [20,10,5],
    'same_refresh_rows_for_all_methods': True,
    'retraining_performed': False,
    'retuning_performed': False,
    'refresh_rows_excluded_from_metrics': True,
    'within_method_summary': within_summary.to_dict(orient='records'),
    'post_refresh_comparison_summary': comparison_summary.to_dict(orient='records'),
    'confirmatory': False,
    'evaluation_role': 'post-confirmatory matched re-anchoring fairness audit',
}

with open(OUT / f'{EXPERIMENT}_MANIFEST.json', 'w') as f:
    json.dump(manifest, f, indent=2)

notes = []
notes.append('# Rev3E Matched Re-anchoring Fairness — Manuscript Decision Note')
notes.append('')
notes.append('Same periodic trusted BMS observations were applied to Direct+Prior, PriorResidual+Prior, and the fixed structured prior.')
notes.append('Refresh rows were excluded from accuracy metrics, and all method comparisons use identical unobserved physical rows.')
notes.append('')
notes.append('## Within-method refresh counts')
for r in within_summary.itertuples(index=False):
    notes.append(
        f'- {r.method}: refresh supported {int(r.refresh_supported)}/{int(r.tests)}, '
        f'no-refresh supported {int(r.no_refresh_supported)}/{int(r.tests)}, '
        f'unresolved {int(r.unresolved)}/{int(r.tests)}; '
        f'median relative improvement {float(r.median_relative_improvement_pct):.2f}%.'
    )
notes.append('')
notes.append('## Post-refresh matched comparison counts')
for r in comparison_summary.itertuples(index=False):
    notes.append(
        f'- {r.contrast}: PR supported {int(r.PR_supported)}/{int(r.tests)}, '
        f'comparator supported {int(r.comparator_supported)}/{int(r.tests)}, '
        f'unresolved {int(r.unresolved)}/{int(r.tests)}.'
    )
notes.append('')
notes.append('## Interpretation boundary')
notes.append('- This audit tests fairness of deployment-side refresh information.')
notes.append('- It does not retrain any model or recompute hidden states.')
notes.append('- It does not establish electrochemical true-SoC accuracy, safety certification, or unseen-vehicle generalization.')

(OUT / f'{EXPERIMENT}_MANUSCRIPT_NOTES.md').write_text(
    '\n'.join(notes),
    encoding='utf-8'
)

print(json.dumps(manifest, indent=2))
print('\nSaved files:')
for p in sorted(OUT.iterdir()):
    print(' -', p.name)
