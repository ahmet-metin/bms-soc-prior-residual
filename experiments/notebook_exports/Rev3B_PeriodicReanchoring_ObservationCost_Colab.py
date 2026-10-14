# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/robustness/Rev3B_PeriodicReanchoring_ObservationCost_Colab.ipynb
# Notebook SHA256: 8f70cf342c07b8e60436089ea61c84143f1e22b728daebfe69723dfb87881ba1
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] Rev3B — Periodic BMS Re-anchoring / Observation-Cost Audit

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
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
REV3_OUT = ROOT / 'Revizyon3' / 'outputs'

R2C_OUT = REV2_OUT / 'R2C_corrected_sparse_prior_curve'
INPUT = R2C_OUT / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet'

OUT = REV3_OUT / 'Rev3B_PeriodicReanchoring_ObservationCost'
FIG_DIR = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = 'REV3B_PERIODIC_REANCHORING_OBSERVATION_COST'
EXPECTED_SOURCE_SHA256 = 'a280f1dd79a2f57a8109cd6d69b395b07a2e54db0e6136bf3628e286af9afb72'

DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
FRACTIONS = [0.10, 1.00]
MODEL = 'PriorResidual+Prior'

# Policy name -> refresh period in seconds. None means no refresh.
POLICIES = {
    'NoRefresh': None,
    '20min': 20 * 60.0,
    '10min': 10 * 60.0,
    '5min': 5 * 60.0,
}
REFRESH_POLICIES = ['20min', '10min', '5min']

BOOT_DRAWS = 10000
CLUSTER_BOOT_DRAWS = 20000
MIN_POPULATION_VEHICLES = 5
RNG_SEED = 73103

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

assert INPUT.exists(), f'Missing corrected R2C predictions: {INPUT}'
input_sha = sha256_file(INPUT)
assert input_sha == EXPECTED_SOURCE_SHA256, (
    'Corrected R2C source hash mismatch',
    input_sha,
    EXPECTED_SOURCE_SHA256,
)

print('Experiment:', EXPERIMENT)
print('Input     :', INPUT)
print('Input SHA :', input_sha)
print('Output    :', OUT)

# %% [markdown] 1. Load the final corrected PriorResidual predictions and fail closed

# %% Code cell 2 (notebook cell 4)
pred = pd.read_parquet(INPUT).copy()

required = {
    'domain', 'fraction', 'model',
    'vehicle_id', 'segment_id', 'compound_segment',
    'time_value', 'elapsed_s', 'target', 'prediction'
}
missing = sorted(required - set(pred.columns))
assert not missing, f'Missing required columns: {missing}'

pred['domain'] = pred['domain'].astype(str)
pred['vehicle_id'] = pred['vehicle_id'].astype(str)
pred['segment_id'] = pred['segment_id'].astype(str)
pred['compound_segment'] = pred['compound_segment'].astype(str)
pred['fraction'] = pred['fraction'].astype(float)
pred['elapsed_s'] = pred['elapsed_s'].astype(float)
pred['target'] = pred['target'].astype(float)
pred['prediction'] = pred['prediction'].astype(float)

pred = pred[
    pred['domain'].isin(DOMAINS)
    & (pred['model'] == MODEL)
    & pred['fraction'].apply(lambda x: any(np.isclose(x, f) for f in FRACTIONS))
].copy()

pred['fraction'] = pred['fraction'].apply(
    lambda x: min(FRACTIONS, key=lambda f: abs(float(x) - f))
)

assert np.isfinite(pred['elapsed_s']).all()
assert (pred['elapsed_s'] >= -1e-9).all()
assert np.isfinite(pred['target']).all()
assert np.isfinite(pred['prediction']).all()

# One seed-averaged final prediction per physical row.
PHYSICAL_KEYS = [
    'domain', 'fraction', 'vehicle_id', 'segment_id',
    'compound_segment', 'time_value', 'elapsed_s'
]
dup = pred.duplicated(PHYSICAL_KEYS)
assert not dup.any(), f'Duplicate physical seed-averaged rows: {int(dup.sum())}'

expected_conditions = len(DOMAINS) * len(FRACTIONS)
actual_conditions = pred[['domain','fraction']].drop_duplicates().shape[0]
assert actual_conditions == expected_conditions, (actual_conditions, expected_conditions)

print('Rows:', len(pred))
print('Conditions:', actual_conditions)
display(
    pred.groupby(['domain','fraction'])
        .agg(
            rows=('prediction','size'),
            segments=('compound_segment','nunique'),
            vehicles=('vehicle_id','nunique'),
            max_elapsed_min=('elapsed_s', lambda x: float(np.max(x)/60.0)),
        )
        .reset_index()
)

# %% [markdown] 2. Freeze the policy contract before outcome inspection

# %% Code cell 3 (notebook cell 6)
preflight = {
    'experiment': EXPERIMENT,
    'status': 'REV3B_PROTOCOL_FROZEN_BEFORE_REFRESH_OUTCOME_INSPECTION',
    'source_experiment': 'R2C_CORRECTED_SPARSE_PRIOR_CURVE',
    'source_sha256': input_sha,
    'target': 'production/vehicle-reported BMS-facing SoC',
    'model': MODEL,
    'domains': DOMAINS,
    'fractions': FRACTIONS,
    'refresh_policies_minutes': [20, 10, 5],
    'refresh_rule': (
        'at first recorded row at or after each scheduled refresh, '
        'offset = target_refresh - raw_prediction_refresh; '
        'apply raw_prediction + latest offset until next refresh'
    ),
    'model_retraining': False,
    'model_retuning': False,
    'structured_prior_recomputed_after_refresh': False,
    'hidden_state_recomputed_after_refresh': False,
    'refresh_rows_excluded_from_primary_metrics': True,
    'matched_unobserved_rows_required': True,
    'primary_estimand': (
        'segment-balanced MAE difference: no-refresh minus refreshed '
        'on identical policy-specific unobserved rows'
    ),
    'primary_family_size': len(DOMAINS) * len(FRACTIONS) * len(REFRESH_POLICIES),
    'primary_bootstrap_unit': 'compound acquisition segment',
    'bootstrap_draws': BOOT_DRAWS,
    'multiplicity': 'Holm across 5 domains x 2 fractions x 3 refresh policies',
    'deployment_refresh_observations_counted_separately_from_training_supervision': True,
    'population_secondary_min_vehicles': MIN_POPULATION_VEHICLES,
    'population_secondary_bootstrap': 'two-stage vehicle -> segment',
    'cluster_bootstrap_draws': CLUSTER_BOOT_DRAWS,
    'EVE_historically_opened': True,
    'confirmatory': False,
    'evaluation_role': 'post-confirmatory deployment observation-cost audit',
}

with open(OUT / f'{EXPERIMENT}_PREFLIGHT.json', 'w') as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% [markdown] 3. Causal refresh scheduler and output-offset re-anchoring

# %% Code cell 4 (notebook cell 8)
def scheduled_refresh_positions(g, period_s):
    """Return integer row positions selected as trusted refresh events.

    Segment-start is the already-required initial anchor and is not counted as a periodic refresh.
    For each k*period_s, choose the first recorded row at or after the scheduled time.
    Duplicate selected rows are collapsed if sampling is sparse.
    """
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


def apply_refresh_policy(g, period_s):
    """Apply causal output-offset refresh inside one physical segment."""
    z = g.sort_values(['elapsed_s', 'time_value'], kind='mergesort').reset_index(drop=True).copy()

    raw = z['prediction'].to_numpy(float)
    target = z['target'].to_numpy(float)
    elapsed = z['elapsed_s'].to_numpy(float)

    refresh_positions = scheduled_refresh_positions(z, period_s)
    refresh_set = set(refresh_positions)

    refreshed = raw.copy()
    correction = 0.0
    last_refresh_elapsed = np.full(len(z), np.nan, dtype=float)
    current_refresh_time = np.nan
    is_refresh = np.zeros(len(z), dtype=bool)

    for i in range(len(z)):
        if i in refresh_set:
            # Trusted BMS observation is accessed only at this designated refresh row.
            correction = float(target[i] - raw[i])
            current_refresh_time = float(elapsed[i])
            is_refresh[i] = True

        refreshed[i] = raw[i] + correction
        last_refresh_elapsed[i] = current_refresh_time

    z['prediction_no_refresh'] = raw
    z['prediction_refreshed'] = refreshed
    z['refresh_event'] = is_refresh
    z['last_refresh_elapsed_s'] = last_refresh_elapsed
    z['time_since_refresh_s'] = np.where(
        np.isfinite(last_refresh_elapsed),
        elapsed - last_refresh_elapsed,
        np.nan,
    )

    # Causality audit.
    m = np.isfinite(z['last_refresh_elapsed_s'].to_numpy(float))
    assert np.all(
        z.loc[m, 'last_refresh_elapsed_s'].to_numpy(float)
        <= z.loc[m, 'elapsed_s'].to_numpy(float) + 1e-9
    )
    assert np.allclose(
        z.loc[z['refresh_event'], 'prediction_refreshed'].to_numpy(float),
        z.loc[z['refresh_event'], 'target'].to_numpy(float),
        rtol=0.0, atol=1e-10,
    )

    return z


policy_frames = []
for (domain, fraction, compound_segment), g in pred.groupby(
    ['domain','fraction','compound_segment'], sort=False
):
    for policy in REFRESH_POLICIES:
        z = apply_refresh_policy(g, POLICIES[policy])
        z['policy'] = policy
        policy_frames.append(z)

policy_rows = pd.concat(policy_frames, ignore_index=True)

print('Policy-expanded rows:', len(policy_rows))
display(
    policy_rows.groupby(['domain','fraction','policy'])
        .agg(
            rows=('prediction','size'),
            refresh_events=('refresh_event','sum'),
            segments=('compound_segment','nunique'),
        )
        .reset_index()
)

# %% [markdown] 4. Observation burden and exact matched-unobserved evaluation

# %% Code cell 5 (notebook cell 10)
# Duration is computed once per physical segment from the seed-averaged corrected test table.
segment_duration = (
    pred.groupby(['domain','fraction','vehicle_id','compound_segment'], as_index=False)
        .agg(
            segment_duration_s=('elapsed_s','max'),
            n_rows_segment=('elapsed_s','size'),
        )
)

duration_summary = (
    segment_duration.groupby(['domain','fraction'], as_index=False)
        .agg(
            total_duration_s=('segment_duration_s','sum'),
            n_segments=('compound_segment','nunique'),
            total_rows=('n_rows_segment','sum'),
        )
)

metric_rows = []
budget_rows = []

for (domain, fraction, policy), g in policy_rows.groupby(
    ['domain','fraction','policy'], sort=False
):
    # Refresh rows are direct observations and MUST NOT enter the primary accuracy estimand.
    u = g[~g['refresh_event']].copy()
    assert len(u) > 0, (domain, fraction, policy)

    u['ae_no_refresh'] = np.abs(
        u['prediction_no_refresh'].to_numpy(float) - u['target'].to_numpy(float)
    )
    u['ae_refreshed'] = np.abs(
        u['prediction_refreshed'].to_numpy(float) - u['target'].to_numpy(float)
    )

    seg = (
        u.groupby(['vehicle_id','compound_segment'], as_index=False)
         .agg(
             MAE_no_refresh=('ae_no_refresh','mean'),
             MAE_refreshed=('ae_refreshed','mean'),
             n_unobserved_rows=('target','size'),
         )
    )
    seg['delta_no_refresh_minus_refreshed'] = (
        seg['MAE_no_refresh'] - seg['MAE_refreshed']
    )
    seg['domain'] = domain
    seg['fraction'] = float(fraction)
    seg['policy'] = policy
    metric_rows.append(seg)

    d = duration_summary[
        (duration_summary['domain'] == domain)
        & np.isclose(duration_summary['fraction'].astype(float), float(fraction))
    ].iloc[0]

    total_hours = float(d['total_duration_s']) / 3600.0
    n_segments = int(d['n_segments'])
    total_rows = int(d['total_rows'])
    n_refresh = int(g['refresh_event'].sum())
    n_unobserved = int((~g['refresh_event']).sum())

    initial_anchor_obs_per_hour = n_segments / total_hours if total_hours > 0 else np.nan
    additional_refresh_obs_per_hour = n_refresh / total_hours if total_hours > 0 else np.nan
    total_obs_per_hour = (n_segments + n_refresh) / total_hours if total_hours > 0 else np.nan

    budget_rows.append({
        'domain': domain,
        'fraction': float(fraction),
        'policy': policy,
        'period_min': float(POLICIES[policy] / 60.0),
        'n_segments': n_segments,
        'total_duration_hours': total_hours,
        'initial_anchor_observations': n_segments,
        'periodic_refresh_observations': n_refresh,
        'initial_anchor_obs_per_hour': initial_anchor_obs_per_hour,
        'additional_refresh_obs_per_hour': additional_refresh_obs_per_hour,
        'total_BMS_observations_per_hour': total_obs_per_hour,
        'total_test_rows': total_rows,
        'unobserved_rows': n_unobserved,
        'unobserved_fraction': n_unobserved / total_rows,
    })

segment_metrics = pd.concat(metric_rows, ignore_index=True)
observation_budget = pd.DataFrame(budget_rows)

# Exact matched-row audit: both predictions come from the same unobserved physical rows.
assert not segment_metrics[['MAE_no_refresh','MAE_refreshed']].isna().any().any()

display(observation_budget)

# %% [markdown] 5. Primary segment-balanced inference

# %% Code cell 6 (notebook cell 12)
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
        return point, np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    boot = x[idx].mean(axis=1)

    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))

    # Positive delta means refresh lowers MAE.
    p_left = (np.sum(boot <= 0.0) + 1.0) / (draws + 1.0)
    p_right = (np.sum(boot >= 0.0) + 1.0) / (draws + 1.0)
    p_two = min(1.0, 2.0 * min(p_left, p_right))
    p_refresh_lower = p_left

    return point, lo, hi, float(p_two), float(p_refresh_lower)


primary_rows = []
for (domain, fraction, policy), g in segment_metrics.groupby(
    ['domain','fraction','policy'], sort=False
):
    seed = (
        RNG_SEED
        + DOMAINS.index(str(domain))*1000
        + int(round(float(fraction)*100))*10
        + REFRESH_POLICIES.index(str(policy))
    )

    point, lo, hi, p_two, p_refresh = paired_bootstrap_mean_delta(
        g['delta_no_refresh_minus_refreshed'].to_numpy(float),
        BOOT_DRAWS,
        seed,
    )

    mae_no = float(g['MAE_no_refresh'].mean())
    mae_ref = float(g['MAE_refreshed'].mean())

    primary_rows.append({
        'domain': domain,
        'fraction': float(fraction),
        'policy': policy,
        'n_segments': int(g['compound_segment'].nunique()),
        'n_vehicles': int(g['vehicle_id'].nunique()),
        'NoRefresh_segment_balanced_MAE': mae_no,
        'Refreshed_segment_balanced_MAE': mae_ref,
        'delta_NoRefresh_minus_Refreshed': point,
        'relative_MAE_improvement_pct': (
            100.0 * point / mae_no if mae_no > 0 else np.nan
        ),
        'CI95_low': lo,
        'CI95_high': hi,
        'two_sided_bootstrap_p': p_two,
        'P_refresh_lower': p_refresh,
    })

primary = pd.DataFrame(primary_rows).sort_values(
    ['domain','fraction','policy']
).reset_index(drop=True)

assert len(primary) == len(DOMAINS) * len(FRACTIONS) * len(REFRESH_POLICIES)
primary['Holm_p'] = holm_adjust(primary['two_sided_bootstrap_p'].to_numpy(float))
primary['Refresh_supported_after_Holm'] = (
    (primary['delta_NoRefresh_minus_Refreshed'] > 0)
    & (primary['Holm_p'] < 0.05)
)
primary['NoRefresh_supported_after_Holm'] = (
    (primary['delta_NoRefresh_minus_Refreshed'] < 0)
    & (primary['Holm_p'] < 0.05)
)
primary['unresolved'] = ~(
    primary['Refresh_supported_after_Holm']
    | primary['NoRefresh_supported_after_Holm']
)

primary = primary.merge(
    observation_budget,
    on=['domain','fraction','policy'],
    how='left',
    validate='one_to_one',
)

display(primary)

# %% [markdown] 6. Population-level secondary sensitivity for domains with at least five test vehicles

# %% Code cell 7 (notebook cell 14)
eligible_domains = (
    pred.groupby('domain')['vehicle_id'].nunique()
        .loc[lambda s: s >= MIN_POPULATION_VEHICLES]
        .index.tolist()
)
print('Population-eligible domains:', eligible_domains)


def two_stage_vehicle_segment_bootstrap(g, draws, seed):
    # g has one row per physical segment.
    vehicles = sorted(g['vehicle_id'].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)

    # Equal-vehicle point estimand.
    vehicle_means = (
        g.groupby('vehicle_id')['delta_no_refresh_minus_refreshed'].mean()
    )
    point = float(vehicle_means.mean())

    boot = np.empty(draws, dtype=float)
    by_vehicle = {
        v: g[g['vehicle_id'].astype(str) == v]['delta_no_refresh_minus_refreshed'].to_numpy(float)
        for v in vehicles
    }

    for b in range(draws):
        sampled_vehicles = rng.choice(vehicles, size=len(vehicles), replace=True)
        per_vehicle = []
        for v in sampled_vehicles:
            x = by_vehicle[str(v)]
            xs = rng.choice(x, size=len(x), replace=True)
            per_vehicle.append(float(np.mean(xs)))
        boot[b] = float(np.mean(per_vehicle))

    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    p_left = (np.sum(boot <= 0.0) + 1.0) / (draws + 1.0)
    p_right = (np.sum(boot >= 0.0) + 1.0) / (draws + 1.0)
    p_two = min(1.0, 2.0 * min(p_left, p_right))

    return point, lo, hi, float(p_two), float(p_left)


cluster_rows = []
for (domain, fraction, policy), g in segment_metrics[
    segment_metrics['domain'].isin(eligible_domains)
].groupby(['domain','fraction','policy'], sort=False):
    seed = (
        RNG_SEED + 50000
        + DOMAINS.index(str(domain))*1000
        + int(round(float(fraction)*100))*10
        + REFRESH_POLICIES.index(str(policy))
    )
    point, lo, hi, p_two, p_refresh = two_stage_vehicle_segment_bootstrap(
        g, CLUSTER_BOOT_DRAWS, seed
    )
    cluster_rows.append({
        'domain': domain,
        'fraction': float(fraction),
        'policy': policy,
        'n_vehicles': int(g['vehicle_id'].nunique()),
        'n_segments': int(g['compound_segment'].nunique()),
        'equal_vehicle_delta_NoRefresh_minus_Refreshed': point,
        'CI95_low': lo,
        'CI95_high': hi,
        'two_sided_bootstrap_p': p_two,
        'P_refresh_lower': p_refresh,
    })

cluster = pd.DataFrame(cluster_rows)
if len(cluster):
    cluster['Holm_p_secondary_family'] = holm_adjust(
        cluster['two_sided_bootstrap_p'].to_numpy(float)
    )
    cluster['Refresh_supported_after_Holm'] = (
        (cluster['equal_vehicle_delta_NoRefresh_minus_Refreshed'] > 0)
        & (cluster['Holm_p_secondary_family'] < 0.05)
    )
    cluster['NoRefresh_supported_after_Holm'] = (
        (cluster['equal_vehicle_delta_NoRefresh_minus_Refreshed'] < 0)
        & (cluster['Holm_p_secondary_family'] < 0.05)
    )
    cluster['unresolved'] = ~(
        cluster['Refresh_supported_after_Holm']
        | cluster['NoRefresh_supported_after_Holm']
    )

display(cluster)

# %% [markdown] 7. Observation-cost summary and manuscript-facing figure

# %% Code cell 8 (notebook cell 16)
decision = primary.copy()

decision['MAE_improvement_per_added_obs_per_hour'] = np.where(
    decision['additional_refresh_obs_per_hour'] > 0,
    decision['delta_NoRefresh_minus_Refreshed']
    / decision['additional_refresh_obs_per_hour'],
    np.nan,
)

display(
    decision[
        [
            'domain','fraction','policy',
            'NoRefresh_segment_balanced_MAE',
            'Refreshed_segment_balanced_MAE',
            'relative_MAE_improvement_pct',
            'additional_refresh_obs_per_hour',
            'total_BMS_observations_per_hour',
            'unobserved_fraction',
            'CI95_low','CI95_high','Holm_p',
            'Refresh_supported_after_Holm',
            'NoRefresh_supported_after_Holm',
            'unresolved',
        ]
    ]
)

# One simple manuscript-facing cost/fidelity plot per supervision endpoint.
for fraction in FRACTIONS:
    q = decision[np.isclose(decision['fraction'], fraction)].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for domain in DOMAINS:
        d = q[q['domain'] == domain].sort_values('additional_refresh_obs_per_hour')
        ax.plot(
            d['additional_refresh_obs_per_hour'],
            d['Refreshed_segment_balanced_MAE'],
            marker='o',
            label=domain,
        )
    ax.set_xlabel('Additional periodic BMS observations per acquisition hour')
    ax.set_ylabel('Segment-balanced MAE (pp)')
    ax.set_title(f'Periodic re-anchoring observation cost — {int(fraction*100)}% training targets')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f'{EXPERIMENT}_CostFidelity_{int(fraction*100)}pct.png',
        dpi=220,
    )
    plt.close(fig)

# %% [markdown] 8. Save complete audit package

# %% Code cell 9 (notebook cell 18)
segment_metrics.to_csv(
    OUT / f'{EXPERIMENT}_SEGMENT_LEVEL_MATCHED_UNOBSERVED.csv',
    index=False,
)
observation_budget.to_csv(
    OUT / f'{EXPERIMENT}_OBSERVATION_BUDGET.csv',
    index=False,
)
primary.to_csv(
    OUT / f'{EXPERIMENT}_PRIMARY_PAIRED_BOOTSTRAP.csv',
    index=False,
)
cluster.to_csv(
    OUT / f'{EXPERIMENT}_CLUSTER_AWARE_SECONDARY.csv',
    index=False,
)
decision.to_csv(
    OUT / f'{EXPERIMENT}_DECISION_TABLE.csv',
    index=False,
)

manifest = {
    'experiment': EXPERIMENT,
    'status': 'REV3B_PERIODIC_REANCHORING_OBSERVATION_COST_COMPLETE',
    'source_experiment': 'R2C_CORRECTED_SPARSE_PRIOR_CURVE',
    'source_file': str(INPUT),
    'source_sha256': input_sha,
    'target': 'production/vehicle-reported BMS-facing SoC',
    'model': MODEL,
    'domains': DOMAINS,
    'fractions': FRACTIONS,
    'refresh_policies_minutes': [20,10,5],
    'retraining_performed': False,
    'retuning_performed': False,
    'refresh_mechanism': 'causal output-offset re-anchoring',
    'full_model_state_recomputed': False,
    'refresh_rows_excluded_from_primary_metrics': True,
    'matched_unobserved_rows_verified': True,
    'deployment_refresh_observations_separate_from_training_supervision': True,
    'primary_tests': int(len(primary)),
    'primary_family_size': int(len(DOMAINS)*len(FRACTIONS)*len(REFRESH_POLICIES)),
    'bootstrap_draws': BOOT_DRAWS,
    'Refresh_supported_after_Holm': int(primary['Refresh_supported_after_Holm'].sum()),
    'NoRefresh_supported_after_Holm': int(primary['NoRefresh_supported_after_Holm'].sum()),
    'unresolved': int(primary['unresolved'].sum()),
    'population_eligible_domains': eligible_domains,
    'population_secondary_tests': int(len(cluster)),
    'cluster_refresh_supported': int(cluster['Refresh_supported_after_Holm'].sum()) if len(cluster) else 0,
    'cluster_no_refresh_supported': int(cluster['NoRefresh_supported_after_Holm'].sum()) if len(cluster) else 0,
    'cluster_unresolved': int(cluster['unresolved'].sum()) if len(cluster) else 0,
    'EVE_historically_opened': True,
    'confirmatory': False,
    'evaluation_role': 'post-confirmatory deployment observation-cost audit',
}

with open(OUT / f'{EXPERIMENT}_MANIFEST.json', 'w') as f:
    json.dump(manifest, f, indent=2)

notes = []
notes.append('# Rev3B Periodic Re-anchoring / Observation-Cost — Manuscript Note')
notes.append('')
notes.append(
    'Primary model: final corrected PriorResidual+Prior predictions from R2C; '
    '10% and 100% training-target endpoints.'
)
notes.append('')
notes.append(
    'Refresh is a causal deployment-side output offset. Refresh rows are excluded '
    'from the primary accuracy analysis, and every refreshed prediction is compared '
    'with the no-refresh prediction on the identical unobserved physical rows.'
)
notes.append('')
notes.append('## Primary decision counts')
notes.append('')
notes.append(f"- Refresh supported after Holm: {int(primary['Refresh_supported_after_Holm'].sum())}/{len(primary)}")
notes.append(f"- No-refresh supported after Holm: {int(primary['NoRefresh_supported_after_Holm'].sum())}/{len(primary)}")
notes.append(f"- Unresolved: {int(primary['unresolved'].sum())}/{len(primary)}")
notes.append('')
notes.append('## Reporting boundary')
notes.append('')
notes.append(
    '- Deployment refresh observations are not counted as training labels.\n'
    '- This experiment quantifies an observation-cost/fidelity trade-off for BMS-facing SoC reconstruction.\n'
    '- It does not claim electrochemical true-SoC estimation, safety certification, or independent external validation.\n'
    '- Output-offset refresh does not recompute the learned model state or structured prior.'
)

(OUT / f'{EXPERIMENT}_MANUSCRIPT_NOTES.md').write_text(
    '\n'.join(notes),
    encoding='utf-8',
)

print(json.dumps(manifest, indent=2))
print('\nSaved outputs:')
for p in sorted(OUT.iterdir()):
    print(' -', p.name)
