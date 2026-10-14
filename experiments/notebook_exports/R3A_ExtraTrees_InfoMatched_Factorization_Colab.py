# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/benchmark_ensemble/R3A_ExtraTrees_InfoMatched_Factorization_Colab.ipynb
# Notebook SHA256: 1d572bfe928ebc7dd038f857a97e866224abb273ef5a994aa2bc50c4213b5127
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] Revizyon3 — ExtraTrees Information-Matched Target-Factorization Robustness

# %% Code cell 1 (notebook cell 2)

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, json, random, time, gc, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
OUTPUTS = ROOT / 'outputs'
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
REV3 = ROOT / 'Revizyon3'

OUT = REV3 / 'outputs' / 'R3A_ET_InfoMatchedFactorization'
PRED_DIR = OUT / 'predictions'
FIG_DIR = OUT / 'figures'
for p in [OUT, PRED_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = 'R3A_ET_INFO_MATCHED_FACTORIZATION'

DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
FRACTIONS = [0.01, 0.10, 1.00]
SEEDS = [42, 43, 44, 45, 46]

DIRECT = 'ET_DirectPlusPrior'
PR = 'ET_PriorResidualPlusPrior'
METHODS = [DIRECT, PR]

TREE_N_ESTIMATORS = 300
TREE_MIN_SAMPLES_LEAF = 2
TREE_MAX_FEATURES = 1.0

BOOTSTRAP_DRAWS = 10000
CLUSTER_BOOTSTRAP_DRAWS = 20000
MIN_POPULATION_VEHICLES = 5

PRIOR_CALIBRATION_VERSION = 'P0_supervision_matched_v1'
EPS = 1e-12

print('Experiment:', EXPERIMENT)
print('Output:', OUT)

# %% Code cell 2 (notebook cell 3)

def find_unique_file(root, filename):
    matches = list(Path(root).rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected exactly one {filename!r} under {root}; found {len(matches)}: {matches}'
        )
    return matches[0]

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def stable_hash_u01(s):
    return int(hashlib.md5(str(s).encode('utf-8')).hexdigest()[:12], 16) / float(16**12)

def fraction_key(f):
    return round(float(f), 8)

def safe_tag(s):
    return str(s).replace('+','Plus').replace('/','_').replace(' ','_')

def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))

def regression_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    e = pred - y
    ae = np.abs(e)
    return {
        'MAE': float(ae.mean()),
        'RMSE': float(np.sqrt(np.mean(e**2))),
        'P95': float(np.percentile(ae, 95)),
        'Bias': float(e.mean()),
    }

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

def exact_sparse_subset(df, fraction, seed):
    if float(fraction) >= 0.999999:
        return df.copy()
    pieces = []
    for _, g in df.groupby('vehicle_id', sort=False):
        k = max(1, int(round(float(fraction) * len(g))))
        keys = (
            g['vehicle_id'].astype(str) + '|' +
            g['segment_id'].astype(str) + '|' +
            g['time_value'].astype(str) + '|' + str(int(seed))
        )
        order = keys.map(stable_hash_u01).sort_values().index
        pieces.append(g.loc[order[:k]].copy())
    return pd.concat(pieces, ignore_index=False)

def identity_string(df):
    z = (
        df['vehicle_id'].astype(str) + '|' +
        df['segment_id'].astype(str) + '|' +
        df['time_value'].astype(str)
    ).sort_values()
    return '\n'.join(z.tolist())

def identity_sha256(df):
    return hashlib.sha256(identity_string(df).encode('utf-8')).hexdigest()

# %% [markdown] 1. Governance inputs

# %% Code cell 3 (notebook cell 5)

CANON = OUTPUTS / 'model_screening_10' / 'canonical'

MANIFEST_11B = find_unique_file(OUTPUTS, '11B_manifest.json')

R2B_OUT = REV2_OUT / 'R2B_effective_supervision_budget'
R2B_MANIFEST = R2B_OUT / 'R2B_EFFECTIVE_SUPERVISION_BUDGET_MANIFEST.json'

R2C_CORR_OUT = REV2_OUT / 'R2C_corrected_sparse_prior_curve'
R2C_PRIOR_AUDIT = (
    R2C_CORR_OUT /
    'R2C_CORRECTED_SPARSE_PRIOR_CURVE_PRIOR_CALIBRATION_BY_CONDITION.csv'
)

P0_AUDIT_DIR = REV2_OUT / 'P0_sparse_prior_calibration_audit_v2'
P0_SELECTOR_AUDIT_FILE = P0_AUDIT_DIR / 'P0_SELECTOR_MANIFEST_SHA256.csv'

for p in [R2B_MANIFEST, R2C_PRIOR_AUDIT, P0_SELECTOR_AUDIT_FILE]:
    assert p.exists(), p

with open(MANIFEST_11B) as f:
    man11b = json.load(f)
with open(R2B_MANIFEST) as f:
    r2b_manifest = json.load(f)

r2c_prior_audit = pd.read_csv(R2C_PRIOR_AUDIT)
p0_selector_audit = pd.read_csv(P0_SELECTOR_AUDIT_FILE)

assert man11b['strict_causality'] is True
assert r2b_manifest['anchor_label_overlap_is_double_counted'] is False
assert r2b_manifest['test_targets_count_as_supervision'] is False
assert set(DOMAINS).issubset(set(r2c_prior_audit['domain'].astype(str)))

EXCLUDED_TOKENS = [str(x).lower() for x in man11b['feature_rule']['excluded_tokens']]

print('Governance inputs: PASS')
print('Excluded feature tokens:', EXCLUDED_TOKENS)

# %% Code cell 4 (notebook cell 6)

# Load the exact canonical data used by the corrected Revizyon2 pipeline.
DOMAIN_DATA = {}
DOMAIN_FEATURES = {}
DOMAIN_META = {}

for domain in ['VED', 'KU', 'SCUT', 'TUM']:
    pq = CANON / f'{domain}_canonical.parquet'
    js = CANON / f'{domain}_metadata.json'
    assert pq.exists(), pq
    assert js.exists(), js

    d = pd.read_parquet(pq).copy()
    with open(js) as f:
        meta = json.load(f)

    raw_features = list(meta['feature_columns'])
    base_features = [
        c for c in raw_features
        if not any(tok in c.lower() for tok in EXCLUDED_TOKENS)
    ]
    assert 'soc_phy' not in base_features
    features = base_features + ['soc_phy']

    required = {
        'vehicle_id','segment_id','time_value','elapsed_s','split',
        'target','soc0','coord','soc_phy', *raw_features
    }
    missing = sorted(required - set(d.columns))
    assert not missing, (domain, missing)

    d = d[d['split'].isin(['train','val','test'])].copy()
    d['vehicle_id'] = d['vehicle_id'].astype(str)
    d['segment_id'] = d['segment_id'].astype(str)
    d['compound_segment'] = d['vehicle_id'] + '||' + d['segment_id']
    d['soc_phy_frozen'] = d['soc_phy'].to_numpy(float)

    DOMAIN_DATA[domain] = d
    DOMAIN_FEATURES[domain] = features
    DOMAIN_META[domain] = meta

# EVE uses the frozen pre-unseal train/validation canonicalization and the one-shot test canonical table.
EVE_TV = OUTPUTS / 'eve_12B_v4_frozen_trainval' / '12B_v4_EVE_canonical_trainval.parquet'
EVE_META = OUTPUTS / 'eve_12B_v4_frozen_trainval' / '12B_v4_EVE_canonical_metadata.json'
EVE_TEST = OUTPUTS / 'eve_12C_one_shot_confirmatory' / '12C_EVE_confirmatory_canonical.parquet'

for p in [EVE_TV, EVE_META, EVE_TEST]:
    assert p.exists(), p

eve_tv = pd.read_parquet(EVE_TV).copy()
eve_test = pd.read_parquet(EVE_TEST).copy()
with open(EVE_META) as f:
    eve_meta = json.load(f)

eve_features = list(eve_meta['feature_columns'])
if 'soc_phy' not in eve_features:
    eve_features = eve_features + ['soc_phy']

for d in [eve_tv, eve_test]:
    required = {'vehicle_id','segment_id','time_value','elapsed_s','target','soc0','coord','soc_phy'}
    missing = sorted(required - set(d.columns))
    assert not missing, ('EVE', missing)
    d['vehicle_id'] = d['vehicle_id'].astype(str)
    d['segment_id'] = d['segment_id'].astype(str)
    d['compound_segment'] = d['vehicle_id'] + '||' + d['segment_id']
    d['soc_phy_frozen'] = d['soc_phy'].to_numpy(float)

eve_tv = eve_tv[eve_tv['split'].isin(['train','val'])].copy()
eve_test['split'] = 'test'

DOMAIN_DATA['EVE'] = pd.concat([eve_tv, eve_test], ignore_index=True)
DOMAIN_FEATURES['EVE'] = eve_features
DOMAIN_META['EVE'] = eve_meta

feature_rows = []
for domain in DOMAINS:
    for c in DOMAIN_FEATURES[domain]:
        feature_rows.append({'domain': domain, 'feature': c})

feature_contract = pd.DataFrame(feature_rows)
feature_contract.to_csv(OUT / f'{EXPERIMENT}_FEATURE_CONTRACT.csv', index=False)

for domain in DOMAINS:
    print(
        domain,
        DOMAIN_DATA[domain]['split'].value_counts().to_dict(),
        'n_features=', len(DOMAIN_FEATURES[domain]),
        'features=', DOMAIN_FEATURES[domain]
    )

# %% [markdown] 2. Exact supervision-matched structured prior

# %% Code cell 5 (notebook cell 8)

GROUP_COLS = {
    'VED': [],
    'KU': ['vehicle_id'],
    'SCUT': ['vehicle_id'],
    'TUM': ['vehicle_id'],
    'EVE': [],
}

CALIBRATION_RULES = {
    'VED':  {'kind':'abs',        'threshold':1e-2},
    'KU':   {'kind':'abs',        'threshold':1e-4},
    'SCUT': {'kind':'abs',        'threshold':1e-4},
    'TUM':  {'kind':'pos',        'threshold':1e-2},
    'EVE':  {'kind':'abs_strict', 'threshold':1e-8},
}

def fit_beta_allowed(g, domain):
    x = g['coord'].to_numpy(float)
    y = g['target'].to_numpy(float) - g['soc0'].to_numpy(float)
    rule = CALIBRATION_RULES[domain]

    finite = np.isfinite(x) & np.isfinite(y)
    if rule['kind'] == 'pos':
        eligible = x >= rule['threshold']
    elif rule['kind'] == 'abs_strict':
        eligible = np.abs(x) > rule['threshold']
    else:
        eligible = np.abs(x) >= rule['threshold']

    m = finite & eligible
    x, y = x[m], y[m]

    if len(x) == 0 or float(np.dot(x, x)) <= EPS:
        return np.nan, int(len(x))

    return float(np.dot(x, y) / np.dot(x, x)), int(len(x))

def normalized_group_key(key):
    if isinstance(key, tuple):
        return '||'.join(map(str, key))
    return str(key)

def fit_condition_prior(domain, fraction, seed):
    train = DOMAIN_DATA[domain][DOMAIN_DATA[domain]['split'] == 'train'].copy()
    selected = exact_sparse_subset(train, fraction, seed).copy()
    selector_sha = identity_sha256(selected)

    # Exact selector identity must match the completed P0 audit.
    q = p0_selector_audit[
        (p0_selector_audit['domain'].astype(str) == str(domain)) &
        np.isclose(p0_selector_audit['fraction'].astype(float), float(fraction)) &
        (p0_selector_audit['seed'].astype(int) == int(seed))
    ]
    assert len(q) == 1, (domain, fraction, seed, len(q))
    assert selector_sha == str(q['selector_sha256'].iloc[0]), (
        'selector hash mismatch', domain, fraction, seed
    )

    group_cols = GROUP_COLS[domain]
    beta_map, n_info_map = {}, {}

    if group_cols:
        for key, g in selected.groupby(group_cols, sort=False):
            k = normalized_group_key(key)
            b, n_info = fit_beta_allowed(g, domain)
            if not np.isfinite(b):
                raise RuntimeError(
                    f'{domain} fraction={fraction} seed={seed} group={k}: '
                    'no informative retained rows'
                )
            beta_map[k] = float(b)
            n_info_map[k] = int(n_info)

        expected_groups = set(train[group_cols[0]].astype(str).unique().tolist())
        assert set(beta_map) == expected_groups
    else:
        b, n_info = fit_beta_allowed(selected, domain)
        if not np.isfinite(b):
            raise RuntimeError(
                f'{domain} fraction={fraction} seed={seed}: no informative retained rows'
            )
        beta_map['__POOLED__'] = float(b)
        n_info_map['__POOLED__'] = int(n_info)

    # Semantic cross-check against completed corrected R2C calibration.
    ref = r2c_prior_audit[
        (r2c_prior_audit['domain'].astype(str) == str(domain)) &
        np.isclose(r2c_prior_audit['fraction'].astype(float), float(fraction)) &
        (r2c_prior_audit['seed'].astype(int) == int(seed))
    ].copy()

    assert len(ref) >= 1, (domain, fraction, seed, 'missing R2C prior audit rows')
    assert ref['prior_contract_sha256'].astype(str).nunique() == 1
    assert ref['selector_sha256'].astype(str).nunique() == 1
    assert selector_sha == str(ref['selector_sha256'].iloc[0])

    ref_groups = set(ref['group'].astype(str).tolist())
    local_groups = set(map(str, beta_map.keys()))
    assert ref_groups == local_groups, (
        'R2C prior group mismatch', domain, fraction, seed,
        sorted(ref_groups), sorted(local_groups)
    )

    beta_ref = {str(r.group): float(r.beta_corrected) for r in ref.itertuples(index=False)}
    ninfo_ref = {
        str(r.group): int(r.n_informative_retained_targets)
        for r in ref.itertuples(index=False)
    }

    for group in sorted(local_groups):
        b_local = float(beta_map[group])
        b_ref = float(beta_ref[group])
        assert np.isclose(b_local, b_ref, rtol=1e-12, atol=1e-12), (
            'R2C beta mismatch', domain, fraction, seed, group, b_local, b_ref
        )
        assert int(n_info_map[group]) == int(ninfo_ref[group])

    return {
        'selector_sha256': selector_sha,
        'beta_map': beta_map,
        'n_info_map': n_info_map,
        'prior_contract_sha256': str(ref['prior_contract_sha256'].iloc[0]),
    }

def apply_condition_prior(df, domain, beta_map):
    z = df.copy()
    group_cols = GROUP_COLS[domain]

    if group_cols:
        keys = z[group_cols[0]].astype(str)
        bvec = keys.map(beta_map)
        if bvec.isna().any():
            raise RuntimeError(
                f'{domain}: missing corrected prior slope for '
                f'{sorted(keys[bvec.isna()].unique())}'
            )
        beta_vec = bvec.to_numpy(float)
    else:
        beta_vec = np.full(
            len(z), float(beta_map['__POOLED__']), dtype=float
        )

    z['soc_phy'] = (
        z['soc0'].to_numpy(float) +
        beta_vec * z['coord'].to_numpy(float)
    )
    z['residual_target'] = (
        z['target'].to_numpy(float) -
        z['soc_phy'].to_numpy(float)
    )

    assert np.isfinite(z['soc_phy'].to_numpy(float)).all()
    assert np.isfinite(z['residual_target'].to_numpy(float)).all()
    return z

PRIOR_CONTRACT_LOOKUP = {}
prior_rows = []

for domain in DOMAINS:
    for fraction in FRACTIONS:
        for seed in SEEDS:
            c = fit_condition_prior(domain, fraction, seed)
            PRIOR_CONTRACT_LOOKUP[
                (domain, fraction_key(fraction), int(seed))
            ] = c

            for group, beta in c['beta_map'].items():
                prior_rows.append({
                    'domain': domain,
                    'fraction': float(fraction),
                    'seed': int(seed),
                    'group': group,
                    'selector_sha256': c['selector_sha256'],
                    'prior_contract_sha256': c['prior_contract_sha256'],
                    'n_informative_retained_targets': int(c['n_info_map'][group]),
                    'beta_corrected': float(beta),
                })

prior_manifest = pd.DataFrame(prior_rows)
prior_manifest.to_csv(
    OUT / f'{EXPERIMENT}_PRIOR_CALIBRATION_BY_CONDITION.csv',
    index=False
)

print('Verified prior contracts:', len(PRIOR_CONTRACT_LOOKUP))

# %% Code cell 6 (notebook cell 9)

class TrainOnlyImputer:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        self.median_ = med.astype(np.float32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.where(
            np.isfinite(X),
            X,
            self.median_[None, :]
        ).astype(np.float32)

def prepare_split_frame(d, split):
    z = d[d['split'] == split].copy()
    z = z.sort_values(
        ['compound_segment', 'time_value'],
        kind='mergesort'
    ).reset_index(drop=True)
    z['seq_pos'] = np.arange(len(z), dtype=np.int64)
    return z

def build_condition_pack(domain, fraction, seed):
    c = PRIOR_CONTRACT_LOOKUP[
        (domain, fraction_key(fraction), int(seed))
    ]

    corrected = apply_condition_prior(
        DOMAIN_DATA[domain],
        domain,
        c['beta_map']
    )

    tr = prepare_split_frame(corrected, 'train')
    va = prepare_split_frame(corrected, 'val')
    te = prepare_split_frame(corrected, 'test')

    selected = exact_sparse_subset(tr, fraction, seed).copy()
    assert identity_sha256(selected) == c['selector_sha256']

    features = DOMAIN_FEATURES[domain]

    pp = TrainOnlyImputer().fit(
        tr[features].to_numpy(float)
    )

    return {
        'train': tr,
        'val': va,
        'test': te,
        'selected': selected,
        'features': features,
        'X_train': pp.transform(tr[features].to_numpy(float)),
        'X_val': pp.transform(va[features].to_numpy(float)),
        'X_test': pp.transform(te[features].to_numpy(float)),
        'selector_sha256': c['selector_sha256'],
        'prior_contract_sha256': c['prior_contract_sha256'],
    }

# Effective supervision budget: union of retained labels and mandatory segment-start anchors.
budget_rows = []

for domain in DOMAINS:
    tr0 = prepare_split_frame(DOMAIN_DATA[domain], 'train')

    anchors = (
        tr0.sort_values(
            ['compound_segment','elapsed_s','time_value'],
            kind='mergesort'
        )
        .drop_duplicates('compound_segment', keep='first')
    )
    anchor_ids = set(anchors['seq_pos'].astype(int))

    for fraction in FRACTIONS:
        for seed in SEEDS:
            sel = exact_sparse_subset(tr0, fraction, seed)
            label_ids = set(sel['seq_pos'].astype(int))
            union = label_ids | anchor_ids

            budget_rows.append({
                'domain': domain,
                'fraction': float(fraction),
                'seed': int(seed),
                'n_train': len(tr0),
                'n_labels': len(label_ids),
                'n_anchors': len(anchor_ids),
                'n_overlap': len(label_ids & anchor_ids),
                'n_unique_fit_observations': len(union),
                'nominal_fraction': len(label_ids) / len(tr0),
                'effective_fraction': len(union) / len(tr0),
            })

budget_seed = pd.DataFrame(budget_rows)
budget_seed.to_csv(
    OUT / f'{EXPERIMENT}_EFFECTIVE_BUDGET_BY_SEED.csv',
    index=False
)

budget_summary = (
    budget_seed
    .groupby(['domain','fraction'], as_index=False)
    .agg(
        nominal_fraction=('nominal_fraction','mean'),
        effective_fraction=('effective_fraction','mean'),
        n_labels=('n_labels','mean'),
        n_anchors=('n_anchors','mean'),
        n_unique_fit_observations=('n_unique_fit_observations','mean')
    )
)

budget_summary.to_csv(
    OUT / f'{EXPERIMENT}_EFFECTIVE_BUDGET_SUMMARY.csv',
    index=False
)

display(budget_summary)

# %% [markdown] 3. Freeze the experiment before model outcomes

# %% Code cell 7 (notebook cell 11)

plan_rows = []
for domain in DOMAINS:
    for fraction in FRACTIONS:
        for seed in SEEDS:
            for method in METHODS:
                plan_rows.append({
                    'domain': domain,
                    'fraction': float(fraction),
                    'seed': int(seed),
                    'method': method,
                    'n_estimators': TREE_N_ESTIMATORS,
                    'min_samples_leaf': TREE_MIN_SAMPLES_LEAF,
                    'max_features': TREE_MAX_FEATURES,
                })

plan = pd.DataFrame(plan_rows)
assert len(plan) == 5 * 3 * 5 * 2
plan.to_csv(
    OUT / f'{EXPERIMENT}_FROZEN_RUN_PLAN.csv',
    index=False
)

preflight = {
    'experiment': EXPERIMENT,
    'status': 'PROTOCOL_FROZEN_BEFORE_OUTCOME_INSPECTION',
    'target': 'production/vehicle-reported BMS-facing SoC',
    'primary_question': (
        'Does Prior-Residual target factorization remain beneficial '
        'under an information-matched ExtraTrees backbone?'
    ),
    'domains': DOMAINS,
    'fractions': FRACTIONS,
    'seeds': SEEDS,
    'methods': METHODS,
    'same_input_information': True,
    'same_structured_prior_input': True,
    'same_sparse_target_identities': True,
    'same_prior_calibration': True,
    'same_model_family': 'ExtraTreesRegressor',
    'same_tree_hyperparameters': True,
    'tree_hyperparameters': {
        'n_estimators': TREE_N_ESTIMATORS,
        'min_samples_leaf': TREE_MIN_SAMPLES_LEAF,
        'max_features': TREE_MAX_FEATURES,
    },
    'only_intended_difference': (
        'Direct BMS-SoC target versus prior-residual target/reconstruction'
    ),
    'primary_estimand': 'segment-balanced MAE after five-seed final-prediction averaging',
    'primary_family_size': 15,
    'primary_bootstrap_unit': 'compound acquisition segment',
    'bootstrap_draws': BOOTSTRAP_DRAWS,
    'multiplicity': 'Holm across 5 domains x 3 fractions',
    'secondary_population_domains': ['SCUT','TUM'],
    'secondary_population_bootstrap': 'two-stage vehicle -> segment',
    'secondary_cluster_draws': CLUSTER_BOOTSTRAP_DRAWS,
    'minimum_population_vehicles': MIN_POPULATION_VEHICLES,
    'EVE_historically_opened': True,
    'confirmatory': False,
    'evaluation_role': 'post-confirmatory cross-backbone target-factorization robustness',
}

with open(OUT / f'{EXPERIMENT}_PREFLIGHT.json', 'w') as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% Code cell 8 (notebook cell 12)

def condition_fingerprint(domain, fraction, seed, method, pack):
    payload = {
        'experiment': EXPERIMENT,
        'domain': domain,
        'fraction': float(fraction),
        'seed': int(seed),
        'method': method,
        'features': list(pack['features']),
        'selector_sha256': pack['selector_sha256'],
        'prior_contract_sha256': pack['prior_contract_sha256'],
        'tree': {
            'n_estimators': TREE_N_ESTIMATORS,
            'min_samples_leaf': TREE_MIN_SAMPLES_LEAF,
            'max_features': TREE_MAX_FEATURES,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()

def build_et(seed):
    return ExtraTreesRegressor(
        n_estimators=TREE_N_ESTIMATORS,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        max_features=TREE_MAX_FEATURES,
        random_state=int(seed),
        n_jobs=-1,
    )

def train_one(domain, fraction, seed, method, pack=None):
    set_all_seeds(seed)

    if pack is None:
        pack = build_condition_pack(domain, fraction, seed)

    tr = pack['train']
    te = pack['test']
    selected = pack['selected']

    idx = selected['seq_pos'].astype(int).to_numpy()
    Xtr = pack['X_train'][idx]
    Xte = pack['X_test']

    if method == DIRECT:
        ytr = selected['target'].to_numpy(float)
    elif method == PR:
        ytr = selected['residual_target'].to_numpy(float)
    else:
        raise ValueError(method)

    model = build_et(seed)

    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw = np.asarray(model.predict(Xte), float)
    infer_s = time.perf_counter() - t1

    if method == DIRECT:
        pred = raw
    else:
        pred = te['soc_phy'].to_numpy(float) + raw

    y = te['target'].to_numpy(float)
    met = regression_metrics(y, pred)

    out = te[
        [
            'vehicle_id','segment_id','compound_segment',
            'time_value','elapsed_s','target','soc_phy'
        ]
    ].copy()

    out['domain'] = domain
    out['fraction'] = float(fraction)
    out['method'] = method
    out['seed'] = int(seed)
    out['prediction'] = pred
    out['abs_error'] = np.abs(pred - y)

    tag = f'{domain}_fraction{fraction}_{safe_tag(method)}_seed{seed}'
    pred_path = PRED_DIR / f'{tag}.parquet'
    out.to_parquet(pred_path, index=False)

    fp = condition_fingerprint(
        domain, fraction, seed, method, pack
    )

    return {
        'domain': domain,
        'fraction': float(fraction),
        'seed': int(seed),
        'method': method,
        'selector_sha256': pack['selector_sha256'],
        'prior_contract_sha256': pack['prior_contract_sha256'],
        'run_contract_fingerprint': fp,
        'n_features': len(pack['features']),
        'n_train_labels': int(len(selected)),
        'n_test': int(len(te)),
        'train_seconds': float(train_s),
        'inference_seconds': float(infer_s),
        'prediction_file': str(pred_path),
        'prediction_sha256': sha256_file(pred_path),
        **met,
    }

# %% [markdown] 4. Run 150 matched fits with restart-safe incremental saving

# %% Code cell 9 (notebook cell 14)

RUNS_INC = OUT / f'{EXPERIMENT}_RUNS_INCREMENTAL.csv'

expected_keys = {
    (str(r.domain), float(r.fraction), int(r.seed), str(r.method))
    for r in plan.itertuples(index=False)
}

completed = set()
valid_rows = []

if RUNS_INC.exists():
    old = pd.read_csv(RUNS_INC)

    for r in old.itertuples(index=False):
        key = (
            str(r.domain),
            float(r.fraction),
            int(r.seed),
            str(r.method)
        )
        if key not in expected_keys:
            continue

        p = Path(str(r.prediction_file))

        try:
            pack = build_condition_pack(
                key[0], key[1], key[2]
            )
            expected_fp = condition_fingerprint(
                key[0], key[1], key[2], key[3], pack
            )

            if (
                p.exists()
                and sha256_file(p) == str(r.prediction_sha256)
                and str(r.selector_sha256) == pack['selector_sha256']
                and str(r.prior_contract_sha256) == pack['prior_contract_sha256']
                and str(r.run_contract_fingerprint) == expected_fp
            ):
                completed.add(key)
                valid_rows.append(r._asdict())
        except Exception:
            pass

    if len(valid_rows) != len(old):
        if valid_rows:
            pd.DataFrame(valid_rows).to_csv(RUNS_INC, index=False)
        else:
            RUNS_INC.unlink(missing_ok=True)

print('Total run slots:', len(plan))
print('Valid completed:', len(completed))
print('Remaining:', len(plan) - len(completed))

domain_order = {d:i for i,d in enumerate(DOMAINS)}

condition_plan = (
    plan[['domain','fraction','seed']]
    .drop_duplicates()
    .assign(_do=lambda x: x['domain'].map(domain_order))
    .sort_values(['_do','fraction','seed'])
    .drop(columns='_do')
)

for c in condition_plan.itertuples(index=False):
    domain = str(c.domain)
    fraction = float(c.fraction)
    seed = int(c.seed)

    pending = [
        m for m in METHODS
        if (domain, fraction, seed, m) not in completed
    ]
    if not pending:
        print('SKIP condition complete:', domain, fraction, seed)
        continue

    # Build once; exact same information/preprocessing/prior for both methods.
    pack = build_condition_pack(domain, fraction, seed)

    for method in pending:
        key = (domain, fraction, seed, method)
        print('\nRUN', key)

        result = train_one(
            domain, fraction, seed, method, pack=pack
        )

        row = pd.DataFrame([result])

        if RUNS_INC.exists():
            prev = pd.read_csv(RUNS_INC)
            prev = pd.concat([prev, row], ignore_index=True)
        else:
            prev = row

        prev = prev.drop_duplicates(
            ['domain','fraction','seed','method'],
            keep='last'
        )
        prev.to_csv(RUNS_INC, index=False)

        completed.add(key)
        print(
            'COMPLETE',
            len(completed), '/', len(plan),
            'MAE=', round(float(result['MAE']), 6)
        )

        gc.collect()

runs = pd.read_csv(RUNS_INC)
runs = runs.drop_duplicates(
    ['domain','fraction','seed','method'],
    keep='last'
)

assert len(runs) == 150, len(runs)
assert runs.groupby(
    ['domain','fraction','method']
)['seed'].nunique().eq(5).all()

runs.to_csv(
    OUT / f'{EXPERIMENT}_RUNS.csv',
    index=False
)

print('All matched ExtraTrees runs complete:', len(runs))

# %% [markdown] 5. Seed-level descriptive summary

# %% Code cell 10 (notebook cell 16)

summary = (
    runs
    .groupby(['domain','fraction','method'], as_index=False)
    .agg(
        MAE_mean=('MAE','mean'),
        MAE_std=('MAE','std'),
        RMSE_mean=('RMSE','mean'),
        RMSE_std=('RMSE','std'),
        P95_mean=('P95','mean'),
        Bias_mean=('Bias','mean'),
        n_train_labels=('n_train_labels','first'),
        n_test=('n_test','first'),
        train_seconds_mean=('train_seconds','mean'),
        inference_seconds_mean=('inference_seconds','mean'),
    )
    .merge(
        budget_summary,
        on=['domain','fraction'],
        how='left',
        validate='many_to_one'
    )
)

summary.to_csv(
    OUT / f'{EXPERIMENT}_RUN_SUMMARY.csv',
    index=False
)

display(
    summary.sort_values(
        ['domain','fraction','MAE_mean']
    )
)

# %% [markdown] 6. Correct five-seed final-prediction averaging

# %% Code cell 11 (notebook cell 18)

parts = []

for r in runs.itertuples(index=False):
    p = Path(str(r.prediction_file))
    assert p.exists(), p
    assert sha256_file(p) == str(r.prediction_sha256)

    z = pd.read_parquet(p).copy()
    parts.append(z)

seed_pred = pd.concat(parts, ignore_index=True)

PHYSICAL_KEYS = [
    'domain','fraction','method',
    'vehicle_id','segment_id','compound_segment',
    'time_value','elapsed_s'
]

coverage = (
    seed_pred
    .groupby(PHYSICAL_KEYS, dropna=False)
    .agg(
        n_seeds=('seed','nunique'),
        target_min=('target','min'),
        target_max=('target','max'),
        soc_phy_min=('soc_phy','min'),
        soc_phy_max=('soc_phy','max'),
    )
    .reset_index()
)

assert coverage['n_seeds'].eq(5).all()
assert np.max(
    np.abs(
        coverage['target_max'].to_numpy(float) -
        coverage['target_min'].to_numpy(float)
    )
) <= 1e-12

coverage['soc_phy_seed_range'] = (
    coverage['soc_phy_max'] - coverage['soc_phy_min']
)

coverage.to_csv(
    OUT / f'{EXPERIMENT}_CROSS_SEED_ROW_COVERAGE_AUDIT.csv',
    index=False
)

# Target is invariant; average final SoC predictions by physical row.
avg_pred = (
    seed_pred
    .groupby(
        PHYSICAL_KEYS,
        as_index=False,
        dropna=False
    )
    .agg(
        target=('target','first'),
        prediction=('prediction','mean'),
        soc_phy_seed_mean=('soc_phy','mean'),
        soc_phy_seed_range=('soc_phy', lambda x: float(np.max(x)-np.min(x))),
    )
)

avg_pred['abs_error'] = np.abs(
    avg_pred['prediction'].to_numpy(float) -
    avg_pred['target'].to_numpy(float)
)

avg_pred.to_parquet(
    OUT / f'{EXPERIMENT}_SEED_AVERAGED_PREDICTIONS.parquet',
    index=False
)

segment_level = (
    avg_pred
    .groupby(
        ['domain','fraction','method','vehicle_id','compound_segment'],
        as_index=False
    )['abs_error'].mean()
    .rename(columns={'abs_error':'segment_MAE'})
)

segment_summary = (
    segment_level
    .groupby(['domain','fraction','method'], as_index=False)
    .agg(
        segment_balanced_MAE=('segment_MAE','mean'),
        n_segments=('compound_segment','nunique'),
        n_vehicles=('vehicle_id','nunique')
    )
)

segment_summary.to_csv(
    OUT / f'{EXPERIMENT}_SEGMENT_BALANCED_SUMMARY.csv',
    index=False
)

print('Seed-averaged physical prediction rows:', len(avg_pred))
display(segment_summary)

# %% [markdown] 7. Primary paired segment-bootstrap family

# %% Code cell 12 (notebook cell 20)

def paired_segment_boot(domain, fraction, draws=BOOTSTRAP_DRAWS, seed=9001):
    q = avg_pred[
        (avg_pred['domain'] == domain) &
        np.isclose(avg_pred['fraction'], fraction) &
        avg_pred['method'].isin([DIRECT, PR])
    ].copy()

    seg = (
        q.groupby(
            ['method','compound_segment'],
            as_index=False
        )['abs_error'].mean()
    )

    d = (
        seg[seg['method'] == DIRECT]
        .set_index('compound_segment')['abs_error']
    )
    p = (
        seg[seg['method'] == PR]
        .set_index('compound_segment')['abs_error']
    )

    common = sorted(set(d.index) & set(p.index))
    assert len(common) > 0
    assert len(common) == len(d) == len(p)

    dv = d.loc[common].to_numpy(float)
    pv = p.loc[common].to_numpy(float)

    point = float(dv.mean() - pv.mean())

    rng = np.random.default_rng(seed)
    n = len(common)
    samples = np.empty(draws, dtype=float)

    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(
            dv[idx].mean() - pv[idx].mean()
        )

    p_left = (np.sum(samples <= 0) + 1) / (draws + 1)
    p_right = (np.sum(samples >= 0) + 1) / (draws + 1)
    p2 = float(min(1.0, 2 * min(p_left, p_right)))

    return {
        'domain': domain,
        'fraction': float(fraction),
        'n_segments': int(n),
        'Direct_segment_balanced_MAE': float(dv.mean()),
        'PR_segment_balanced_MAE': float(pv.mean()),
        'delta_Direct_minus_PR': point,
        'relative_PR_improvement_pct': (
            100.0 * point / float(dv.mean())
            if float(dv.mean()) != 0 else np.nan
        ),
        'CI95_low': float(np.percentile(samples, 2.5)),
        'CI95_high': float(np.percentile(samples, 97.5)),
        'two_sided_bootstrap_p': p2,
        'P_PR_lower': float(np.mean(samples > 0)),
    }

rows = []
k = 0

for domain in DOMAINS:
    for fraction in FRACTIONS:
        rows.append(
            paired_segment_boot(
                domain,
                fraction,
                seed=9001 + 101*k
            )
        )
        k += 1

primary = pd.DataFrame(rows)
assert len(primary) == 15

primary['Holm_p'] = holm_adjust(
    primary['two_sided_bootstrap_p'].to_numpy(float)
)

primary['PR_supported_after_Holm'] = (
    (primary['CI95_low'] > 0) &
    (primary['Holm_p'] < 0.05)
)

primary['Direct_supported_after_Holm'] = (
    (primary['CI95_high'] < 0) &
    (primary['Holm_p'] < 0.05)
)

primary['unresolved'] = ~(
    primary['PR_supported_after_Holm'] |
    primary['Direct_supported_after_Holm']
)

primary.to_csv(
    OUT / f'{EXPERIMENT}_PRIMARY_PAIRED_BOOTSTRAP.csv',
    index=False
)

display(primary)

# %% [markdown] 8. Secondary vehicle-aware population sensitivity

# %% Code cell 13 (notebook cell 22)

def hierarchical_vehicle_segment_boot(
    domain,
    fraction,
    draws=CLUSTER_BOOTSTRAP_DRAWS,
    seed=18001
):
    q = avg_pred[
        (avg_pred['domain'] == domain) &
        np.isclose(avg_pred['fraction'], fraction) &
        avg_pred['method'].isin([DIRECT, PR])
    ].copy()

    seg = (
        q.groupby(
            ['vehicle_id','compound_segment','method'],
            as_index=False
        )['abs_error'].mean()
    )

    wide = seg.pivot(
        index=['vehicle_id','compound_segment'],
        columns='method',
        values='abs_error'
    ).dropna()

    assert DIRECT in wide.columns and PR in wide.columns

    wide['delta_Direct_minus_PR'] = (
        wide[DIRECT] - wide[PR]
    )

    vehicle_ids = sorted(
        wide.index.get_level_values('vehicle_id').unique()
    )
    nveh = len(vehicle_ids)

    if nveh < MIN_POPULATION_VEHICLES:
        return None

    by_vehicle = {}
    for v in vehicle_ids:
        arr = (
            wide.xs(v, level='vehicle_id')
            ['delta_Direct_minus_PR']
            .to_numpy(float)
        )
        assert len(arr) > 0
        by_vehicle[v] = arr

    vehicle_point = np.array(
        [by_vehicle[v].mean() for v in vehicle_ids],
        dtype=float
    )
    point = float(vehicle_point.mean())

    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)

    for b in range(draws):
        chosen = rng.choice(vehicle_ids, size=nveh, replace=True)
        vehicle_draw_means = []

        for v in chosen:
            arr = by_vehicle[v]
            idx = rng.integers(0, len(arr), size=len(arr))
            vehicle_draw_means.append(float(arr[idx].mean()))

        samples[b] = float(np.mean(vehicle_draw_means))

    p_left = (np.sum(samples <= 0) + 1) / (draws + 1)
    p_right = (np.sum(samples >= 0) + 1) / (draws + 1)
    p2 = float(min(1.0, 2 * min(p_left, p_right)))

    return {
        'domain': domain,
        'fraction': float(fraction),
        'n_vehicles': int(nveh),
        'n_segments': int(len(wide)),
        'equal_vehicle_delta_Direct_minus_PR': point,
        'CI95_low': float(np.percentile(samples, 2.5)),
        'CI95_high': float(np.percentile(samples, 97.5)),
        'two_sided_bootstrap_p': p2,
        'P_PR_lower': float(np.mean(samples > 0)),
    }

cluster_rows = []
k = 0

for domain in DOMAINS:
    for fraction in FRACTIONS:
        r = hierarchical_vehicle_segment_boot(
            domain,
            fraction,
            seed=18001 + 101*k
        )
        if r is not None:
            cluster_rows.append(r)
        k += 1

cluster = pd.DataFrame(cluster_rows)

if len(cluster):
    cluster['Holm_p_secondary_family'] = holm_adjust(
        cluster['two_sided_bootstrap_p'].to_numpy(float)
    )

    cluster['PR_supported_after_Holm'] = (
        (cluster['CI95_low'] > 0) &
        (cluster['Holm_p_secondary_family'] < 0.05)
    )

    cluster['Direct_supported_after_Holm'] = (
        (cluster['CI95_high'] < 0) &
        (cluster['Holm_p_secondary_family'] < 0.05)
    )

    cluster['unresolved'] = ~(
        cluster['PR_supported_after_Holm'] |
        cluster['Direct_supported_after_Holm']
    )

cluster.to_csv(
    OUT / f'{EXPERIMENT}_CLUSTER_AWARE_SECONDARY.csv',
    index=False
)

display(cluster)

# %% [markdown] 9. Compact final decision table and integrity manifest

# %% Code cell 14 (notebook cell 24)

decision = (
    primary
    .merge(
        budget_summary[
            ['domain','fraction','nominal_fraction','effective_fraction']
        ],
        on=['domain','fraction'],
        how='left',
        validate='one_to_one'
    )
    .sort_values(['domain','fraction'])
)

decision.to_csv(
    OUT / f'{EXPERIMENT}_DECISION_TABLE.csv',
    index=False
)

n_pr_point = int(
    (decision['delta_Direct_minus_PR'] > 0).sum()
)
n_direct_point = int(
    (decision['delta_Direct_minus_PR'] < 0).sum()
)
n_pr_supported = int(
    decision['PR_supported_after_Holm'].sum()
)
n_direct_supported = int(
    decision['Direct_supported_after_Holm'].sum()
)
n_unresolved = int(
    decision['unresolved'].sum()
)

cluster_pr_supported = (
    int(cluster['PR_supported_after_Holm'].sum())
    if len(cluster) else 0
)
cluster_direct_supported = (
    int(cluster['Direct_supported_after_Holm'].sum())
    if len(cluster) else 0
)
cluster_unresolved = (
    int(cluster['unresolved'].sum())
    if len(cluster) else 0
)

manifest = {
    **preflight,
    'status': 'R3A_ET_INFO_MATCHED_FACTORIZATION_COMPLETE',
    'run_slots_complete': int(len(runs)),
    'primary_tests_complete': int(len(decision)),
    'PR_lower_point_conditions': n_pr_point,
    'Direct_lower_point_conditions': n_direct_point,
    'PR_supported_after_Holm': n_pr_supported,
    'Direct_supported_after_Holm': n_direct_supported,
    'unresolved': n_unresolved,
    'population_eligible_secondary_tests': int(len(cluster)),
    'cluster_PR_supported': cluster_pr_supported,
    'cluster_Direct_supported': cluster_direct_supported,
    'cluster_unresolved': cluster_unresolved,
    'cross_seed_grouping_excludes_soc_phy': True,
    'all_physical_rows_have_five_seeds': True,
    'all_results_reported': True,
}

with open(OUT / f'{EXPERIMENT}_MANIFEST.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
display(decision)

# %% Code cell 15 (notebook cell 25)

# Auto-generated interpretation note for later manuscript integration.
lines = [
    '# R3A ExtraTrees information-matched factorization — manuscript note',
    '',
    'Primary design: same ExtraTrees family, same input telemetry, same corrected '
    'supervision-matched prior, same retained TRAIN target identities, same splits, '
    'same imputation, same tree hyperparameters and same seeds. The only intended '
    'difference is direct BMS-SoC target versus prior-residual target/reconstruction.',
    '',
    f'- Prior–Residual has lower point-estimate segment-balanced MAE in '
    f'{n_pr_point}/15 conditions.',
    f'- Direct has lower point-estimate segment-balanced MAE in '
    f'{n_direct_point}/15 conditions.',
    f'- Prior–Residual is supported after Holm in {n_pr_supported}/15 conditions.',
    f'- Direct is supported after Holm in {n_direct_supported}/15 conditions.',
    f'- Unresolved primary contrasts: {n_unresolved}/15.',
    '',
    'Population-level secondary sensitivity is restricted to domains with >=5 test vehicles.',
    f'- Eligible secondary tests: {len(cluster)}.',
    f'- Prior–Residual supported: {cluster_pr_supported}.',
    f'- Direct supported: {cluster_direct_supported}.',
    f'- Unresolved: {cluster_unresolved}.',
    '',
    'Interpretation rule: use this experiment only as cross-backbone robustness '
    'for target factorization. Do not claim unseen-vehicle generalization, '
    'external validation, or electrochemical true-SoC estimation.'
]

note_path = OUT / f'{EXPERIMENT}_MANUSCRIPT_NOTES.md'
note_path.write_text('\n'.join(lines), encoding='utf-8')

print(note_path.read_text())

# %% [markdown] Stop rule
