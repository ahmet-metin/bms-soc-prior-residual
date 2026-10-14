# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/benchmark_ensemble/R2D_CorrectedSparsePrior_LiteratureBenchmark_Colab_FIXED_V3.ipynb
# Notebook SHA256: 50e54f5fbb36f699ad9aa9e3123a7d17cb07a1b69c3215805890414bbd124bb0
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R2D-C — Corrected Literature-Adapted Benchmark with Supervision-Matched Prior Calibration

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, json, random, time, gc, contextlib, warnings, subprocess, sys, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost'])
    from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
OUTPUTS = ROOT / 'outputs'
REV1_OUT = ROOT / 'Revizyon1' / 'outputs'
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
CANON = OUTPUTS / 'model_screening_10' / 'canonical'

OLD_R2D_OUT = REV2_OUT / 'R2D_literature_benchmark'
OUT = REV2_OUT / 'R2D_corrected_sparse_prior_benchmark'
PRED_DIR = OUT / 'predictions'
CKPT_DIR = OUT / 'checkpoints'
HISTORY_DIR = OUT / 'history'
FIG_DIR = OUT / 'figures'
QUARANTINE_DIR = CKPT_DIR / 'stale_contract_checkpoints'
for p in [OUT, PRED_DIR, CKPT_DIR, HISTORY_DIR, FIG_DIR, QUARANTINE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = 'R2D_CORRECTED_SPARSE_PRIOR_BENCHMARK'
OLD_EXPERIMENT = 'R2D_LITERATURE_BENCHMARK'

DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
FRACTIONS = [0.10, 1.00]
SEEDS = [42, 43, 44, 45, 46]

OURS = 'PRSoC_CommonInfo_ET_Residual'
HOU = 'Hou2024_Adapted_XGB_Residual'
CHE_COMP = 'Che2026_Adapted_RF_Compensation'
CHE_CORR = 'Che2026_Adapted_RF_Correction'
GUO = 'Guo2026_PRKL_Inspired_GRU_Residual'
COMMON_INFO_METHODS = [OURS, HOU, CHE_COMP, CHE_CORR, GUO]
PRIMARY_LITERATURE = [HOU, CHE_COMP, GUO]

SEQ_LEN = 16
DROPOUT = 0.10
GRU_HIDDEN = 64
GRU_HEAD = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MIN_DELTA = 1e-6
RESUME_SAVE_EVERY = 10
BOOTSTRAP_DRAWS = 10000

TREE_N_ESTIMATORS = 300
TREE_MIN_SAMPLES_LEAF = 2
FULL_PRIOR_REUSE_TOL_PCT = 1e-9
PRIOR_CALIBRATION_VERSION = 'P0_supervision_matched_v1'
EPS = 1e-12

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ENABLED = DEVICE.type == 'cuda'
BATCH_SIZE = 512 if AMP_ENABLED else 128

print('Experiment:', EXPERIMENT)
print('Device:', DEVICE)
print('Output:', OUT)

# %% Code cell 2 (notebook cell 3)
def find_unique_file(root, filename):
    matches = list(Path(root).rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one {filename!r} under {root}; found {len(matches)}: {matches}')
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

def exact_sparse_subset(df, fraction, seed):
    if float(fraction) >= 0.999999:
        return df.copy()
    pieces = []
    for _, g in df.groupby('vehicle_id', sort=False):
        k = max(1, int(round(float(fraction) * len(g))))
        keys = (
            g['vehicle_id'].astype(str) + '|' + g['segment_id'].astype(str) + '|' +
            g['time_value'].astype(str) + '|' + str(seed)
        )
        order = keys.map(stable_hash_u01).sort_values().index
        pieces.append(g.loc[order[:k]].copy())
    return pd.concat(pieces, ignore_index=False)

def identity_string(df):
    z = (
        df['vehicle_id'].astype(str) + '|' + df['segment_id'].astype(str) + '|' +
        df['time_value'].astype(str)
    ).sort_values()
    return '\n'.join(z.tolist())

def identity_sha256(df):
    return hashlib.sha256(identity_string(df).encode('utf-8')).hexdigest()

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

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def safe_tag(s):
    return str(s).replace('+','Plus').replace('/','_').replace(' ','_')

def fraction_key(f):
    return round(float(f), 8)

# %% Code cell 3 (notebook cell 4)
# Governance inputs: corrected R2C calibration contract + original R2D source results.
MANIFEST_11B = find_unique_file(OUTPUTS, '11B_manifest.json')
R2B_OUT = REV2_OUT / 'R2B_effective_supervision_budget'
R2B_MANIFEST = R2B_OUT / 'R2B_EFFECTIVE_SUPERVISION_BUDGET_MANIFEST.json'
assert R2B_MANIFEST.exists(), R2B_MANIFEST
R2C_CORR_OUT = REV2_OUT / 'R2C_corrected_sparse_prior_curve'
R2C_BUDGET = R2C_CORR_OUT / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_FROZEN_DOMAIN_TRAINING_BUDGET.csv'
R2C_PRIOR_AUDIT = R2C_CORR_OUT / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_PRIOR_CALIBRATION_BY_CONDITION.csv'
assert R2C_BUDGET.exists(), R2C_BUDGET
assert R2C_PRIOR_AUDIT.exists(), R2C_PRIOR_AUDIT
P0_AUDIT_DIR = REV2_OUT / 'P0_sparse_prior_calibration_audit_v2'
P0_FULL_GATE = P0_AUDIT_DIR / 'P0_FULL_FRACTION_REPRODUCTION_GATE.csv'
P0_SELECTOR_AUDIT_FILE = P0_AUDIT_DIR / 'P0_SELECTOR_MANIFEST_SHA256.csv'
assert P0_FULL_GATE.exists(), P0_FULL_GATE
assert P0_SELECTOR_AUDIT_FILE.exists(), P0_SELECTOR_AUDIT_FILE
OLD_R2D_RUNS = OLD_R2D_OUT / f'{OLD_EXPERIMENT}_RUNS.csv'

with open(MANIFEST_11B) as f:
    man11b = json.load(f)
with open(R2B_MANIFEST) as f:
    r2b_manifest = json.load(f)

r2c_budget = pd.read_csv(R2C_BUDGET)
r2c_prior_audit = pd.read_csv(R2C_PRIOR_AUDIT)
p0_full_gate = pd.read_csv(P0_FULL_GATE)
p0_selector_audit = pd.read_csv(P0_SELECTOR_AUDIT_FILE)
old_r2d_runs = pd.read_csv(OLD_R2D_RUNS)

assert man11b['strict_causality'] is True
assert int(man11b['sequence_length']) == SEQ_LEN
assert r2b_manifest['anchor_label_overlap_is_double_counted'] is False
assert r2b_manifest['test_targets_count_as_supervision'] is False
assert set(r2c_budget['domain']) == set(DOMAINS)
assert set(p0_full_gate['domain']) == set(DOMAINS)
assert len(old_r2d_runs) == 250

EXCLUDED_TOKENS = [str(x).lower() for x in man11b['feature_rule']['excluded_tokens']]
GRU_BUDGET = {
    r.domain: {'max_epochs': int(r.max_epochs), 'patience': int(r.patience)}
    for r in r2c_budget.itertuples(index=False)
}

full_gate_map = {str(r.domain): float(r.max_rel_diff_pct) for r in p0_full_gate.itertuples(index=False)}
FULL_REUSE_DOMAINS = [d for d in DOMAINS if full_gate_map[d] <= FULL_PRIOR_REUSE_TOL_PCT]
assert set(FULL_REUSE_DOMAINS) == {'VED','KU','SCUT','EVE'}
assert 'TUM' not in FULL_REUSE_DOMAINS

print('Governance inputs: PASS')
print('Full-label reuse domains:', FULL_REUSE_DOMAINS)
display(r2c_budget)
display(p0_full_gate)

# %% Code cell 4 (notebook cell 5)
# Load canonical data. Preserve the old canonical prior only for audit/reuse validation.
DOMAIN_DATA = {}
DOMAIN_FEATURES = {}
DOMAIN_META = {}

for domain in ['VED','KU','SCUT','TUM']:
    pq = CANON / f'{domain}_canonical.parquet'
    js = CANON / f'{domain}_metadata.json'
    assert pq.exists(), pq
    assert js.exists(), js
    d = pd.read_parquet(pq).copy()
    with open(js) as f:
        meta = json.load(f)

    raw_features = list(meta['feature_columns'])
    base_features = [c for c in raw_features if not any(tok in c.lower() for tok in EXCLUDED_TOKENS)]
    assert 'soc_phy' not in base_features
    features = base_features + ['soc_phy']

    required = {'vehicle_id','segment_id','time_value','elapsed_s','split','target','soc0','coord','soc_phy',*raw_features}
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

EVE_TV = OUTPUTS / 'eve_12B_v4_frozen_trainval' / '12B_v4_EVE_canonical_trainval.parquet'
EVE_META = OUTPUTS / 'eve_12B_v4_frozen_trainval' / '12B_v4_EVE_canonical_metadata.json'
EVE_TEST = OUTPUTS / 'eve_12C_one_shot_confirmatory' / '12C_EVE_confirmatory_canonical.parquet'

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
        feature_rows.append({'domain':domain,'feature':c})
pd.DataFrame(feature_rows).to_csv(OUT / f'{EXPERIMENT}_FEATURE_CONTRACT.csv', index=False)

for domain in DOMAINS:
    print(domain, DOMAIN_DATA[domain]['split'].value_counts().to_dict(), 'n_features=', len(DOMAIN_FEATURES[domain]))

# %% Code cell 5 (notebook cell 6)
# Exact corrected prior calibration copied from the completed R2C-C supervision-matched audit.
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

def infer_frozen_beta(g):
    x = g['coord'].to_numpy(float)
    delta = g['soc_phy_frozen'].to_numpy(float) - g['soc0'].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(delta) & (np.abs(x) > 1e-10)
    if not np.any(m):
        return np.nan
    return float(np.median(delta[m] / x[m]))

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
    if len(x) == 0 or float(np.dot(x,x)) <= EPS:
        return np.nan, int(len(x))
    return float(np.dot(x,y) / np.dot(x,x)), int(len(x))

def normalized_group_key(key):
    if isinstance(key, tuple):
        return '||'.join(map(str,key))
    return str(key)

def fit_condition_prior(domain, fraction, seed):
    train = DOMAIN_DATA[domain][DOMAIN_DATA[domain]['split']=='train'].copy()
    selected = exact_sparse_subset(train, fraction, seed).copy()
    selector_sha = identity_sha256(selected)

    q = p0_selector_audit[
        (p0_selector_audit['domain'].astype(str)==str(domain)) &
        np.isclose(p0_selector_audit['fraction'].astype(float), float(fraction)) &
        (p0_selector_audit['seed'].astype(int)==int(seed))
    ]
    assert len(q) == 1, (domain,fraction,seed,len(q))
    assert selector_sha == str(q['selector_sha256'].iloc[0]), ('selector hash mismatch',domain,fraction,seed)

    group_cols = GROUP_COLS[domain]
    beta_map, n_info_map, frozen_map = {}, {}, {}

    if group_cols:
        for key, g in selected.groupby(group_cols, sort=False):
            k = normalized_group_key(key)
            b, n_info = fit_beta_allowed(g, domain)
            if not np.isfinite(b):
                raise RuntimeError(f'{domain} fraction={fraction} seed={seed} group={k}: no informative retained rows')
            beta_map[k] = float(b)
            n_info_map[k] = int(n_info)
        for key, g in train.groupby(group_cols, sort=False):
            frozen_map[normalized_group_key(key)] = infer_frozen_beta(g)
        expected_groups = set(train[group_cols[0]].astype(str).unique().tolist())
        assert set(beta_map) == expected_groups
    else:
        b, n_info = fit_beta_allowed(selected, domain)
        if not np.isfinite(b):
            raise RuntimeError(f'{domain} fraction={fraction} seed={seed}: no informative retained rows')
        beta_map['__POOLED__'] = float(b)
        n_info_map['__POOLED__'] = int(n_info)
        frozen_map['__POOLED__'] = infer_frozen_beta(train)

    payload = {
        'version': PRIOR_CALIBRATION_VERSION,
        'domain': domain,
        'fraction': float(fraction),
        'seed': int(seed),
        'selector_sha256': selector_sha,
        'group_cols': group_cols,
        'calibration_rule': CALIBRATION_RULES[domain],
        'beta_map': {k:float(beta_map[k]) for k in sorted(beta_map)},
    }
    contract_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

    # Semantic cross-check against the completed corrected R2C prior audit.
    # Do not require a freshly serialized JSON payload to reproduce the archived SHA byte-for-byte:
    # tiny floating-point/serialization differences across environments can change the hash even when
    # the scientific prior is identical. Instead, verify the actual contract contents strictly, then
    # adopt the archived R2C contract hash as the canonical identifier.
    ref = r2c_prior_audit[
        (r2c_prior_audit['domain'].astype(str)==str(domain)) &
        np.isclose(r2c_prior_audit['fraction'].astype(float), float(fraction)) &
        (r2c_prior_audit['seed'].astype(int)==int(seed))
    ].copy()
    assert len(ref) >= 1, (domain, fraction, seed, 'missing R2C prior audit rows')
    assert ref['prior_contract_sha256'].astype(str).nunique() == 1
    assert ref['selector_sha256'].astype(str).nunique() == 1
    assert selector_sha == str(ref['selector_sha256'].iloc[0]), ('R2C selector mismatch', domain, fraction, seed)

    ref_groups = set(ref['group'].astype(str).tolist())
    local_groups = set(map(str, beta_map.keys()))
    assert ref_groups == local_groups, ('R2C prior group mismatch', domain, fraction, seed, sorted(ref_groups), sorted(local_groups))

    beta_ref = {str(r.group): float(r.beta_corrected) for r in ref.itertuples(index=False)}
    ninfo_ref = {str(r.group): int(r.n_informative_retained_targets) for r in ref.itertuples(index=False)}
    for group in sorted(local_groups):
        b_local = float(beta_map[group])
        b_ref = float(beta_ref[group])
        if not np.isclose(b_local, b_ref, rtol=1e-12, atol=1e-12):
            raise AssertionError(('R2C beta mismatch', domain, fraction, seed, group, b_local, b_ref, b_local-b_ref))
        assert int(n_info_map[group]) == int(ninfo_ref[group]), (
            'R2C informative-row mismatch', domain, fraction, seed, group, int(n_info_map[group]), int(ninfo_ref[group])
        )

    local_contract_sha = contract_sha
    contract_sha = str(ref['prior_contract_sha256'].iloc[0])

    return {
        'selector_sha256': selector_sha,
        'beta_map': beta_map,
        'n_info_map': n_info_map,
        'frozen_map': frozen_map,
        'prior_contract_sha256': contract_sha,
        'local_recomputed_contract_sha256': local_contract_sha,
        'payload': payload,
    }

def apply_condition_prior(df, domain, beta_map):
    z = df.copy()
    group_cols = GROUP_COLS[domain]
    if group_cols:
        keys = z[group_cols[0]].astype(str)
        bvec = keys.map(beta_map)
        if bvec.isna().any():
            raise RuntimeError(f'{domain}: missing corrected prior slope for {sorted(keys[bvec.isna()].unique())}')
        beta_vec = bvec.to_numpy(float)
    else:
        beta_vec = np.full(len(z), float(beta_map['__POOLED__']), dtype=float)
    z['soc_phy'] = z['soc0'].to_numpy(float) + beta_vec * z['coord'].to_numpy(float)
    z['residual_target'] = z['target'].to_numpy(float) - z['soc_phy'].to_numpy(float)
    assert np.isfinite(z['soc_phy'].to_numpy(float)).all()
    assert np.isfinite(z['residual_target'].to_numpy(float)).all()
    return z

PRIOR_CONTRACT_LOOKUP = {}
prior_rows = []
for domain in DOMAINS:
    for fraction in FRACTIONS:
        for seed in SEEDS:
            c = fit_condition_prior(domain,fraction,seed)
            PRIOR_CONTRACT_LOOKUP[(domain,fraction_key(fraction),int(seed))] = c
            for group,beta in c['beta_map'].items():
                frozen = float(c['frozen_map'].get(group,np.nan))
                prior_rows.append({
                    'domain':domain,'fraction':float(fraction),'seed':int(seed),'group':group,
                    'selector_sha256':c['selector_sha256'],'prior_contract_sha256':c['prior_contract_sha256'],
                    'n_informative_retained_targets':int(c['n_info_map'][group]),
                    'beta_corrected':float(beta),'beta_frozen':frozen,
                    'beta_diff':float(beta-frozen) if np.isfinite(frozen) else np.nan,
                })

prior_manifest = pd.DataFrame(prior_rows)
prior_manifest.to_csv(OUT / f'{EXPERIMENT}_PRIOR_CALIBRATION_BY_CONDITION.csv', index=False)
print('Corrected prior contracts:', len(PRIOR_CONTRACT_LOOKUP), 'conditions')

# %% Code cell 6 (notebook cell 7)
# Corrected frozen run plan: 150 new fits + 100 verified full-label reuses.
plan_rows = []
for domain in DOMAINS:
    for fraction in FRACTIONS:
        for method in COMMON_INFO_METHODS:
            for seed in SEEDS:
                reuse = bool(np.isclose(fraction,1.0) and domain in FULL_REUSE_DOMAINS)
                plan_rows.append({
                    'domain':domain,'fraction':float(fraction),'method':method,'seed':int(seed),
                    'action':'reuse_verified_old_R2D_full' if reuse else 'train_corrected_R2D',
                    'reason':'full_prior_reproduced_and_protocol_unchanged' if reuse else (
                        'sparse_prior_must_be_refit' if fraction < 1 else 'TUM_full_prior_not_reproduced'
                    )
                })
plan = pd.DataFrame(plan_rows)
assert len(plan) == 250
assert int((plan['action']=='train_corrected_R2D').sum()) == 150
assert int((plan['action']=='reuse_verified_old_R2D_full').sum()) == 100
plan.to_csv(OUT / f'{EXPERIMENT}_FROZEN_RUN_PLAN.csv', index=False)

preflight = {
    'experiment':EXPERIMENT,
    'status':'R2D_CORRECTED_PROTOCOL_FROZEN_BEFORE_CORRECTED_OUTCOME_INSPECTION',
    'target':'vehicle-reported/BMS-facing SoC',
    'domains':DOMAINS,'fractions':FRACTIONS,'seeds':SEEDS,
    'primary_literature_comparators':PRIMARY_LITERATURE,
    'secondary_comparator':CHE_CORR,
    'ours_common_information_comparator':OURS,
    'prior_calibration_version':PRIOR_CALIBRATION_VERSION,
    'prior_correction':'target-dependent slope refit per domain/fraction/seed from retained TRAIN targets only',
    'validation_targets_used_for_prior_fit':False,
    'test_targets_used_for_prior_fit':False,
    'sparse_old_R2D_reuse_allowed':False,
    'full_reuse_domains':FULL_REUSE_DOMAINS,
    'TUM_full_reuse_allowed':False,
    'total_run_slots':250,
    'new_corrected_training_runs':150,
    'verified_full_reuse_runs':100,
    'bootstrap_draws':BOOTSTRAP_DRAWS,
    'primary_statistical_family_size':30,
    'primary_contrast':'literature comparator segment-balanced MAE minus OURS common-info segment-balanced MAE',
    'positive_delta_means_ours_lower_error':True,
    'holm_family':'3 primary literature methods x 5 domains x 2 fractions',
    'Che_direct_correction_is_secondary':True,
    'EVE_historically_opened':True,
    'confirmatory':False,
    'evaluation_role':'corrective post-confirmatory literature-adapted benchmark',
    'not_exact_paper_reproduction':True,
}
with open(OUT / f'{EXPERIMENT}_PREFLIGHT.json','w') as f:
    json.dump(preflight,f,indent=2)
print(json.dumps(preflight,indent=2))

# %% Code cell 7 (notebook cell 8)
class TrainOnlyImputerScaler:
    def fit(self, X):
        X = np.asarray(X,dtype=np.float64)
        med = np.nanmedian(X,axis=0)
        med = np.where(np.isfinite(med),med,0.0)
        Xi = np.where(np.isfinite(X),X,med[None,:])
        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)
        std = np.where(np.isfinite(std) & (std>1e-8),std,1.0)
        self.median_ = med.astype(np.float32)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self
    def impute(self,X):
        X = np.asarray(X,dtype=np.float64)
        return np.where(np.isfinite(X),X,self.median_[None,:]).astype(np.float32)
    def transform(self,X):
        Xi = self.impute(X)
        return ((Xi-self.mean_[None,:])/self.std_[None,:]).astype(np.float32)

class TargetScaler:
    def fit(self,y):
        y=np.asarray(y,dtype=np.float64)
        self.mean_=float(np.mean(y)); self.std_=float(np.std(y))
        if not np.isfinite(self.std_) or self.std_<1e-8: self.std_=1.0
        return self
    def transform(self,y):
        return ((np.asarray(y,float)-self.mean_)/self.std_).astype(np.float32)
    def inverse(self,y):
        return np.asarray(y,float)*self.std_+self.mean_

def prepare_split_frame(d,split):
    z=d[d['split']==split].copy()
    z=z.sort_values(['compound_segment','time_value'],kind='mergesort').reset_index(drop=True)
    z['seq_pos']=np.arange(len(z),dtype=np.int64)
    return z

def build_condition_pack(domain,fraction,seed):
    c=PRIOR_CONTRACT_LOOKUP[(domain,fraction_key(fraction),int(seed))]
    corrected=apply_condition_prior(DOMAIN_DATA[domain],domain,c['beta_map'])
    tr=prepare_split_frame(corrected,'train')
    va=prepare_split_frame(corrected,'val')
    te=prepare_split_frame(corrected,'test')
    selected=exact_sparse_subset(tr,fraction,seed).copy()
    assert identity_sha256(selected)==c['selector_sha256']

    f=DOMAIN_FEATURES[domain]
    pp=TrainOnlyImputerScaler().fit(tr[f].to_numpy(float))
    return {
        'train':tr,'val':va,'test':te,'selected':selected,
        'X_train_raw':pp.impute(tr[f].to_numpy(float)),
        'X_val_raw':pp.impute(va[f].to_numpy(float)),
        'X_test_raw':pp.impute(te[f].to_numpy(float)),
        'X_train_scaled':pp.transform(tr[f].to_numpy(float)),
        'X_val_scaled':pp.transform(va[f].to_numpy(float)),
        'X_test_scaled':pp.transform(te[f].to_numpy(float)),
        'selector_sha256':c['selector_sha256'],
        'prior_contract_sha256':c['prior_contract_sha256'],
    }

# Effective supervision accounting, identical for every method within a condition.
budget_rows=[]
for domain in DOMAINS:
    base=DOMAIN_DATA[domain]
    tr0=prepare_split_frame(base,'train')
    anchors=(tr0.sort_values(['compound_segment','elapsed_s','time_value'],kind='mergesort')
               .drop_duplicates('compound_segment',keep='first'))
    anchor_ids=set(anchors['seq_pos'].astype(int))
    for fraction in FRACTIONS:
        for seed in SEEDS:
            sel=exact_sparse_subset(tr0,fraction,seed)
            label_ids=set(sel['seq_pos'].astype(int))
            union=label_ids|anchor_ids
            budget_rows.append({
                'domain':domain,'fraction':float(fraction),'seed':int(seed),'n_train':len(tr0),
                'n_labels':len(label_ids),'n_anchors':len(anchor_ids),'n_overlap':len(label_ids&anchor_ids),
                'n_unique_fit_observations':len(union),'nominal_fraction':len(label_ids)/len(tr0),
                'effective_fraction':len(union)/len(tr0),
            })
budget_seed=pd.DataFrame(budget_rows)
budget_seed.to_csv(OUT / f'{EXPERIMENT}_EFFECTIVE_BUDGET_BY_SEED.csv',index=False)
budget_summary=(budget_seed.groupby(['domain','fraction'],as_index=False)
                .agg(nominal_fraction=('nominal_fraction','mean'),effective_fraction=('effective_fraction','mean'),
                     n_labels=('n_labels','mean'),n_anchors=('n_anchors','mean'),
                     n_unique_fit_observations=('n_unique_fit_observations','mean')))
budget_summary.to_csv(OUT / f'{EXPERIMENT}_EFFECTIVE_BUDGET_SUMMARY.csv',index=False)
display(budget_summary)

# %% Code cell 8 (notebook cell 9)
# Model-native implementations.

def build_tree_model(method, seed):
    if method == OURS:
        return ExtraTreesRegressor(
            n_estimators=TREE_N_ESTIMATORS,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if method in [CHE_COMP, CHE_CORR]:
        return RandomForestRegressor(
            n_estimators=TREE_N_ESTIMATORS,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if method == HOU:
        return XGBRegressor(
            n_estimators=TREE_N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(method)

class GRUResidualNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, GRU_HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(GRU_HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HEAD),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(GRU_HEAD, 1),
        )
    def forward(self, x):
        h, _ = self.gru(x)
        z = self.norm(h[:, -1, :])
        return self.head(z).squeeze(-1)

class CausalSeqDataset(Dataset):
    def __init__(self, frame, X_scaled, positions, target_scaler, seq_len=SEQ_LEN):
        self.frame = frame.reset_index(drop=True)
        self.X = X_scaled
        self.positions = np.asarray(positions, int)
        self.target_scaler = target_scaler
        self.seq_len = int(seq_len)
        self.seg_start = (
            self.frame.groupby("compound_segment", sort=False)["seq_pos"].min().astype(int).to_dict()
        )
    def __len__(self):
        return len(self.positions)
    def __getitem__(self, i):
        pos = int(self.positions[i])
        row = self.frame.iloc[pos]
        s0 = int(self.seg_start[row["compound_segment"]])
        start = max(s0, pos-self.seq_len+1)
        x = self.X[start:pos+1]
        n = len(x)
        pad = self.seq_len-n
        out = np.zeros((self.seq_len, self.X.shape[1]+1), dtype=np.float32)
        out[pad:, :-1] = x
        out[pad:, -1] = 1.0
        y = float(row["residual_target"])
        ys = float(self.target_scaler.transform([y])[0])
        return torch.from_numpy(out), torch.tensor(ys, dtype=torch.float32), pos

@torch.no_grad()
def predict_gru(model, loader, target_scaler):
    model.eval()
    ps, poss = [], []
    for xb, _, pos in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        p = model(xb).detach().cpu().numpy()
        ps.append(p)
        poss.append(np.asarray(pos))
    return np.concatenate(poss).astype(int), target_scaler.inverse(np.concatenate(ps))

# %% Code cell 9 (notebook cell 10)
def _cpu_byte_rng_tensor(x):
    if torch.is_tensor(x):
        return x.detach().to(device='cpu',dtype=torch.uint8).contiguous()
    return torch.as_tensor(x,dtype=torch.uint8,device='cpu').contiguous()

def capture_rng_state():
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state()}
    if torch.cuda.is_available(): state['torch_cuda']=torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    if not state: return
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(_cpu_byte_rng_tensor(state['torch_cpu']))
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all([_cpu_byte_rng_tensor(s) for s in state['torch_cuda']])

def safe_torch_load(path,map_location):
    try: return torch.load(path,map_location=map_location,weights_only=False)
    except TypeError: return torch.load(path,map_location=map_location)

def make_grad_scaler():
    if not AMP_ENABLED: return None
    try: return torch.amp.GradScaler('cuda',enabled=True)
    except Exception: return torch.cuda.amp.GradScaler(enabled=True)

def autocast_context():
    if not AMP_ENABLED: return contextlib.nullcontext()
    try: return torch.amp.autocast(device_type='cuda',enabled=True)
    except Exception: return torch.cuda.amp.autocast(enabled=True)

def condition_contract(domain,fraction,method,seed):
    c=PRIOR_CONTRACT_LOOKUP[(domain,fraction_key(fraction),int(seed))]
    payload={
        'experiment':EXPERIMENT,'domain':domain,'fraction':float(fraction),'method':method,'seed':int(seed),
        'features':DOMAIN_FEATURES[domain],'selector_sha256':c['selector_sha256'],
        'prior_contract_sha256':c['prior_contract_sha256'],
        'tree_n_estimators':TREE_N_ESTIMATORS,'tree_min_samples_leaf':TREE_MIN_SAMPLES_LEAF,
    }
    if method==GUO:
        payload.update({'seq_len':SEQ_LEN,'hidden':GRU_HIDDEN,'head':GRU_HEAD,'dropout':DROPOUT,
                        'lr':LR,'weight_decay':WEIGHT_DECAY,'max_epochs':GRU_BUDGET[domain]['max_epochs'],
                        'patience':GRU_BUDGET[domain]['patience']})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode('utf-8')).hexdigest()

def quarantine_if_stale(tag,best_path,resume_path,history_path,current_fp):
    if not resume_path.exists():
        return None
    st=safe_torch_load(resume_path,DEVICE)
    old_fp=st.get('contract_fingerprint',st.get('contract'))
    if old_fp==current_fp:
        return st
    stamp=f'{safe_tag(tag)}_oldfp_{str(old_fp)[:12]}_newfp_{current_fp[:12]}'
    for p,suffix in [(resume_path,'resume.pt'),(best_path,'best.pt'),(history_path,'history.csv')]:
        if p.exists():
            dest=QUARANTINE_DIR / f'{stamp}_{suffix}'
            if dest.exists(): dest.unlink()
            shutil.move(str(p),str(dest))
    print('STALE CONTRACT quarantined -> restart fresh:',tag)
    return None

# %% Code cell 10 (notebook cell 11)
def result_base(domain,fraction,method,seed,pack):
    return {
        'domain':domain,'fraction':float(fraction),'method':method,'seed':int(seed),
        'selector_sha256':pack['selector_sha256'],'prior_contract_sha256':pack['prior_contract_sha256'],
        'run_contract_fingerprint':condition_contract(domain,fraction,method,seed),
        'n_train_labels':int(len(pack['selected'])),'n_test':int(len(pack['test'])),
    }

def train_tree_run(domain,fraction,method,seed):
    set_all_seeds(seed)
    pack=build_condition_pack(domain,fraction,seed)
    tr,te,selected=pack['train'],pack['test'],pack['selected']
    idx=selected['seq_pos'].astype(int).to_numpy()
    Xtr=pack['X_train_raw'][idx]; Xte=pack['X_test_raw']
    ytr=selected['target'].to_numpy(float) if method==CHE_CORR else selected['residual_target'].to_numpy(float)

    model=build_tree_model(method,seed)
    t0=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-t0
    t1=time.perf_counter(); raw=np.asarray(model.predict(Xte),float); infer_s=time.perf_counter()-t1
    pred=raw if method==CHE_CORR else te['soc_phy'].to_numpy(float)+raw
    met=regression_metrics(te['target'].to_numpy(float),pred)

    out=te[['vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','soc_phy']].copy()
    out['domain']=domain; out['fraction']=float(fraction); out['method']=method; out['seed']=int(seed)
    out['prediction']=pred; out['abs_error']=np.abs(pred-out['target'].to_numpy(float))
    tag=f'{domain}_fraction{fraction}_{safe_tag(method)}_seed{seed}'
    pred_path=PRED_DIR / f'{tag}.parquet'
    out.to_parquet(pred_path,index=False)

    return {**result_base(domain,fraction,method,seed,pack),'source_experiment':EXPERIMENT,
            'reused_full_endpoint':False,'train_seconds':train_s,'inference_seconds':infer_s,
            'prediction_file':str(pred_path),'prediction_sha256':sha256_file(pred_path),**met}

# %% Code cell 11 (notebook cell 12)
def train_gru_run(domain,fraction,seed):
    set_all_seeds(seed)
    method=GUO
    pack=build_condition_pack(domain,fraction,seed)
    tr,va,te,selected=pack['train'],pack['val'],pack['test'],pack['selected']
    target_scaler=TargetScaler().fit(selected['residual_target'].to_numpy(float))

    trpos=selected['seq_pos'].astype(int).to_numpy(); vap=va['seq_pos'].astype(int).to_numpy(); tep=te['seq_pos'].astype(int).to_numpy()
    trds=CausalSeqDataset(tr,pack['X_train_scaled'],trpos,target_scaler)
    vads=CausalSeqDataset(va,pack['X_val_scaled'],vap,target_scaler)
    teds=CausalSeqDataset(te,pack['X_test_scaled'],tep,target_scaler)
    trloader=DataLoader(trds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=AMP_ENABLED)
    valoader=DataLoader(vads,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)
    teloader=DataLoader(teds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)

    model=GRUResidualNet(len(DOMAIN_FEATURES[domain])+1).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    loss_fn=nn.MSELoss(); scaler=make_grad_scaler()
    max_epochs=int(GRU_BUDGET[domain]['max_epochs']); patience=int(GRU_BUDGET[domain]['patience'])

    tag=f'{domain}_fraction{fraction}_{safe_tag(method)}_seed{seed}'
    best_path=CKPT_DIR / f'{tag}_best.pt'; resume_path=CKPT_DIR / f'{tag}_resume.pt'; history_path=HISTORY_DIR / f'history_{tag}.csv'
    fp=condition_contract(domain,fraction,method,seed)

    best_val=np.inf; best_epoch=0; stale=0; start_epoch=1; elapsed_before=0.0; history=[]
    st=quarantine_if_stale(tag,best_path,resume_path,history_path,fp)
    if st is not None:
        model.load_state_dict(st['model_state']); opt.load_state_dict(st['optimizer_state'])
        if scaler is not None and st.get('scaler_state'): scaler.load_state_dict(st['scaler_state'])
        best_val=float(st['best_val']); best_epoch=int(st['best_epoch']); stale=int(st['stale'])
        start_epoch=int(st['epoch'])+1; elapsed_before=float(st.get('elapsed_seconds',0.0))
        history=list(st.get('history',[])); restore_rng_state(st.get('rng_state'))
        print('RESUME',tag,'epoch',start_epoch)

    t0=time.perf_counter(); final_epoch=start_epoch-1; stopped_by_patience=False
    for epoch in range(start_epoch,max_epochs+1):
        final_epoch=epoch; model.train()
        for xb,yb,_ in trloader:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
            with autocast_context():
                yp=model(xb); loss=loss_fn(yp,yb)
            if scaler is None:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
                scaler.step(opt); scaler.update()

        vpos,vres=predict_gru(model,valoader,target_scaler); vrows=va.iloc[vpos]
        vpred=vrows['soc_phy'].to_numpy(float)+vres
        vmae=regression_metrics(vrows['target'].to_numpy(float),vpred)['MAE']
        history.append({'epoch':epoch,'val_MAE':vmae})
        if vmae < best_val-MIN_DELTA:
            best_val=vmae; best_epoch=epoch; stale=0
            torch.save({'model_state':model.state_dict(),'best_val':best_val,'best_epoch':best_epoch,
                        'contract_fingerprint':fp,'selector_sha256':pack['selector_sha256'],
                        'prior_contract_sha256':pack['prior_contract_sha256'],
                        'target_scaler':{'mean':target_scaler.mean_,'std':target_scaler.std_}},best_path)
        else:
            stale+=1

        if epoch % RESUME_SAVE_EVERY==0 or stale>=patience or epoch==max_epochs:
            pd.DataFrame(history).to_csv(history_path,index=False)
            torch.save({'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),
                        'scaler_state':scaler.state_dict() if scaler is not None else {},'epoch':epoch,
                        'best_val':best_val,'best_epoch':best_epoch,'stale':stale,'history':history,
                        'rng_state':capture_rng_state(),'elapsed_seconds':elapsed_before+(time.perf_counter()-t0),
                        'contract_fingerprint':fp,'selector_sha256':pack['selector_sha256'],
                        'prior_contract_sha256':pack['prior_contract_sha256']},resume_path)
        if stale>=patience:
            stopped_by_patience=True; break

    assert best_path.exists()
    bst=safe_torch_load(best_path,DEVICE)
    assert bst.get('contract_fingerprint')==fp
    model.load_state_dict(bst['model_state'])
    t1=time.perf_counter(); tpos,tres=predict_gru(model,teloader,target_scaler); infer_s=time.perf_counter()-t1
    trows=te.iloc[tpos].copy(); pred=trows['soc_phy'].to_numpy(float)+tres
    met=regression_metrics(trows['target'].to_numpy(float),pred)

    out=trows[['vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','soc_phy']].copy()
    out['domain']=domain; out['fraction']=float(fraction); out['method']=method; out['seed']=int(seed)
    out['prediction']=pred; out['abs_error']=np.abs(pred-out['target'].to_numpy(float))
    pred_path=PRED_DIR / f'{tag}.parquet'; out.to_parquet(pred_path,index=False)

    if resume_path.exists(): resume_path.unlink()
    pd.DataFrame(history).to_csv(history_path,index=False)
    return {**result_base(domain,fraction,method,seed,pack),'source_experiment':EXPERIMENT,
            'reused_full_endpoint':False,'train_seconds':elapsed_before+(time.perf_counter()-t0),
            'inference_seconds':infer_s,'best_epoch':int(bst['best_epoch']),'final_epoch':int(final_epoch),
            'stopped_by_patience':bool(stopped_by_patience),'hit_epoch_cap':bool(final_epoch>=max_epochs and not stopped_by_patience),
            'best_val_MAE':float(bst['best_val']),'prediction_file':str(pred_path),
            'prediction_sha256':sha256_file(pred_path),**met}

# %% Code cell 12 (notebook cell 13)
def materialize_reused_full(domain,method,seed):
    fraction=1.0
    q=old_r2d_runs[
        (old_r2d_runs['domain'].astype(str)==domain) & np.isclose(old_r2d_runs['fraction'].astype(float),1.0) &
        (old_r2d_runs['method'].astype(str)==method) & (old_r2d_runs['seed'].astype(int)==int(seed))
    ]
    assert len(q)==1,(domain,method,seed,len(q))
    r=q.iloc[0]
    src=Path(str(r['prediction_file']))
    assert src.exists(),src
    assert sha256_file(src)==str(r['prediction_sha256'])
    assert domain in FULL_REUSE_DOMAINS

    pack=build_condition_pack(domain,1.0,seed)
    z=pd.read_parquet(src).copy()
    # Full-prior gate implies exact/near-exact slope reproduction. Check rowwise prior to tight numeric tolerance.
    te=pack['test']
    key_cols=['vehicle_id','segment_id','time_value']
    a=z[key_cols+['soc_phy']].copy(); b=te[key_cols+['soc_phy']].copy()
    m=a.merge(b,on=key_cols,suffixes=('_old','_corrected'),validate='one_to_one')
    assert len(m)==len(z)==len(te)
    max_prior_diff=float(np.max(np.abs(m['soc_phy_old']-m['soc_phy_corrected'])))
    assert max_prior_diff <= 1e-8,(domain,method,seed,max_prior_diff)

    tag=f'{domain}_fraction1.0_{safe_tag(method)}_seed{seed}_REUSED_FULL'
    dest=PRED_DIR / f'{tag}.parquet'
    shutil.copy2(src,dest)
    met=regression_metrics(z['target'].to_numpy(float),z['prediction'].to_numpy(float))
    return {**result_base(domain,1.0,method,seed,pack),'source_experiment':OLD_EXPERIMENT,
            'reused_full_endpoint':True,'reuse_reason':'full_prior_reproduced_and_protocol_unchanged',
            'old_prediction_file':str(src),'max_abs_soc_phy_old_vs_corrected':max_prior_diff,
            'train_seconds':float(r.get('train_seconds',np.nan)),'inference_seconds':float(r.get('inference_seconds',np.nan)),
            'prediction_file':str(dest),'prediction_sha256':sha256_file(dest),**met}

RUNS_INC=OUT / f'{EXPERIMENT}_RUNS_INCREMENTAL.csv'
expected_keys={(str(r.domain),float(r.fraction),str(r.method),int(r.seed)) for r in plan.itertuples(index=False)}
completed=set(); valid_rows=[]
if RUNS_INC.exists():
    old=pd.read_csv(RUNS_INC)
    for r in old.itertuples(index=False):
        key=(str(r.domain),float(r.fraction),str(r.method),int(r.seed))
        if key not in expected_keys: continue
        p=Path(str(r.prediction_file))
        try:
            c=PRIOR_CONTRACT_LOOKUP[(key[0],fraction_key(key[1]),key[3])]
            expected_fp=condition_contract(*key)
            if (p.exists() and sha256_file(p)==str(r.prediction_sha256) and
                str(r.selector_sha256)==c['selector_sha256'] and str(r.prior_contract_sha256)==c['prior_contract_sha256'] and
                str(r.run_contract_fingerprint)==expected_fp):
                completed.add(key); valid_rows.append(r._asdict())
        except Exception:
            pass
    if len(valid_rows)!=len(old):
        if valid_rows:
            pd.DataFrame(valid_rows).to_csv(RUNS_INC,index=False)
        else:
            RUNS_INC.unlink(missing_ok=True)

print('Total slots:',len(plan))
print('Already valid completed:',len(completed))
print('Remaining:',len(plan)-len(completed))

method_order={OURS:0,HOU:1,CHE_COMP:2,CHE_CORR:3,GUO:4}
domain_order={d:i for i,d in enumerate(['VED','SCUT','TUM','KU','EVE'])}
plan2=plan.copy(); plan2['_mo']=plan2['method'].map(method_order); plan2['_do']=plan2['domain'].map(domain_order)
plan2=plan2.sort_values(['action','_mo','_do','fraction','seed']).drop(columns=['_mo','_do'])

for r in plan2.itertuples(index=False):
    key=(r.domain,float(r.fraction),r.method,int(r.seed))
    if key in completed:
        print('SKIP completed:',key); continue
    print('\nRUN',key,'action=',r.action)
    if r.action=='reuse_verified_old_R2D_full':
        result=materialize_reused_full(r.domain,r.method,int(r.seed))
    elif r.method==GUO:
        result=train_gru_run(r.domain,float(r.fraction),int(r.seed))
    else:
        result=train_tree_run(r.domain,float(r.fraction),r.method,int(r.seed))

    row=pd.DataFrame([result])
    if RUNS_INC.exists():
        prev=pd.read_csv(RUNS_INC); prev=pd.concat([prev,row],ignore_index=True)
    else:
        prev=row
    prev=prev.drop_duplicates(['domain','fraction','method','seed'],keep='last')
    prev.to_csv(RUNS_INC,index=False)
    completed.add(key)
    print('COMPLETE',len(completed),'/',len(plan))
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

runs=pd.read_csv(RUNS_INC)
runs['reused_full_endpoint']=runs['reused_full_endpoint'].astype(str).str.lower().eq('true')
runs=runs.drop_duplicates(['domain','fraction','method','seed'],keep='last')
assert len(runs)==250,len(runs)
assert runs.groupby(['domain','fraction','method'])['seed'].nunique().eq(5).all()
assert int(runs['reused_full_endpoint'].astype(bool).sum())==100
runs.to_csv(OUT / f'{EXPERIMENT}_RUNS.csv',index=False)
print('All corrected R2D run slots complete:',len(runs))

# %% Code cell 13 (notebook cell 14)
# Five-seed summary.
summary=(runs.groupby(['domain','fraction','method'],as_index=False)
         .agg(MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),RMSE_mean=('RMSE','mean'),RMSE_std=('RMSE','std'),
              P95_mean=('P95','mean'),Bias_mean=('Bias','mean'),n_train_labels=('n_train_labels','first'),
              n_test=('n_test','first'),train_seconds_mean=('train_seconds','mean'),
              inference_seconds_mean=('inference_seconds','mean'),reused_runs=('reused_full_endpoint','sum'))
         .merge(budget_summary,on=['domain','fraction'],how='left',validate='many_to_one'))
summary.to_csv(OUT / f'{EXPERIMENT}_COMMON_INFO_SUMMARY.csv',index=False)
display(summary.sort_values(['domain','fraction','MAE_mean']))

# %% Code cell 14 (notebook cell 15)
# Seed-averaged predictions + segment-balanced MAE for every method/condition.
parts=[]
for r in runs.itertuples(index=False):
    p=Path(str(r.prediction_file)); assert p.exists(); assert sha256_file(p)==str(r.prediction_sha256)
    parts.append(pd.read_parquet(p).copy())
seed_pred=pd.concat(parts,ignore_index=True)
keys=['domain','fraction','method','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','soc_phy']
avg_pred=seed_pred.groupby(keys,as_index=False,dropna=False)['prediction'].mean()
avg_pred['abs_error']=np.abs(avg_pred['prediction']-avg_pred['target'])
avg_pred.to_parquet(OUT / f'{EXPERIMENT}_SEED_AVERAGED_PREDICTIONS.parquet',index=False)

segment_summary=(avg_pred.groupby(['domain','fraction','method','compound_segment'],as_index=False)['abs_error'].mean()
                 .groupby(['domain','fraction','method'],as_index=False)['abs_error'].mean()
                 .rename(columns={'abs_error':'segment_balanced_MAE'}))
segment_summary.to_csv(OUT / f'{EXPERIMENT}_SEGMENT_BALANCED_SUMMARY.csv',index=False)
print('Seed-averaged rows:',len(avg_pred))

# %% Code cell 15 (notebook cell 16)
# Primary paired compound-segment bootstrap:
# literature comparator minus OURS common-info ET residual.
# Positive delta => OURS has lower segment-balanced MAE.

def paired_boot(domain, fraction, lit_method, draws=BOOTSTRAP_DRAWS, seed=9001):
    q = avg_pred[
        (avg_pred["domain"] == domain)
        & np.isclose(avg_pred["fraction"], fraction)
        & avg_pred["method"].isin([OURS, lit_method])
    ].copy()

    seg = (
        q.groupby(["method","compound_segment"], as_index=False)["abs_error"].mean()
    )

    a = seg[seg["method"] == OURS].set_index("compound_segment")["abs_error"]
    b = seg[seg["method"] == lit_method].set_index("compound_segment")["abs_error"]

    common = sorted(set(a.index) & set(b.index))
    assert len(common) > 0
    av = a.loc[common].to_numpy(float)
    bv = b.loc[common].to_numpy(float)

    point = float(bv.mean()-av.mean())

    rng = np.random.default_rng(seed)
    n = len(common)
    samples = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0,n,size=n)
        samples[i] = float(bv[idx].mean()-av[idx].mean())

    p_left = (np.sum(samples <= 0)+1)/(draws+1)
    p_right = (np.sum(samples >= 0)+1)/(draws+1)
    p2 = float(min(1.0, 2*min(p_left,p_right)))

    return {
        "domain": domain,
        "fraction": float(fraction),
        "literature_method": lit_method,
        "n_segments": n,
        "ours_segment_balanced_MAE": float(av.mean()),
        "literature_segment_balanced_MAE": float(bv.mean()),
        "delta_literature_minus_ours": point,
        "CI95_low": float(np.percentile(samples,2.5)),
        "CI95_high": float(np.percentile(samples,97.5)),
        "two_sided_bootstrap_p": p2,
        "P_ours_lower": float(np.mean(samples>0)),
    }

rows = []
k = 0
for method in PRIMARY_LITERATURE:
    for domain in DOMAINS:
        for fraction in FRACTIONS:
            rows.append(paired_boot(domain, fraction, method, seed=9001+k*101))
            k += 1

boot = pd.DataFrame(rows)
assert len(boot) == 30
boot["Holm_p"] = holm_adjust(boot["two_sided_bootstrap_p"].to_numpy(float))
boot["ours_supported_after_Holm"] = (
    (boot["CI95_low"] > 0)
    & (boot["Holm_p"] < 0.05)
)
boot["literature_supported_after_Holm"] = (
    (boot["CI95_high"] < 0)
    & (boot["Holm_p"] < 0.05)
)
boot["statistically_unresolved"] = ~(
    boot["ours_supported_after_Holm"] | boot["literature_supported_after_Holm"]
)

boot.to_csv(OUT / f'{EXPERIMENT}_PRIMARY_PAIRED_BOOTSTRAP.csv', index=False)
display(boot)

# %% Code cell 16 (notebook cell 17)
# Secondary Che correction sensitivity — descriptive + unadjusted bootstrap only.

sec_rows = []
for i, domain in enumerate(DOMAINS):
    for j, fraction in enumerate(FRACTIONS):
        sec_rows.append(paired_boot(
            domain, fraction, CHE_CORR,
            seed=13001+i*100+j
        ))
secondary_che = pd.DataFrame(sec_rows)
secondary_che.to_csv(OUT / f"{EXPERIMENT}_SECONDARY_CHE_CORRECTION_BOOTSTRAP.csv", index=False)
display(secondary_che)

# %% Code cell 17 (notebook cell 18)
# Corrected-vs-original audit for affected conditions only.
old_summary=pd.read_csv(OLD_R2D_OUT / f'{OLD_EXPERIMENT}_COMMON_INFO_SUMMARY.csv')
new_summary=summary.copy()
cmp=old_summary.merge(new_summary,on=['domain','fraction','method'],suffixes=('_old','_corrected'),how='inner')
cmp['affected_by_prior_correction']=((cmp['fraction']<1.0) | (cmp['domain']=='TUM'))
cmp['delta_corrected_minus_old_MAE']=cmp['MAE_mean_corrected']-cmp['MAE_mean_old']
cmp.to_csv(OUT / f'{EXPERIMENT}_CORRECTED_VS_ORIGINAL_AUDIT.csv',index=False)
display(cmp[cmp['affected_by_prior_correction']].sort_values(['domain','fraction','method']))

# %% Code cell 18 (notebook cell 19)
# Manuscript-ready primary table includes both row-weighted and segment-balanced MAE.
primary_table=(summary[summary['method'].isin([OURS,HOU,CHE_COMP,GUO])]
               .merge(segment_summary,on=['domain','fraction','method'],how='left',validate='one_to_one'))
primary_table=primary_table[[
    'domain','fraction','method','MAE_mean','MAE_std','RMSE_mean','P95_mean','segment_balanced_MAE',
    'nominal_fraction','effective_fraction','n_train_labels','n_test','reused_runs'
]]
primary_table.to_csv(OUT / f'{EXPERIMENT}_MANUSCRIPT_COMMON_INFO_TABLE.csv',index=False)

rank_table=primary_table.copy()
rank_table['MAE_rank']=rank_table.groupby(['domain','fraction'])['MAE_mean'].rank(method='min')
rank_table.to_csv(OUT / f'{EXPERIMENT}_DESCRIPTIVE_RANKS.csv',index=False)

scope=pd.DataFrame([
    {'method':HOU,'paper':'Hou et al., Energy 2024','preserved_component':'XGBoost residual correction around a rough prior',
     'shared_adaptation':'supervision-matched PR-SoC structured prior + same causal EV telemetry',
     'unavailable_original_component':'Rint/OCV-model parameterization','claim':'literature-adapted, not reproduced','primary_family':True},
    {'method':CHE_COMP,'paper':'Che et al., Nature Communications 2026','preserved_component':'ML residual compensation around model-based prior',
     'shared_adaptation':'Random Forest + supervision-matched PR-SoC prior + same causal EV telemetry',
     'unavailable_original_component':'online ECM/filter states, R/Rp/C/OCV mechanistic features','claim':'literature-adapted, not reproduced','primary_family':True},
    {'method':CHE_CORR,'paper':'Che et al., Nature Communications 2026','preserved_component':'direct correction using prior as an input',
     'shared_adaptation':'Random Forest + supervision-matched PR-SoC prior + same causal EV telemetry',
     'unavailable_original_component':'online ECM/filter mechanistic features','claim':'secondary literature-adapted sensitivity','primary_family':False},
    {'method':GUO,'paper':'Guo et al., Journal of Energy Chemistry 2026','preserved_component':'GRU residual learning around a recursive prior',
     'shared_adaptation':'GRU + supervision-matched PR-SoC prior + same strict-causal telemetry history',
     'unavailable_original_component':'SPM-EKF states and identified electrochemical parameters','claim':'PRKL-inspired, not reproduced','primary_family':True},
    {'method':OURS,'paper':'Current study','preserved_component':'ExtraTrees residual correction around the structured prior',
     'shared_adaptation':'same supervision-matched prior + same causal telemetry','unavailable_original_component':'N/A',
     'claim':'common-information PR-SoC reference','primary_family':False},
])
scope.to_csv(OUT / f'{EXPERIMENT}_METHOD_SCOPE.csv',index=False)
display(primary_table.sort_values(['domain','fraction','MAE_mean']))

# %% Code cell 19 (notebook cell 20)
# Figures: corrected common-information benchmark.
for domain in DOMAINS:
    for fraction in FRACTIONS:
        q=primary_table[(primary_table['domain']==domain)&np.isclose(primary_table['fraction'],fraction)].sort_values('MAE_mean')
        plt.figure(figsize=(8,5))
        plt.bar(q['method'],q['MAE_mean'],yerr=q['MAE_std'],capsize=3)
        plt.ylabel('Test MAE (percentage points)')
        plt.title(f"{domain} — {'10% nominal labels' if fraction==0.1 else 'full labels'}")
        plt.xticks(rotation=25,ha='right'); plt.tight_layout()
        plt.savefig(FIG_DIR / f'{domain}_fraction_{fraction}_corrected_literature_benchmark.png',dpi=220,bbox_inches='tight')
        plt.show()

# %% Code cell 20 (notebook cell 21)
# Outcome-independent manuscript notes.
n_primary=len(boot)
ours_supported=int(boot['ours_supported_after_Holm'].sum())
lit_supported=int(boot['literature_supported_after_Holm'].sum())
unresolved=int(boot['statistically_unresolved'].sum())
notes=[
    '# R2D-C Corrected Literature-Adapted Benchmark Notes','',
    '## Correction','',
    '- Every 10% condition uses a supervision-matched prior calibrated only from retained TRAIN targets.',
    '- TUM 100% was also retrained because its current canonical full-training slope did not reproduce the historical prior.',
    '- 100% VED/KU/SCUT/EVE runs were reused only after prediction SHA and rowwise corrected-prior equality checks.',
    '- Validation/test targets never enter prior calibration.','',
    '## Primary inference','',
    f'- Primary comparisons: {n_primary}',
    f'- OURS common-information ET residual supported after Holm: {ours_supported}',
    f'- Literature comparator supported after Holm: {lit_supported}',
    f'- Statistically unresolved: {unresolved}','',
    'Primary family excludes the Che direct-correction sensitivity. All methods are literature-adapted, not exact reproductions.','',
    '## Reporting boundary','',
    'EVE is post-confirmatory because the test set had already been opened historically.',
]
(OUT / f'{EXPERIMENT}_MANUSCRIPT_NOTES.md').write_text('\n'.join(notes),encoding='utf-8')
print('\n'.join(notes))

# %% Code cell 21 (notebook cell 22)
# Final integrity manifest.
assert len(runs)==250
assert runs.groupby(['domain','fraction','method'])['seed'].nunique().eq(5).all()
assert len(primary_table)==5*2*4
assert len(boot)==30
assert int(runs['reused_full_endpoint'].sum())==100
assert int((~runs['reused_full_endpoint']).sum())==150

manifest={
    **preflight,
    'status':'R2D_CORRECTED_SPARSE_PRIOR_BENCHMARK_COMPLETE',
    'run_slots_complete':int(len(runs)),
    'new_corrected_runs_complete':int((~runs['reused_full_endpoint']).sum()),
    'verified_full_reuse_runs_complete':int(runs['reused_full_endpoint'].sum()),
    'primary_tests_complete':int(len(boot)),
    'ours_supported_after_Holm':int(boot['ours_supported_after_Holm'].sum()),
    'literature_supported_after_Holm':int(boot['literature_supported_after_Holm'].sum()),
    'statistically_unresolved':int(boot['statistically_unresolved'].sum()),
    'all_results_reported':True,
}
with open(OUT / f'{EXPERIMENT}_MANIFEST.json','w') as f:
    json.dump(manifest,f,indent=2)
print(json.dumps(manifest,indent=2))

# %% [markdown] Expected outputs
