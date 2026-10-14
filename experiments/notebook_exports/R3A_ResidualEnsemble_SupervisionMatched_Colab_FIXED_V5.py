# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/benchmark_ensemble/R3A_ResidualEnsemble_SupervisionMatched_Colab_FIXED_V5.ipynb
# Notebook SHA256: 8f15a112dafbf1c0295f8f38a686eb98279cb4ad0a2f1a81fa96f6455a838400
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R3A — Supervision-Matched Residual Expert Ensemble

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, json, random, time, gc, contextlib, warnings, subprocess, sys, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import minimize

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

R2D_CORR_OUT = REV2_OUT / 'R2D_corrected_sparse_prior_benchmark'
OUT = REV2_OUT / 'R3A_residual_ensemble'
PRED_DIR = OUT / 'predictions'
CKPT_DIR = OUT / 'checkpoints'
HISTORY_DIR = OUT / 'history'
FIG_DIR = OUT / 'figures'
QUARANTINE_DIR = CKPT_DIR / 'stale_contract_checkpoints'
for p in [OUT, PRED_DIR, CKPT_DIR, HISTORY_DIR, FIG_DIR, QUARANTINE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = 'R3A_RESIDUAL_ENSEMBLE'
PRIOR_CALIBRATION_VERSION = 'P0_supervision_matched_v1'

DOMAINS = ['VED', 'KU', 'SCUT', 'TUM', 'EVE']
FRACTIONS = [0.10, 1.00]
SEEDS = [42, 43, 44, 45, 46]

ET = 'ET_Residual'
XGB = 'XGB_Residual'
GRU = 'GRU_Residual'
EXPERTS = [ET, XGB, GRU]

EQUAL = 'EqualMean_ResidualEnsemble'
STATIC = 'StaticConvex_ResidualEnsemble'
GATED = 'DisagreementGated_ResidualEnsemble'
BESTVAL = 'ValidationSelected_SingleExpert'
ENSEMBLES = [EQUAL, STATIC, GATED, BESTVAL]
ALL_METHODS = EXPERTS + ENSEMBLES

# R2D-matched expert hyperparameters.
TREE_N_ESTIMATORS = 300
TREE_MIN_SAMPLES_LEAF = 2
SEQ_LEN = 16
DROPOUT = 0.10
GRU_HIDDEN = 64
GRU_HEAD = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MIN_DELTA = 1e-6
RESUME_SAVE_EVERY = 10

# Ensemble protocol, frozen before test inspection.
VALIDATION_EARLYSTOP_SHARE = 0.50
GATE_QUANTILES = [1/3, 2/3]
GATE_MIN_CAL_ROWS = 20
GATE_MIN_CAL_SEGMENTS = 5
BOOTSTRAP_DRAWS = 10000
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
# Governance: corrected R2C/P0 contracts are mandatory. Corrected R2D is optional provenance only.
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
P0_SELECTOR_AUDIT_FILE = P0_AUDIT_DIR / 'P0_SELECTOR_MANIFEST_SHA256.csv'
assert P0_SELECTOR_AUDIT_FILE.exists(), P0_SELECTOR_AUDIT_FILE

# Optional provenance cross-check only. R3A trains ET/XGB/GRU independently,
# therefore a missing/incomplete corrected-R2D manifest must not block execution.
R2D_MANIFEST = R2D_CORR_OUT / 'R2D_CORRECTED_SPARSE_PRIOR_BENCHMARK_MANIFEST.json'
r2d_manifest = None
if R2D_MANIFEST.exists():
    try:
        with open(R2D_MANIFEST) as f:
            candidate = json.load(f)
        if candidate.get('status') == 'R2D_CORRECTED_SPARSE_PRIOR_BENCHMARK_COMPLETE':
            r2d_manifest = candidate
        else:
            warnings.warn(
                f'Corrected R2D manifest exists but status is {candidate.get("status")!r}; '
                'continuing because R3A is independently trained.'
            )
    except Exception as exc:
        warnings.warn(f'Could not read corrected R2D manifest ({exc}); continuing independently.')
else:
    warnings.warn('Corrected R2D manifest not found; continuing because R3A trains all base experts independently.')

with open(MANIFEST_11B) as f:
    man11b = json.load(f)
with open(R2B_MANIFEST) as f:
    r2b_manifest = json.load(f)

r2c_budget = pd.read_csv(R2C_BUDGET)
r2c_prior_audit = pd.read_csv(R2C_PRIOR_AUDIT)
p0_selector_audit = pd.read_csv(P0_SELECTOR_AUDIT_FILE)

assert man11b['strict_causality'] is True
assert int(man11b['sequence_length']) == SEQ_LEN
assert r2b_manifest['anchor_label_overlap_is_double_counted'] is False
assert r2b_manifest['test_targets_count_as_supervision'] is False
assert set(r2c_budget['domain']) == set(DOMAINS)

EXCLUDED_TOKENS = [str(x).lower() for x in man11b['feature_rule']['excluded_tokens']]
GRU_BUDGET = {
    r.domain: {'max_epochs': int(r.max_epochs), 'patience': int(r.patience)}
    for r in r2c_budget.itertuples(index=False)
}
assert set(GRU_BUDGET) == set(DOMAINS)
assert all(v['max_epochs'] >= 1 and v['patience'] >= 1 for v in GRU_BUDGET.values())

print('Governance inputs: PASS')
print('Corrected R2D provenance:', 'COMPLETE' if r2d_manifest is not None else 'not required / unavailable')
display(r2c_budget)

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
# Freeze R3A protocol before inspecting R3A test outcomes.
preflight = {
    'experiment': EXPERIMENT,
    'status': 'R3A_PROTOCOL_FROZEN_BEFORE_TEST_OUTCOME_INSPECTION',
    'target': 'vehicle-reported/BMS-facing SoC',
    'domains': DOMAINS,
    'fractions': FRACTIONS,
    'seeds': SEEDS,
    'experts': EXPERTS,
    'ensemble_methods': ENSEMBLES,
    'prior_calibration_version': PRIOR_CALIBRATION_VERSION,
    'prior_contract': 'exact corrected R2C supervision-matched prior per domain/fraction/seed',
    'validation_partition': 'compound segments deterministically split into disjoint early-stop and ensemble-calibration subsets',
    'validation_earlystop_share': VALIDATION_EARLYSTOP_SHARE,
    'ensemble_calibration_targets': 'same nominal fraction as TRAIN condition, sampled only inside calibration validation segments',
    'ensemble_calibration_label_cost_reported_separately': True,
    'test_targets_used_for_ensemble_fit': False,
    'test_predictions_used_for_ensemble_fit': False,
    'gate_signal': 'expert residual-prediction disagreement only; no target-dependent gate assignment',
    'gate_quantiles': GATE_QUANTILES,
    'gate_regimes': ['low_disagreement','mid_disagreement','high_disagreement'],
    'gate_weight_fit': 'non-negative convex MAE minimization on sparse calibration targets within each regime',
    'gate_fallback': 'static convex weights when a regime has insufficient calibration rows/segments',
    'primary_method': GATED,
    'primary_comparator': ET,
    'primary_inference': 'paired compound-segment bootstrap on seed-averaged test predictions',
    'primary_family_size': 10,
    'bootstrap_draws': BOOTSTRAP_DRAWS,
    'holm_family': '5 domains x 2 fractions for gated-vs-ET primary contrast',
    'EVE_historically_opened': True,
    'confirmatory': False,
    'evaluation_role': 'post-confirmatory residual-expert ensemble experiment',
    'corrected_R2D_manifest_required': False,
    'corrected_R2D_manifest_complete_when_started': bool(r2d_manifest is not None),
}
with open(OUT / f'{EXPERIMENT}_PREFLIGHT.json','w') as f:
    json.dump(preflight, f, indent=2)
print(json.dumps(preflight, indent=2))

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
def build_tree_model(method, seed):
    if method == ET:
        return ExtraTreesRegressor(
            n_estimators=TREE_N_ESTIMATORS,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if method == XGB:
        return XGBRegressor(
            n_estimators=TREE_N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='reg:squarederror',
            tree_method='hist',
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
            nn.Linear(GRU_HIDDEN, GRU_HEAD), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(GRU_HEAD, 1)
        )
    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.norm(h[:, -1, :])).squeeze(-1)

class CausalSeqDataset(Dataset):
    def __init__(self, frame, X_scaled, positions, target_scaler, seq_len=SEQ_LEN):
        self.frame = frame.reset_index(drop=True)
        self.X = X_scaled
        self.positions = np.asarray(positions, int)
        self.target_scaler = target_scaler
        self.seq_len = int(seq_len)
        self.seg_start = self.frame.groupby('compound_segment', sort=False)['seq_pos'].min().astype(int).to_dict()
    def __len__(self): return len(self.positions)
    def __getitem__(self, i):
        pos = int(self.positions[i]); row = self.frame.iloc[pos]
        s0 = int(self.seg_start[row['compound_segment']]); start = max(s0, pos-self.seq_len+1)
        x = self.X[start:pos+1]; n = len(x); pad = self.seq_len-n
        out = np.zeros((self.seq_len, self.X.shape[1]+1), dtype=np.float32)
        out[pad:, :-1] = x; out[pad:, -1] = 1.0
        ys = float(self.target_scaler.transform([float(row['residual_target'])])[0])
        return torch.from_numpy(out), torch.tensor(ys, dtype=torch.float32), pos

@torch.no_grad()
def predict_gru(model, loader, target_scaler):
    model.eval(); ps=[]; poss=[]
    for xb, _, pos in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        ps.append(model(xb).detach().cpu().numpy()); poss.append(np.asarray(pos))
    return np.concatenate(poss).astype(int), target_scaler.inverse(np.concatenate(ps))

# %% Code cell 9 (notebook cell 10)
def _cpu_byte_rng_tensor(x):
    if torch.is_tensor(x): return x.detach().to(device='cpu',dtype=torch.uint8).contiguous()
    return torch.as_tensor(x,dtype=torch.uint8,device='cpu').contiguous()

def capture_rng_state():
    st={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state()}
    if torch.cuda.is_available(): st['torch_cuda']=torch.cuda.get_rng_state_all()
    return st

def restore_rng_state(state):
    if not state: return
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(_cpu_byte_rng_tensor(state['torch_cpu']))
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

# %% Code cell 10 (notebook cell 11)
def deterministic_val_partition(val_frame):
    seg = (val_frame.groupby('compound_segment', as_index=False)['time_value'].min()
           .sort_values(['time_value','compound_segment'], kind='mergesort').reset_index(drop=True))
    if len(seg) < 2:
        raise RuntimeError(
            f'R3A requires at least two validation compound segments for disjoint early-stop/calibration sets; found {len(seg)}.'
        )
    h = seg['compound_segment'].map(lambda s: stable_hash_u01('R3A_VAL|' + str(s)))
    es_segments = set(seg.loc[h < VALIDATION_EARLYSTOP_SHARE, 'compound_segment'])
    cal_segments = set(seg.loc[h >= VALIDATION_EARLYSTOP_SHARE, 'compound_segment'])
    if not es_segments or not cal_segments:
        cut = max(1, min(len(seg)-1, len(seg)//2))
        es_segments = set(seg.iloc[:cut]['compound_segment'])
        cal_segments = set(seg.iloc[cut:]['compound_segment'])
    assert es_segments and cal_segments and es_segments.isdisjoint(cal_segments)
    assert es_segments | cal_segments == set(seg['compound_segment'])
    return es_segments, cal_segments

def exact_sparse_subset_with_salt(df, fraction, seed, salt):
    if len(df) == 0:
        raise RuntimeError('Cannot sample ensemble-calibration targets from an empty validation-calibration frame.')
    if float(fraction) >= 0.999999:
        return df.copy()
    pieces=[]
    for _, g in df.groupby('vehicle_id', sort=False):
        k=max(1,int(round(float(fraction)*len(g))))
        keys=(g['vehicle_id'].astype(str)+'|'+g['segment_id'].astype(str)+'|'+
              g['time_value'].astype(str)+'|'+str(seed)+'|'+str(salt))
        order=keys.map(stable_hash_u01).sort_values().index
        pieces.append(g.loc[order[:k]].copy())
    if not pieces:
        raise RuntimeError('No vehicle groups available for ensemble-calibration sampling.')
    return pd.concat(pieces, ignore_index=False)

def build_r3a_pack(domain, fraction, seed):
    pack = build_condition_pack(domain, fraction, seed)
    es_segments, cal_segments = deterministic_val_partition(pack['val'])
    val_es = pack['val'][pack['val']['compound_segment'].isin(es_segments)].copy()
    val_cal = pack['val'][pack['val']['compound_segment'].isin(cal_segments)].copy()
    assert len(val_es) > 0 and len(val_cal) > 0
    cal_selected = exact_sparse_subset_with_salt(val_cal, fraction, seed, 'ENSEMBLE_CAL')
    assert len(cal_selected) > 0
    assert cal_selected['seq_pos'].is_unique
    assert cal_selected['seq_pos'].between(0, len(pack['val'])-1).all()
    pack.update({
        'val_es': val_es,
        'val_cal': val_cal,
        'cal_selected': cal_selected,
        'es_segments': es_segments,
        'cal_segments': cal_segments,
    })
    return pack

# Full preflight across all 50 domain/fraction/seed conditions before any model training.
val_audit=[]
for domain in DOMAINS:
    for fraction in FRACTIONS:
        for seed in SEEDS:
            p=build_r3a_pack(domain,fraction,seed)
            assert len(p['train']) > 0 and len(p['val']) > 0 and len(p['test']) > 0
            assert p['train']['seq_pos'].is_unique and p['val']['seq_pos'].is_unique and p['test']['seq_pos'].is_unique
            assert p['selected']['seq_pos'].between(0, len(p['train'])-1).all()
            assert p['val_es']['seq_pos'].between(0, len(p['val'])-1).all()
            assert p['val_cal']['seq_pos'].between(0, len(p['val'])-1).all()
            assert p['X_train_raw'].shape[0] == len(p['train'])
            assert p['X_val_raw'].shape[0] == len(p['val'])
            assert p['X_test_raw'].shape[0] == len(p['test'])
            assert np.isfinite(p['train']['soc_phy'].to_numpy(float)).all()
            assert np.isfinite(p['val']['soc_phy'].to_numpy(float)).all()
            assert np.isfinite(p['test']['soc_phy'].to_numpy(float)).all()
            val_audit.append({
                'domain':domain,'fraction':fraction,'seed':seed,
                'n_val_es_rows':len(p['val_es']),'n_val_es_segments':len(p['es_segments']),
                'n_val_cal_rows':len(p['val_cal']),'n_val_cal_segments':len(p['cal_segments']),
                'n_calibration_targets':len(p['cal_selected']),
                'calibration_nominal_fraction':len(p['cal_selected'])/len(p['val_cal']),
            })
val_audit=pd.DataFrame(val_audit)
assert len(val_audit) == len(DOMAINS)*len(FRACTIONS)*len(SEEDS)
val_audit.to_csv(OUT / f'{EXPERIMENT}_VALIDATION_PARTITION_AND_CALIBRATION_BUDGET.csv',index=False)
print('All-condition preflight: PASS —', len(val_audit), 'conditions')
display(val_audit.groupby(['domain','fraction'],as_index=False).agg(
    cal_targets_mean=('n_calibration_targets','mean'),
    cal_fraction_mean=('calibration_nominal_fraction','mean'),
    es_segments=('n_val_es_segments','mean'),cal_segments=('n_val_cal_segments','mean')))

# %% Code cell 11 (notebook cell 12)
def r3a_contract(domain, fraction, expert, seed):
    c=PRIOR_CONTRACT_LOOKUP[(domain,fraction_key(fraction),int(seed))]
    payload={
        'experiment':EXPERIMENT,'domain':domain,'fraction':float(fraction),'expert':expert,'seed':int(seed),
        'selector_sha256':c['selector_sha256'],'prior_contract_sha256':c['prior_contract_sha256'],
        'features':DOMAIN_FEATURES[domain],'tree_n_estimators':TREE_N_ESTIMATORS,
        'tree_min_samples_leaf':TREE_MIN_SAMPLES_LEAF,'seq_len':SEQ_LEN,
        'validation_partition_rule':'R3A_VAL deterministic compound-segment hash',
    }
    if expert==GRU:
        payload.update({'hidden':GRU_HIDDEN,'head':GRU_HEAD,'dropout':DROPOUT,'lr':LR,
                        'weight_decay':WEIGHT_DECAY,'max_epochs':GRU_BUDGET[domain]['max_epochs'],
                        'patience':GRU_BUDGET[domain]['patience']})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def save_prediction_frame(frame, domain, fraction, expert, seed, split_name, residual_pred):
    z=frame[['seq_pos','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','soc_phy','residual_target']].copy()
    z['domain']=domain; z['fraction']=float(fraction); z['expert']=expert; z['seed']=int(seed); z['split_name']=split_name
    z['residual_prediction']=np.asarray(residual_pred,float)
    z['prediction']=z['soc_phy'].to_numpy(float)+z['residual_prediction'].to_numpy(float)
    z['abs_error']=np.abs(z['prediction']-z['target'])
    return z

def train_tree_expert(domain,fraction,expert,seed):
    set_all_seeds(seed); pack=build_r3a_pack(domain,fraction,seed)
    idx=pack['selected']['seq_pos'].astype(int).to_numpy()
    model=build_tree_model(expert,seed)
    t0=time.perf_counter(); model.fit(pack['X_train_raw'][idx],pack['selected']['residual_target'].to_numpy(float)); train_s=time.perf_counter()-t0
    frames=[]
    for split_name,frame,X in [('val_cal',pack['val_cal'],pack['X_val_raw']),('test',pack['test'],pack['X_test_raw'])]:
        pos=frame['seq_pos'].astype(int).to_numpy(); res=np.asarray(model.predict(X[pos]),float)
        frames.append(save_prediction_frame(frame,domain,fraction,expert,seed,split_name,res))
    out=pd.concat(frames,ignore_index=True)
    path=PRED_DIR/f'{domain}_fraction{fraction}_{safe_tag(expert)}_seed{seed}.parquet'; out.to_parquet(path,index=False)
    test=out[out['split_name']=='test']; met=regression_metrics(test['target'],test['prediction'])
    return {'domain':domain,'fraction':float(fraction),'expert':expert,'seed':int(seed),'train_seconds':train_s,
            'prediction_file':str(path),'prediction_sha256':sha256_file(path),'run_contract_fingerprint':r3a_contract(domain,fraction,expert,seed),**met}

def train_gru_expert(domain,fraction,seed):
    set_all_seeds(seed); expert=GRU; pack=build_r3a_pack(domain,fraction,seed)
    tr,va,te,selected=pack['train'],pack['val'],pack['test'],pack['selected']
    scaler_y=TargetScaler().fit(selected['residual_target'].to_numpy(float))
    trpos=selected['seq_pos'].astype(int).to_numpy()
    es_pos=pack['val_es']['seq_pos'].astype(int).to_numpy(); cal_pos=pack['val_cal']['seq_pos'].astype(int).to_numpy(); test_pos=te['seq_pos'].astype(int).to_numpy()
    trloader=DataLoader(CausalSeqDataset(tr,pack['X_train_scaled'],trpos,scaler_y),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=AMP_ENABLED)
    esloader=DataLoader(CausalSeqDataset(va,pack['X_val_scaled'],es_pos,scaler_y),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)
    calloader=DataLoader(CausalSeqDataset(va,pack['X_val_scaled'],cal_pos,scaler_y),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)
    testloader=DataLoader(CausalSeqDataset(te,pack['X_test_scaled'],test_pos,scaler_y),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)

    model=GRUResidualNet(len(DOMAIN_FEATURES[domain])+1).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); loss_fn=nn.MSELoss(); amp=make_grad_scaler()
    max_epochs=int(GRU_BUDGET[domain]['max_epochs']); patience=int(GRU_BUDGET[domain]['patience'])
    tag=f'{domain}_fraction{fraction}_{safe_tag(expert)}_seed{seed}'
    best_path=CKPT_DIR/f'{tag}_best.pt'; resume_path=CKPT_DIR/f'{tag}_resume.pt'; hist_path=HISTORY_DIR/f'history_{tag}.csv'
    fp=r3a_contract(domain,fraction,expert,seed)
    best=np.inf; best_epoch=0; stale=0; start=1; history=[]; elapsed0=0.0
    if resume_path.exists():
        st=safe_torch_load(resume_path,DEVICE)
        if st.get('contract_fingerprint')==fp:
            model.load_state_dict(st['model_state']); opt.load_state_dict(st['optimizer_state']);
            if amp is not None and st.get('scaler_state'): amp.load_state_dict(st['scaler_state'])
            best=float(st['best']); best_epoch=int(st['best_epoch']); stale=int(st['stale']); start=int(st['epoch'])+1
            history=list(st.get('history',[])); elapsed0=float(st.get('elapsed_seconds',0.0)); restore_rng_state(st.get('rng_state'))
        else:
            for p in [resume_path,best_path,hist_path]:
                if p.exists():
                    dest = QUARANTINE_DIR/(p.stem+'_stale_'+fp[:10]+p.suffix)
                    if dest.exists(): dest.unlink()
                    shutil.move(str(p), str(dest))
    t0=time.perf_counter(); final_epoch=start-1; stopped=False
    for epoch in range(start,max_epochs+1):
        final_epoch=epoch; model.train()
        for xb,yb,_ in trloader:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
            with autocast_context(): loss=loss_fn(model(xb),yb)
            if amp is None:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
            else:
                amp.scale(loss).backward(); amp.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); amp.step(opt); amp.update()
        pos,res=predict_gru(model,esloader,scaler_y); rows=va.iloc[pos]
        pred=rows['soc_phy'].to_numpy(float)+res; es_mae=regression_metrics(rows['target'],pred)['MAE']
        history.append({'epoch':epoch,'earlystop_MAE':es_mae})
        if es_mae < best-MIN_DELTA:
            best=es_mae; best_epoch=epoch; stale=0
            torch.save({'model_state':model.state_dict(),'contract_fingerprint':fp,'best':best,'best_epoch':best_epoch},best_path)
        else: stale+=1
        if epoch%RESUME_SAVE_EVERY==0 or stale>=patience or epoch==max_epochs:
            pd.DataFrame(history).to_csv(hist_path,index=False)
            torch.save({'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'scaler_state':amp.state_dict() if amp is not None else {},
                        'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'rng_state':capture_rng_state(),
                        'elapsed_seconds':elapsed0+(time.perf_counter()-t0),'contract_fingerprint':fp},resume_path)
        if stale>=patience: stopped=True; break
    train_elapsed = elapsed0 + (time.perf_counter()-t0)
    if not best_path.exists():
        raise RuntimeError(f'Best GRU checkpoint was not created for {tag}.')
    bst=safe_torch_load(best_path,DEVICE); assert bst['contract_fingerprint']==fp; model.load_state_dict(bst['model_state'])
    frames=[]
    for split_name,frame,loader in [('val_cal',pack['val_cal'],calloader),('test',te,testloader)]:
        pos,res=predict_gru(model,loader,scaler_y); rows=(va if split_name=='val_cal' else te).iloc[pos].copy()
        frames.append(save_prediction_frame(rows,domain,fraction,expert,seed,split_name,res))
    out=pd.concat(frames,ignore_index=True); path=PRED_DIR/f'{tag}.parquet'; out.to_parquet(path,index=False)
    resume_path.unlink(missing_ok=True); pd.DataFrame(history).to_csv(hist_path,index=False)
    test=out[out['split_name']=='test']; met=regression_metrics(test['target'],test['prediction'])
    return {'domain':domain,'fraction':float(fraction),'expert':expert,'seed':int(seed),'train_seconds':train_elapsed,
            'best_epoch':int(bst['best_epoch']),'best_earlystop_MAE':float(bst['best']),'final_epoch':int(final_epoch),
            'stopped_by_patience':bool(stopped),'hit_epoch_cap':bool(final_epoch>=max_epochs and not stopped),
            'prediction_file':str(path),'prediction_sha256':sha256_file(path),'run_contract_fingerprint':fp,**met}

# %% Code cell 12 (notebook cell 13)
BASE_RUNS_INC=OUT/f'{EXPERIMENT}_BASE_EXPERT_RUNS_INCREMENTAL.csv'
expected={(d,float(f),e,int(s)) for d in DOMAINS for f in FRACTIONS for e in EXPERTS for s in SEEDS}
completed=set(); valid_rows=[]
if BASE_RUNS_INC.exists():
    try:
        old=pd.read_csv(BASE_RUNS_INC)
    except pd.errors.EmptyDataError:
        old=pd.DataFrame()
    for r in old.itertuples(index=False):
        try:
            key=(str(r.domain),float(r.fraction),str(r.expert),int(r.seed))
            p=Path(str(r.prediction_file))
            if (key in expected and p.exists() and sha256_file(p)==str(r.prediction_sha256)
                    and str(r.run_contract_fingerprint)==r3a_contract(*key)):
                completed.add(key); valid_rows.append(r._asdict())
        except Exception:
            pass
    if valid_rows:
        pd.DataFrame(valid_rows).drop_duplicates(['domain','fraction','expert','seed'],keep='last').to_csv(BASE_RUNS_INC,index=False)
    else:
        BASE_RUNS_INC.unlink(missing_ok=True)

print('Base expert slots:',len(expected),'already complete:',len(completed),'remaining:',len(expected)-len(completed))
for expert in EXPERTS:
    for domain in ['VED','SCUT','TUM','KU','EVE']:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                key=(domain,float(fraction),expert,int(seed))
                if key in completed: continue
                print('RUN',key)
                res=train_gru_expert(domain,fraction,seed) if expert==GRU else train_tree_expert(domain,fraction,expert,seed)
                df=pd.DataFrame([res])
                if BASE_RUNS_INC.exists():
                    try: prev=pd.read_csv(BASE_RUNS_INC)
                    except pd.errors.EmptyDataError: prev=pd.DataFrame()
                else:
                    prev=pd.DataFrame()
                prev=pd.concat([prev,df],ignore_index=True)
                prev=prev[prev.apply(lambda x: (str(x['domain']),float(x['fraction']),str(x['expert']),int(x['seed'])) in expected,axis=1)]
                prev=prev.drop_duplicates(['domain','fraction','expert','seed'],keep='last')
                prev.to_csv(BASE_RUNS_INC,index=False); completed.add(key)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

base_runs=pd.read_csv(BASE_RUNS_INC).drop_duplicates(['domain','fraction','expert','seed'],keep='last')
base_keys={(str(r.domain),float(r.fraction),str(r.expert),int(r.seed)) for r in base_runs.itertuples(index=False)}
assert base_keys == expected, (len(base_keys), len(expected), sorted(expected-base_keys)[:10], sorted(base_keys-expected)[:10])
assert len(base_runs)==len(expected)
base_runs.to_csv(OUT/f'{EXPERIMENT}_BASE_EXPERT_RUNS.csv',index=False)
print('Base experts complete:',len(base_runs))

# %% Code cell 13 (notebook cell 14)
def fit_convex_mae(P, y):
    P=np.asarray(P,float); y=np.asarray(y,float); k=P.shape[1]
    x0=np.full(k,1.0/k)
    def obj(w): return float(np.mean(np.abs(P@w-y)))
    cons={'type':'eq','fun':lambda w: np.sum(w)-1.0}
    res=minimize(obj,x0,method='SLSQP',bounds=[(0.0,1.0)]*k,constraints=cons,options={'maxiter':500,'ftol':1e-10})
    if not res.success or not np.isfinite(res.x).all():
        mae=np.mean(np.abs(P-y[:,None]),axis=0); inv=1.0/np.maximum(mae,1e-8); w=inv/inv.sum()
    else:
        w=np.clip(res.x,0,1); w=w/w.sum()
    return w

def align_expert_predictions(domain,fraction,seed,split_name):
    parts=[]
    identity=['seq_pos','vehicle_id','segment_id','compound_segment','time_value','elapsed_s']
    invariant=['target','soc_phy','residual_target']
    for expert in EXPERTS:
        r=base_runs[(base_runs.domain==domain)&np.isclose(base_runs.fraction,fraction)&(base_runs.expert==expert)&(base_runs.seed==seed)]
        assert len(r)==1, (domain,fraction,seed,expert,len(r))
        path=Path(str(r.iloc[0].prediction_file))
        assert path.exists(), path
        assert sha256_file(path)==str(r.iloc[0].prediction_sha256)
        z=pd.read_parquet(path)
        z=z[z.split_name==split_name].copy()
        assert len(z)>0, (domain,fraction,seed,expert,split_name,'empty prediction split')
        assert not z.duplicated(identity).any(), (domain,fraction,seed,expert,split_name,'duplicate row identity')
        keep=identity+invariant+['residual_prediction']
        z=z[keep].rename(columns={
            'target':f'target__{expert}','soc_phy':f'soc_phy__{expert}',
            'residual_target':f'residual_target__{expert}','residual_prediction':expert
        })
        parts.append(z)
    out=parts[0]
    for p in parts[1:]:
        out=out.merge(p,on=identity,how='inner',validate='one_to_one')
    expected_rows=len(parts[0])
    assert len(out)==expected_rows, (domain,fraction,seed,split_name,len(out),expected_rows)
    ref=EXPERTS[0]
    for expert in EXPERTS[1:]:
        for col in invariant:
            a=out[f'{col}__{ref}'].to_numpy(float); b=out[f'{col}__{expert}'].to_numpy(float)
            if not np.allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True):
                raise AssertionError((domain,fraction,seed,split_name,col,ref,expert,float(np.nanmax(np.abs(a-b)))))
    out['target']=out[f'target__{ref}']
    out['soc_phy']=out[f'soc_phy__{ref}']
    out['residual_target']=out[f'residual_target__{ref}']
    drop=[f'{col}__{expert}' for expert in EXPERTS for col in invariant]
    return out.drop(columns=drop)

def regime_id(disagreement, q1, q2):
    return np.where(disagreement<=q1,0,np.where(disagreement<=q2,1,2)).astype(int)

weight_rows=[]; pred_parts=[]; run_rows=[]
for domain in DOMAINS:
    for fraction in FRACTIONS:
        for seed in SEEDS:
            cal=align_expert_predictions(domain,fraction,seed,'val_cal')
            test=align_expert_predictions(domain,fraction,seed,'test')
            pack=build_r3a_pack(domain,fraction,seed)
            cal_ids=set(pack['cal_selected']['seq_pos'].astype(int))
            fitmask=cal['seq_pos'].astype(int).isin(cal_ids).to_numpy()
            assert fitmask.sum()==len(pack['cal_selected']) and fitmask.sum()>0
            Pcal=cal[EXPERTS].to_numpy(float); ycal=cal['residual_target'].to_numpy(float)
            Ptest=test[EXPERTS].to_numpy(float)

            # No-target equal mean.
            w_equal=np.full(len(EXPERTS),1/len(EXPERTS))
            # Sparse calibration static convex weights.
            w_static=fit_convex_mae(Pcal[fitmask],ycal[fitmask])
            cal_mae=[np.mean(np.abs(Pcal[fitmask,j]-ycal[fitmask])) for j in range(len(EXPERTS))]
            best_j=int(np.argmin(cal_mae)); w_best=np.eye(len(EXPERTS))[best_j]

            # Target-independent gate assignment from expert disagreement; thresholds frozen on all calibration predictions.
            dis_cal=np.std(Pcal,axis=1); q1,q2=np.quantile(dis_cal,GATE_QUANTILES)
            reg_cal=regime_id(dis_cal,q1,q2); reg_test=regime_id(np.std(Ptest,axis=1),q1,q2)
            W=[]
            for rg in range(3):
                m=fitmask & (reg_cal==rg)
                nrows=int(m.sum()); nseg=int(cal.loc[m,'compound_segment'].nunique()) if nrows else 0
                if nrows>=GATE_MIN_CAL_ROWS and nseg>=GATE_MIN_CAL_SEGMENTS:
                    w=fit_convex_mae(Pcal[m],ycal[m]); fallback=False
                else:
                    w=w_static.copy(); fallback=True
                W.append(w)
                for j,e in enumerate(EXPERTS):
                    weight_rows.append({'domain':domain,'fraction':float(fraction),'seed':seed,'weight_type':'regime','regime':rg,
                                        'expert':e,'weight':float(w[j]),'n_cal_rows':nrows,'n_cal_segments':nseg,'fallback_to_static':fallback,
                                        'q1':float(q1),'q2':float(q2)})
            for name,w in [('equal',w_equal),('static',w_static),('bestval',w_best)]:
                for j,e in enumerate(EXPERTS):
                    weight_rows.append({'domain':domain,'fraction':float(fraction),'seed':seed,'weight_type':name,'regime':-1,
                                        'expert':e,'weight':float(w[j]),'n_cal_rows':int(fitmask.sum()),
                                        'n_cal_segments':int(cal.loc[fitmask,'compound_segment'].nunique()),'fallback_to_static':False,
                                        'q1':float(q1),'q2':float(q2)})

            pred_map={
                EQUAL:Ptest@w_equal,
                STATIC:Ptest@w_static,
                BESTVAL:Ptest@w_best,
                GATED:np.sum(Ptest*np.vstack([W[r] for r in reg_test]),axis=1),
            }
            # Expert predictions are also materialized in the same method format for fair summaries.
            for j,e in enumerate(EXPERTS): pred_map[e]=Ptest[:,j]

            for method,rpred in pred_map.items():
                z=test[['seq_pos','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','soc_phy','residual_target']].copy()
                z['domain']=domain; z['fraction']=float(fraction); z['seed']=int(seed); z['method']=method
                z['residual_prediction']=rpred; z['prediction']=z['soc_phy'].to_numpy(float)+rpred
                z['abs_error']=np.abs(z['prediction']-z['target'])
                pred_parts.append(z)
                met=regression_metrics(z['target'],z['prediction'])
                run_rows.append({'domain':domain,'fraction':float(fraction),'seed':seed,'method':method,
                                 'n_train_labels':len(pack['selected']),'n_calibration_labels':int(fitmask.sum()),
                                 'calibration_fraction':float(fitmask.mean()),**met})

weights=pd.DataFrame(weight_rows); weights.to_csv(OUT/f'{EXPERIMENT}_ENSEMBLE_WEIGHTS.csv',index=False)
all_pred=pd.concat(pred_parts,ignore_index=True); all_pred.to_parquet(OUT/f'{EXPERIMENT}_ALL_TEST_PREDICTIONS.parquet',index=False)
runs=pd.DataFrame(run_rows); runs.to_csv(OUT/f'{EXPERIMENT}_RUNS.csv',index=False)
print('Ensemble conditions complete:',runs.groupby(['domain','fraction','method']).size().shape[0])
display(weights.head(30))

# %% Code cell 14 (notebook cell 15)
# Five-seed row metrics and seed-averaged segment-balanced metrics.
summary=(runs.groupby(['domain','fraction','method'],as_index=False)
         .agg(MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),RMSE_mean=('RMSE','mean'),RMSE_std=('RMSE','std'),
              P95_mean=('P95','mean'),Bias_mean=('Bias','mean'),n_train_labels=('n_train_labels','first'),
              n_calibration_labels=('n_calibration_labels','mean'),calibration_fraction=('calibration_fraction','mean'))
         .merge(budget_summary,on=['domain','fraction'],how='left',validate='many_to_one'))

keys=['domain','fraction','method','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target']
# soc_phy/residual_target are seed-specific under supervision-matched sparse prior calibration,
# so they MUST NOT be grouping keys when averaging final SoC predictions across seeds.
avg_pred=all_pred.groupby(keys,as_index=False,dropna=False)['prediction'].mean()
avg_pred['abs_error']=np.abs(avg_pred['prediction']-avg_pred['target'])
seg=(avg_pred.groupby(['domain','fraction','method','compound_segment'],as_index=False)['abs_error'].mean()
     .groupby(['domain','fraction','method'],as_index=False)['abs_error'].mean()
     .rename(columns={'abs_error':'segment_balanced_MAE'}))
summary=summary.merge(seg,on=['domain','fraction','method'],how='left',validate='one_to_one')
summary.to_csv(OUT/f'{EXPERIMENT}_SUMMARY.csv',index=False)
avg_pred.to_parquet(OUT/f'{EXPERIMENT}_SEED_AVERAGED_PREDICTIONS.parquet',index=False)
display(summary.sort_values(['domain','fraction','MAE_mean']))

# Diversity diagnostics among the three experts, based only on seed-averaged test residual errors.
div=[]
for domain in DOMAINS:
    for fraction in FRACTIONS:
        q=avg_pred[(avg_pred.domain==domain)&np.isclose(avg_pred.fraction,fraction)&avg_pred.method.isin(EXPERTS)]
        row_id=['compound_segment','time_value','elapsed_s']
        assert not q.duplicated(row_id+['method']).any()
        piv=q.pivot(index=row_id,columns='method',values='prediction')
        target=q.drop_duplicates(row_id).set_index(row_id)['target']
        target=target.loc[piv.index]
        err=piv.sub(target,axis=0)
        for i,a in enumerate(EXPERTS):
            for b in EXPERTS[i+1:]:
                div.append({'domain':domain,'fraction':float(fraction),'expert_a':a,'expert_b':b,
                            'error_correlation':float(err[a].corr(err[b])),
                            'prediction_disagreement_MAE':float(np.mean(np.abs(piv[a]-piv[b])))})
diversity=pd.DataFrame(div); diversity.to_csv(OUT/f'{EXPERIMENT}_EXPERT_DIVERSITY.csv',index=False)
display(diversity)

# %% Code cell 15 (notebook cell 16)
def paired_method_boot(domain,fraction,method_a,method_b,draws=BOOTSTRAP_DRAWS,seed=12345):
    q=avg_pred[(avg_pred.domain==domain)&np.isclose(avg_pred.fraction,fraction)&avg_pred.method.isin([method_a,method_b])]
    s=q.groupby(['method','compound_segment'],as_index=False)['abs_error'].mean()
    a=s[s.method==method_a].set_index('compound_segment')['abs_error']; b=s[s.method==method_b].set_index('compound_segment')['abs_error']
    common=sorted(set(a.index)&set(b.index))
    if not common:
        raise RuntimeError((domain,fraction,method_a,method_b,'no common compound segments'))
    av=a.loc[common].to_numpy(float); bv=b.loc[common].to_numpy(float)
    point=float(av.mean()-bv.mean()) # A minus B; negative => A better
    rng=np.random.default_rng(seed); n=len(common); draws_arr=np.empty(draws)
    for i in range(draws):
        ix=rng.integers(0,n,size=n); draws_arr[i]=float(av[ix].mean()-bv[ix].mean())
    p_left=(np.sum(draws_arr<=0)+1)/(draws+1); p_right=(np.sum(draws_arr>=0)+1)/(draws+1)
    return {'domain':domain,'fraction':float(fraction),'method_a':method_a,'method_b':method_b,'n_segments':n,
            'A_segment_MAE':float(av.mean()),'B_segment_MAE':float(bv.mean()),'delta_A_minus_B':point,
            'CI95_low':float(np.percentile(draws_arr,2.5)),'CI95_high':float(np.percentile(draws_arr,97.5)),
            'two_sided_bootstrap_p':float(min(1,2*min(p_left,p_right)))}

primary=[]; k=0
for domain in DOMAINS:
    for fraction in FRACTIONS:
        primary.append(paired_method_boot(domain,fraction,GATED,ET,seed=7001+k*101)); k+=1
boot=pd.DataFrame(primary); boot['Holm_p']=holm_adjust(boot['two_sided_bootstrap_p'])
boot['gated_supported_after_Holm']=(boot['CI95_high']<0)&(boot['Holm_p']<0.05)
boot['ET_supported_after_Holm']=(boot['CI95_low']>0)&(boot['Holm_p']<0.05)
boot['unresolved']=~(boot.gated_supported_after_Holm|boot.ET_supported_after_Holm)
boot.to_csv(OUT/f'{EXPERIMENT}_PRIMARY_GATED_VS_ET_BOOTSTRAP.csv',index=False)
display(boot)

secondary=[]; k=0
for comparator in [STATIC,EQUAL,BESTVAL,XGB,GRU]:
    for domain in DOMAINS:
        for fraction in FRACTIONS:
            secondary.append(paired_method_boot(domain,fraction,GATED,comparator,seed=17001+k*37)); k+=1
secondary=pd.DataFrame(secondary); secondary.to_csv(OUT/f'{EXPERIMENT}_SECONDARY_BOOTSTRAPS_UNADJUSTED.csv',index=False)

# %% Code cell 16 (notebook cell 17)
for domain in DOMAINS:
    q=summary[(summary.domain==domain)&summary.method.isin([ET,XGB,GRU,EQUAL,STATIC,GATED])].copy()
    for fraction in FRACTIONS:
        z=q[np.isclose(q.fraction,fraction)].sort_values('MAE_mean')
        plt.figure(figsize=(9,5))
        plt.bar(z['method'],z['MAE_mean'],yerr=z['MAE_std'],capsize=3)
        plt.ylabel('Test MAE (percentage points)')
        plt.title(f'{domain} — residual ensemble — {"10%" if fraction==0.1 else "100%"} training labels')
        plt.xticks(rotation=28,ha='right'); plt.tight_layout()
        plt.savefig(FIG_DIR/f'{domain}_fraction{fraction}_R3A_residual_ensemble.png',dpi=220,bbox_inches='tight')
        plt.show()

# Compact manuscript-ready table.
manu=summary[summary.method.isin([ET,XGB,GRU,EQUAL,STATIC,GATED])][[
    'domain','fraction','method','MAE_mean','MAE_std','RMSE_mean','P95_mean','segment_balanced_MAE',
    'nominal_fraction','effective_fraction','n_calibration_labels','calibration_fraction']]
manu.to_csv(OUT/f'{EXPERIMENT}_MANUSCRIPT_TABLE.csv',index=False)

# %% Code cell 17 (notebook cell 18)
n_support=int(boot.gated_supported_after_Holm.sum()); n_et=int(boot.ET_supported_after_Holm.sum()); n_un=int(boot.unresolved.sum())
point=summary.pivot_table(index=['domain','fraction'],columns='method',values='MAE_mean').reset_index()
point['gated_minus_best_expert_test_MAE']=point[GATED]-point[EXPERTS].min(axis=1)
notes=[
    '# R3A Residual Ensemble Notes','',
    '## Protocol','',
    '- Three residual experts: ExtraTrees, XGBoost, GRU.',
    '- Same supervision-matched structured prior and same retained TRAIN target identities per condition.',
    '- Validation compound segments are split into disjoint GRU early-stop and ensemble-calibration sets.',
    '- Ensemble calibration uses only the same nominal fraction of target rows as the TRAIN condition and its label cost is reported separately.',
    '- Gate assignment uses expert prediction disagreement only; no test target is used for weights or thresholds.','',
    '## Primary inference','',
    f'- Gated ensemble supported over ET after Holm: {n_support}/10 conditions.',
    f'- ET supported over gated ensemble after Holm: {n_et}/10 conditions.',
    f'- Unresolved: {n_un}/10 conditions.','',
    '## Interpretation boundary','',
    'If the gated model does not improve over static convex or equal averaging, do not force the gating contribution.',
    'A negative result still quantifies whether corrected R2D expert diversity is exploitable after supervision matching.',
]
(OUT/f'{EXPERIMENT}_MANUSCRIPT_NOTES.md').write_text('\n'.join(notes),encoding='utf-8')
point.to_csv(OUT/f'{EXPERIMENT}_GATED_VS_BEST_EXPERT_DESCRIPTIVE.csv',index=False)
print('\n'.join(notes)); display(point)

# %% Code cell 18 (notebook cell 19)
assert len(base_runs)==5*2*3*5
assert len(runs)==5*2*len(ALL_METHODS)*5
assert len(boot)==10
assert summary.groupby(['domain','fraction','method']).size().eq(1).all()
manifest={**preflight,
    'status':'R3A_RESIDUAL_ENSEMBLE_COMPLETE',
    'base_expert_runs':int(len(base_runs)),
    'method_seed_runs':int(len(runs)),
    'primary_tests':int(len(boot)),
    'gated_supported_after_Holm':int(boot.gated_supported_after_Holm.sum()),
    'ET_supported_after_Holm':int(boot.ET_supported_after_Holm.sum()),
    'unresolved':int(boot.unresolved.sum()),
    'all_results_reported':True,
}
with open(OUT/f'{EXPERIMENT}_MANIFEST.json','w') as f: json.dump(manifest,f,indent=2)
print(json.dumps(manifest,indent=2))

# %% [markdown] Expected outputs
