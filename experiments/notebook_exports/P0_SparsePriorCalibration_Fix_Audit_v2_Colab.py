# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/P0_SparsePriorCalibration_Fix_Audit_v2_Colab.ipynb
# Notebook SHA256: 844a85ca43890c4499034bf905413d6276dda1be332d7b9a6263dba92dde901c
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] P0 — Sparse Prior Calibration v2 Audit / Fix Preparation

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
OUTPUTS = ROOT / 'outputs'
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
CANON = OUTPUTS / 'model_screening_10' / 'canonical'

OUT = REV2_OUT / 'P0_sparse_prior_calibration_audit_v2'
OUT.mkdir(parents=True, exist_ok=True)

DEV_DOMAINS = ['VED','KU','SCUT','TUM']
ALL_DOMAINS = DEV_DOMAINS + ['EVE']
FRACTIONS = [0.01,0.02,0.05,0.10,0.20,0.50,1.00]
SEEDS = [42,43,44,45,46]

# Preserve the calibration granularity documented by the frozen canonical metadata.
# VED: pooled train-only slope; EVE: single train-only slope.
# KU/SCUT/TUM: vehicle-specific train-only slope.
GROUP_COLS = {
    'VED': [],
    'KU': ['vehicle_id'],
    'SCUT': ['vehicle_id'],
    'TUM': ['vehicle_id'],
    'EVE': [],
}

EPS = 1e-12
print('Output:', OUT)

# %% Code cell 2 (notebook cell 3)
def stable_hash_u01(s):
    return int(hashlib.md5(str(s).encode('utf-8')).hexdigest()[:12], 16) / float(16**12)

def exact_sparse_subset(df, fraction, seed):
    """Exact selector contract copied from the frozen R2C notebook."""
    if float(fraction) >= 0.999999:
        return df.copy()
    pieces=[]
    for _, g in df.groupby('vehicle_id', sort=False):
        k=max(1, int(round(float(fraction)*len(g))))
        keys=(g['vehicle_id'].astype(str)+'|'+g['segment_id'].astype(str)+'|'+g['time_value'].astype(str)+'|'+str(seed))
        order=keys.map(stable_hash_u01).sort_values().index
        pieces.append(g.loc[order[:k]].copy())
    return pd.concat(pieces, ignore_index=False)

def identity_string(df):
    z=(df['vehicle_id'].astype(str)+'|'+df['segment_id'].astype(str)+'|'+df['time_value'].astype(str)).sort_values()
    return '\n'.join(z.tolist())

def identity_sha256(df):
    return hashlib.sha256(identity_string(df).encode('utf-8')).hexdigest()

def regression_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); e=p-y; ae=np.abs(e)
    return {'MAE':float(ae.mean()), 'RMSE':float(np.sqrt(np.mean(e**2))),
            'P95':float(np.percentile(ae,95)), 'Bias':float(e.mean())}

def infer_old_beta(g):
    """Infer the frozen slope encoded in soc_phy = soc0 + beta*coord."""
    coord=g['coord'].to_numpy(float)
    delta=(g['soc_phy']-g['soc0']).to_numpy(float)
    mask=np.isfinite(coord)&np.isfinite(delta)&(np.abs(coord)>1e-10)
    ratios=delta[mask]/coord[mask]
    if len(ratios)==0:
        return np.nan
    # Frozen prior should encode a constant slope inside each calibration group.
    return float(np.median(ratios))

# Historical prior-calibration eligibility recovered from the frozen source notebooks.
# VED: estimate_beta(...): genuine SoC-change anchors, |q_Ah| >= 0.01 Ah.
# KU/SCUT: fit_beta(...): canonical rows are genuine SoC anchors, |q_signed_Ah| >= 1e-4 Ah.
# TUM: fit_gamma_zero_intercept(...): genuine SoC anchors, distance_km >= 0.01 km.
# EVE: 12B-v4 train-only dense BMS targets, |coord| > 1e-8 Ah.
CALIBRATION_RULES = {
    'VED':  {'kind':'abs', 'threshold':1e-2},
    'KU':   {'kind':'abs', 'threshold':1e-4},
    'SCUT': {'kind':'abs', 'threshold':1e-4},
    'TUM':  {'kind':'pos', 'threshold':1e-2},
    'EVE':  {'kind':'abs_strict', 'threshold':1e-8},
}

def fit_beta_allowed(g, domain):
    """OLS-through-origin using only retained target rows and the historical domain eligibility rule."""
    x=g['coord'].to_numpy(float)
    y=(g['target']-g['soc0']).to_numpy(float)
    rule=CALIBRATION_RULES[domain]
    finite=np.isfinite(x)&np.isfinite(y)
    if rule['kind']=='pos':
        eligible=(x >= rule['threshold'])
    elif rule['kind']=='abs_strict':
        eligible=(np.abs(x) > rule['threshold'])
    else:
        eligible=(np.abs(x) >= rule['threshold'])
    mask=finite & eligible
    x=x[mask]; y=y[mask]
    if len(x)==0 or float(np.dot(x,x))<=EPS:
        return np.nan, int(len(x))
    return float(np.dot(x,y)/np.dot(x,x)), int(len(x))

def add_compound(df):
    d=df.copy()
    d['vehicle_id']=d['vehicle_id'].astype(str)
    d['segment_id']=d['segment_id'].astype(str)
    d['compound_segment']=d['vehicle_id']+'||'+d['segment_id']
    return d

def load_all():
    data={}; meta={}
    for domain in DEV_DOMAINS:
        p=CANON/f'{domain}_canonical.parquet'
        m=CANON/f'{domain}_metadata.json'
        assert p.exists(), p; assert m.exists(), m
        d=add_compound(pd.read_parquet(p))
        with open(m) as f: md=json.load(f)
        data[domain]=d[d['split'].isin(['train','val','test'])].copy(); meta[domain]=md
    eve_tv=add_compound(pd.read_parquet(OUTPUTS/'eve_12B_v4_frozen_trainval'/'12B_v4_EVE_canonical_trainval.parquet'))
    eve_test=add_compound(pd.read_parquet(OUTPUTS/'eve_12C_one_shot_confirmatory'/'12C_EVE_confirmatory_canonical.parquet'))
    eve_tv=eve_tv[eve_tv['split'].isin(['train','val'])].copy(); eve_test['split']='test'
    data['EVE']=pd.concat([eve_tv,eve_test], ignore_index=True)
    with open(OUTPUTS/'eve_12B_v4_frozen_trainval'/'12B_v4_EVE_canonical_metadata.json') as f: meta['EVE']=json.load(f)
    for domain,d in data.items():
        req={'vehicle_id','segment_id','compound_segment','time_value','split','target','soc0','coord','soc_phy'}
        miss=sorted(req-set(d.columns)); assert not miss,(domain,miss)
    return data,meta

DATA,META=load_all()
for d in ALL_DOMAINS:
    print(d, DATA[d]['split'].value_counts().to_dict(), 'vehicles=',DATA[d]['vehicle_id'].nunique(), 'segments=',DATA[d]['compound_segment'].nunique())

# %% Code cell 3 (notebook cell 4)
# 1) Recover the frozen dense-prior slopes encoded by soc_phy.
old_beta_rows=[]
for domain in ALL_DOMAINS:
    tr=DATA[domain].query("split == 'train'").copy()
    group_cols=GROUP_COLS[domain]
    if group_cols:
        groups=tr.groupby(group_cols, sort=False)
    else:
        groups=[('__POOLED__',tr)]
    for key,g in groups:
        if isinstance(key,tuple): key='||'.join(map(str,key))
        old_beta_rows.append({'domain':domain,'group':str(key),'old_beta_from_soc_phy':infer_old_beta(g),'n_train_rows':len(g)})
old_beta=pd.DataFrame(old_beta_rows)
display(old_beta)
old_beta.to_csv(OUT/'P0_FROZEN_BETA_REFERENCE.csv',index=False)

# %% Code cell 4 (notebook cell 5)
# 2) Refit beta under the exact sparse target selector.
# Mandatory segment-start anchors provide soc0 for every segment. They do not add a nonzero-coordinate
# regression equation by themselves; the retained target rows supply (target-soc0, coord) pairs.

beta_rows=[]; selector_rows=[]; budget_rows=[]; prior_rows=[]

for domain in ALL_DOMAINS:
    full=DATA[domain]
    train=full.query("split == 'train'").copy()
    test=full.query("split == 'test'").copy()
    group_cols=GROUP_COLS[domain]

    # exact anchor identity = first row per compound segment in train after stable chronological ordering
    sort_cols=['compound_segment'] + [c for c in ['elapsed_s','time_value'] if c in train.columns]
    anchors=(train.sort_values(sort_cols,kind='mergesort').drop_duplicates('compound_segment',keep='first'))
    anchor_ids=set(identity_string(anchors).split('\n')) if len(anchors) else set()

    for frac in FRACTIONS:
        for seed in SEEDS:
            sel=exact_sparse_subset(train,frac,seed).copy()
            selector_rows.append({'domain':domain,'fraction':frac,'seed':seed,'n_selected':len(sel),'selector_sha256':identity_sha256(sel)})
            selected_ids=set(identity_string(sel).split('\n')) if len(sel) else set()
            union_ids=selected_ids|anchor_ids
            budget_rows.append({'domain':domain,'fraction':frac,'seed':seed,'n_train_rows':len(train),'n_selected_labels':len(selected_ids),'n_train_anchors':len(anchor_ids),'n_unique_fit_observations':len(union_ids),'effective_fit_fraction':len(union_ids)/len(train)})

            beta_map={}
            if group_cols:
                # Fit only from retained rows within each frozen calibration group.
                for key,g in sel.groupby(group_cols,sort=False):
                    if isinstance(key,tuple): key_norm='||'.join(map(str,key))
                    else: key_norm=str(key)
                    b,ninfo=fit_beta_allowed(g, domain)
                    if not np.isfinite(b):
                        raise RuntimeError(f'{domain} frac={frac} seed={seed} group={key_norm}: no informative retained rows for beta')
                    beta_map[key_norm]=b
                    old_match=old_beta[(old_beta.domain==domain)&(old_beta.group==key_norm)]
                    old=float(old_match.old_beta_from_soc_phy.iloc[0]) if len(old_match) else np.nan
                    beta_rows.append({'domain':domain,'fraction':frac,'seed':seed,'group':key_norm,'n_informative_labels':ninfo,'beta_sparse':b,'beta_frozen':old,'beta_diff':b-old,'beta_rel_diff_pct':100*(b-old)/abs(old) if np.isfinite(old) and abs(old)>EPS else np.nan})
            else:
                b,ninfo=fit_beta_allowed(sel, domain)
                if not np.isfinite(b): raise RuntimeError(f'{domain} frac={frac} seed={seed}: no informative retained rows for beta')
                beta_map['__POOLED__']=b
                old=float(old_beta[(old_beta.domain==domain)&(old_beta.group=='__POOLED__')].old_beta_from_soc_phy.iloc[0])
                beta_rows.append({'domain':domain,'fraction':frac,'seed':seed,'group':'__POOLED__','n_informative_labels':ninfo,'beta_sparse':b,'beta_frozen':old,'beta_diff':b-old,'beta_rel_diff_pct':100*(b-old)/abs(old) if abs(old)>EPS else np.nan})

            # Evaluate changed prior on test rows only; test targets are never used in the fit.
            tt=test.copy()
            if group_cols:
                keys=tt[group_cols].astype(str).agg('||'.join,axis=1) if len(group_cols)>1 else tt[group_cols[0]].astype(str)
                bvec=keys.map(beta_map)
                if bvec.isna().any():
                    missing=sorted(keys[bvec.isna()].unique().tolist())
                    raise RuntimeError(f'{domain}: sparse beta unavailable for test groups {missing}')
                newp=tt['soc0'].to_numpy(float)+bvec.to_numpy(float)*tt['coord'].to_numpy(float)
            else:
                newp=tt['soc0'].to_numpy(float)+beta_map['__POOLED__']*tt['coord'].to_numpy(float)
            oldp=tt['soc_phy'].to_numpy(float); y=tt['target'].to_numpy(float)
            oldm=regression_metrics(y,oldp); newm=regression_metrics(y,newp)
            prior_rows.append({'domain':domain,'fraction':frac,'seed':seed,
                               'old_prior_MAE':oldm['MAE'],'sparse_prior_MAE':newm['MAE'],
                               'prior_MAE_delta':newm['MAE']-oldm['MAE'],
                               'mean_abs_prior_change':float(np.mean(np.abs(newp-oldp))),
                               'p95_abs_prior_change':float(np.percentile(np.abs(newp-oldp),95)),
                               'max_abs_prior_change':float(np.max(np.abs(newp-oldp)))})

beta_audit=pd.DataFrame(beta_rows)
selector_manifest=pd.DataFrame(selector_rows)
budget=pd.DataFrame(budget_rows)
prior_audit=pd.DataFrame(prior_rows)

beta_audit.to_csv(OUT/'P0_SPARSE_BETA_AUDIT.csv',index=False)
selector_manifest.to_csv(OUT/'P0_SELECTOR_MANIFEST_SHA256.csv',index=False)
budget.to_csv(OUT/'P0_EFFECTIVE_BUDGET_WITH_SPARSE_BETA.csv',index=False)
prior_audit.to_csv(OUT/'P0_PRIOR_IMPACT_ON_TEST.csv',index=False)

print('beta rows',len(beta_audit),'prior rows',len(prior_audit))
display(beta_audit.head(20))
display(prior_audit.groupby(['domain','fraction']).agg(mean_abs_prior_change=('mean_abs_prior_change','mean'), sparse_prior_MAE=('sparse_prior_MAE','mean'), old_prior_MAE=('old_prior_MAE','mean')).reset_index())

# %% Code cell 5 (notebook cell 6)
# 3) Full-fraction reproduction gate.
# At fraction=1.0, the allowed target set is the full training set. Therefore the sparse-fit formula
# should reproduce the frozen prior slope closely. If it does not, do NOT launch expensive reruns yet;
# first inspect the original canonical beta-fitting rule.

full=beta_audit[beta_audit['fraction']==1.0].copy()
full['abs_beta_diff']=full['beta_diff'].abs()
print(full[['domain','seed','group','beta_sparse','beta_frozen','abs_beta_diff']].to_string(index=False))

summary=(full.groupby('domain',as_index=False)
         .agg(max_abs_beta_diff=('abs_beta_diff','max'), max_rel_diff_pct=('beta_rel_diff_pct',lambda s: float(np.nanmax(np.abs(s))))))
summary.to_csv(OUT/'P0_FULL_FRACTION_REPRODUCTION_GATE.csv',index=False)
display(summary)

# This is deliberately a diagnostic gate, not a hard-coded tolerance assertion, because some frozen
# domains may have used robust filtering/weighting when the canonical beta was created. The table tells
# us whether the simple OLS-through-origin reconstruction is the correct historical fit rule.

# %% Code cell 6 (notebook cell 7)
# 4) Governance/status artifact.
status={
 'issue':'Current sparse learner fractions reuse a structured prior calibrated from the full training target split.',
 'fix_principle':'For each domain/fraction/seed, calibrate every target-dependent prior parameter only from retained training target rows plus mandatory segment-start anchors.',
 'selector_contract':'Exact R2C deterministic per-vehicle hash selector, seeds 42-46.',
 'calibration_granularity':GROUP_COLS,
 'test_targets_used_for_fitting':False,
 'validation_targets_used_for_beta_fitting':False,
 'existing_results_overwritten':False,
 'next_step':'After confirming the full-fraction reproduction gate, generate fraction/seed-specific soc_phy and rerun all sparse claims that depend on the prior.'
}
with open(OUT/'P0_STATUS.json','w') as f: json.dump(status,f,indent=2)
print(json.dumps(status,indent=2))

# %% [markdown] What to send back after running
