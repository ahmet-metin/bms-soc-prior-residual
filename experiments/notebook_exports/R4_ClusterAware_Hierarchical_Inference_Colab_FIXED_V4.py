# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/inference/R4_ClusterAware_Hierarchical_Inference_Colab_FIXED_V4.ipynb
# Notebook SHA256: f2738f5b1c137f42d86e8a3b13ea831885e21b85b0fbbaeec02c4deb59e95b1d
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R4 — Cluster-Aware / Hierarchical Inference Audit — FIXED V4

# %% Code cell 1 (notebook cell 2)

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json, itertools, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'

R2C = REV2_OUT / 'R2C_corrected_sparse_prior_curve'
R2D = REV2_OUT / 'R2D_corrected_cross_seed_aggregation'
R3A = REV2_OUT / 'R3A_residual_ensemble'

OUT = REV2_OUT / 'R4_cluster_aware_inference_v4'
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENT = 'R4_CLUSTER_AWARE_HIERARCHICAL_INFERENCE'
CODE_VERSION = 'R4_vehicle_segment_hierarchical_v4_corrected_R2D_seedavg'

DOMAINS = ['VED','KU','SCUT','TUM','EVE']
R2C_FRACTIONS = [0.01,0.02,0.05,0.10,0.20,0.50,1.00]
ENDPOINT_FRACTIONS = [0.10,1.00]

R2C_DIRECT = 'Direct+Prior'
R2C_PR = 'PriorResidual+Prior'

R2D_OURS = 'PRSoC_CommonInfo_ET_Residual'
R2D_LIT = [
    'Hou2024_Adapted_XGB_Residual',
    'Che2026_Adapted_RF_Compensation',
    'Guo2026_PRKL_Inspired_GRU_Residual',
]

R3A_GATED = 'DisagreementGated_ResidualEnsemble'
R3A_ET = 'ET_Residual'

BOOT_DRAWS = 20000
MIN_VEHICLES_FOR_POPULATION_INFERENCE = 5
EPS = 1e-12

print('Experiment:', EXPERIMENT)
print('Code version:', CODE_VERSION)
print('Output:', OUT)

# %% Code cell 2 (notebook cell 3)

# ---------- Source file contracts / completion governance ----------

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

R2C_MANIFEST = R2C / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_MANIFEST.json'
R2D_MANIFEST = R2D / 'R2D_CROSS_SEED_AGGREGATION_FIX_MANIFEST.json'
R3A_MANIFEST = R3A / 'R3A_RESIDUAL_ENSEMBLE_MANIFEST.json'

R2C_PRED = R2C / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet'
R2D_PRED = R2D / 'R2D_CROSS_SEED_AGGREGATION_FIX_SEED_AVERAGED_PREDICTIONS.parquet'
R3A_PRED = R3A / 'R3A_RESIDUAL_ENSEMBLE_SEED_AVERAGED_PREDICTIONS.parquet'

R2C_BUDGET = R2C / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_EFFECTIVE_BUDGET_SUMMARY.csv'
R2C_RUNS = R2C / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_ALL_CURVE_RUNS.csv'
R2C_PRIMARY = R2C / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_PRIMARY_EFFECTIVE_AULEC_BOOTSTRAP.csv'

R2D_PRIMARY = R2D / 'R2D_CROSS_SEED_AGGREGATION_FIX_PRIMARY_PAIRED_BOOTSTRAP.csv'
R3A_PRIMARY = R3A / 'R3A_RESIDUAL_ENSEMBLE_PRIMARY_GATED_VS_ET_BOOTSTRAP.csv'

required = [
    R2C_PRED,R2D_PRED,R3A_PRED,
    R2C_BUDGET,R2C_RUNS,R2C_PRIMARY,
    R2D_MANIFEST,R3A_MANIFEST,
    R2D_PRIMARY,R3A_PRIMARY,
]
missing=[str(p) for p in required if not p.exists()]
assert not missing, ('Missing required R4 inputs', missing)

# R2D/R3A have fail-closed final manifests.
m2d=load_json(R2D_MANIFEST)
m3a=load_json(R3A_MANIFEST)
assert m2d.get('status') == 'R2D_CROSS_SEED_AGGREGATION_FIX_COMPLETE', m2d.get('status')
assert m2d.get('models_retrained') is False
assert m2d.get('all_physical_rows_have_five_seeds') is True
assert m2d.get('soc_phy_is_grouping_key') is False
assert int(m2d.get('primary_tests_complete',-1)) == 30
assert m3a.get('status') == 'R3A_RESIDUAL_ENSEMBLE_COMPLETE', m3a.get('status')

# R2C: use manifest if present; otherwise verify the final output bundle itself.
r2c_manifest_present=R2C_MANIFEST.exists()
if r2c_manifest_present:
    m2c=load_json(R2C_MANIFEST)
    assert m2c.get('status') == 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_COMPLETE', m2c.get('status')
    r2c_completion_source='final_manifest'
    r2c_source_status=m2c.get('status')
else:
    m2c={}
    r2c_runs=pd.read_csv(R2C_RUNS)
    r2c_primary=pd.read_csv(R2C_PRIMARY)
    r2c_budget_check=pd.read_csv(R2C_BUDGET)
    expected_r2c_keys = 5*7*2*5
    required_run_cols = {'domain','fraction','model','seed'}
    assert required_run_cols.issubset(r2c_runs.columns), (required_run_cols-set(r2c_runs.columns), r2c_runs.columns.tolist())
    n_unique_r2c = len(r2c_runs[list(required_run_cols)].drop_duplicates())
    assert n_unique_r2c == expected_r2c_keys, ('R2C unique run-key count', n_unique_r2c, expected_r2c_keys)
    assert len(r2c_primary)==5, ('R2C primary test count',len(r2c_primary))
    assert len(r2c_budget_check)==5*7, ('R2C budget count',len(r2c_budget_check))
    assert set(r2c_primary['domain'].astype(str))==set(DOMAINS)
    assert r2c_primary['effective_AULEC_supported_after_Holm'].notna().all()
    r2c_completion_source='verified_final_output_bundle'
    r2c_source_status='R2C_CORRECTED_SPARSE_PRIOR_CURVE_COMPLETE_BY_OUTPUT_BUNDLE'

print('Governance inputs: PASS')
print('R2C completion:', r2c_completion_source)
print('R2D:', m2d['status'])
print('R3A:', m3a['status'])

# %% Code cell 3 (notebook cell 4)

# ---------- Statistical helpers ----------

def holm_adjust(p_values):
    p=np.asarray(p_values,float)
    assert np.isfinite(p).all()
    m=len(p)
    order=np.argsort(p)
    out=np.empty(m,float)
    running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx])
        out[idx]=min(1.0,running)
    return out

def bootstrap_two_sided_p(samples):
    s=np.asarray(samples,float)
    assert np.isfinite(s).all()
    p_left=(np.sum(s<=0)+1.0)/(len(s)+1.0)
    p_right=(np.sum(s>=0)+1.0)/(len(s)+1.0)
    return float(min(1.0,2.0*min(p_left,p_right)))

def normalized_log_auc(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    order=np.argsort(x); x=x[order]; y=y[order]
    assert len(x)>=2
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(x>0), x
    lx=np.log10(x)
    width=float(lx[-1]-lx[0])
    assert width>EPS
    trapz=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    return float(trapz(y,lx)/width)

def exact_vehicle_signflip_p(vehicle_deltas):
    d=np.asarray(vehicle_deltas,float)
    d=d[np.isfinite(d)]
    n=len(d)
    if n==0:
        return np.nan
    obs=abs(float(np.mean(d)))
    # Expected n is small here; exact enumeration avoids asymptotic approximations.
    if n<=20:
        ge=0; total=0
        for signs in itertools.product((-1.0,1.0), repeat=n):
            val=abs(float(np.mean(d*np.asarray(signs,float))))
            ge += int(val >= obs-1e-15)
            total += 1
        return float(ge/total)
    # Deterministic fallback for any future larger dataset.
    rng=np.random.default_rng(88991+n)
    B=100000
    ge=0
    for _ in range(B):
        signs=rng.choice([-1.0,1.0],size=n)
        ge += int(abs(float(np.mean(d*signs))) >= obs-1e-15)
    return float((ge+1)/(B+1))

def inference_status(nv):
    nv=int(nv)
    if nv>=MIN_VEHICLES_FOR_POPULATION_INFERENCE:
        return 'population_inference_eligible'
    if nv==1:
        return 'within_vehicle_only_single_vehicle'
    return 'sensitivity_only_few_vehicles'

def build_analysis_segment(df):
    z=df.copy()
    z['vehicle_id']=z['vehicle_id'].astype(str)
    if 'segment_id' in z.columns:
        z['segment_id']=z['segment_id'].astype(str)
        z['analysis_segment']=z['vehicle_id']+'||'+z['segment_id']
    elif 'compound_segment' in z.columns:
        z['compound_segment']=z['compound_segment'].astype(str)
        # In the stored R2C/R2D/R3A files compound_segment is normally already vehicle-qualified.
        # Avoid accidentally producing vehicle||vehicle||segment if segment_id is absent in a future export.
        pref=z['vehicle_id']+'||'
        already=np.fromiter((cs.startswith(p) for cs,p in zip(z['compound_segment'],pref)), dtype=bool, count=len(z))
        z['analysis_segment']=np.where(already, z['compound_segment'], pref+z['compound_segment'])
    else:
        raise RuntimeError('Prediction table has neither segment_id nor compound_segment.')
    assert not z['analysis_segment'].isna().any()
    return z

def hierarchical_bootstrap_segment_delta(seg_delta,draws,seed):
    d=seg_delta[['vehicle_id','analysis_segment','delta']].copy()
    assert len(d)>0
    assert not d.duplicated(['vehicle_id','analysis_segment']).any()
    d['vehicle_id']=d['vehicle_id'].astype(str)
    vehicles=sorted(d.vehicle_id.unique())
    per_v={v:d[d.vehicle_id==v]['delta'].to_numpy(float) for v in vehicles}
    assert all(len(x)>0 and np.isfinite(x).all() for x in per_v.values())

    vehicle_means=np.asarray([per_v[v].mean() for v in vehicles],float)
    point=float(vehicle_means.mean())

    rng=np.random.default_rng(seed)
    nv=len(vehicles)
    boot=np.empty(draws,float)
    for b in range(draws):
        sampled_v=rng.integers(0,nv,size=nv)
        vals=[]
        for vi in sampled_v:
            x=per_v[vehicles[int(vi)]]
            ii=rng.integers(0,len(x),size=len(x))
            vals.append(float(x[ii].mean()))
        boot[b]=float(np.mean(vals))

    return {
        'vehicle_balanced_delta':point,
        'CI95_low':float(np.percentile(boot,2.5)),
        'CI95_high':float(np.percentile(boot,97.5)),
        'hierarchical_bootstrap_p':bootstrap_two_sided_p(boot),
        'n_vehicles':nv,
        'n_segments':len(d),
        'vehicle_means':vehicle_means,
    }

def leave_one_vehicle_out(seg_delta):
    d=seg_delta.copy()
    d['vehicle_id']=d['vehicle_id'].astype(str)
    vehicles=sorted(d.vehicle_id.unique())
    full=float(d.groupby('vehicle_id')['delta'].mean().mean())
    rows=[]
    if len(vehicles)<2:
        return full,rows
    for v in vehicles:
        q=d[d.vehicle_id!=v]
        rows.append({
            'dropped_vehicle':v,
            'LOVO_delta':float(q.groupby('vehicle_id')['delta'].mean().mean())
        })
    return full,rows

def apply_full_family_holm(df,pcol,outcol,expected_n):
    q=df.copy()
    assert len(q)==expected_n,(outcol,len(q),expected_n)
    q[outcol]=holm_adjust(q[pcol].to_numpy(float))
    return q

# %% Code cell 4 (notebook cell 5)

# ---------- Load actual prediction contracts ----------

r2c=build_analysis_segment(pd.read_parquet(R2C_PRED))
r2d=build_analysis_segment(pd.read_parquet(R2D_PRED))
r3a=build_analysis_segment(pd.read_parquet(R3A_PRED))
budget=pd.read_csv(R2C_BUDGET)

contracts=[
    ('R2C',r2c,{'domain','fraction','model','vehicle_id','compound_segment','target','prediction'}),
    ('R2D',r2d,{'domain','fraction','method','vehicle_id','compound_segment','target','prediction'}),
    ('R3A',r3a,{'domain','fraction','method','vehicle_id','compound_segment','target','prediction'}),
]
for name,df,req in contracts:
    miss=sorted(req-set(df.columns))
    assert not miss,(name,'missing columns',miss,df.columns.tolist())
    assert set(df.domain.astype(str))==set(DOMAINS),(name,sorted(df.domain.astype(str).unique()))
    assert np.isfinite(pd.to_numeric(df.target,errors='coerce')).all(),(name,'non-finite target')
    assert np.isfinite(pd.to_numeric(df.prediction,errors='coerce')).all(),(name,'non-finite prediction')
    df['abs_error']=np.abs(df.prediction.to_numpy(float)-df.target.to_numpy(float))


# Identity uniqueness: one corrected seed-averaged prediction per physical test row/method/fraction.
# R2D V4 input is the cross-seed aggregation fix, so duplicates here are a hard failure.
for name,df,method_col in [('R2C',r2c,'model'),('R2D',r2d,'method'),('R3A',r3a,'method')]:
    id_cols=['domain','fraction',method_col,'vehicle_id','analysis_segment']
    if 'time_value' in df.columns:
        id_cols.append('time_value')
    dup=int(df.duplicated(id_cols).sum())
    assert dup==0,(name,'duplicate seed-averaged prediction identities',dup,id_cols)

# Exact method/model names.
assert {R2C_DIRECT,R2C_PR}.issubset(set(r2c.model.astype(str)))
assert {R2D_OURS,*R2D_LIT}.issubset(set(r2d.method.astype(str)))
assert {R3A_GATED,R3A_ET}.issubset(set(r3a.method.astype(str)))

# Exact fraction contracts.
assert np.allclose(sorted(r2c.fraction.astype(float).unique()),R2C_FRACTIONS)
assert np.allclose(sorted(r2d.fraction.astype(float).unique()),ENDPOINT_FRACTIONS)
assert np.allclose(sorted(r3a.fraction.astype(float).unique()),ENDPOINT_FRACTIONS)

# Budget contract.
required_budget={'domain','fraction','effective_fit_fraction_mean','actual_nominal_fraction_mean'}
assert required_budget.issubset(budget.columns),(required_budget-set(budget.columns),budget.columns.tolist())
assert len(budget)==35
assert set(budget.domain.astype(str))==set(DOMAINS)

audit=[]
for name,df in [('R2C',r2c),('R2D',r2d),('R3A',r3a)]:
    for domain in DOMAINS:
        q=df[df.domain.astype(str)==domain]
        nv=q.vehicle_id.astype(str).nunique()
        audit.append({
            'experiment':name,'domain':domain,
            'n_vehicles':int(nv),
            'n_analysis_segments':int(q.analysis_segment.nunique()),
            'n_rows':int(len(q)),
            'population_inference_status':inference_status(nv),
        })

cluster_audit=pd.DataFrame(audit)
cluster_audit.to_csv(OUT/f'{EXPERIMENT}_CLUSTER_STRUCTURE_AUDIT.csv',index=False)
display(cluster_audit)

print('Prediction + budget contracts: PASS')

# %% Code cell 5 (notebook cell 6)

# ---------- R2C: hierarchical effective-supervision AULEC ----------

r2c_rows=[]
r2c_lovo=[]

for di,domain in enumerate(DOMAINS):
    q=r2c[
        (r2c.domain.astype(str)==domain) &
        r2c.model.isin([R2C_DIRECT,R2C_PR])
    ].copy()

    fracs=np.asarray(sorted(q.fraction.astype(float).unique()),float)
    assert np.allclose(fracs,R2C_FRACTIONS)

    b=budget[budget.domain.astype(str)==domain].sort_values('fraction')
    assert np.allclose(b.fraction.astype(float).to_numpy(),fracs)
    x=b.effective_fit_fraction_mean.to_numpy(float)

    s=(q.groupby(['vehicle_id','analysis_segment','fraction','model'],as_index=False)
         ['abs_error'].mean())

    vehicles=sorted(s.vehicle_id.astype(str).unique())
    per_v={}
    vehicle_deltas=[]
    all_D=[]; all_P=[]

    for v in vehicles:
        sv=s[s.vehicle_id.astype(str)==v]
        segs=sorted(sv.analysis_segment.unique())
        mats={}
        for model in [R2C_DIRECT,R2C_PR]:
            p=(sv[sv.model==model]
               .pivot(index='analysis_segment',columns='fraction',values='abs_error')
               .reindex(index=segs,columns=fracs))
            assert not p.isna().any().any(),(domain,v,model,'incomplete 7-point curve')
            mats[model]=p.to_numpy(float)

        D=mats[R2C_DIRECT]; P=mats[R2C_PR]
        per_v[v]=(D,P)
        all_D.append(D); all_P.append(P)
        vehicle_deltas.append(
            normalized_log_auc(x,P.mean(axis=0))-
            normalized_log_auc(x,D.mean(axis=0))
        )

    DD=np.vstack(all_D); PP=np.vstack(all_P)
    segment_point=normalized_log_auc(x,PP.mean(axis=0))-normalized_log_auc(x,DD.mean(axis=0))
    vehicle_point=float(np.mean(vehicle_deltas))

    rng=np.random.default_rng(51001+di*101)
    nv=len(vehicles)
    boot=np.empty(BOOT_DRAWS,float)
    for bb in range(BOOT_DRAWS):
        sampled_v=rng.integers(0,nv,size=nv)
        vals=[]
        for vi in sampled_v:
            D,P=per_v[vehicles[int(vi)]]
            ii=rng.integers(0,len(D),size=len(D))
            vals.append(
                normalized_log_auc(x,P[ii].mean(axis=0))-
                normalized_log_auc(x,D[ii].mean(axis=0))
            )
        boot[bb]=float(np.mean(vals))

    r2c_rows.append({
        'domain':domain,
        'n_vehicles':nv,
        'n_segments':int(len(DD)),
        'segment_balanced_delta_PR_minus_Direct_AULEC':float(segment_point),
        'vehicle_balanced_delta_PR_minus_Direct_AULEC':vehicle_point,
        'CI95_low':float(np.percentile(boot,2.5)),
        'CI95_high':float(np.percentile(boot,97.5)),
        'hierarchical_bootstrap_p':bootstrap_two_sided_p(boot),
        'exact_vehicle_signflip_p':exact_vehicle_signflip_p(vehicle_deltas),
        'population_inference_status':inference_status(nv),
        'population_inference_eligible':bool(nv>=MIN_VEHICLES_FOR_POPULATION_INFERENCE),
    })

    if nv>=2:
        full=float(np.mean(vehicle_deltas))
        for j,v in enumerate(vehicles):
            loo=float(np.mean([xv for k,xv in enumerate(vehicle_deltas) if k!=j]))
            r2c_lovo.append({
                'family':'R2C_AULEC','domain':domain,'fraction':np.nan,
                'comparator':'PR-minus-Direct','dropped_vehicle':v,
                'full_delta':full,'LOVO_delta':loo,
            })

r2c_cluster=pd.DataFrame(r2c_rows)
r2c_cluster=apply_full_family_holm(
    r2c_cluster,'hierarchical_bootstrap_p','Holm_p_hierarchical_full_family',5
)
r2c_cluster=apply_full_family_holm(
    r2c_cluster,'exact_vehicle_signflip_p','Holm_p_exact_signflip_full_family',5
)

r2c_cluster['PR_supported_population_cluster_bootstrap']=(
    r2c_cluster.population_inference_eligible &
    (r2c_cluster.CI95_high<0) &
    (r2c_cluster.Holm_p_hierarchical_full_family<0.05)
)
r2c_cluster['Direct_supported_population_cluster_bootstrap']=(
    r2c_cluster.population_inference_eligible &
    (r2c_cluster.CI95_low>0) &
    (r2c_cluster.Holm_p_hierarchical_full_family<0.05)
)

r2c_cluster.to_csv(OUT/f'{EXPERIMENT}_R2C_AULEC_CLUSTER_AWARE.csv',index=False)
display(r2c_cluster)

# %% Code cell 6 (notebook cell 7)

# ---------- R2D: hierarchical literature-adapted benchmark ----------

r2d_rows=[]
r2d_lovo=[]
counter=0

for lit in R2D_LIT:
    for domain in DOMAINS:
        for fraction in ENDPOINT_FRACTIONS:
            q=r2d[
                (r2d.domain.astype(str)==domain) &
                np.isclose(r2d.fraction.astype(float),fraction) &
                r2d.method.isin([R2D_OURS,lit])
            ].copy()

            s=(q.groupby(['vehicle_id','analysis_segment','method'],as_index=False)
                 ['abs_error'].mean())

            ours=s[s.method==R2D_OURS].set_index(['vehicle_id','analysis_segment'])['abs_error']
            comp=s[s.method==lit].set_index(['vehicle_id','analysis_segment'])['abs_error']
            common=ours.index.intersection(comp.index)
            assert len(common)>0,(domain,fraction,lit,'no common test segments')

            d=pd.DataFrame({
                'vehicle_id':[str(k[0]) for k in common],
                'analysis_segment':[str(k[1]) for k in common],
                # Positive delta => literature comparator has larger error => OURS better.
                'delta':comp.loc[common].to_numpy(float)-ours.loc[common].to_numpy(float),
            })

            hb=hierarchical_bootstrap_segment_delta(d,BOOT_DRAWS,61001+counter*53)
            counter+=1
            vdelta=d.groupby('vehicle_id')['delta'].mean().to_numpy(float)
            nv=hb['n_vehicles']

            r2d_rows.append({
                'domain':domain,'fraction':float(fraction),'literature_method':lit,
                'n_vehicles':nv,'n_segments':hb['n_segments'],
                'segment_balanced_delta_literature_minus_ours':float(d.delta.mean()),
                'vehicle_balanced_delta_literature_minus_ours':hb['vehicle_balanced_delta'],
                'CI95_low':hb['CI95_low'],'CI95_high':hb['CI95_high'],
                'hierarchical_bootstrap_p':hb['hierarchical_bootstrap_p'],
                'exact_vehicle_signflip_p':exact_vehicle_signflip_p(vdelta),
                'population_inference_status':inference_status(nv),
                'population_inference_eligible':bool(nv>=MIN_VEHICLES_FOR_POPULATION_INFERENCE),
            })

            full,lrows=leave_one_vehicle_out(d)
            for rr in lrows:
                r2d_lovo.append({
                    'family':'R2D','domain':domain,'fraction':float(fraction),
                    'comparator':lit,'dropped_vehicle':rr['dropped_vehicle'],
                    'full_delta':full,'LOVO_delta':rr['LOVO_delta'],
                })

r2d_cluster=pd.DataFrame(r2d_rows)
assert len(r2d_cluster)==30

r2d_cluster=apply_full_family_holm(
    r2d_cluster,'hierarchical_bootstrap_p','Holm_p_hierarchical_full_family',30
)
r2d_cluster=apply_full_family_holm(
    r2d_cluster,'exact_vehicle_signflip_p','Holm_p_exact_signflip_full_family',30
)

r2d_cluster['ours_supported_population_cluster_bootstrap']=(
    r2d_cluster.population_inference_eligible &
    (r2d_cluster.CI95_low>0) &
    (r2d_cluster.Holm_p_hierarchical_full_family<0.05)
)
r2d_cluster['literature_supported_population_cluster_bootstrap']=(
    r2d_cluster.population_inference_eligible &
    (r2d_cluster.CI95_high<0) &
    (r2d_cluster.Holm_p_hierarchical_full_family<0.05)
)
r2d_cluster['population_cluster_unresolved']=~(
    r2d_cluster.ours_supported_population_cluster_bootstrap |
    r2d_cluster.literature_supported_population_cluster_bootstrap
)

r2d_cluster.to_csv(OUT/f'{EXPERIMENT}_R2D_CLUSTER_AWARE.csv',index=False)
display(r2d_cluster)

# %% Code cell 7 (notebook cell 8)

# ---------- R3A: hierarchical Gated vs ET ----------

r3a_rows=[]
r3a_lovo=[]

for ci,(domain,fraction) in enumerate(
    [(d,f) for d in DOMAINS for f in ENDPOINT_FRACTIONS]
):
    q=r3a[
        (r3a.domain.astype(str)==domain) &
        np.isclose(r3a.fraction.astype(float),fraction) &
        r3a.method.isin([R3A_GATED,R3A_ET])
    ].copy()

    s=(q.groupby(['vehicle_id','analysis_segment','method'],as_index=False)
         ['abs_error'].mean())

    gated=s[s.method==R3A_GATED].set_index(['vehicle_id','analysis_segment'])['abs_error']
    et=s[s.method==R3A_ET].set_index(['vehicle_id','analysis_segment'])['abs_error']
    common=gated.index.intersection(et.index)
    assert len(common)>0,(domain,fraction,'no common test segments')

    d=pd.DataFrame({
        'vehicle_id':[str(k[0]) for k in common],
        'analysis_segment':[str(k[1]) for k in common],
        # Negative delta => Gated lower error => Gated better.
        'delta':gated.loc[common].to_numpy(float)-et.loc[common].to_numpy(float),
    })

    hb=hierarchical_bootstrap_segment_delta(d,BOOT_DRAWS,71001+ci*71)
    vdelta=d.groupby('vehicle_id')['delta'].mean().to_numpy(float)
    nv=hb['n_vehicles']

    r3a_rows.append({
        'domain':domain,'fraction':float(fraction),
        'n_vehicles':nv,'n_segments':hb['n_segments'],
        'segment_balanced_delta_gated_minus_ET':float(d.delta.mean()),
        'vehicle_balanced_delta_gated_minus_ET':hb['vehicle_balanced_delta'],
        'CI95_low':hb['CI95_low'],'CI95_high':hb['CI95_high'],
        'hierarchical_bootstrap_p':hb['hierarchical_bootstrap_p'],
        'exact_vehicle_signflip_p':exact_vehicle_signflip_p(vdelta),
        'population_inference_status':inference_status(nv),
        'population_inference_eligible':bool(nv>=MIN_VEHICLES_FOR_POPULATION_INFERENCE),
    })

    full,lrows=leave_one_vehicle_out(d)
    for rr in lrows:
        r3a_lovo.append({
            'family':'R3A','domain':domain,'fraction':float(fraction),
            'comparator':'Gated-minus-ET','dropped_vehicle':rr['dropped_vehicle'],
            'full_delta':full,'LOVO_delta':rr['LOVO_delta'],
        })

r3a_cluster=pd.DataFrame(r3a_rows)
assert len(r3a_cluster)==10

r3a_cluster=apply_full_family_holm(
    r3a_cluster,'hierarchical_bootstrap_p','Holm_p_hierarchical_full_family',10
)
r3a_cluster=apply_full_family_holm(
    r3a_cluster,'exact_vehicle_signflip_p','Holm_p_exact_signflip_full_family',10
)

r3a_cluster['gated_supported_population_cluster_bootstrap']=(
    r3a_cluster.population_inference_eligible &
    (r3a_cluster.CI95_high<0) &
    (r3a_cluster.Holm_p_hierarchical_full_family<0.05)
)
r3a_cluster['ET_supported_population_cluster_bootstrap']=(
    r3a_cluster.population_inference_eligible &
    (r3a_cluster.CI95_low>0) &
    (r3a_cluster.Holm_p_hierarchical_full_family<0.05)
)
r3a_cluster['population_cluster_unresolved']=~(
    r3a_cluster.gated_supported_population_cluster_bootstrap |
    r3a_cluster.ET_supported_population_cluster_bootstrap
)

r3a_cluster.to_csv(OUT/f'{EXPERIMENT}_R3A_CLUSTER_AWARE.csv',index=False)
display(r3a_cluster)

# %% Code cell 8 (notebook cell 9)

# ---------- Leave-one-vehicle-out stability ----------

lovo=pd.DataFrame(r2c_lovo+r2d_lovo+r3a_lovo)

if len(lovo):
    def same_direction(a,b):
        if not np.isfinite(a) or not np.isfinite(b) or abs(a)<=EPS or abs(b)<=EPS:
            return np.nan
        return bool(np.sign(a)==np.sign(b))

    lovo['same_direction_as_full']=[
        same_direction(a,b) for a,b in zip(lovo.full_delta,lovo.LOVO_delta)
    ]
    lovo.to_csv(OUT/f'{EXPERIMENT}_LEAVE_ONE_VEHICLE_OUT.csv',index=False)

    lovo_summary=(
        lovo.groupby(['family','domain','fraction','comparator'],dropna=False,as_index=False)
        .agg(
            n_LOVO=('dropped_vehicle','size'),
            full_delta=('full_delta','first'),
            LOVO_min=('LOVO_delta','min'),
            LOVO_max=('LOVO_delta','max'),
            same_direction_rate=('same_direction_as_full','mean'),
        )
    )
    lovo_summary.to_csv(OUT/f'{EXPERIMENT}_LEAVE_ONE_VEHICLE_OUT_SUMMARY.csv',index=False)
else:
    lovo_summary=pd.DataFrame()

print('LOVO rows:',len(lovo))
display(lovo_summary)

# %% Code cell 9 (notebook cell 10)

# ---------- Corrected segment-bootstrap vs V4 cluster-aware conclusions ----------

orig2c=pd.read_csv(R2C_PRIMARY)
orig2d=pd.read_csv(R2D_PRIMARY)
orig3a=pd.read_csv(R3A_PRIMARY)

# Strict source contracts before comparison.
assert len(orig2c)==5
assert len(orig2d)==30
assert len(orig3a)==10
assert {'domain','effective_AULEC_supported_after_Holm'}.issubset(orig2c.columns)
assert {'domain','fraction','literature_method','ours_supported_after_Holm','literature_supported_after_Holm'}.issubset(orig2d.columns)
assert {'domain','fraction','gated_supported_after_Holm','ET_supported_after_Holm'}.issubset(orig3a.columns)

cmp=[]

for r in r2c_cluster.itertuples(index=False):
    o=orig2c[orig2c.domain.astype(str)==str(r.domain)]
    assert len(o)==1
    cmp.append({
        'family':'R2C_AULEC','domain':r.domain,'fraction':np.nan,
        'comparator':'PR-minus-Direct','n_vehicles':r.n_vehicles,
        'population_inference_status':r.population_inference_status,
        'original_segment_supported':bool(o.iloc[0].effective_AULEC_supported_after_Holm),
        'population_cluster_supported':bool(r.PR_supported_population_cluster_bootstrap),
        'segment_delta':r.segment_balanced_delta_PR_minus_Direct_AULEC,
        'vehicle_balanced_delta':r.vehicle_balanced_delta_PR_minus_Direct_AULEC,
        'cluster_CI_low':r.CI95_low,'cluster_CI_high':r.CI95_high,
    })

for r in r2d_cluster.itertuples(index=False):
    o=orig2d[
        (orig2d.domain.astype(str)==str(r.domain)) &
        np.isclose(orig2d.fraction.astype(float),float(r.fraction)) &
        (orig2d.literature_method.astype(str)==str(r.literature_method))
    ]
    assert len(o)==1
    cmp.append({
        'family':'R2D','domain':r.domain,'fraction':r.fraction,
        'comparator':r.literature_method,'n_vehicles':r.n_vehicles,
        'population_inference_status':r.population_inference_status,
        'original_segment_supported':bool(
            o.iloc[0].ours_supported_after_Holm or
            o.iloc[0].literature_supported_after_Holm
        ),
        'population_cluster_supported':bool(
            r.ours_supported_population_cluster_bootstrap or
            r.literature_supported_population_cluster_bootstrap
        ),
        'segment_delta':r.segment_balanced_delta_literature_minus_ours,
        'vehicle_balanced_delta':r.vehicle_balanced_delta_literature_minus_ours,
        'cluster_CI_low':r.CI95_low,'cluster_CI_high':r.CI95_high,
    })

for r in r3a_cluster.itertuples(index=False):
    o=orig3a[
        (orig3a.domain.astype(str)==str(r.domain)) &
        np.isclose(orig3a.fraction.astype(float),float(r.fraction))
    ]
    assert len(o)==1
    cmp.append({
        'family':'R3A','domain':r.domain,'fraction':r.fraction,
        'comparator':'Gated-minus-ET','n_vehicles':r.n_vehicles,
        'population_inference_status':r.population_inference_status,
        'original_segment_supported':bool(
            o.iloc[0].gated_supported_after_Holm or
            o.iloc[0].ET_supported_after_Holm
        ),
        'population_cluster_supported':bool(
            r.gated_supported_population_cluster_bootstrap or
            r.ET_supported_population_cluster_bootstrap
        ),
        'segment_delta':r.segment_balanced_delta_gated_minus_ET,
        'vehicle_balanced_delta':r.vehicle_balanced_delta_gated_minus_ET,
        'cluster_CI_low':r.CI95_low,'cluster_CI_high':r.CI95_high,
    })

comparison=pd.DataFrame(cmp)
assert len(comparison)==5+30+10
comparison.to_csv(OUT/f'{EXPERIMENT}_ORIGINAL_VS_CLUSTER_AWARE.csv',index=False)
display(comparison)

# %% Code cell 10 (notebook cell 11)

# ---------- Final fail-closed manifest + manuscript notes ----------

assert len(r2c_cluster)==5
assert len(r2d_cluster)==30
assert len(r3a_cluster)==10
assert len(comparison)==45

# Full-family adjusted p values must be finite and in [0,1].
for df,col in [
    (r2c_cluster,'Holm_p_hierarchical_full_family'),
    (r2d_cluster,'Holm_p_hierarchical_full_family'),
    (r3a_cluster,'Holm_p_hierarchical_full_family'),
]:
    x=df[col].to_numpy(float)
    assert np.isfinite(x).all() and np.all((x>=0)&(x<=1))

eligible2c=int(r2c_cluster.population_inference_eligible.sum())
eligible2d=int(r2d_cluster.population_inference_eligible.sum())
eligible3a=int(r3a_cluster.population_inference_eligible.sum())

manifest={
    'experiment':EXPERIMENT,
    'status':'R4_CLUSTER_AWARE_HIERARCHICAL_INFERENCE_COMPLETE',
    'code_version':CODE_VERSION,
    'source_status':{
        'R2C':r2c_source_status,
        'R2C_completion_source':r2c_completion_source,
        'R2D_cross_seed_aggregation':m2d.get('status'),
        'R3A':m3a.get('status'),
    },
    'retraining_performed':False,
    'R2D_seed_aggregation_source':'R2D_CROSS_SEED_AGGREGATION_FIX_COMPLETE',
    'R2D_soc_phy_used_as_cross_seed_grouping_key':False,
    'highest_identifiable_cluster':'vehicle_id',
    'nested_unit':'analysis_segment = vehicle_id || segment_id',
    'point_estimand':'equal-vehicle mean of within-vehicle segment effects',
    'bootstrap':'two-stage vehicle then segment-within-vehicle bootstrap',
    'bootstrap_draws':BOOT_DRAWS,
    'additional_small_cluster_test':'exact paired sign-flip over vehicle-level effects',
    'minimum_vehicles_for_population_inference_reporting_convention':MIN_VEHICLES_FOR_POPULATION_INFERENCE,
    'multiplicity_policy':'Holm applied to full predeclared families before population-support eligibility filter',
    'holm_family_sizes':{'R2C':5,'R2D':30,'R3A':10},
    'R2C_tests':5,
    'R2C_population_eligible':eligible2c,
    'R2C_PR_supported_population_cluster_bootstrap':int(r2c_cluster.PR_supported_population_cluster_bootstrap.sum()),
    'R2D_tests':30,
    'R2D_population_eligible':eligible2d,
    'R2D_ours_supported_population_cluster_bootstrap':int(r2d_cluster.ours_supported_population_cluster_bootstrap.sum()),
    'R2D_literature_supported_population_cluster_bootstrap':int(r2d_cluster.literature_supported_population_cluster_bootstrap.sum()),
    'R3A_tests':10,
    'R3A_population_eligible':eligible3a,
    'R3A_gated_supported_population_cluster_bootstrap':int(r3a_cluster.gated_supported_population_cluster_bootstrap.sum()),
    'R3A_ET_supported_population_cluster_bootstrap':int(r3a_cluster.ET_supported_population_cluster_bootstrap.sum()),
    'EVE_across_vehicle_population_inference':False,
    'all_results_reported':True,
}

with open(OUT/f'{EXPERIMENT}_MANIFEST.json','w') as f:
    json.dump(manifest,f,indent=2)

notes=[
    '# R4 Cluster-Aware / Hierarchical Inference — V2','',
    '## What was re-analysed','',
    '- No model was retrained.',
    '- Corrected R2C, **cross-seed-corrected R2D-C**, and R3A five-seed averaged TEST predictions were reused.',
    '- Vehicle was treated as the highest identifiable independent cluster; segments were nested within vehicle.',
    '- Point estimates use equal vehicle weighting, not weighting vehicles by their number of segments.',
    '- Uncertainty uses a two-stage vehicle→segment hierarchical bootstrap.',
    '- Exact vehicle-level sign-flip and leave-one-vehicle-out analyses are additional sensitivity checks.','',
    '## Multiplicity','',
    '- Holm correction is applied to the complete predeclared families: R2C=5, R2D=30, R3A=10.',
    f'- Across-vehicle population-support wording is permitted only for contrasts with at least {MIN_VEHICLES_FOR_POPULATION_INFERENCE} distinct vehicles.','',
    '## R2C','',
    f'- Population-eligible domains: {eligible2c}/5.',
    f"- PR supported after full-family Holm + population eligibility: {int(r2c_cluster.PR_supported_population_cluster_bootstrap.sum())}/{eligible2c if eligible2c else 0}.",'',
    '## R2D','',
    f'- Population-eligible contrasts: {eligible2d}/30.',
    f"- OURS supported: {int(r2d_cluster.ours_supported_population_cluster_bootstrap.sum())}/{eligible2d if eligible2d else 0}.",
    f"- Literature comparator supported: {int(r2d_cluster.literature_supported_population_cluster_bootstrap.sum())}/{eligible2d if eligible2d else 0}.",'',
    '## R3A','',
    f'- Population-eligible contrasts: {eligible3a}/10.',
    f"- Gated supported: {int(r3a_cluster.gated_supported_population_cluster_bootstrap.sum())}/{eligible3a if eligible3a else 0}.",
    f"- ET supported: {int(r3a_cluster.ET_supported_population_cluster_bootstrap.sum())}/{eligible3a if eligible3a else 0}.",'',
    '## Boundary','',
    '- Segment-bootstrap inference remains valid as uncertainty over the observed segment corpus.',
    '- Cluster-aware inference is the stricter sensitivity analysis for statements intended to generalize across vehicles.',
    '- EVE is single-vehicle and historically opened/post-confirmatory; no across-vehicle inferential claim is made from EVE.',
]
(OUT/f'{EXPERIMENT}_MANUSCRIPT_NOTES.md').write_text('\n'.join(notes),encoding='utf-8')

print(json.dumps(manifest,indent=2))
print('\n'.join(notes))

# %% [markdown] Expected outputs
