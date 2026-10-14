"""Recompute R7 equal-segment label-efficiency summaries from *frozen* seed-averaged predictions.

No training, fitting, model choice, or EVE retesting is performed here.
Input predictions must have one mean-over-seeds prediction per physical test row.
EVE test starts are excluded *only* for the REV8 post-anchor primary estimand.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DOMAINS = ('VED','KU','SCUT','TUM','EVE')
FRACTIONS = (0.01,0.02,0.05,0.10,0.20,0.50,1.0)
MODELS = ('Direct+Prior','PriorResidual+Prior')
PAIR_KEYS = ['domain','fraction','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target']

def require_columns(df, required, description):
    missing = sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'{description}: missing fields: {missing}')

def normalize_predictions(predictions: pd.DataFrame, *, eve_postanchor: bool = True,
                          strict_grid: bool = True) -> pd.DataFrame:
    """Validate exact paired test rows without leaking seed-level replicates into observations."""
    require_columns(predictions, PAIR_KEYS+['model','prediction'], 'R2C seed-averaged predictions')
    z = predictions.copy()
    z['domain'] = z['domain'].astype(str)
    z['model'] = z['model'].astype(str)
    z['fraction'] = pd.to_numeric(z['fraction'],errors='raise').astype(float)
    for col in ('target','prediction','elapsed_s'):
        z[col] = pd.to_numeric(z[col],errors='raise').astype(float)
        if not np.isfinite(z[col]).all(): raise ValueError(f'{col}: nonfinite entries')
    if (z['elapsed_s'] < -1e-9).any(): raise ValueError('negative elapsed seconds')
    for f in z['fraction'].unique():
        if not any(np.isclose(f,x,atol=1e-10,rtol=0) for x in FRACTIONS):
            raise ValueError(f'Unrecognized nominal label fraction: {f}')
    z['fraction']=z['fraction'].map(lambda f:min(FRACTIONS,key=lambda a:abs(a-f)))
    if (~z['domain'].isin(DOMAINS)).any() or (~z['model'].isin(MODELS)).any():
        raise ValueError('Unknown domain or model type')
    if z.duplicated(PAIR_KEYS+['model']).any():
        raise ValueError('Duplicate physical predictions; pass seed-AVERAGED not seed-level predictions')
    matches = z.groupby(PAIR_KEYS,dropna=False)['model'].nunique()
    if not (matches==2).all():
        raise ValueError('Direct+Prior and PriorResidual+Prior must cover identical physical rows')
    if strict_grid:
        grid={(d,f,m) for d in DOMAINS for f in FRACTIONS for m in MODELS}
        actual=set(zip(z.domain,z.fraction,z.model))
        if grid!=actual: raise ValueError(f'Expected 70 domain/fraction/model conditions; got {len(actual)}, missing {sorted(grid-actual)[:5]}')
    if eve_postanchor:
        # EVE 12C's original observed segment-start BMS anchor is not a scored prediction.
        z=z[(z['domain']!='EVE') | (z['elapsed_s']>1e-9)].copy()
        if not z.loc[z.domain=='EVE'].empty and strict_grid:
            if z.loc[z.domain=='EVE','compound_segment'].nunique()!=509:
                raise ValueError('REV8 expects 509 post-anchor-evaluable EVE segments; input/preprocessing differs')
    return z

def normalized_log_auc(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    order=np.argsort(x);x=x[order];y=y[order]
    if len(x)!=7 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(x<=0) or not np.all(np.diff(x)>0):
        raise ValueError('AULEC requires exactly seven strictly increasing positive observation rates and finite errors')
    logx=np.log10(x)
    trapz=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    return float(trapz(y,logx)/(logx[-1]-logx[0]))

def summarize_r7(predictions:pd.DataFrame, budget:pd.DataFrame, *, eve_postanchor=True,
                 strict_grid=True):
    z=normalize_predictions(predictions,eve_postanchor=eve_postanchor,strict_grid=strict_grid)
    require_columns(budget,['domain','fraction','effective_fit_fraction_mean','actual_nominal_fraction_mean'], 'Effective fit budget')
    b=budget.copy();b['fraction']=b['fraction'].astype(float)
    if b.duplicated(['domain','fraction']).any():raise ValueError('Duplicate effective budget condition')
    z['abs_error']=np.abs(z['prediction']-z['target'])
    seg=(z.groupby(['domain','fraction','model','compound_segment'],as_index=False)
           .agg(segment_MAE=('abs_error','mean'),n_rows=('abs_error','size')))
    long=(seg.groupby(['domain','fraction','model'],as_index=False)
          .agg(n_segments=('compound_segment','nunique'),n_test_rows=('n_rows','sum'),
               segment_balanced_MAE=('segment_MAE','mean'),segment_MAE_std=('segment_MAE','std'),
               segment_MAE_median=('segment_MAE','median')))
    wide=(long.pivot(index=['domain','fraction'],columns='model',values='segment_balanced_MAE')
          .reset_index().rename(columns={MODELS[0]:'Direct_segment_balanced_MAE',MODELS[1]:'PR_segment_balanced_MAE'}))
    required_budget=['domain','fraction','actual_nominal_fraction_mean','effective_fit_fraction_mean']
    optional_budget=['anchor_overhead_pp_mean','n_selected_labels_mean','n_unique_fit_observations_mean']
    cols=required_budget+[x for x in optional_budget if x in b]
    wide=wide.merge(b[cols],on=['domain','fraction'],how='left',validate='one_to_one')
    if wide['effective_fit_fraction_mean'].isna().any():raise ValueError('Missing effective-budget mapping')
    counts=(long[long.model==MODELS[0]][['domain','fraction','n_segments','n_test_rows']])
    wide=wide.merge(counts,on=['domain','fraction'],validate='one_to_one')
    wide['Delta_PR_minus_Direct']=wide['PR_segment_balanced_MAE']-wide['Direct_segment_balanced_MAE']
    wide['PR_relative_improvement_pct']=100*(1-wide['PR_segment_balanced_MAE']/wide['Direct_segment_balanced_MAE'])
    wide['PR_lower_segment_balanced_MAE']=wide['Delta_PR_minus_Direct']<0
    wide['nominal_pct']=wide['actual_nominal_fraction_mean']*100
    wide['effective_pct']=wide['effective_fit_fraction_mean']*100
    wide['_order']=wide.domain.map({d:i for i,d in enumerate(DOMAINS)})
    wide=wide.sort_values(['_order','fraction']).drop(columns=['_order']).reset_index(drop=True)
    if strict_grid and (len(wide)!=35 or len(long)!=70):raise ValueError('Incomplete 35/70-condition grid')
    auc=[]
    for domain,g in wide.groupby('domain',sort=False):
        if len(g)!=7:
            if strict_grid:raise ValueError(f'Missing AULEC conditions in {domain}')
            continue
        x=g['effective_fit_fraction_mean']
        auc.append({'domain':domain,'Direct_effective_AULEC_segment_balanced':normalized_log_auc(x,g['Direct_segment_balanced_MAE']),
                    'PR_effective_AULEC_segment_balanced':normalized_log_auc(x,g['PR_segment_balanced_MAE'])})
    return long,wide,pd.DataFrame(auc)

def main(argv=None):
    p=argparse.ArgumentParser(description='Recompute R7 from 5-seed-averaged frozen R2C predictions, not from archived summary values')
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--budget',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--include-eve-anchor',action='store_true',help='Reproduce historical anchor-INCLUDED analysis, NOT REV8 primary')
    args=p.parse_args(argv)
    long,wide,auc=summarize_r7(pd.read_parquet(args.predictions),pd.read_csv(args.budget),eve_postanchor=not args.include_eve_anchor)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    long.to_csv(args.output_dir/'R7_70_conditions.csv',index=False)
    wide.to_csv(args.output_dir/'R7_35_conditions.csv',index=False)
    auc.to_csv(args.output_dir/'R7_AULEC_computed.csv',index=False)
    (args.output_dir/'analysis_manifest.json').write_text(json.dumps({'analysis':'R7','source_predictions':str(args.predictions),'source_budget':str(args.budget),
       'EVE_postanchor':not args.include_eve_anchor,'training_performed':False,'inference_from_predictions':True},indent=2))
    print(auc.to_string(index=False))
if __name__=='__main__':main()
