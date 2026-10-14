"""Causal periodic observation-cost sensitivity (Rev3B) from frozen PR predictions.

Observed BMS labels are accessed ONLY at scheduled refresh rows. Refresh rows are
excluded from both matched comparison errors. This is a *post-hoc descriptive*
simulation on already-opened BMS test sets, NOT label-free prediction.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from .postprocess import DOMAINS, require_columns, normalize_predictions

POLICIES={'20min':1200.0,'10min':600.0,'5min':300.0}

def scheduled_refresh_positions(rows,seconds):
    if seconds<=0:raise ValueError('Refresh period must be positive')
    times=rows['elapsed_s'].to_numpy(float)
    if len(times)==0:return []
    if np.any(np.diff(times)<0):raise ValueError('Rows must be sorted by elapsed time')
    if times[-1]<seconds:return []
    schedule=np.arange(seconds,times[-1]+1e-9,seconds)
    positions=[]
    for t in schedule:
        j=int(np.searchsorted(times,t,side='left'))
        if j<len(times):positions.append(j)
    # Physical BMS start-anchor is not a periodic refresh event.
    return sorted(set(positions))

def apply_refresh_policy(rows,seconds):
    z=rows.sort_values(['elapsed_s','time_value'],kind='mergesort').reset_index(drop=True).copy()
    raw=z.prediction.to_numpy(float); target=z.target.to_numpy(float)
    elapsed=z.elapsed_s.to_numpy(float)
    picks=set(scheduled_refresh_positions(z,seconds))
    new=raw.copy();flags=np.zeros(len(z),bool);offset=0.0
    for i in range(len(z)):
        if i in picks:
            # One actual BMS observation is acquired here and counted as cost.
            offset=float(target[i]-raw[i]);flags[i]=True
        new[i]=raw[i]+offset
    z['prediction_no_refresh']=raw;z['prediction_refreshed']=new;z['refresh_event']=flags
    if not np.allclose(new[flags],target[flags],rtol=0,atol=1e-10):
        raise ValueError('Refresh offset fails to reproduce the BMS observation at the refresh instant')
    return z

def summarize_periodic_refresh(predictions,*,strict_grid=True):
    # Paired Direct/PR validates physical rows; only PR is refreshed for Rev3B.
    p=normalize_predictions(predictions,eve_postanchor=False,strict_grid=strict_grid)
    p=p[p.model.eq('PriorResidual+Prior') & p.fraction.isin([0.1,1.0])].copy()
    expected={(d,f) for d in DOMAINS for f in (0.1,1.0)}
    if strict_grid and set(zip(p.domain,p.fraction)) != expected:
        raise ValueError('Incomplete 10-condition refresh policy grid')
    if p.duplicated(['domain','fraction','compound_segment','time_value']).any():
        raise ValueError('Duplicate test row in seed-averaged inputs')
    rows=[]
    for (d,f,seg),g in p.groupby(['domain','fraction','compound_segment'],sort=False):
        for policy,period in POLICIES.items():
            z=apply_refresh_policy(g,period)
            z['policy']=policy;rows.append(z)
    all_rows=pd.concat(rows,ignore_index=True)
    unobserved=all_rows[~all_rows.refresh_event].copy()
    unobserved['baseline_abs_error']=(unobserved.target-unobserved.prediction_no_refresh).abs()
    unobserved['refreshed_abs_error']=(unobserved.target-unobserved.prediction_refreshed).abs()
    # Each baseline/refreshed error must be calculated on the SAME policy-specific unobserved rows.
    seg=(unobserved.groupby(['domain','fraction','compound_segment','policy'],as_index=False)
         .agg(baseline_MAE=('baseline_abs_error','mean'),refreshed_MAE=('refreshed_abs_error','mean'),
              n_unobserved=('target','size')))
    seg['gain']=seg.baseline_MAE-seg.refreshed_MAE
    sm=(seg.groupby(['domain','fraction','policy'],as_index=False)
        .agg(n_segments=('compound_segment','nunique'),
             matched_no_refresh_MAE=('baseline_MAE','mean'),
             refreshed_MAE=('refreshed_MAE','mean'),
             improvement_pp=('gain','mean')))
    sm['relative_improvement_pct']=100*sm.improvement_pp/sm.matched_no_refresh_MAE
    # Exposure is counted once per physical segment, not once per policy or model.
    exposure=(p.groupby(['domain','fraction','compound_segment'],as_index=False)
              .agg(duration_s=('elapsed_s',lambda x:float(np.max(x)))))
    duration=exposure.groupby(['domain','fraction'],as_index=False).duration_s.sum().rename(columns={'duration_s':'exposure_s'})
    acquisitions=(all_rows.groupby(['domain','fraction','policy'],as_index=False)
                  .agg(refresh_events=('refresh_event','sum')))
    sm=sm.merge(acquisitions,on=['domain','fraction','policy'],validate='one_to_one').merge(duration,on=['domain','fraction'],validate='many_to_one')
    if (sm.exposure_s<=0).any():raise ValueError('Nonpositive test acquisition exposure')
    sm['additional_BMS_observations_per_hour']=sm.refresh_events*3600/sm.exposure_s
    nseg=exposure.groupby(['domain','fraction'],as_index=False).agg(initial_anchor_observations=('compound_segment','nunique'))
    sm=sm.merge(nseg,on=['domain','fraction'],validate='many_to_one')
    sm['total_BMS_observations_per_hour']=(sm.refresh_events+sm.initial_anchor_observations)*3600/sm.exposure_s
    return sm,seg

def main(argv=None):
    p=argparse.ArgumentParser(description='Rev3B fixed 20/10/5-min causal refresh sensitivity; no retraining')
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    sm,seg=summarize_periodic_refresh(pd.read_parquet(args.predictions))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sm.to_csv(args.output_dir/'Rev3B_matched_refresh_summary.csv',index=False)
    seg.to_csv(args.output_dir/'Rev3B_matched_segment_errors.csv',index=False)
    print(f'PASS: {len(sm)} fixed domain/fraction/policy contrasts (descriptive; not fresh confirmatory testing)')
if __name__=='__main__':main()
