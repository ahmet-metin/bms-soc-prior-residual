"""R6 anchor-age descriptive analysis from matched, seed-averaged R2C predictions.

Faithful to the frozen R6 bin edges, segment-balanced MAE and paired percentile
bootstrap. Confidence intervals describe observed segment variability; not
fleet-wide inference or a null-centered hypothesis test.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .postprocess import require_columns,MODELS,DOMAINS,normalize_predictions

AGE_BINS=[0,5,10,20,40,np.inf]
AGE_LABELS=['0–5','5–10','10–20','20–40','≥40']

def bootstrap_segment_delta(x,draws=10000,seed=60261):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)==0:return float('nan'),float('nan'),float('nan')
    point=float(x.mean())
    if len(x)<2:return point,float('nan'),float('nan')
    rng=np.random.default_rng(seed)
    # Avoid allocating huge draws × segment arrays (for a Colab RAM-limited run).
    pieces=[]
    for start in range(0,draws,128):
        idx=rng.integers(0,len(x),size=(min(128,draws-start),len(x)))
        pieces.extend(x[idx].mean(axis=1))
    return point,*map(float,np.percentile(pieces,[2.5,97.5]))

def summarize_anchor_age(predictions,*,draws=10000,strict_grid=True,min_segments=10):
    pred=normalize_predictions(predictions,eve_postanchor=False,strict_grid=strict_grid)
    pred=pred[pred.fraction.isin([0.1,1.0])].copy()
    keys=['domain','fraction','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target']
    wide=pred.pivot(index=keys,columns='model',values='prediction').reset_index()
    if wide[list(MODELS)].isna().any().any():raise ValueError('Unpaired benchmark test rows')
    wide['Direct_abs_error']=(wide[MODELS[0]]-wide.target).abs()
    wide['PR_abs_error']=(wide[MODELS[1]]-wide.target).abs()
    wide['anchor_age_min']=wide.elapsed_s/60
    wide['anchor_age_bin']=pd.cut(wide.anchor_age_min,AGE_BINS,labels=AGE_LABELS,right=False,include_lowest=True)
    if wide.anchor_age_bin.isna().any():raise ValueError('Elapsed-time outside allowed bins')
    seg=(wide.groupby(['domain','fraction','vehicle_id','compound_segment','anchor_age_bin'],observed=True,as_index=False)
         .agg(Direct_MAE=('Direct_abs_error','mean'),PR_MAE=('PR_abs_error','mean'),n_rows=('target','size')))
    seg['Delta_PR_minus_Direct']=seg.PR_MAE-seg.Direct_MAE
    summary=(seg.groupby(['domain','fraction','anchor_age_bin'],observed=True,as_index=False)
             .agg(n_segments=('compound_segment','nunique'),n_vehicles=('vehicle_id','nunique'),
                  n_rows=('n_rows','sum'),Direct_segment_balanced_MAE=('Direct_MAE','mean'),
                  PR_segment_balanced_MAE=('PR_MAE','mean'),Delta_PR_minus_Direct=('Delta_PR_minus_Direct','mean')))
    summary['interpretable_bin']=summary.n_segments>=min_segments
    summary['PR_relative_improvement_pct']=100*(1-summary.PR_segment_balanced_MAE/summary.Direct_segment_balanced_MAE)
    ci=[]
    for (domain,fraction,age_bin),g in seg.groupby(['domain','fraction','anchor_age_bin'],observed=True):
        rngseed=60261+DOMAINS.index(domain)*1000+int(round(float(fraction)*100))*10+AGE_LABELS.index(str(age_bin))
        _,lo,hi=bootstrap_segment_delta(g.Delta_PR_minus_Direct,draws=draws,seed=rngseed)
        ci.append({'domain':domain,'fraction':fraction,'anchor_age_bin':age_bin,'CI95_low':lo,'CI95_high':hi})
    summary=summary.merge(pd.DataFrame(ci),on=['domain','fraction','anchor_age_bin'],validate='one_to_one')
    return seg,summary

def farthest_supported_age_bin(summary):
    """One manuscript-facing R6 robustness row per domain and supervision fraction."""
    rows=[]
    for (domain,fraction),g in summary.groupby(['domain','fraction'],sort=False):
        gi=g[g.interpretable_bin].copy()
        gi['_order']=gi.anchor_age_bin.astype(str).map({b:i for i,b in enumerate(AGE_LABELS)})
        gi=gi.sort_values('_order')
        far=gi.iloc[-1] if len(gi) else None
        rows.append({'domain':domain,'fraction':float(fraction),
            'n_interpretable_bins':int(len(gi)),
            'PR_better_bins':int((gi.Delta_PR_minus_Direct<0).sum()),
            'farthest_interpretable_bin':str(far.anchor_age_bin) if far is not None else '',
            'farthest_bin_delta':float(far.Delta_PR_minus_Direct) if far is not None else float('nan')})
    return pd.DataFrame(rows)

def main(argv=None):
    p=argparse.ArgumentParser(description='R6 descriptive anchor-age robustness from fixed five-seed mean predictions')
    p.add_argument('--predictions',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--draws',type=int,default=10000)
    args=p.parse_args(argv)
    if args.draws<1: p.error('draws must be >=1')
    seg,summ=summarize_anchor_age(pd.read_parquet(args.predictions),draws=args.draws)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    seg.to_csv(args.output_dir/'R6_segment_age.csv',index=False)
    summ.to_csv(args.output_dir/'R6_anchor_age_summary.csv',index=False)
    farthest_supported_age_bin(summ).to_csv(args.output_dir/'R6_domain_farthest_supported_bin.csv',index=False)
    print(f'Saved {len(summ)} age bins to {args.output_dir}. Descriptive CIs, not fleet inference.')
if __name__=='__main__':main()
