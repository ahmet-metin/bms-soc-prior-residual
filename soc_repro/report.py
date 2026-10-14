"""One-command post-training manuscript analysis/figure generation.

Input is the frozen R2C five-seed-AVERAGED prediction parquet plus effective
training-observation budget. This module NEVER triggers training.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .postprocess import summarize_r7, DOMAINS
from .robustness import summarize_anchor_age, AGE_LABELS, farthest_supported_age_bin
from .reanchoring import summarize_periodic_refresh

def figure2(wide, output):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12.4,7.15),constrained_layout=True)
    for i,domain in enumerate(DOMAINS):
        ax=axes.flat[i]
        g=wide[wide.domain==domain].sort_values('fraction')
        x=g.effective_fit_fraction_mean.to_numpy(float)*100
        ax.plot(x,g.Direct_segment_balanced_MAE,'-o',markersize=4,linewidth=1.6,label='Direct+Prior')
        ax.plot(x,g.PR_segment_balanced_MAE,'-o',markersize=4,linewidth=1.6,label='PriorResidual+Prior')
        ax.set_title(domain);ax.set_xlabel('Effective fit observations (%)')
        if i in (0,3):ax.set_ylabel('Segment-balanced MAE (SoC pp)')
        if x.max()/x.min()>10:ax.set_xscale('log')
        ax.grid(alpha=.3)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)

def figure3(age_summary, output):
    import matplotlib.pyplot as plt
    fig,axarr=plt.subplots(2,5,figsize=(17,6.5),sharey=False,constrained_layout=True)
    for j,frac in enumerate((0.1,1.0)):
        for k,domain in enumerate(DOMAINS):
            ax=axarr[j,k]
            g=age_summary[(age_summary.domain==domain)&np.isclose(age_summary.fraction,frac)].copy()
            g['order']=g.anchor_age_bin.astype(str).map({b:i for i,b in enumerate(AGE_LABELS)})
            g=g.sort_values('order')
            ax.plot(g.order,g.Direct_segment_balanced_MAE,'-o',label='Direct+Prior')
            ax.plot(g.order,g.PR_segment_balanced_MAE,'-o',label='PriorResidual+Prior')
            ax.set_title(f'{domain} · {frac:g} labels')
            ax.set_xticks(range(5),AGE_LABELS,rotation=45)
            if k==0:ax.set_ylabel('Segment-balanced MAE (SoC pp)')
    axarr[0,0].legend(fontsize=7)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)

def figure4(reanchor_summary, output):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for j,frac in enumerate((0.1,1.0)):
        ax=axs[j]
        for domain in DOMAINS:
            g=reanchor_summary[(reanchor_summary.domain==domain)&np.isclose(reanchor_summary.fraction,frac)].sort_values('additional_BMS_observations_per_hour')
            ax.plot(g.additional_BMS_observations_per_hour,g.relative_improvement_pct,'-o',label=domain)
        ax.set_title(f'{frac:g} retained label fraction');ax.set_xlabel('Additional trusted BMS observations / exposure hour')
        if j==0:ax.set_ylabel('MAE improvement vs matched unobserved rows (%)')
        ax.grid(alpha=.25)
    axs[1].legend(loc='best',fontsize=8)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description='Generate REV8 Figures 2/3/4 and R7/R6/Rev3B derived tables from predictions, without retraining')
    p.add_argument('--predictions',type=Path,required=True,help='R2C five-seed-averaged per-physical-row predictions')
    p.add_argument('--budget',type=Path,required=True,help='R2C effective observation budget summary')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--bootstrap-draws',type=int,default=10000)
    p.add_argument('--figure-format',choices=['pdf','png'],default='pdf')
    args=p.parse_args(argv)
    if args.bootstrap_draws<1:p.error('bootstrap-draws must be positive')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    prediction=pd.read_parquet(args.predictions)
    long,wide,aulec=summarize_r7(prediction,pd.read_csv(args.budget),eve_postanchor=True)
    long.to_csv(out/'Table_R7_Long70.csv',index=False)
    wide.to_csv(out/'Table_R7_35conditions_REV8.csv',index=False)
    aulec.to_csv(out/'Table_AULEC_from_predictions.csv',index=False)
    figure2(wide,out/f'Figure2_LabelEfficiency_5panel.{args.figure_format}')
    _,age=summarize_anchor_age(prediction,draws=args.bootstrap_draws)
    age.to_csv(out/'Table_AnchorAge_R6.csv',index=False)
    farthest_supported_age_bin(age).to_csv(out/'Table_AnchorAge_FarthestBin_R6.csv',index=False)
    figure3(age,out/f'Figure3_AnchorAge_Combined.{args.figure_format}')
    refresh,seg=summarize_periodic_refresh(prediction)
    refresh.to_csv(out/'Table_Reanchoring_Rev3B.csv',index=False)
    seg.to_csv(out/'Table_Reanchoring_Segments_Rev3B.csv',index=False)
    figure4(refresh,out/f'Figure4_Reanchoring_CostFidelity.{args.figure_format}')
    (out/'PROVENANCE.json').write_text(json.dumps({
        'R2C_predictions_source':str(args.predictions),'R2C_effective_budget_source':str(args.budget),
        'analysis':'from existing seed-averaged test predictions','retrained_models':False,
        'EVE_postanchor_primary_only_for_figure2':True,
        'EVE_one_shot_confirmatory_reopened':False,
        'figures_covered':[2,3,4],
        'limits':'Figure1 is a study-design artwork, trajectories and ΔSoC are separate. R2D/R3A/R4 are separate training/ensemble notebooks.'},indent=2))
    print('PASS: exported R7, R6, Rev3B analyses and Figures 2/3/4 from provided frozen predictions.')
    print(aulec.to_string(index=False))
if __name__=='__main__':main()
