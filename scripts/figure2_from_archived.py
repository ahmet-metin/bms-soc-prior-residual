"""Regenerate Figure 2 from archived R7 and post-anchor EVE AGGREGATE CSVs (NOT fresh rerun).
Run: python regenerate_figure2_postanchor.py  (requires pandas/numpy/matplotlib)
Does not retrain models or change any other manuscript files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
r7=pd.read_csv(root.parent/'references/R7_35_conditions_ARCHIVED.csv')
eve=pd.read_csv(root.parent/'references/EVE_postanchor_14_conditions_ARCHIVED.csv')
fig,axs=plt.subplots(2,3,figsize=(12.4,7.15),constrained_layout=True)
for i,dom in enumerate(['VED','KU','SCUT','TUM','EVE']):
    ax=axs.flat[i]
    sub=r7[r7.domain==dom].sort_values('effective_fit_fraction_mean')
    xx=sub.effective_fit_fraction_mean.to_numpy(float)*100
    if dom=='EVE':
        direct=eve[eve.model=='Direct+Prior'].sort_values('fraction').segment_balanced_MAE_after.to_numpy(float)
        pr=eve[eve.model=='PriorResidual+Prior'].sort_values('fraction').segment_balanced_MAE_after.to_numpy(float)
    else:
        direct=sub.Direct_segment_balanced_MAE.to_numpy(float)
        pr=sub.PR_segment_balanced_MAE.to_numpy(float)
    ax.plot(xx,direct,'-o',lw=1.6,markersize=4,label='Direct+Prior')
    ax.plot(xx,pr,'-o',lw=1.6,markersize=4,label='PriorResidual+Prior')
    ax.set_title(dom,fontsize=13,pad=4)
    ax.grid(True,axis='y',lw=0.6,alpha=0.42)
    ax.set_xlabel('Effective fit observations (%)',fontsize=10)
    if i in (0,3):ax.set_ylabel('Segment-balanced MAE (SoC pp)',fontsize=10)
    if xx.max()/xx.min()>10:ax.set_xscale('log')
    else:ax.set_xlim(max(0,xx.min()*0.94),103)
    ax.tick_params(axis='both',labelsize=9)
axs.flat[5].axis('off')
h,l=axs.flat[0].get_legend_handles_labels()
axs.flat[5].legend(h,l,loc='center',frameon=False,fontsize=11)
out=root.parent/'outputs'/'Figure2_LabelEfficiency_5panel.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,bbox_inches='tight')
print('ARCHIVED aggregate Figure2 output:',out)
plt.close(fig)
