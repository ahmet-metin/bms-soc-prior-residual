#!/usr/bin/env python3
"""Recompute archived primary AULEC; does not retrain ML models."""
import csv, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DIR=ROOT/'references'
def rows(p):
 with open(p,newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def auc(x,y):
 assert len(x)==len(y)==7
 assert all(x[i+1]>x[i]>0 for i in range(6))
 lx=list(map(math.log,x))
 return sum((lx[i+1]-lx[i])*(y[i]+y[i+1])/2 for i in range(6))/(lx[-1]-lx[0])
r7=rows(DIR/'R7_35_conditions_ARCHIVED.csv')
eve=rows(DIR/'EVE_postanchor_14_conditions_ARCHIVED.csv')
exp={r['domain']:(float(r['published_direct']),float(r['published_prior_residual'])) for r in rows(DIR/'PAPER_AULEC_EXPECTED_REV8.csv')}
assert len(r7)==35 and len(eve)==14 and len(exp)==5
assert len({(r['domain'],r['fraction']) for r in r7})==35
for domain in ['VED','KU','SCUT','TUM','EVE']:
 if domain!='EVE':
  dd=sorted((r for r in r7 if r['domain']==domain),key=lambda r:float(r['fraction']))
  x=[float(r['effective_fit_fraction_mean']) for r in dd]
  d=auc(x,[float(r['Direct_segment_balanced_MAE']) for r in dd])
  p=auc(x,[float(r['PR_segment_balanced_MAE']) for r in dd])
 else:
  ds=sorted((r for r in eve if r['model']=='Direct+Prior'),key=lambda r:float(r['fraction']))
  ps=sorted((r for r in eve if r['model']=='PriorResidual+Prior'),key=lambda r:float(r['fraction']))
  assert [z['fraction'] for z in ds]==[z['fraction'] for z in ps]
  x=[float(r['effective_fit_fraction_mean']) for r in ds]
  d=auc(x,[float(r['segment_balanced_MAE_after']) for r in ds])
  p=auc(x,[float(r['segment_balanced_MAE_after']) for r in ps])
 ed,ep=exp[domain]
 assert abs(d-ed)<7e-5 and abs(p-ep)<7e-5,(domain,d,p,ed,ep)
 assert p<d
 print(f'PASS {domain:4} Direct={d:.6f} PR={p:.6f} paper=({ed:.4f},{ep:.4f})')
print('PASS archived AULEC arithmetic: 5/5. No model was retrained.')
