"""Audited preprocessed-input validation (not a raw-data extractor).

Users must lawfully obtain the original BMS telemetry. Raw-to-canonical
reconstruction remains in archived provenance notebooks; this command avoids
silently inventing missing metadata or replacing their frozen split logic.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .postprocess import DOMAINS

REQUIRED={'vehicle_id','segment_id','time_value','elapsed_s','target','soc0','coord','soc_phy','split'}

def check_canonical(path,metadata_path,domain):
    path=Path(path);metadata_path=Path(metadata_path)
    if not path.is_file() or not metadata_path.is_file():raise FileNotFoundError(path if not path.is_file() else metadata_path)
    with open(metadata_path) as f:metadata=json.load(f)
    d=pd.read_parquet(path)
    absent=(REQUIRED-set(d.columns))
    if absent:raise ValueError(f'{domain}: missing canonical columns {sorted(absent)}')
    features=list(metadata['feature_columns']);miss=[f for f in features if f not in d]
    if miss:raise ValueError(f'{domain}: missing features {miss}')
    if not set(d.split.unique()).issubset({'train','val','test'}):
        raise ValueError('Invalid split labels')
    if not {'train','val'}.issubset(set(d.split.unique())):raise ValueError('Missing training or validation split')
    if d.duplicated(['vehicle_id','segment_id','time_value']).any():raise ValueError('Duplicate canonical physical endpoint')
    if d.assign(_seg=d.vehicle_id.astype(str)+'||'+d.segment_id.astype(str)).groupby('_seg').split.nunique().max()>1:
        raise ValueError('Segment crosses train/validation/test boundaries')
    if not np.isfinite(d[['target','soc0','coord','soc_phy','elapsed_s']].to_numpy(float)).all():
        raise ValueError('Non-finite critical columns')
    if (d['elapsed_s'] < -1e-9).any():raise ValueError('Negative elapsed seconds')
    if domain!='EVE' and str(metadata.get('domain',''))!=domain:raise ValueError('Wrong domain metadata')
    return {'domain':domain,'rows':len(d),'features':len(features),
            'split_counts':{str(k):int(v) for k,v in d.split.value_counts().to_dict().items()},'schema_pass':True}

def main(argv=None):
    p=argparse.ArgumentParser(description='Validate user-provided frozen canonical data; DOES NOT download or create raw data')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--domain',choices=DOMAINS,default=None)
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    root=args.root.expanduser().resolve();reports=[]
    for domain in ([args.domain] if args.domain else DOMAINS):
        if domain=='EVE':
            cp=root/'outputs/eve_12B_v4_frozen_trainval/12B_v4_EVE_canonical_trainval.parquet'
            mp=root/'outputs/eve_12B_v4_frozen_trainval/12B_v4_EVE_canonical_metadata.json'
            test=root/'outputs/eve_12C_one_shot_confirmatory/12C_EVE_confirmatory_canonical.parquet'
            if not test.is_file():raise FileNotFoundError(test)
        else:
            cp=root/'outputs/model_screening_10/canonical'/f'{domain}_canonical.parquet'
            mp=root/'outputs/model_screening_10/canonical'/f'{domain}_metadata.json'
        report=check_canonical(cp,mp,domain);reports.append(report)
        print(json.dumps(report,ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'scope':'frozen canonical only; no raw-to-canonical reexecution','reports':reports},indent=2))
if __name__=='__main__':main()
