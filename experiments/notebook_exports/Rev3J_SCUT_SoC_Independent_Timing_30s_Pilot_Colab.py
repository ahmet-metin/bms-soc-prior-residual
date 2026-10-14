# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/fixed_time_sensitivity/Rev3J_SCUT_SoC_Independent_Timing_30s_Pilot_Colab.ipynb
# Notebook SHA256: fabae4ba342c6f90c4158187d9951d2271ac0a83294d5b44ed71ad645e54c760
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] Rev3J SCUT — SoC-independent fixed-time canonical sampling

# %% Code cell 1 (notebook cell 2)
DOMAIN='SCUT'

# Target = last observable BMS-reported SoC, NOT true electrochemical SoC.
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow openpyxl python-calamine pandas torch numpy requests
from google.colab import drive
try:
    drive.mount('/content/drive')
except ValueError:
    print('Google Drive already mounted')
from pathlib import Path
import json, hashlib, random, gc, math, time, zipfile, re, subprocess, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
ROOT=Path('/content/drive/MyDrive/NissanLeaf')
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRID_SECONDS=30.0
HISTORY_S=120.0
SEQ_LEN=16
DENSE_LAYERS=3
GROWTH=19
FUSION_WIDTH=64
DROPOUT=0.10
FRACTIONS=[0.10,1.0]
SEEDS=[42,43,44,45,46]
EPOCH_LIMIT=200
PATIENCE=20
LR=1e-3
WEIGHT_DECAY=1e-5
BATCH_SIZE=256 if DEVICE.type=='cuda' else 128
MIN_BETA_INFORMATIVE=20
STATS=['last','mean','std','min','max','delta']
print('device:',DEVICE,'domain:',DOMAIN)

OUT=ROOT/'Revizyon3'/'outputs'/'Rev3J_SOC_INDEPENDENT_TIMING_SCUT_PILOT'
OUT.mkdir(parents=True,exist_ok=True)

# %% [markdown] 1. Reconstruct SCUT raw signals from actual spreadsheets (not SoC-change anchor cache)

# %% Code cell 2 (notebook cell 4)
def _excel_engine():
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except Exception:
        return "openpyxl"


EXCEL_ENGINE = _excel_engine()

print("Excel engine:", EXCEL_ENGINE)


def read_table_auto(
    path,
    nrows=None,
    sheet_name=0,
):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last = None

        for enc in [
            "utf-8",
            "utf-8-sig",
            "gbk",
            "latin1",
        ]:
            try:
                return pd.read_csv(
                    path,
                    encoding=enc,
                    low_memory=False,
                    nrows=nrows,
                )
            except Exception as e:
                last = e

        raise last

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(
            path,
            sheet_name=sheet_name,
            nrows=nrows,
            engine=EXCEL_ENGINE,
        )

    raise ValueError(
        f"Unsupported file: {path}"
    )


def excel_sheet_names(path):
    path = Path(path)

    if path.suffix.lower() not in [
        ".xlsx",
        ".xls",
    ]:
        return [None]

    with pd.ExcelFile(
        path,
        engine=EXCEL_ENGINE,
    ) as xf:
        return list(xf.sheet_names)


def scut_tables(vid):
    root = (
        SCUT_ROOT
        / "extracted"
        / f"vehicle_{vid}"
    )

    files = []

    for ext in [
        "*.xlsx",
        "*.XLSX",
        "*.xls",
        "*.XLS",
        "*.csv",
        "*.CSV",
    ]:
        files.extend(
            root.rglob(ext)
        )

    return sorted(
        set(files)
    )


def scut_sources(vid):
    out = []

    for p in scut_tables(vid):
        if p.suffix.lower() in [
            ".xlsx",
            ".xls",
        ]:
            for sheet in excel_sheet_names(p):
                out.append(
                    {
                        "path": p,
                        "sheet_name": sheet,
                        "source_name": (
                            f"{p.name}::{sheet}"
                        ),
                    }
                )

        else:
            out.append(
                {
                    "path": p,
                    "sheet_name": None,
                    "source_name": p.name,
                }
            )

    return out


def _norm_header(x):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x)
        .replace("\ufeff", "")
        .strip()
        .lower(),
    )


SCUT_ROLE_ALIASES = {
    "timestamp": [
        "time",
        "timestamp",
        "datetime",
    ],
    "speed_kmh": [
        "vhc_speed",
        "vehicle_speed",
        "speed",
    ],
    "charging_signal": [
        "charging_signal",
        "chargingstatus",
        "chargestatus",
    ],
    "voltage_v": [
        "hv_voltage",
        "packvoltage",
        "batteryvoltage",
    ],
    "current_a": [
        "hv_current",
        "packcurrent",
        "batterycurrent",
    ],
    "soc_pct": [
        "bcell_soc",
        "soc",
        "batterysoc",
    ],
    "temp_max": [
        "bcell_maxTemp",
        "maxtemp",
    ],
    "temp_min": [
        "bcell_minTemp",
        "mintemp",
    ],
    "temp_single": [
        "bcell_Temp",
        "bcelltemp",
        "batterytemp",
    ],
}


def resolve_scut_schema(columns):
    by_norm = {
        _norm_header(c): c
        for c in columns
    }

    out = {}

    for role, aliases in (
        SCUT_ROLE_ALIASES.items()
    ):
        hit = None

        for a in aliases:
            key = _norm_header(a)

            if key in by_norm:
                hit = by_norm[key]
                break

        out[role] = hit

    required = [
        "timestamp",
        "speed_kmh",
        "charging_signal",
        "voltage_v",
        "current_a",
        "soc_pct",
    ]

    missing = [
        x
        for x in required
        if out[x] is None
    ]

    return out, missing

# %% Code cell 3 (notebook cell 5)
SCUT_ROOT=ROOT/'dataset'/'external_validation'/'scut'
SCUT_VEHICLES=list(range(1,8))
SCUT_DRIVING_SIGNAL_BY_VEHICLE={1:3,2:3,3:3,4:3,5:3,6:3,7:0}
SCUT_REPO_API='https://api.github.com/repos/Translab-SCUT/Electric-vehicle-operation-data/contents'
GRID_TOLERANCE_SECONDS=12.0
MIN_COORD=1e-4
N_VEHICLES=7
SCUT_MIN_SESSION_SECONDS=45.0

def scut_sources(vid):
    root=SCUT_ROOT/'extracted'/f'vehicle_{vid}'
    fl=[]
    for ext in ('*.xlsx','*.xls','*.csv','*.XLSX','*.XLS','*.CSV'):fl+=list(root.rglob(ext))
    fl=sorted(set(fl))
    if not fl:
        archive=SCUT_ROOT/'archives'/f'vehicle#{vid}.zip'
        if not archive.exists():
            import requests
            repo=requests.get(SCUT_REPO_API,timeout=120);repo.raise_for_status()
            hits=[z for z in repo.json() if z.get('name')==f'vehicle#{vid}.zip']
            assert len(hits)==1
            u=hits[0]['download_url']; resp=requests.get(u,timeout=240);resp.raise_for_status()
            archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(resp.content)
        root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as zipf: zipf.extractall(root)
        fl=sorted(set(f for ext in ('*.xlsx','*.xls','*.csv') for f in root.rglob(ext)))
    if not fl: raise FileNotFoundError(f'No SCUT spreadsheets in {root}')
    for f in fl:
        if f.suffix.lower() in ('.xlsx','.xls'):
            with pd.ExcelFile(f,engine=EXCEL_ENGINE) as e:
                for name in e.sheet_names: yield (f,name,f'{f.name}::{name}')
        else: yield(f,None,f.name)

# 7B schema and timestamp parser reused; replacing 7B SOC-jump segmentation below.

# %% Code cell 4 (notebook cell 6)
def parse_scut_time_axis(values):
    s = pd.Series(values)

    num = pd.to_numeric(
        s,
        errors="coerce",
    )

    finite = num[
        np.isfinite(num)
    ]

    if not len(finite):
        return pd.to_datetime(
            s,
            errors="coerce",
        )

    def parse_one(v):
        try:
            iv = int(
                round(float(v))
            )

            txt = str(
                abs(iv)
            ).zfill(10)

            return pd.to_datetime(
                "2020" + txt,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )

        except Exception:
            return pd.NaT

    parsed = num.map(
        parse_one
    )

    ok = float(
        parsed.notna().mean()
    )

    if ok < 0.90:
        raise RuntimeError(
            "SCUT packed time parse rate "
            f"too low: {ok:.3f}"
        )

    # Handle possible year rollover.
    arr = parsed.copy()

    prev = None
    add_year = 0

    for idx, val in arr.items():
        if pd.isna(val):
            continue

        candidate = (
            val
            + pd.DateOffset(
                years=add_year
            )
        )

        if (
            prev is not None
            and candidate
            < prev - pd.Timedelta(
                days=180
            )
        ):
            add_year += 1
            candidate = (
                val
                + pd.DateOffset(
                    years=add_year
                )
            )

        arr.loc[idx] = candidate
        prev = candidate

    return pd.to_datetime(
        arr
    )


def normalize_soc_percent(values):
    s = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).astype(float)

    finite = s[
        np.isfinite(s)
    ]

    if len(finite):
        if (
            np.nanpercentile(
                finite,
                99,
            )
            <= 1.5
        ):
            s *= 100.0

    return s

# %% Code cell 5 (notebook cell 7)
def canonicalize_scut(
    path,
    vid,
    sheet_name=0,
    source_name=None,
):
    raw = read_table_auto(
        path,
        sheet_name=sheet_name,
    )

    mp, missing = (
        resolve_scut_schema(
            raw.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{Path(path).name}"
            f"[{sheet_name}]: "
            f"unresolved roles {missing}"
        )

    d = pd.DataFrame(
        index=raw.index
    )

    d["timestamp"] = (
        parse_scut_time_axis(
            raw[
                mp["timestamp"]
            ]
        )
    )

    d["charging_signal"] = (
        pd.to_numeric(
            raw[
                mp[
                    "charging_signal"
                ]
            ],
            errors="coerce",
        )
    )

    d["speed_kmh"] = (
        pd.to_numeric(
            raw[
                mp["speed_kmh"]
            ],
            errors="coerce",
        )
    )

    d["voltage_v"] = (
        pd.to_numeric(
            raw[
                mp["voltage_v"]
            ],
            errors="coerce",
        )
    )

    d["current_a"] = (
        pd.to_numeric(
            raw[
                mp["current_a"]
            ],
            errors="coerce",
        )
    )

    d["soc_pct"] = (
        normalize_soc_percent(
            raw[
                mp["soc_pct"]
            ]
        ).to_numpy()
    )

    # CRITICAL 7B FIX:
    driving_code = (
        SCUT_DRIVING_SIGNAL_BY_VEHICLE[
            vid
        ]
    )

    d = d[
        d["charging_signal"]
        == driving_code
    ].copy()

    if (
        mp.get("temp_max")
        is not None
        and mp.get("temp_min")
        is not None
    ):
        mx = pd.to_numeric(
            raw.loc[
                d.index,
                mp["temp_max"],
            ],
            errors="coerce",
        )

        mn = pd.to_numeric(
            raw.loc[
                d.index,
                mp["temp_min"],
            ],
            errors="coerce",
        )

        d["temp_c_optional"] = (
            mx + mn
        ) / 2.0

    elif (
        mp.get("temp_single")
        is not None
    ):
        d["temp_c_optional"] = (
            pd.to_numeric(
                raw.loc[
                    d.index,
                    mp[
                        "temp_single"
                    ],
                ],
                errors="coerce",
            )
        )

    else:
        d["temp_c_optional"] = np.nan

    d["dataset"] = "SCUT"
    d["vehicle_id"] = (
        f"SCUT_{vid}"
    )
    d["source_file"] = (
        source_name
        or Path(path).name
    )

    return d[
        [
            "dataset",
            "vehicle_id",
            "timestamp",
            "current_a",
            "voltage_v",
            "speed_kmh",
            "soc_pct",
            "temp_c_optional",
            "source_file",
        ]
    ]

# %% Code cell 6 (notebook cell 8)
def fixed_time_indices(t,interval_s=GRID_SECONDS,tolerance_s=GRID_TOLERANCE_SECONDS):
    t=np.asarray(t,dtype=float)
    assert len(t)>0 and np.isfinite(t).all() and (np.diff(t)>0).all(), 'Nonmonotone time input'
    target_times=t[0]+np.arange(int(np.floor((t[-1]-t[0])/interval_s))+1)*interval_s
    inds=np.searchsorted(t,target_times,side='right')-1
    ok=(inds>=0)&((target_times-t[inds])<=tolerance_s)
    chosen=np.unique(inds[ok])
    assert len(chosen)>0 and chosen[0]==0
    return chosen

def feature_stats(v):
    v=np.asarray(v,float)
    v=v[np.isfinite(v)]
    if len(v)==0: return [np.nan]*6
    return [float(v[-1]),float(v.mean()),float(v.std(ddof=0)),float(v.min()),float(v.max()),float(v[-1]-v[0])]

def assign_non_target_chronological_split(z):
    # Only segment start time and count are accessed, never target differences/SoC.
    seg=z.groupby(['vehicle_id','compound_segment'],as_index=False).agg(start_time=('time_value','min'),n_grid=('time_value','size'))
    pieces=[]
    for vid,g in seg.groupby('vehicle_id',sort=False):
        g=g.sort_values(['start_time','compound_segment'],kind='mergesort').reset_index(drop=True)
        n=len(g)
        assert n>=5, f'{vid}: fewer than 5 segments'
        ntr=max(1,int(np.floor(0.60*n)))
        nv=max(1,int(np.floor(0.20*n)))
        assert ntr+nv<n
        g['split']='test';g.loc[:ntr-1,'split']='train';g.loc[ntr:ntr+nv-1,'split']='val'
        pieces.append(g)
    mp=pd.concat(pieces,ignore_index=True)
    z=z.merge(mp[['vehicle_id','compound_segment','split']],on=['vehicle_id','compound_segment'],validate='many_to_one')
    # chronological no same-session leakage
    assert z.groupby('compound_segment').split.nunique().max()==1
    check=z.groupby(['vehicle_id','split'],as_index=False).agg(segments=('compound_segment','nunique'),rows=('target','size'))
    print(check.to_string(index=False));check.to_csv(OUT/f'{DOMAIN}_FIXEDTIME_SPLIT_AUDIT.csv',index=False)
    return z

# %% Code cell 7 (notebook cell 9)
def non_target_scut_sessions(d,source_name):
    d=d.copy()
    for c in ['current_a','voltage_v','speed_kmh']:
        d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    # IMPORTANT: soc_pct is intentionally NOT in filtering or session boundary criteria.
    d=d.dropna(subset=['timestamp','current_a','voltage_v','speed_kmh'])
    d=d[d.voltage_v.between(20,1000)&(d.current_a.abs()<=2000)&d.speed_kmh.between(0,300)].copy()
    d=d.sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    if len(d)<3:return pd.DataFrame()
    dt=d.timestamp.diff().dt.total_seconds()
    med=float(dt[dt>0].median()) if (dt>0).any() else 1.0
    gap=max(30.0,5.0*med)
    new_sess=dt.isna()|(dt<=0)|(dt>gap)
    d['session_local']=new_sess.cumsum()
    d['segment_id']=d.vehicle_id.astype(str)+'_'+source_name+'_'+d.session_local.astype(str)
    # No soc_jump, no target-change-count threshold.
    return d.drop(columns='session_local')

def build_scut_segments(z):
    data=[];aud=[]
    for sid,g in z.groupby('segment_id',sort=False):
        g=g.sort_values('timestamp').reset_index(drop=True).copy()
        if len(g)<3:continue
        t=g.timestamp.astype('int64').to_numpy(float)/1e9
        if t[-1]-t[0]<SCUT_MIN_SESSION_SECONDS:continue
        dtime=np.diff(t,prepend=t[0]);I=g.current_a.to_numpy(float)
        Iprev=np.r_[I[0],I[:-1]]
        g['q_signed_ah']=np.cumsum(0.5*(I+Iprev)*dtime/3600)
        g['q_abs_ah']=np.cumsum(0.5*(np.abs(I)+np.abs(Iprev))*dtime/3600)
        g['elapsed_s']=t-t[0]
        # indices selected without SoC; BMS SoC read only AFTER time-grid is frozen.
        ix=fixed_time_indices(t)
        if len(ix)<2:continue
        gg=g.iloc[ix].copy().reset_index(drop=True)
        # The first grid time must have an observed BMS SoC for permitted initialization.
        # Exclude missing-initialization segments; never select by change-event count.
        first_soc=float(pd.to_numeric(gg.soc_pct.iloc[0],errors='coerce'))
        if not (np.isfinite(first_soc) and 0.0<=first_soc<=100.0):continue
        gg['soc0']=first_soc;g['soc0']=first_soc
        raw_soc=pd.to_numeric(g.soc_pct,errors='coerce')
        g['soc_pct']=raw_soc.where(raw_soc.between(0.0,100.0)).ffill()
        # Label lookup AFTER time selection; forward held reported SoC.
        if not np.isfinite(g.iloc[ix].soc_pct.to_numpy(float)).all():continue
        signals=['current_a','voltage_v','speed_kmh','q_abs_ah','elapsed_s','soc0']
        arrays={s:g[s].to_numpy(float) for s in signals}
        for j,pos in enumerate(ix):
            left=int(np.searchsorted(t,t[pos]-HISTORY_S,side='left'))
            r={'domain':DOMAIN,'vehicle_id':str(g.vehicle_id.iloc[0]),'segment_id':str(sid),
               'compound_segment':str(sid),'time_value':float(t[pos]),
               'elapsed_s':float(g.elapsed_s.iloc[pos]),'is_start':j==0,
               'target':float(g.soc_pct.iloc[pos]),'soc0':float(g.soc0.iloc[pos]),
               'coord':float(g.q_signed_ah.iloc[pos])}
            for s in signals:
                for stat,value in zip(STATS,feature_stats(arrays[s][left:pos+1])):
                    r[f'x__{s}__{stat}']=value
            data.append(r)
        changes=int((g.soc_pct.diff().abs()>1e-6).sum())
        coincidences=int((g.soc_pct.diff().abs()>1e-6).iloc[ix].sum())
        aud.append({'segment_id':sid,'n_raw':len(g),'n_time_grid':len(ix),
                    'n_soc_change_events':changes,'grid_points_also_change_events':coincidences})
    return data,aud

allrows=[];all_aud=[];source_rows=[]
for vid in SCUT_VEHICLES:
    parts=[];ns=0
    for path,sheet,source in scut_sources(vid):
        d=canonicalize_scut(path,vid,sheet_name=sheet,source_name=source)
        d=non_target_scut_sessions(d,source)
        if not len(d):continue
        rows,audit=build_scut_segments(d)
        parts.extend(rows);all_aud.extend(audit)
        source_rows.append({'vehicle_id':f'SCUT_{vid}','source_file':source,'rows_retained':len(rows),
                            'segments_retained':len(audit)})
        ns+=1
    if not parts:raise RuntimeError(f'SCUT_{vid}: no valid raw-grid segments; inspect SCUT timestamps and tolerance')
    allrows.extend(parts)
    print('SCUT vehicle',vid,'source tables',ns,'grid rows',len(parts))
grid=pd.DataFrame(allrows)
FEATURES=[f'x__{s}__{stat}' for s in ['current_a','voltage_v','speed_kmh','q_abs_ah','elapsed_s','soc0'] for stat in STATS]
assert len(FEATURES)==36 and grid.vehicle_id.nunique()==7
assert np.isfinite(grid[FEATURES+['coord','target']].to_numpy(float)).all()
assert (grid.groupby('compound_segment').is_start.sum()==1).all()
grid=assign_non_target_chronological_split(grid)
grid.to_parquet(OUT/f'{DOMAIN}_FIXEDTIME_CANONICAL.parquet',index=False)
pd.DataFrame(all_aud).to_csv(OUT/f'{DOMAIN}_FIXEDTIME_GRID_AUDIT.csv',index=False)
pd.DataFrame(source_rows).to_csv(OUT/f'{DOMAIN}_FIXEDTIME_SOURCE_AUDIT.csv',index=False)
print('FINAL SCUT fixed-grid:',len(grid),'rows',grid.compound_segment.nunique(),'segments')

# %% [markdown] 2. Freeze protocol and checkpoint identity; ensure data and split audits are saved

# %% Code cell 8 (notebook cell 11)
import hashlib
CONTRACT={
    'domain':DOMAIN,'grid_seconds':GRID_SECONDS,'grid_tolerance_s':GRID_TOLERANCE_SECONDS,
    'history_seconds':HISTORY_S,'split_method':'by vehicle chronological segment count 60-20-20',
    'no_soc_event_sampling':True,'initial_soc0_allowed':True,'initial_rows_scored':False,
    'fractions':FRACTIONS,'seeds':SEEDS,'features':FEATURES,'n_grid_rows':len(grid),
    'n_vehicles':N_VEHICLES,'n_segments':int(grid.compound_segment.nunique()),
    'epoch_limit':EPOCH_LIMIT,'patience':PATIENCE,'batch_size':BATCH_SIZE,
    'code_source':'Rev3I VED DenseRecurrentNet information-matched learner',
}
PROTOCOL_HASH=hashlib.sha256(json.dumps(CONTRACT,sort_keys=True).encode()).hexdigest()
manifest=OUT/f'{DOMAIN}_FIXEDTIME_PROTOCOL_MANIFEST.json'
if manifest.exists():
    prior=json.loads(manifest.read_text())
    assert prior['protocol_sha256']==PROTOCOL_HASH, 'Existing experiment differs from configuration: do not mix runs; use a new output directory'
else:
    manifest.write_text(json.dumps({'protocol_sha256':PROTOCOL_HASH,**CONTRACT},indent=2))
PRED_DIR=OUT/'predictions';PRED_DIR.mkdir(parents=True,exist_ok=True)
CKPT_DIR=OUT/'checkpoints';CKPT_DIR.mkdir(parents=True,exist_ok=True)
print('Frozen protocol hash:',PROTOCOL_HASH)

# %% [markdown] 3. Matched DenseGRU model (direct vs residual targets; all other information shared)

# %% Code cell 9 (notebook cell 13)
class TrainOnlyPreprocessor:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        Xi = np.where(np.isfinite(X), X, med[None, :])

        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)

        self.median_ = med.astype(np.float32)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, self.median_[None, :])
        return ((X - self.mean_[None, :]) / self.std_[None, :]).astype(np.float32)

class TargetScaler:
    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y))
        if not np.isfinite(self.std_) or self.std_ < 1e-8:
            self.std_ = 1.0
        return self

    def transform(self, y):
        return (
            (np.asarray(y, float) - self.mean_) / self.std_
        ).astype(np.float32)

    def inverse(self, y):
        return np.asarray(y, float) * self.std_ + self.mean_

class CausalSequenceDataset(Dataset):
    def __init__(
        self,
        frame,
        X_scaled,
        endpoint_positions,
        target_scaler,
        target_col,
        seq_len,
    ):
        self.frame = frame.reset_index(drop=True)
        self.X = X_scaled
        self.positions = np.asarray(endpoint_positions, dtype=int)
        self.target_scaler = target_scaler
        self.target_col = target_col
        self.seq_len = int(seq_len)

        self.seg_start = (
            self.frame.groupby("compound_segment", sort=False)["seq_pos"]
            .min()
            .astype(int)
            .to_dict()
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        pos = int(self.positions[i])
        row = self.frame.iloc[pos]
        seg0 = int(self.seg_start[row["compound_segment"]])

        start = max(seg0, pos - self.seq_len + 1)

        x = self.X[start:pos + 1]
        n = len(x)
        pad = self.seq_len - n

        # +1 channel = strict causal validity mask.
        out = np.zeros(
            (self.seq_len, self.X.shape[1] + 1),
            dtype=np.float32,
        )

        if n:
            out[pad:, :-1] = x
            out[pad:, -1] = 1.0

        y_raw = float(row[self.target_col])
        y_scaled = float(self.target_scaler.transform([y_raw])[0])

        return (
            torch.from_numpy(out),
            torch.tensor(y_scaled, dtype=torch.float32),
            pos,
        )

def prepare_split_frame(domain_df, split):
    z = domain_df[domain_df["split"] == split].copy()

    z = z.sort_values(
        ["compound_segment", "time_value"],
        kind="mergesort",
    ).reset_index(drop=True)

    z["seq_pos"] = np.arange(len(z), dtype=np.int64)
    return z

class DenseRecurrentNet(nn.Module):
    # Exact R1A development architecture.
    def __init__(self, input_dim):
        super().__init__()

        self.rnn1 = nn.GRU(
            input_size=input_dim,
            hidden_size=GROWTH,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.GRU(
            input_size=input_dim + GROWTH,
            hidden_size=GROWTH,
            num_layers=1,
            batch_first=True,
        )

        self.rnn3 = nn.GRU(
            input_size=input_dim + 2 * GROWTH,
            hidden_size=GROWTH,
            num_layers=1,
            batch_first=True,
        )

        self.feature_dropout = nn.Dropout(DROPOUT)

        self.dense_fusion = nn.Linear(
            DENSE_LAYERS * GROWTH,
            FUSION_WIDTH,
        )

        self.input_skip = nn.Linear(
            input_dim,
            FUSION_WIDTH,
        )

        self.fusion_norm = nn.LayerNorm(FUSION_WIDTH)

        self.head = nn.Sequential(
            nn.Linear(FUSION_WIDTH, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h1, _ = self.rnn1(x)
        h1_next = self.feature_dropout(h1)

        h2, _ = self.rnn2(
            torch.cat([x, h1_next], dim=-1)
        )
        h2_next = self.feature_dropout(h2)

        h3, _ = self.rnn3(
            torch.cat([x, h1_next, h2_next], dim=-1)
        )

        endpoint_dense = torch.cat(
            [
                h1[:, -1, :],
                h2[:, -1, :],
                h3[:, -1, :],
            ],
            dim=-1,
        )

        fused = (
            self.dense_fusion(endpoint_dense)
            + self.input_skip(x[:, -1, :])
        )

        fused = torch.relu(self.fusion_norm(fused))
        return self.head(fused).squeeze(-1)

# %% Code cell 10 (notebook cell 14)
def stable_hash_u01(s):
    return int(hashlib.md5(str(s).encode('utf-8')).hexdigest()[:12],16)/float(16**12)
def select_fit_targets(frame,fraction,seed):
    # Segment starts form a separate, mandatory observation budget and are NOT chosen for ML training.
    candidates=frame[~frame.is_start].copy()
    pieces=[]
    for vid,g in candidates.groupby('vehicle_id',sort=False):
        count=max(1,int(round(fraction*len(g))))
        identity=g.vehicle_id.astype(str)+'|'+g.segment_id.astype(str)+'|'+g.time_value.astype(str)+'|'+str(seed)
        take=identity.map(stable_hash_u01).sort_values(kind='mergesort').index[:count]
        pieces.append(g.loc[take].copy())
    return pd.concat(pieces).sort_index()

def fit_beta_allowed(selected):
    x=selected['coord'].to_numpy(float)
    y=selected['target'].to_numpy(float)-selected['soc0'].to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)&(np.abs(x)>=0.01)
    n=int(ok.sum())
    if n<MIN_BETA_TRAIN_TARGETS:raise RuntimeError(f'Only {n} informative retained labels: predeclared minimum={MIN_BETA_TRAIN_TARGETS}; no fallback permitted.')
    den=float(x[ok]@x[ok]);assert den>0
    return float(x[ok]@y[ok]/den),n

# %% Code cell 11 (notebook cell 15)
def stable_hash_u01(s):
    return int(hashlib.md5(str(s).encode('utf-8')).hexdigest()[:12],16)/float(16**12)

def select_fit_targets(frame,fraction,seed):
    # Pick target IDs using ONLY vehicle, segment, grid timestamp, seed.
    candidates=frame.loc[~frame.is_start]
    pieces=[]
    for vid,g in candidates.groupby('vehicle_id',sort=False):
        k=max(1,int(round(fraction*len(g))))
        identities=g.vehicle_id.astype(str)+'|'+g.compound_segment.astype(str)+'|'+g.time_value.astype(str)+'|'+str(seed)
        pieces.append(g.loc[identities.map(stable_hash_u01).sort_values(kind='mergesort').index[:k]])
    return pd.concat(pieces).sort_index()

def fit_prior_by_vehicle(chosen):
    betas={}; audits=[]
    for vid,g in chosen.groupby('vehicle_id',sort=False):
        x=g.coord.to_numpy(float)
        y=g.target.to_numpy(float)-g.soc0.to_numpy(float)
        valid=np.isfinite(x)&np.isfinite(y)&(np.abs(x)>=MIN_COORD)
        n=int(valid.sum())
        if n<MIN_BETA_INFORMATIVE:
            raise RuntimeError(f'{vid}: only {n} informative {DOMAIN} train targets; need >= {MIN_BETA_INFORMATIVE}; no fallback allowed')
        den=float(x[valid]@x[valid]);assert den>0
        b=float(x[valid]@y[valid]/den)
        betas[str(vid)]=b
        audits.append({'vehicle_id':str(vid),'beta':b,'n_beta_informative':n,'n_train_labels':len(g)})
    missing=set(grid.vehicle_id.unique())-set(betas)
    assert not missing,f'Missing vehicles for prior: {missing}'
    return betas,audits

train_base=grid[grid.split=='train'].copy()
plan=[]
for frac in FRACTIONS:
    for seed in SEEDS:
        chosen=select_fit_targets(train_base,frac,seed)
        b,ba=fit_prior_by_vehicle(chosen)
        start_count=train_base.compound_segment.nunique()
        for row in ba:
            r=dict(row);r.update({'domain':DOMAIN,'fraction':frac,'seed':seed,'mandatory_start_anchors':start_count,
                 'selected_nonstart_labels':len(chosen),'train_rows':len(train_base),
                 'effective_obs_fraction':(len(chosen)+start_count)/len(train_base)})
            plan.append(r)
plan=pd.DataFrame(plan)
plan.to_csv(OUT/f'{DOMAIN}_FIXEDTIME_SUPERVISION_AND_PRIOR_PLAN.csv',index=False)
print(plan.groupby(['fraction','seed']).agg(n_betas=('beta','size'),n_targets=('selected_nonstart_labels','first'),effective_obs_fraction=('effective_obs_fraction','first')).to_string())

# %% [markdown] 4. Run 20 matched training jobs (5 seeds × 2 fractions × 2 models); save each checkpoint & prediction

# %% Code cell 12 (notebook cell 17)
def regression_metrics(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'P95':float(np.percentile(np.abs(e),95))}

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

@torch.no_grad()
def predict(model,loader,tsc):
    model.eval();ind=[];ys=[]
    for xb,yb,pos in loader:
        yhat=model(xb.to(DEVICE)).detach().cpu().numpy()
        ys.append(tsc.inverse(yhat));ind.append(np.asarray(pos))
    return np.concatenate(ind).astype(int),np.concatenate(ys)


def regression_metrics(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'P95':float(np.percentile(np.abs(e),95))}

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
@torch.no_grad()
def predict(model,loader,scaler):
    model.eval();ii=[];raw=[]
    for xb,yb,pos in loader:
        raw.append(scaler.inverse(model(xb.to(DEVICE)).detach().cpu().numpy()))
        ii.append(np.asarray(pos))
    return np.concatenate(ii).astype(int),np.concatenate(raw)

def make_run_pack(frac,seed):
    chosen=select_fit_targets(train_base,frac,seed)
    beta_map,ba=fit_prior_by_vehicle(chosen)
    sections={}
    for split in ['train','val','test']:
        z=grid[grid.split==split].copy()
        z['soc_phy']=z.soc0.to_numpy(float)+z.vehicle_id.map(beta_map).to_numpy(float)*z.coord.to_numpy(float)
        z['pr_target']=z.target.to_numpy(float)-z.soc_phy.to_numpy(float)
        assert np.isfinite(z.soc_phy).all()
        sections[split]=prepare_split_frame(z,split)
    keys=set(zip(chosen.vehicle_id.astype(str),chosen.compound_segment.astype(str),chosen.time_value.astype(float)))
    tr=sections['train']
    chosen_mask=np.fromiter((k in keys for k in zip(tr.vehicle_id.astype(str),tr.compound_segment.astype(str),tr.time_value.astype(float))),dtype=bool,count=len(tr))
    assert chosen_mask.sum()==len(chosen)
    train_pos=tr.loc[chosen_mask,'seq_pos'].to_numpy(int)
    val_pos=sections['val'].loc[~sections['val'].is_start,'seq_pos'].to_numpy(int)
    test_pos=sections['test'].loc[~sections['test'].is_start,'seq_pos'].to_numpy(int)
    assert len(val_pos)>0 and len(test_pos)>0
    pre=TrainOnlyPreprocessor().fit(tr[FEATURES+['soc_phy']].to_numpy(float))
    xs={split:pre.transform(z[FEATURES+['soc_phy']].to_numpy(float)) for split,z in sections.items()}
    return sections,xs,train_pos,val_pos,test_pos,beta_map

def train_run(frac,seed,model_name):
    tag=f'{DOMAIN}_f{frac:.3f}_{model_name}_s{seed}'
    outfile=PRED_DIR/f'{tag}.parquet';ckpt=CKPT_DIR/f'{tag}.pt'
    if outfile.exists() and ckpt.exists():
        d=pd.read_parquet(outfile)
        assert d.seed.nunique()==1 and int(d.seed.iloc[0])==seed and d.model.nunique()==1
        assert d.compound_segment.nunique()>0
        print('RESUME:',tag)
        return d
    set_seed(seed)
    sections,xs,tr_pos,val_pos,test_pos,beta_map=make_run_pack(frac,seed)
    tar='target' if model_name=='DirectPlusPrior' else 'pr_target'
    scaler=TargetScaler().fit(sections['train'].iloc[tr_pos][tar].to_numpy(float))
    def loader(split,positions,shuffle):
        ds=CausalSequenceDataset(sections[split],xs[split],positions,scaler,tar,SEQ_LEN)
        return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=0,pin_memory=DEVICE.type=='cuda')
    tr=loader('train',tr_pos,True);val=loader('val',val_pos,False);te=loader('test',test_pos,False)
    model=DenseRecurrentNet(len(FEATURES)+2).to(DEVICE)
    model_params=sum(p.numel() for p in model.parameters())
    # Architecture is frozen; input dimension is domain-specific; each pair MUST match.
    assert model_params>10000 and model_params<50000, model_params
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    lossfn=nn.MSELoss()
    best=float('inf');best_state=None;stale=0;best_epoch=0
    for epoch in range(1,EPOCH_LIMIT+1):
        model.train()
        for xb,yb,_ in tr:
            xb=xb.to(DEVICE);yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True);loss=lossfn(model(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('NaN training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        ii,praw=predict(model,val,scaler)
        vr=sections['val'].iloc[ii]
        pv=praw+(vr.soc_phy.to_numpy(float) if tar=='pr_target' else 0.0)
        maev=float(np.abs(pv-vr.target.to_numpy(float)).mean())
        if maev<best-1e-6:
            best=maev;stale=0;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if epoch==1 or epoch%25==0:print(tag,'epoch',epoch,'val-MAE',round(maev,5),'best',round(best,5))
        if stale>=PATIENCE:break
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'fraction':frac,'seed':seed,'model':model_name,'domain':DOMAIN,
                'parameters':model_params,'best_epoch':best_epoch,'validation_mae':best,
                'beta_map':beta_map,'protocol_sha256':PROTOCOL_HASH},ckpt)
    ii,praw=predict(model,te,scaler)
    zr=sections['test'].iloc[ii].copy()
    yhat=praw+(zr.soc_phy.to_numpy(float) if tar=='pr_target' else 0)
    assert not zr.is_start.any()
    out=zr[['domain','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target']].copy()
    out['prediction']=yhat;out['model']=model_name;out['fraction']=frac;out['seed']=seed
    out['validation_mae']=best;out['best_epoch']=best_epoch
    out.to_parquet(outfile,index=False)
    return out

all_predictions=[]
for f in FRACTIONS:
    for seed in SEEDS:
        for model_name in ['DirectPlusPrior','PriorResidualPlusPrior']:
            all_predictions.append(train_run(f,seed,model_name))
preds=pd.concat(all_predictions,ignore_index=True)
preds.to_parquet(OUT/f'{DOMAIN}_FIXEDTIME_ALL_SEED_PREDICTIONS.parquet',index=False)
print('FINISHED',DOMAIN,'prediction records',len(preds))

# %% [markdown] 5. Five-seed-averaged test scores + all-vehicle heterogeneity; descriptive intervals only

# %% Code cell 13 (notebook cell 19)
keys=['domain','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','fraction','model']
coverage=preds.groupby(keys,dropna=False).seed.nunique()
assert coverage.min()==coverage.max()==len(SEEDS), 'Missing or duplicate seed predictions'
avg=preds.groupby(keys,dropna=False,as_index=False).agg(prediction=('prediction','mean'))
pair_coverage=avg.groupby(['domain','vehicle_id','compound_segment','time_value','fraction']).model.nunique()
assert pair_coverage.min()==pair_coverage.max()==2, 'Direct/PR test-row identity mismatch'
avg['abs_error']=(avg.prediction-avg.target).abs()
avg.to_parquet(OUT/f'{DOMAIN}_FIXEDTIME_SEED_AVG_TEST_PREDICTIONS.parquet',index=False)
results=[];bs=[];per_vehicle=[]
for f in FRACTIONS:
    g=avg[avg.fraction==f]
    seg=g.groupby(['model','vehicle_id','compound_segment'],as_index=False).agg(seg_MAE=('abs_error','mean'))
    w=seg.pivot(index=['vehicle_id','compound_segment'],columns='model',values='seg_MAE').dropna()
    assert len(w)>=10 and len(w.index.get_level_values(0).unique())==N_VEHICLES
    delta=(w['PriorResidualPlusPrior']-w['DirectPlusPrior']).to_numpy(float)
    rng=np.random.default_rng(7301+int(100*f))
    ix=rng.integers(0,len(delta),size=(10000,len(delta)))
    ci=np.percentile(delta[ix].mean(axis=1),[2.5,97.5])
    vehicle_means=w.groupby(level='vehicle_id').mean()
    vdelta=(vehicle_means['PriorResidualPlusPrior']-vehicle_means['DirectPlusPrior']).to_numpy(float)
    vidx=rng.integers(0,len(vdelta),size=(10000,len(vdelta)))
    vci=np.percentile(vdelta[vidx].mean(axis=1),[2.5,97.5])
    bs.append({'fraction':f,'n_segments':len(w),'n_vehicles':len(vdelta),
          'mean_paired_segment_MAE_diff_PR_minus_Direct':delta.mean(),
          'segment_bootstrap_CI95_low':ci[0],'segment_bootstrap_CI95_high':ci[1],
          'vehicle_balanced_difference':vdelta.mean(),'vehicle_bootstrap_CI95_low':vci[0],
          'vehicle_bootstrap_CI95_high':vci[1],'n_vehicles_PR_better':int((vdelta<0).sum()),
          'descriptive_only':True})
    for v,row in vehicle_means.iterrows():
        per_vehicle.append({'fraction':f,'vehicle_id':str(v),'Direct_segMAE':float(row['DirectPlusPrior']),
           'PR_segMAE':float(row['PriorResidualPlusPrior']),
           'PR_minus_Direct':float(row['PriorResidualPlusPrior']-row['DirectPlusPrior'])})
    for model in ['DirectPlusPrior','PriorResidualPlusPrior']:
        a=g[g.model==model]
        results.append({'fraction':f,'model':model,'row_MAE':a.abs_error.mean(),
                       'segment_balanced_MAE':w[model].mean(),
                       'vehicle_balanced_MAE':vehicle_means[model].mean(),
                       'n_test_rows':len(a),'n_test_segments':len(w),'n_test_vehicles':N_VEHICLES})
summary=pd.DataFrame(results);conf=pd.DataFrame(bs);vsummary=pd.DataFrame(per_vehicle)
summary.to_csv(OUT/f'{DOMAIN}_FIXEDTIME_METRICS.csv',index=False)
conf.to_csv(OUT/f'{DOMAIN}_FIXEDTIME_PAIRED_BOOTSTRAP.csv',index=False)
vsummary.to_csv(OUT/f'{DOMAIN}_FIXEDTIME_PER_VEHICLE.csv',index=False)
print(summary.to_string(index=False));print(conf.to_string(index=False))
print(vsummary.to_string(index=False))
print('NOTE: VED original + SCUT/TUM fixed grid test populations differ; not directly comparable absolute MAEs.')
print('NOTE: n=7 vehicles; bootstrap CIs descriptive only. Neither AULEC nor Holm significance is claimed.')

# %% [markdown] 6. Interpretation and audit gate
