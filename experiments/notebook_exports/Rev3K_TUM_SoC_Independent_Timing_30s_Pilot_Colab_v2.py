# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/fixed_time_sensitivity/Rev3K_TUM_SoC_Independent_Timing_30s_Pilot_Colab_v2.ipynb
# Notebook SHA256: 230c7a7432d41d29d06f1c28b271e9e97b956d9c641d9a3be3d2c374856aef45
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] Rev3K TUM — SoC-independent fixed-time canonical sampling

# %% Code cell 1 (notebook cell 2)
DOMAIN='TUM'

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

OUT=ROOT/'Revizyon3'/'outputs'/'Rev3K_SOC_INDEPENDENT_TIMING_TUM_PILOT'
OUT.mkdir(parents=True,exist_ok=True)

# %% [markdown] 1. Read seven TUM/FTM raw vehicle parquet streams; derive native-speed time grid *before* looking at BMS SoC

# %% Code cell 2 (notebook cell 4)
# Data prep follows audited Notebook 9 v2 for signal dictionary + long-format Arrow reading.
GRID_TOLERANCE_SECONDS=2.0
MIN_COORD=0.01 # km
N_VEHICLES=7
MIN_SEGMENT_DURATION_S=60.0
SPEED_INTEGRATION_MAX_GAP_S=2.0
SPEED_MOVING_KMH=1.0
DRIVING_MEMORY_S=60.0
SPEED_SEGMENT_GAP_S=20.0
SOC_LABEL_MAX_AGE_S=10.0
VOLTAGE_MAX_AGE_S=3.0
REPO_URL='https://github.com/TUMFTM/electric-vehicle-uds-dataset.git'
LOCAL_REPO=Path('/content/tum_ev_uds_dataset')
DATA_DIR=LOCAL_REPO/'data'/'uds_data'

# 7 native vehicle files are public, distributed via Git LFS.
# No SoC-timestamp alignment until AFTER fixed-grid time selection.
if not (DATA_DIR.exists() and len(list(DATA_DIR.glob('*.parquet')))==7 and all(p.stat().st_size>4096 for p in DATA_DIR.glob('*.parquet'))):
    if LOCAL_REPO.exists():shutil.rmtree(LOCAL_REPO)
    subprocess.run(['bash','-lc','apt-get -qq update && apt-get -qq install -y git-lfs'],check=True)
    subprocess.run(['git','lfs','install'],check=True)
    subprocess.run(['git','clone','--depth','1',REPO_URL,str(LOCAL_REPO)],check=True)
    subprocess.run(['git','lfs','pull','-I','data/uds_data/*.parquet'],cwd=LOCAL_REPO,check=True)
assert DATA_DIR.exists()
PARQUET_FILES=sorted(DATA_DIR.glob('*.parquet'))
assert len(PARQUET_FILES)==7 and all(p.stat().st_size>4096 for p in PARQUET_FILES)
OVERVIEW_PATH=LOCAL_REPO/'data'/'value_overview.csv'
overview=pd.read_csv(OVERVIEW_PATH)
PRIMARY_RAW_SIGNALS=['hv_soc','hv_battery_voltage','vehicle_speed']
REQUESTED_SIGNALS=PRIMARY_RAW_SIGNALS

# %% Code cell 3 (notebook cell 5)
TIME_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "date_time",
    "ts",
    "epoch",
    "t",
]

ID_CANDIDATES = [
    "value_id",
    "valueid",
    "signal_id",
    "id",
]

VALUE_CANDIDATES = [
    "value",
    "val",
    "measurement",
    "data",
]


def normalized_column_map(columns):
    return {
        str(c).strip().lower(): c
        for c in columns
    }


def find_candidate_column(columns, candidates):
    mp = normalized_column_map(columns)

    for c in candidates:
        if c in mp:
            return mp[c]

    return None


def smart_to_datetime(s):
    s = pd.Series(s)

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(
            s,
            utc=True,
            errors="coerce",
        )

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(
            s,
            errors="coerce",
        )

        finite = x[np.isfinite(x)]

        if not len(finite):
            return pd.to_datetime(
                s,
                utc=True,
                errors="coerce",
            )

        med = float(np.median(np.abs(finite)))

        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        elif med > 1e8:
            unit = "s"
        else:
            origin = pd.Timestamp("1970-01-01", tz="UTC")
            return origin + pd.to_timedelta(x, unit="s")

        return pd.to_datetime(
            x,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    return pd.to_datetime(
        s,
        utc=True,
        errors="coerce",
    )


def clean_signal_series(time_values, signal_values):
    d = pd.DataFrame(
        {
            "time": smart_to_datetime(
                pd.Series(time_values).reset_index(drop=True)
            ),
            "value": pd.to_numeric(
                pd.Series(signal_values).reset_index(drop=True),
                errors="coerce",
            ),
        }
    )

    d = d[
        d["time"].notna()
        & np.isfinite(d["value"])
    ].copy()

    d = (
        d.sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )

    return d


def arrow_values_for_ids(id_type, ids):
    if pa.types.is_integer(id_type):
        return pa.array(ids, type=id_type)

    if pa.types.is_floating(id_type):
        return pa.array([float(x) for x in ids], type=id_type)

    if pa.types.is_string(id_type) or pa.types.is_large_string(id_type):
        return pa.array([str(x) for x in ids], type=id_type)

    raise TypeError(
        f"Unsupported value_id Arrow type: {id_type}"
    )


def read_requested_signal_series(parquet_path, overview):
    pf = pq.ParquetFile(parquet_path)
    cols = pf.schema_arrow.names

    print("\nSCHEMA:", parquet_path.name, cols)

    time_col = find_candidate_column(cols, TIME_CANDIDATES)

    assert time_col is not None, (
        "Timestamp column not recognized. "
        f"Schema: {cols}"
    )

    # -------------------------
    # WIDE layout
    # -------------------------
    exact_signal_cols = {
        s: s
        for s in REQUESTED_SIGNALS
        if s in cols
    }

    if "hv_soc" in exact_signal_cols:
        read_cols = list(
            dict.fromkeys(
                [time_col, *exact_signal_cols.values()]
            )
        )

        d = pd.read_parquet(
            parquet_path,
            columns=read_cols,
        )

        out = {}

        for signal, col in exact_signal_cols.items():
            out[signal] = clean_signal_series(
                d[time_col],
                d[col],
            )

        return out, {
            "layout": "wide",
            "time_col": str(time_col),
            "id_col": None,
            "value_col": None,
            "all_columns": [str(c) for c in cols],
        }

    # -------------------------
    # LONG layout
    # -------------------------
    id_col = find_candidate_column(cols, ID_CANDIDATES)
    value_col = find_candidate_column(cols, VALUE_CANDIDATES)

    assert id_col is not None and value_col is not None, (
        "Long parquet schema not recognized. "
        f"Schema: {cols}"
    )

    wanted = (
        overview.loc[
            overview["variable_name"].isin(REQUESTED_SIGNALS),
            ["value_id", "variable_name"],
        ]
        .drop_duplicates("value_id")
        .copy()
    )

    wanted_ids = [int(x) for x in wanted["value_id"]]

    ds = pads.dataset(
        str(parquet_path),
        format="parquet",
    )

    id_type = pf.schema_arrow.field(str(id_col)).type

    wanted_arrow = arrow_values_for_ids(
        id_type,
        wanted_ids,
    )

    filt = pads.field(str(id_col)).isin(wanted_arrow)

    table = ds.to_table(
        columns=[
            str(time_col),
            str(id_col),
            str(value_col),
        ],
        filter=filt,
    )

    d = table.to_pandas()

    id_to_name = dict(
        zip(
            wanted["value_id"].astype(int),
            wanted["variable_name"],
        )
    )

    numeric_ids = pd.to_numeric(
        d[id_col],
        errors="coerce",
    )

    d["_signal_name"] = (
        numeric_ids
        .round()
        .astype("Int64")
        .map(id_to_name)
    )

    out = {}

    for signal, g in d.groupby("_signal_name"):
        if pd.isna(signal):
            continue

        out[str(signal)] = clean_signal_series(
            g[time_col],
            g[value_col],
        )

    return out, {
        "layout": "long",
        "time_col": str(time_col),
        "id_col": str(id_col),
        "value_col": str(value_col),
        "all_columns": [str(c) for c in cols],
    }

# %% Code cell 4 (notebook cell 6)
def speed_to_cumulative_distance(speed_df):
    s = speed_df.copy()
    s = s.sort_values("time").reset_index(drop=True)

    s["value"] = pd.to_numeric(
        s["value"],
        errors="coerce",
    )

    s.loc[
        ~s["value"].between(0.0, 260.0),
        "value",
    ] = np.nan

    s = s[
        s["time"].notna()
        & np.isfinite(s["value"])
    ].copy()

    assert len(s) >= 2, "Too few valid speed samples."

    t_ns = s["time"].astype("int64").to_numpy()
    v = s["value"].to_numpy(float)

    dt_s = np.diff(
        t_ns / 1e9,
        prepend=t_ns[0] / 1e9,
    )

    v_prev = np.r_[v[0], v[:-1]]

    dd_km = (
        0.5
        * (v_prev + v)
        * dt_s
        / 3600.0
    )

    valid_gap = (
        np.isfinite(dd_km)
        & (dt_s >= 0.0)
        & (dt_s <= SPEED_INTEGRATION_MAX_GAP_S)
    )

    dd_km = np.where(
        valid_gap,
        np.maximum(dd_km, 0.0),
        0.0,
    )

    out = pd.DataFrame(
        {
            "time": s["time"].to_numpy(),
            "value": np.cumsum(dd_km),
        }
    )

    return out

# %% Code cell 5 (notebook cell 7)
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

# %% Code cell 6 (notebook cell 8)
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads

def compute_speed_driving_segments(speed_df,veh):
    # Native speed clock only; NO SoC sampling, No SoC-change threshold.
    x=speed_df.sort_values('time').reset_index(drop=True).copy()
    x['value']=pd.to_numeric(x.value,errors='coerce')
    x=x[x.value.between(0.0,260.0)].reset_index(drop=True)
    assert len(x)>=2
    # Use motion continuity (60s rolling recent-speed maximum) exactly as reference;
    # timestamps here are native speed measurements, not SoC timestamps.
    moving=x.set_index('time')['value'].rolling('60s',min_periods=1).max().to_numpy()>SPEED_MOVING_KMH
    dt=x.time.diff().dt.total_seconds().to_numpy()
    start=moving & ((~np.r_[False,moving[:-1]]) | (dt>SPEED_SEGMENT_GAP_S))
    nums=np.cumsum(start)
    x['driving']=moving
    x['segment_local']=np.where(moving,nums,-1)
    x=x[x.driving].copy()
    x['segment_id']=str(veh)+'__'+x.segment_local.astype(str)
    return x

def latest_asof(times_sec,signal,max_age_s):
    # Signal may be BMS SoC; called ONLY after time sampling, never for index construction.
    s=signal.sort_values('time')
    src=(pd.to_datetime(s.time,utc=True).astype('int64').to_numpy(float)/1e9)
    v=s.value.to_numpy(float)
    ix=np.searchsorted(src,times_sec,side='right')-1
    ok=(ix>=0)&(times_sec-src[np.maximum(ix,0)]<=max_age_s)
    out=np.full(len(times_sec),np.nan)
    out[ok]=v[ix[ok]]
    return out

def build_tum_one(vehicle,signal_dict):
    for s in PRIMARY_RAW_SIGNALS:
        assert s in signal_dict and len(signal_dict[s])>0,f'{vehicle} missing {s}'
    voltage=signal_dict['hv_battery_voltage']
    voltage=voltage[voltage.value.between(100,1000)].sort_values('time').copy()
    speed=signal_dict['vehicle_speed']
    speed=speed[speed.value.between(0,260)].sort_values('time').copy()
    soc=signal_dict['hv_soc']
    soc=soc[soc.value.between(0,100)].sort_values('time').copy()
    distance=speed_to_cumulative_distance(speed)
    assert np.array_equal(pd.to_datetime(distance.time,utc=True).astype('int64').to_numpy(),pd.to_datetime(speed.time,utc=True).astype('int64').to_numpy())
    sx=compute_speed_driving_segments(speed,vehicle)
    di=distance.rename(columns={'value':'cum_km'})
    di['time']=pd.to_datetime(di['time'],utc=True)
    sx=sx.merge(di,on='time',how='left',validate='one_to_one')
    assert sx.cum_km.notna().all()
    rows=[]; audit=[]
    # Initial voltage may precede the first native-speed timestamp by <=3 seconds.
    # That is an allowable causal as-of observation, even when the [segment-start, t]
    # history window itself contains no voltage packet.
    vt=pd.to_datetime(voltage.time,utc=True).astype('int64').to_numpy(float)/1e9
    vv=voltage.value.to_numpy(float)
    for sid,g in sx.groupby('segment_id',sort=False):
        g=g.sort_values('time').reset_index(drop=True)
        t=pd.to_datetime(g.time,utc=True).astype('int64').to_numpy(float)/1e9
        if len(t)<2 or t[-1]-t[0]<MIN_SEGMENT_DURATION_S:continue
        ix=fixed_time_indices(t)
        if len(ix)<2:continue
        grid_t=t[ix]
        # NOW, and only now, access SoC channel to create targets and the allowed first anchor.
        bms=latest_asof(grid_t,soc,SOC_LABEL_MAX_AGE_S)
        if not np.isfinite(bms[0]): continue
        if not np.isfinite(bms).all():
            # Do not select times based on BMS SoC updates; keep only complete segments.
            continue
        vol=latest_asof(grid_t,voltage,VOLTAGE_MAX_AGE_S)
        if not np.isfinite(vol).all():continue
        dnative=g.cum_km.to_numpy(float)-float(g.cum_km.iloc[0])
        dnative=np.maximum(dnative,0)
        elapsed=t-t[0]
        sp=g.value.to_numpy(float)
        soc0=float(bms[0])
        voltage_asof_fallback_count=0
        for j,pos in enumerate(ix):
            tt=t[pos]
            left=int(np.searchsorted(t,tt-HISTORY_S,side='left'))
            vl=int(np.searchsorted(vt,max(tt-HISTORY_S,t[0]),side='left'))
            vr=int(np.searchsorted(vt,tt,side='right'))
            voltage_window=vv[vl:vr]
            if len(voltage_window)==0:
                # A valid as-of sample can fall just BEFORE segment start.  In
                # particular, at j=0 the 120-s within-segment history is empty,
                # although the voltage signal is fresh at the selected timestamp.
                # Use ONLY the most recent past voltage, never a future value.
                assert np.isfinite(vol[j]), 'No causal fresh voltage at a scored grid point'
                previous_idx=int(np.searchsorted(vt,tt,side='right')-1)
                assert previous_idx>=0, 'Voltage fallback has no past observation'
                age=float(tt-vt[previous_idx])
                assert -1e-4<=age<=VOLTAGE_MAX_AGE_S+1e-4, (
                    'Voltage fallback violates causal staleness gate',age)
                assert np.isclose(vv[previous_idx],vol[j],rtol=1e-7,atol=1e-7)
                voltage_window=np.array([vol[j]],dtype=float)
                voltage_asof_fallback_count+=1
            vals={'hv_battery_voltage':voltage_window,'vehicle_speed':sp[left:pos+1],
                  'distance_km':dnative[left:pos+1],'elapsed_s':elapsed[left:pos+1],
                  'soc0':np.repeat(soc0,pos-left+1)}
            r={'domain':DOMAIN,'vehicle_id':str(vehicle),'segment_id':sid,'compound_segment':sid,
               'time_value':float(tt),'elapsed_s':float(elapsed[pos]),'is_start':j==0,
               'target':float(bms[j]),'soc0':soc0,'coord':float(dnative[pos])}
            for sig,ar in vals.items():
                for stat,value in zip(STATS,feature_stats(ar)):
                    r[f'x__{sig}__{stat}']=value
            rows.append(r)
        # Audit SoC coincidences AFTER time-grid has been frozen; informational only.
        c=np.diff(bms,prepend=bms[0]);nchanges=int((np.abs(c)>1e-6).sum())
        audit.append({'vehicle_id':vehicle,'segment_id':sid,'n_speed_rows':len(g),'n_grid_points':len(ix),
                      'grid_consecutive_soc_changes_audit_ONLY':nchanges,
                      'voltage_causal_asof_fallback_count':voltage_asof_fallback_count})
    return rows,audit

rows=[];audit=[]
for p in PARQUET_FILES:
    vid=p.stem
    sig,meta=read_requested_signal_series(p,overview)
    rr,aa=build_tum_one(vid,sig)
    if not rr:raise RuntimeError(f'{vid} produced no telemetry-time grids')
    print(vid,'grid points',len(rr),'segments',len(aa),
          'voltage-start-asof-fallbacks',sum(a['voltage_causal_asof_fallback_count'] for a in aa))
    rows.extend(rr);audit.extend(aa)
    del sig,rr,aa;gc.collect()
grid=pd.DataFrame(rows)
FEATURES=[f'x__{s}__{st}' for s in ['hv_battery_voltage','vehicle_speed','distance_km','elapsed_s','soc0'] for st in STATS]
assert len(FEATURES)==30 and grid.vehicle_id.nunique()==7
assert np.isfinite(grid[FEATURES+['coord','target']].to_numpy(float)).all()
assert (grid.groupby('compound_segment').is_start.sum()==1).all()
grid=assign_non_target_chronological_split(grid)
grid.to_parquet(OUT/f'{DOMAIN}_FIXEDTIME_CANONICAL.parquet',index=False)
pd.DataFrame(audit).to_csv(OUT/f'{DOMAIN}_FIXEDTIME_GRID_AUDIT.csv',index=False)
print('FINAL TUM grid',len(grid),'rows',grid.compound_segment.nunique(),'segments')

# %% [markdown] 2. Freeze protocol and checkpoint identity; ensure data and split audits are saved

# %% Code cell 7 (notebook cell 10)
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

# %% Code cell 8 (notebook cell 12)
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

# %% Code cell 9 (notebook cell 13)
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

# %% Code cell 10 (notebook cell 14)
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

# %% Code cell 11 (notebook cell 16)
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

# %% Code cell 12 (notebook cell 18)
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
