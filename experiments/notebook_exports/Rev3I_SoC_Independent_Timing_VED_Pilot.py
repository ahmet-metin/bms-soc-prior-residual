# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/fixed_time_sensitivity/Rev3I_SoC_Independent_Timing_VED_Pilot.ipynb
# Notebook SHA256: 40aec5a63879a90e1a7c1ad786bdfb80aa0dd0d39a4b6a06da4a1f192c787242
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] VED — SoC-independent fixed-time canonical sampling (pilot)

# %% Code cell 1 (notebook cell 2)
# Google Colab with a GPU is recommended. A working Drive mount and the VED raw cache are required.
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json, hashlib, random, gc, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT=Path('/content/drive/MyDrive/NissanLeaf')
CACHE=ROOT/'outputs'/'cache'/'leaf_raw_full_v2.parquet'
OUT=ROOT/'Revizyon3'/'outputs'/'Rev3I_SOC_INDEPENDENT_TIMING_VED_PILOT'
OUT.mkdir(parents=True, exist_ok=True)
PRED_DIR=OUT/'predictions'; PRED_DIR.mkdir(exist_ok=True)
CKPT_DIR=OUT/'checkpoints'; CKPT_DIR.mkdir(exist_ok=True)

assert CACHE.exists(), f'Original raw cache missing: {CACHE}. Build the original v2.2 raw cache first.'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, 'raw cache:', CACHE)

LEAF_IDS=[10,455]
GRID_SECONDS=30.0
GRID_TOLERANCE_SECONDS=2.0
HISTORY_S=120.0
GAP_THRESHOLD_S=10.0
SOC_CHANGE_EPS_PP=1e-6   # preprocessing defines old event flags, not used for sampling
MAX_AGE_S={'current_a':2.0,'voltage_v':5.0,'speed_kmh':3.0,'oat_c':180.0,'ac_w':15.0,'heater_w':15.0}
SEQ_LEN=16
DENSE_LAYERS=3
GROWTH=19
FUSION_WIDTH=64
DROPOUT=0.10
FRACTIONS=[0.1,1.0]  # expand only if pilot supports doing so
SEEDS=[42,43,44,45,46]
EPOCH_LIMIT=200
PATIENCE=20
LR=1e-3
WEIGHT_DECAY=1e-5
BATCH_SIZE=256 if DEVICE.type=='cuda' else 128
MIN_BETA_TRAIN_TARGETS=20

# %% [markdown] 1. Reconstruct original VED segmentation and physical coordinates from raw timestamps

# %% Code cell 2 (notebook cell 4)
def assign_chronological_trip_split(df, train_frac=0.60, val_frac=0.20):
    trip_info = (
        df.groupby(['veh_id', 'trip'], as_index=False)
          .agg(
              start_day=('day_num', 'min'),
              start_ts=('timestamp_ms', 'min'),
              n_rows=('trip', 'size'),
          )
    )

    pieces = []
    for vid, g in trip_info.groupby('veh_id', sort=False):
        g = g.sort_values(['start_day', 'start_ts', 'trip']).reset_index(drop=True)
        n = len(g)
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))

        split = np.array(['test'] * n, dtype=object)
        split[:n_train] = 'train'
        split[n_train:n_train+n_val] = 'val'
        g['split'] = split
        pieces.append(g)

    trip_info = pd.concat(pieces, ignore_index=True)
    out = df.merge(
        trip_info[['veh_id', 'trip', 'split']],
        on=['veh_id', 'trip'],
        how='left',
        validate='many_to_one',
    )
    return out, trip_info

def ffill_max_age(values, times_s, max_age_s):
    values = np.asarray(values, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    out = values.copy()

    last_value = np.nan
    last_time = -np.inf
    for i in range(len(out)):
        if np.isfinite(values[i]):
            last_value = values[i]
            last_time = times_s[i]
        elif np.isfinite(last_value) and (times_s[i] - last_time <= max_age_s):
            out[i] = last_value
    return out

def preprocess_segments(df):
    d = df.copy()
    d = d.sort_values(['veh_id', 'trip', 'timestamp_ms']).reset_index(drop=True)
    d['t_abs_s'] = d['timestamp_ms'] / 1000.0

    dt = d.groupby(['veh_id', 'trip'])['t_abs_s'].diff()
    new_seg = dt.isna() | (dt <= 0) | (dt > GAP_THRESHOLD_S)

    d['segment_num'] = new_seg.groupby([d['veh_id'], d['trip']]).cumsum().astype(int)
    d['segment_id'] = (
        d['veh_id'].astype(str) + '_' +
        d['trip'].astype(str) + '_' +
        d['segment_num'].astype(str)
    )

    processed = []
    for sid, g in d.groupby('segment_id', sort=False):
        g = g.sort_values('t_abs_s').copy()

        valid_soc_pos = np.flatnonzero(g['soc_raw'].notna().to_numpy())
        if len(valid_soc_pos) == 0:
            continue

        g = g.iloc[valid_soc_pos[0]:].copy()
        if len(g) < 2:
            continue

        t = g['t_abs_s'].to_numpy(float)
        g['elapsed_s'] = t - t[0]
        g['soc_bms'] = g['soc_raw'].ffill()
        g['soc0'] = float(g['soc_bms'].iloc[0])

        for c, max_age in MAX_AGE_S.items():
            g[c + '_aligned'] = ffill_max_age(g[c].to_numpy(float), t, max_age)

        prev_soc = g['soc_bms'].shift(1)
        g['soc_anchor'] = (
            g['soc_raw'].notna()
            & prev_soc.notna()
            & ((g['soc_bms'] - prev_soc).abs() > SOC_CHANGE_EPS_PP)
        )

        dt_s = g['t_abs_s'].diff().fillna(0.0).to_numpy(float)
        dt_s[(dt_s < 0) | (dt_s > GAP_THRESHOLD_S)] = 0.0
        g['dt_s'] = dt_s

        processed.append(g)

    if not processed:
        raise RuntimeError('No usable continuous segment with BMS SoC initialization.')
    return pd.concat(processed, ignore_index=True)

def add_integrals(df):
    parts = []

    for sid, g in df.groupby('segment_id', sort=False):
        g = g.sort_values('t_abs_s').copy()

        I = g['current_a_aligned'].to_numpy(float)
        V = g['voltage_v_aligned'].to_numpy(float)
        dt = g['dt_s'].to_numpy(float)

        I_prev = np.r_[I[0], I[:-1]]
        current_step_ok = np.isfinite(I) & np.isfinite(I_prev) & (dt > 0)
        dq = np.zeros(len(g), dtype=float)
        dq[current_step_ok] = 0.5 * (I[current_step_ok] + I_prev[current_step_ok]) * dt[current_step_ok] / 3600.0

        g['q_ah'] = np.cumsum(dq)
        g['q_abs_ah'] = np.cumsum(np.abs(dq))
        g['current_observed_s'] = np.cumsum(dt * current_step_ok.astype(float))
        g['observed_elapsed_s'] = np.cumsum(dt)
        denom = np.maximum(g['observed_elapsed_s'].to_numpy(float), 1e-9)
        g['current_coverage_ratio'] = g['current_observed_s'].to_numpy(float) / denom

        # Raw signed pack power; discharge-positive sign later inferred from beta.
        Praw = I * V / 1000.0
        Praw_prev = np.r_[Praw[0], Praw[:-1]]
        power_step_ok = np.isfinite(Praw) & np.isfinite(Praw_prev) & (dt > 0)

        dE_raw = np.zeros(len(g), dtype=float)
        dE_raw[power_step_ok] = 0.5 * (Praw[power_step_ok] + Praw_prev[power_step_ok]) * dt[power_step_ok] / 3600.0

        g['energy_raw_kwh'] = np.cumsum(dE_raw)
        g['power_observed_s'] = np.cumsum(dt * power_step_ok.astype(float))

        parts.append(g)

    return pd.concat(parts, ignore_index=True)

# %% Code cell 3 (notebook cell 5)
raw=pd.read_parquet(CACHE)
raw=raw[raw.veh_id.isin(LEAF_IDS)].copy()
assert set(raw['veh_id'].unique().tolist())==set(LEAF_IDS)
raw2,trip_info=assign_chronological_trip_split(raw)
proc_base=preprocess_segments(raw2)
proc_charge=add_integrals(proc_base)
assert proc_charge['split'].isin(['train','val','test']).all()
assert 'q_ah' in proc_charge and 'q_abs_ah' in proc_charge
assert proc_charge.segment_id.nunique()>0
assert proc_charge['soc0'].notna().all()
print('Raw rows:', len(raw), 'segments:', proc_charge.segment_id.nunique(), 'trip split:', trip_info.split.value_counts().to_dict())

# %% [markdown] 2. Construct a genuinely time-selected grid and causal features

# %% Code cell 4 (notebook cell 7)
SIGNALS=['current_a_aligned','voltage_v_aligned','speed_kmh_aligned','oat_c_aligned','q_abs_ah','elapsed_s','soc0']
STATS=['last','mean','std','min','max','delta']
FEATURES=[f'x__{s}__{st}' for s in SIGNALS for st in STATS]
# Exact R2C VED feature contract: 42 causal signal statistics + shared soc_phy; signed Ah only builds the prior.
assert len(FEATURES)==42

def fixed_time_indices(t,interval_s=GRID_SECONDS, tolerance_s=GRID_TOLERANCE_SECONDS):
    t=np.asarray(t,float)
    assert len(t) and np.isfinite(t).all() and np.all(np.diff(t)>0)
    schedule=t[0]+np.arange(int(np.floor((t[-1]-t[0])/interval_s))+1)*interval_s
    indices=np.searchsorted(t,schedule,side='right')-1
    mask=(indices>=0)&((schedule-t[indices])<=tolerance_s)
    ix=np.unique(indices[mask])
    assert len(ix)>0 and ix[0]==0
    return ix

def feature_stats(x):
    v=np.asarray(x,float)
    v=v[np.isfinite(v)]
    if not len(v): return [np.nan]*6
    return [float(v[-1]),float(v.mean()),float(v.std()),float(v.min()),float(v.max()),float(v[-1]-v[0])]

def prepare_grid(df):
    records=[]
    audit=[]
    for sid,g in df.groupby('segment_id',sort=False):
        g=g.sort_values('t_abs_s').reset_index(drop=True)
        if len(g)<2: continue
        t=g['t_abs_s'].to_numpy(float)
        # The time point selector is intentionally called with timestamps ONLY.
        indices=fixed_time_indices(t)
        # Permanently audit the fact that changing target values cannot alter selected indices.
        assert np.array_equal(indices,fixed_time_indices(t))
        if len(indices)<2: continue
        arr={s:g[s].to_numpy(float) for s in SIGNALS}
        for j,pos in enumerate(indices):
            sec=float(t[pos]); left=int(np.searchsorted(t,sec-HISTORY_S,side='left'))
            rec={'domain':'VED','vehicle_id':str(int(g.at[pos,'veh_id'])),
                 'segment_id':str(sid), 'compound_segment':str(int(g.at[pos,'veh_id']))+'||'+str(sid),
                 'split':str(g.at[pos,'split']), 'time_value':sec,
                 'elapsed_s':float(g.at[pos,'elapsed_s']), 'is_start': j==0,
                 'target':float(g.at[pos,'soc_bms']), 'soc0':float(g.at[pos,'soc0']),
                 'coord':float(g.at[pos,'q_ah']),
                 'source_soc_change_flag_ONLY_FOR_AUDIT':bool(g.at[pos,'soc_anchor'])}
            for s in SIGNALS:
                rec.update({f'x__{s}__{stat}':value for stat,value in zip(STATS,feature_stats(arr[s][left:pos+1]))})
            rec['x__q_signed_ah_endpoint']=float(g.at[pos,'q_ah'])
            records.append(rec)
        audit.append({'segment_id':sid,'n_raw':len(g),'n_time_grid':len(indices),'n_soc_change_events':int(g.soc_anchor.sum()),
                      'grid_points_also_change_events':int(g.iloc[indices].soc_anchor.sum())})
    z=pd.DataFrame(records)
    assert len(z)>0
    z['is_start']=z['is_start'].astype(bool)
    assert z['target'].notna().all()
    finite=np.isfinite(z[FEATURES].to_numpy(float)).all(axis=1)
    # Keep only complete, telemetry-defined segments; never select on SoC-change event flags.
    bad=set(z.loc[~finite,'compound_segment'])
    z=z[~z.compound_segment.isin(bad)].copy()
    counts=z.groupby('compound_segment').size()
    z=z[z.compound_segment.isin(counts[counts>=2].index)].copy()
    z=z.sort_values(['split','compound_segment','time_value'],kind='mergesort').reset_index(drop=True)
    assert (z.groupby('compound_segment')['is_start'].sum()==1).all()
    assert z.groupby('compound_segment')['split'].nunique().max()==1
    return z,pd.DataFrame(audit)

grid,audit=prepare_grid(proc_charge)
grid.to_parquet(OUT/'VED_FIXEDTIME_CANONICAL.parquet',index=False)
audit.to_csv(OUT/'VED_FIXEDTIME_GRID_AUDIT.csv',index=False)
print('Fixed-grid rows:',len(grid),'segments:',grid.compound_segment.nunique())
print(grid.groupby('split').agg(rows=('target','size'),segments=('compound_segment','nunique'),vehicles=('vehicle_id','nunique')))
print('Fraction of grid points coincident with change events (audit only):',grid['source_soc_change_flag_ONLY_FOR_AUDIT'].mean())

# %% [markdown] 3. Supervision-matched prior and training observation budget

# %% Code cell 5 (notebook cell 9)
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

# %% Code cell 6 (notebook cell 10)
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
train_base=grid[grid.split=='train'].copy()
planned=[]
for frac in FRACTIONS:
    for seed in SEEDS:
        lab=select_fit_targets(train_base,frac,seed)
        beta,n=fit_beta_allowed(lab)
        anchor_ids=set(train_base.loc[train_base.is_start,'compound_segment'])
        total_obs=int(len(lab)+len(anchor_ids))
        fit_fraction=total_obs/len(train_base)
        planned.append({'domain':'VED','fraction':frac,'seed':seed,'beta':beta,'n_beta_informative':n,'selected_labels':len(lab),
                        'mandatory_start_anchors':len(anchor_ids),'effective_fit_fraction':fit_fraction,
                        'n_total_fit_rows':len(train_base)})
plan=pd.DataFrame(planned)
plan.to_csv(OUT/'VED_FIXEDTIME_SUPERVISION_AND_PRIOR_PLAN.csv',index=False)
print(plan.to_string(index=False))

# %% [markdown] 4. Five-seed, exactly information-matched training

# %% Code cell 7 (notebook cell 12)
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

def make_run_pack(frac,seed):
    chosen=select_fit_targets(train_base,frac,seed)
    beta,n=fit_beta_allowed(chosen)
    sections={}
    for split in ['train','val','test']:
        z=grid[grid.split==split].copy()
        z['soc_phy']=z.soc0.to_numpy(float)+beta*z.coord.to_numpy(float)
        z['pr_target']=z.target.to_numpy(float)-z.soc_phy.to_numpy(float)
        sections[split]=prepare_split_frame(z,split)
    # Re-identify selected training target rows after sorting: (vehicle, segment, timestamp).
    keys=set(zip(chosen.vehicle_id.astype(str),chosen.segment_id.astype(str),chosen.time_value.astype(float)))
    train=sections['train']
    chosen_mask=np.fromiter((k in keys for k in zip(train.vehicle_id.astype(str),train.segment_id.astype(str),train.time_value.astype(float))),dtype=bool,count=len(train))
    assert chosen_mask.sum()==len(chosen)
    train_positions=train.loc[chosen_mask,'seq_pos'].to_numpy(dtype=int)
    val=sections['val'];test=sections['test']
    val_positions=val.loc[~val.is_start,'seq_pos'].to_numpy(dtype=int)
    test_positions=test.loc[~test.is_start,'seq_pos'].to_numpy(dtype=int)
    assert len(val_positions)>0 and len(test_positions)>0
    pre=TrainOnlyPreprocessor().fit(train[FEATURES+['soc_phy']].to_numpy(float))
    xs={split:pre.transform(z[FEATURES+['soc_phy']].to_numpy(float)) for split,z in sections.items()}
    return sections,xs,train_positions,val_positions,test_positions,beta

def train_run(frac,seed,model_name):
    tag=f'VED_f{frac:.3f}_{model_name}_s{seed}'
    outfile=PRED_DIR/f'{tag}.parquet'; ckpt=CKPT_DIR/f'{tag}.pt'
    if outfile.exists() and ckpt.exists():
        print('SKIP verified files (check hashes/manifest before external publication):',tag)
        return pd.read_parquet(outfile)
    set_seed(seed)
    sections,xs,train_pos,val_pos,test_pos,beta=make_run_pack(frac,seed)
    tar='target' if model_name=='DirectPlusPrior' else 'pr_target'
    scaler=TargetScaler().fit(sections['train'].iloc[train_pos][tar].to_numpy(float))
    def loader(split,positions,shuffle):
        ds=CausalSequenceDataset(sections[split],xs[split],positions,scaler,tar,SEQ_LEN)
        return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=0,pin_memory=DEVICE.type=='cuda')
    tr=loader('train',train_pos,True);val=loader('val',val_pos,False);te=loader('test',test_pos,False)
    model=DenseRecurrentNet(len(FEATURES)+2).to(DEVICE)
    assert sum(p.numel() for p in model.parameters())==23197, 'Capacity mismatch with frozen VED architecture'
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    lossfn=nn.MSELoss()
    best=float('inf');stale=0;best_state=None;best_epoch=0
    for epoch in range(1,EPOCH_LIMIT+1):
        model.train()
        for xb,yb,_ in tr:
            xb=xb.to(DEVICE);yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred=model(xb);loss=lossfn(pred,yb)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        ii,pred_raw=predict(model,val,scaler)
        vr=sections['val'].iloc[ii]
        pred_soc=pred_raw+(vr.soc_phy.to_numpy(float) if tar=='pr_target' else 0)
        val_mae=float(np.mean(np.abs(pred_soc-vr.target.to_numpy(float))))
        if val_mae<best-1e-6:
            best=val_mae;stale=0;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if epoch==1 or epoch%25==0:print(tag,'epoch',epoch,'val-MAE',round(val_mae,5),'best',round(best,5))
        if stale>=PATIENCE:break
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'fraction':frac,'seed':seed,'model':model_name,
                'best_epoch':best_epoch,'validation_mae':best,'beta':beta},ckpt)
    ii,raw=predict(model,te,scaler)
    zr=sections['test'].iloc[ii].copy()
    soc_pred=raw+(zr.soc_phy.to_numpy(float) if tar=='pr_target' else 0)
    out=zr[['domain','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target']].copy()
    out['prediction']=soc_pred;out['model']=model_name;out['fraction']=frac;out['seed']=seed
    out['beta']=beta;out['validation_mae']=best;out['best_epoch']=best_epoch
    assert not zr.is_start.any()
    out.to_parquet(outfile,index=False)
    return out

all_predictions=[]
for frac in FRACTIONS:
    for seed in SEEDS:
        for mod in ['DirectPlusPrior','PriorResidualPlusPrior']:
            all_predictions.append(train_run(frac,seed,mod))
preds=pd.concat(all_predictions,ignore_index=True)
preds.to_parquet(OUT/'VED_FIXEDTIME_ALL_SEED_PREDICTIONS.parquet',index=False)
print('DONE; total prediction rows:',len(preds))

# %% [markdown] 5. Independent test-row aggregation, row/segment MAE and paired intervals

# %% Code cell 8 (notebook cell 14)
keys=['domain','vehicle_id','segment_id','compound_segment','time_value','elapsed_s','target','fraction','model']
coverage=preds.groupby(keys,dropna=False)['seed'].nunique()
assert coverage.min()==coverage.max()==len(SEEDS), 'Seed identity or alignment problem'
avg=preds.groupby(keys,dropna=False,as_index=False).agg(prediction=('prediction','mean'))
avg['abs_error']=(avg.prediction-avg.target).abs()
avg.to_parquet(OUT/'VED_FIXEDTIME_SEED_AVG_TEST_PREDICTIONS.parquet',index=False)
result=[]; bs=[]
for f in FRACTIONS:
    group=avg[avg.fraction==f]
    # Use paired segment keys and calculate each segment's own mean error.
    seg=group.groupby(['model','compound_segment'],as_index=False).agg(seg_MAE=('abs_error','mean'))
    wide=seg.pivot(index='compound_segment',columns='model',values='seg_MAE').dropna()
    assert len(wide)>=10
    delta=(wide['PriorResidualPlusPrior']-wide['DirectPlusPrior']).to_numpy(float)
    rng=np.random.default_rng(7301+int(100*f))
    ix=rng.integers(0,len(delta),size=(10000,len(delta)))
    ci=np.percentile(delta[ix].mean(axis=1),[2.5,97.5])
    bs.append({'fraction':f,'n_segments':len(delta),'PR_minus_Direct_mean_seg_MAE':delta.mean(),
               'CI95_low':ci[0],'CI95_high':ci[1],'descriptive_only':True})
    for m in ['DirectPlusPrior','PriorResidualPlusPrior']:
        u=group[group.model==m]
        result.append({'fraction':f,'model':m,'row_MAE':u.abs_error.mean(),
                       'segment_balanced_MAE':wide[m].mean(),
                       'n_test_rows':len(u),'n_test_segments':len(wide)})
summary=pd.DataFrame(result);ci_df=pd.DataFrame(bs)
summary.to_csv(OUT/'VED_FIXEDTIME_METRICS.csv',index=False)
ci_df.to_csv(OUT/'VED_FIXEDTIME_PAIRED_BOOTSTRAP.csv',index=False)
print(summary.to_string(index=False));print(ci_df.to_string(index=False))
print('Interpretation: This is NEW fixed-time data, so do not mistake the point estimates for a rerun of the published target-event canonical test population.')

# %% [markdown] 6. Decision gate and next domain
