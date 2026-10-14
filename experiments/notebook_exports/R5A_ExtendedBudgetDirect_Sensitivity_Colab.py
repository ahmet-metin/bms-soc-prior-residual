# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/budget_optimization/R5A_ExtendedBudgetDirect_Sensitivity_Colab.ipynb
# Notebook SHA256: 1875a8d20a7c671c12a1ed45f2e0dc2cc28a7aad6ff394bd1e2b5fbd3e594848
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R5A — Extended-Budget Direct Sensitivity

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json, hashlib, random, time, gc, contextlib, warnings, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
OUTPUTS = ROOT / "outputs"
REV2_OUT = ROOT / "Revizyon2" / "outputs"

SRC = REV2_OUT / "R2C_corrected_sparse_prior_curve"
OUT = REV2_OUT / "R5A_extended_budget_direct"
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "seed_predictions"
for p in [OUT, CKPT_DIR, PRED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R5A_EXTENDED_BUDGET_DIRECT_SENSITIVITY"
SOURCE_EXPERIMENT = "R2C_CORRECTED_SPARSE_PRIOR_CURVE"

DEV_DOMAINS = ["VED","KU","SCUT","TUM"]
ALL_DOMAINS = DEV_DOMAINS + ["EVE"]
FRACTIONS = [0.01,0.02,0.05,0.10,0.20,0.50,1.00]
SEEDS = [42,43,44,45,46]

DIRECT = "Direct+Prior"
PR = "PriorResidual+Prior"
MODELS = [DIRECT, PR]

SEQ_LEN = 16
DENSE_LAYERS = 3
GROWTH = 19
FUSION_WIDTH = 64
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 1e-5
MIN_DELTA = 1e-6
GRAD_CLIP = 5.0
RESUME_SAVE_EVERY = 10

BUDGET_MULTIPLIERS = [1,2,4]
PRIMARY_SCENARIOS = [2,4]
BOOTSTRAP_DRAWS = 10000
CLUSTER_BOOTSTRAP_DRAWS = 20000
MIN_VEHICLES_FOR_POPULATION_INFERENCE = 5

# Tight but practical reproduction gates for the 1x rerun.
REPRO_MAX_ABS_PRED_TOL = 5e-3
REPRO_MAE_TOL = 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
BATCH_SIZE = 512 if AMP_ENABLED else 128

SRC_RUNS = SRC / f"{SOURCE_EXPERIMENT}_ALL_CURVE_RUNS.csv"
SRC_SEEDAVG = SRC / f"{SOURCE_EXPERIMENT}_SEED_AVERAGED_PREDICTIONS.parquet"
SRC_BUDGET = SRC / f"{SOURCE_EXPERIMENT}_EFFECTIVE_BUDGET_SUMMARY.csv"
SRC_PRIMARY = SRC / f"{SOURCE_EXPERIMENT}_PRIMARY_EFFECTIVE_AULEC_BOOTSTRAP.csv"

for p in [SRC_RUNS, SRC_SEEDAVG, SRC_BUDGET, SRC_PRIMARY]:
    assert p.exists(), p

print("Experiment:", EXPERIMENT)
print("Device:", DEVICE)
print("Source:", SRC)
print("Output:", OUT)

# %% Code cell 2 (notebook cell 3)
def find_unique_file(root, filename):
    matches = list(Path(root).rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename!r} under {root}; found {len(matches)}")
    return matches[0]

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def stable_hash_u01(s):
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:12],16) / float(16**12)

def regression_metrics(y, pred):
    y = np.asarray(y,float); pred = np.asarray(pred,float)
    e = pred-y; ae = np.abs(e)
    return {
        "MAE":float(ae.mean()),
        "RMSE":float(np.sqrt(np.mean(e**2))),
        "P95":float(np.percentile(ae,95)),
        "Bias":float(e.mean()),
    }

def exact_sparse_subset(df, fraction, seed):
    if float(fraction) >= 0.999999:
        return df.copy()
    pieces=[]
    for _,g in df.groupby("vehicle_id",sort=False):
        k=max(1,int(round(float(fraction)*len(g))))
        keys=(g["vehicle_id"].astype(str)+"|"+g["segment_id"].astype(str)+"|"+
              g["time_value"].astype(str)+"|"+str(seed))
        order=keys.map(stable_hash_u01).sort_values().index
        pieces.append(g.loc[order[:k]].copy())
    return pd.concat(pieces,ignore_index=False)

def identity_string(df):
    z=(df["vehicle_id"].astype(str)+"|"+df["segment_id"].astype(str)+"|"+
       df["time_value"].astype(str)).sort_values()
    return "\n".join(z.tolist())

def identity_sha256(df):
    return hashlib.sha256(identity_string(df).encode("utf-8")).hexdigest()

def normalized_log_auc(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    order=np.argsort(x); x=x[order]; y=y[order]
    assert np.all(x>0)
    lx=np.log10(x)
    width=float(lx[-1]-lx[0])
    assert width>0
    trapz=np.trapezoid if hasattr(np,"trapezoid") else np.trapz
    return float(trapz(y,lx)/width)

def holm_adjust(p_values):
    p=np.asarray(p_values,float)
    assert np.isfinite(p).all()
    m=len(p); order=np.argsort(p)
    out=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx])
        out[idx]=min(1.0,running)
    return out

def bootstrap_two_sided_p(samples):
    s=np.asarray(samples,float)
    p_left=(np.sum(s<=0)+1.0)/(len(s)+1.0)
    p_right=(np.sum(s>=0)+1.0)/(len(s)+1.0)
    return float(min(1.0,2.0*min(p_left,p_right)))

# %% Code cell 3 (notebook cell 4)
# ---------- Source governance and fixed cap-hit plan ----------

source_runs = pd.read_csv(SRC_RUNS)
budget_summary = pd.read_csv(SRC_BUDGET)
source_primary = pd.read_csv(SRC_PRIMARY)

assert len(source_runs) == 5*7*2*5
assert set(source_runs["domain"].astype(str)) == set(ALL_DOMAINS)
assert set(source_runs["model"].astype(str)) == set(MODELS)
assert set(source_runs["seed"].astype(int)) == set(SEEDS)
assert len(source_primary) == 5
assert len(budget_summary) == 5*7

cap_plan = source_runs[
    (source_runs["model"].astype(str)==DIRECT) &
    (source_runs["hit_epoch_cap"].astype(bool))
].copy()

assert len(cap_plan) == 31, ("Expected the pre-observed 31 Direct epoch-cap runs", len(cap_plan))
assert not cap_plan["stopped_by_patience"].astype(bool).any()
assert set(cap_plan["domain"].astype(str)).issubset(set(DEV_DOMAINS))

cap_plan = cap_plan.sort_values(["domain","fraction","seed"]).reset_index(drop=True)
cap_plan.to_csv(OUT/f"{EXPERIMENT}_FROZEN_CAP_HIT_PLAN.csv",index=False)

preflight = {
    "experiment":EXPERIMENT,
    "status":"R5A_PLAN_FROZEN",
    "source_experiment":SOURCE_EXPERIMENT,
    "source_curve_runs":int(len(source_runs)),
    "direct_epoch_cap_runs":int(len(cap_plan)),
    "budget_multipliers":BUDGET_MULTIPLIERS,
    "primary_scenarios":PRIMARY_SCENARIOS,
    "only_original_cap_hit_direct_runs_retrained":True,
    "prior_residual_retrained":False,
    "architecture_changed":False,
    "features_changed":False,
    "label_selector_changed":False,
    "prior_changed":False,
    "optimizer_changed":False,
    "learning_rate_changed":False,
    "patience_changed":False,
    "test_used_for_budget_selection":False,
    "one_continuous_trajectory_per_cap_condition":True,
    "one_x_reproduction_gate":{
        "max_abs_prediction_tolerance":REPRO_MAX_ABS_PRED_TOL,
        "mae_tolerance":REPRO_MAE_TOL,
    },
}
with open(OUT/f"{EXPERIMENT}_PREFLIGHT.json","w") as f:
    json.dump(preflight,f,indent=2)

display(cap_plan[["domain","fraction","seed","best_epoch","final_epoch","max_epochs","patience","test_MAE"]])
print("Cap-hit counts:", cap_plan.groupby("domain").size().to_dict())

# %% Code cell 4 (notebook cell 5)
# ---------- Load exact development-domain canonical feature contracts ----------

CANON = OUTPUTS / "model_screening_10" / "canonical"
MANIFEST_11B = find_unique_file(OUTPUTS, "11B_manifest.json")
with open(MANIFEST_11B) as f:
    man11b=json.load(f)

assert man11b["strict_causality"] is True
assert int(man11b["sequence_length"]) == SEQ_LEN
EXCLUDED_TOKENS=[str(x).lower() for x in man11b["feature_rule"]["excluded_tokens"]]

DOMAIN_DATA={}
DOMAIN_FEATURES={}

for domain in DEV_DOMAINS:
    p=CANON/f"{domain}_canonical.parquet"
    m=CANON/f"{domain}_metadata.json"
    assert p.exists() and m.exists()
    d=pd.read_parquet(p).copy()
    with open(m) as f:
        meta=json.load(f)
    raw_features=list(meta["feature_columns"])
    base_features=[c for c in raw_features if not any(t in c.lower() for t in EXCLUDED_TOKENS)]
    assert "soc_phy" not in base_features
    features=base_features+["soc_phy"]

    required={"vehicle_id","segment_id","time_value","elapsed_s","split","target","soc0","coord","soc_phy",*raw_features}
    missing=sorted(required-set(d.columns))
    assert not missing,(domain,missing)

    d=d[d["split"].isin(["train","val","test"])].copy()
    d["vehicle_id"]=d["vehicle_id"].astype(str)
    d["segment_id"]=d["segment_id"].astype(str)
    d["compound_segment"]=d["vehicle_id"]+"||"+d["segment_id"]
    d["soc_phy_frozen"]=d["soc_phy"].to_numpy(float)

    DOMAIN_DATA[domain]=d
    DOMAIN_FEATURES[domain]=features

    print(domain,d["split"].value_counts().to_dict(),"features",len(features))

# %% Code cell 5 (notebook cell 6)
class TrainOnlyPreprocessor:
    def fit(self,X):
        X=np.asarray(X,dtype=np.float64)
        med=np.nanmedian(X,axis=0)
        med=np.where(np.isfinite(med),med,0.0)
        Xi=np.where(np.isfinite(X),X,med[None,:])
        mean=Xi.mean(axis=0); std=Xi.std(axis=0)
        std=np.where(np.isfinite(std)&(std>1e-8),std,1.0)
        self.median_=med.astype(np.float32)
        self.mean_=mean.astype(np.float32)
        self.std_=std.astype(np.float32)
        return self
    def transform(self,X):
        X=np.asarray(X,dtype=np.float64)
        X=np.where(np.isfinite(X),X,self.median_[None,:])
        return ((X-self.mean_[None,:])/self.std_[None,:]).astype(np.float32)

class TargetScaler:
    def fit(self,y):
        y=np.asarray(y,dtype=np.float64)
        self.mean_=float(np.mean(y)); self.std_=float(np.std(y))
        if not np.isfinite(self.std_) or self.std_<1e-8:
            self.std_=1.0
        return self
    def transform(self,y):
        return ((np.asarray(y,float)-self.mean_)/self.std_).astype(np.float32)
    def inverse(self,y):
        return np.asarray(y,float)*self.std_+self.mean_

def prepare_split_frame(domain_df,split):
    z=domain_df[domain_df["split"]==split].copy()
    z=z.sort_values(["compound_segment","time_value"],kind="mergesort").reset_index(drop=True)
    z["seq_pos"]=np.arange(len(z),dtype=np.int64)
    return z

GROUP_COLS={
    "VED":[],
    "KU":["vehicle_id"],
    "SCUT":["vehicle_id"],
    "TUM":["vehicle_id"],
}

def apply_source_prior(df,domain,beta_map):
    z=df.copy()
    group_cols=GROUP_COLS[domain]
    if group_cols:
        keys=z[group_cols[0]].astype(str)
        bvec=keys.map(beta_map)
        assert not bvec.isna().any(),(domain,sorted(keys[bvec.isna()].unique().tolist()))
        beta_vec=bvec.to_numpy(float)
    else:
        beta_vec=np.full(len(z),float(beta_map["__POOLED__"]),dtype=float)
    z["soc_phy"]=z["soc0"].to_numpy(float)+beta_vec*z["coord"].to_numpy(float)
    return z

def build_condition_pack(source_row):
    domain=str(source_row.domain)
    fraction=float(source_row.fraction)
    seed=int(source_row.seed)
    base=DOMAIN_DATA[domain].copy()

    beta_json=str(source_row.beta_map_json) if pd.notna(source_row.beta_map_json) else ""
    if beta_json.strip() and beta_json.strip().lower()!="nan":
        beta_map=json.loads(beta_json)
        corrected=apply_source_prior(base,domain,beta_map)
        prior_mode="source_corrected_beta_map"
    else:
        # Verified 100% historical endpoint whose frozen prior passed the R2C P0 reproduction gate.
        corrected=base.copy()
        corrected["soc_phy"]=corrected["soc_phy_frozen"].to_numpy(float)
        beta_map={}
        prior_mode="verified_frozen_full_endpoint"

    train_frame=prepare_split_frame(corrected,"train")
    val_frame=prepare_split_frame(corrected,"val")
    test_frame=prepare_split_frame(corrected,"test")
    labeled=exact_sparse_subset(train_frame,fraction,seed).copy()

    src_selector=str(source_row.selector_sha256) if pd.notna(source_row.selector_sha256) else ""
    if src_selector.strip() and src_selector.strip().lower()!="nan":
        assert identity_sha256(labeled)==src_selector,(domain,fraction,seed,"selector mismatch")

    features=DOMAIN_FEATURES[domain]
    pre=TrainOnlyPreprocessor().fit(train_frame[features].to_numpy(float))

    return {
        "train_frame":train_frame,
        "val_frame":val_frame,
        "test_frame":test_frame,
        "labeled":labeled,
        "X_train":pre.transform(train_frame[features].to_numpy(float)),
        "X_val":pre.transform(val_frame[features].to_numpy(float)),
        "X_test":pre.transform(test_frame[features].to_numpy(float)),
        "prior_mode":prior_mode,
        "beta_map":beta_map,
    }

class CausalSequenceDataset(Dataset):
    def __init__(self,frame,X_scaled,endpoint_positions,target_scaler,seq_len):
        self.frame=frame.reset_index(drop=True)
        self.X=X_scaled
        self.positions=np.asarray(endpoint_positions,dtype=int)
        self.target_scaler=target_scaler
        self.seq_len=int(seq_len)
        self.seg_start=(self.frame.groupby("compound_segment",sort=False)["seq_pos"].min().astype(int).to_dict())
    def __len__(self):
        return len(self.positions)
    def __getitem__(self,i):
        pos=int(self.positions[i]); row=self.frame.iloc[pos]
        seg0=int(self.seg_start[row["compound_segment"]])
        start=max(seg0,pos-self.seq_len+1)
        x=self.X[start:pos+1]; n=len(x); pad=self.seq_len-n
        out=np.zeros((self.seq_len,self.X.shape[1]+1),dtype=np.float32)
        if n:
            out[pad:,:-1]=x
            out[pad:,-1]=1.0
        y=float(self.target_scaler.transform([float(row["target"])])[0])
        return torch.from_numpy(out),torch.tensor(y,dtype=torch.float32),pos

# %% Code cell 6 (notebook cell 7)
class DenseRecurrentNet(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.rnn1=nn.GRU(input_size=input_dim,hidden_size=GROWTH,num_layers=1,batch_first=True)
        self.rnn2=nn.GRU(input_size=input_dim+GROWTH,hidden_size=GROWTH,num_layers=1,batch_first=True)
        self.rnn3=nn.GRU(input_size=input_dim+2*GROWTH,hidden_size=GROWTH,num_layers=1,batch_first=True)
        self.feature_dropout=nn.Dropout(DROPOUT)
        self.dense_fusion=nn.Linear(DENSE_LAYERS*GROWTH,FUSION_WIDTH)
        self.input_skip=nn.Linear(input_dim,FUSION_WIDTH)
        self.fusion_norm=nn.LayerNorm(FUSION_WIDTH)
        self.head=nn.Sequential(
            nn.Linear(FUSION_WIDTH,32),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(32,1)
        )
    def forward(self,x):
        h1,_=self.rnn1(x); h1n=self.feature_dropout(h1)
        h2,_=self.rnn2(torch.cat([x,h1n],dim=-1)); h2n=self.feature_dropout(h2)
        h3,_=self.rnn3(torch.cat([x,h1n,h2n],dim=-1))
        endpoint=torch.cat([h1[:,-1,:],h2[:,-1,:],h3[:,-1,:]],dim=-1)
        fused=self.dense_fusion(endpoint)+self.input_skip(x[:,-1,:])
        fused=torch.relu(self.fusion_norm(fused))
        return self.head(fused).squeeze(-1)

def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,"cudnn"):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def capture_rng_state():
    st={"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"]=torch.cuda.get_rng_state_all()
    return st

def _cpu_byte_rng_tensor(x):
    if torch.is_tensor(x):
        return x.detach().to(device="cpu",dtype=torch.uint8).contiguous()
    return torch.as_tensor(x,dtype=torch.uint8,device="cpu").contiguous()

def restore_rng_state(st):
    if not st: return
    random.setstate(st["python"]); np.random.set_state(st["numpy"])
    torch.set_rng_state(_cpu_byte_rng_tensor(st["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in st:
        torch.cuda.set_rng_state_all([_cpu_byte_rng_tensor(x) for x in st["torch_cuda"]])

def safe_torch_load(path,map_location):
    try:
        return torch.load(path,map_location=map_location,weights_only=False)
    except TypeError:
        return torch.load(path,map_location=map_location)

def make_grad_scaler():
    if not AMP_ENABLED:
        try: return torch.amp.GradScaler("cuda",enabled=False)
        except Exception: return None
    try: return torch.amp.GradScaler("cuda",enabled=True)
    except Exception: return torch.cuda.amp.GradScaler(enabled=True)

def autocast_context():
    if not AMP_ENABLED:
        return contextlib.nullcontext()
    try: return torch.amp.autocast(device_type="cuda",enabled=True)
    except Exception: return torch.cuda.amp.autocast(enabled=True)

@torch.no_grad()
def predict_loader(model,loader,target_scaler):
    model.eval(); preds=[]; pos=[]
    for xb,_,p in loader:
        xb=xb.to(DEVICE,non_blocking=True)
        y=model(xb)
        preds.append(y.detach().cpu().numpy())
        pos.append(np.asarray(p))
    return np.concatenate(pos).astype(int),target_scaler.inverse(np.concatenate(preds))

# %% Code cell 7 (notebook cell 8)
def extended_contract_fingerprint(source_row):
    payload={
        "experiment":EXPERIMENT,
        "source_experiment":str(source_row.source_experiment),
        "domain":str(source_row.domain),
        "fraction":float(source_row.fraction),
        "seed":int(source_row.seed),
        "model":DIRECT,
        "source_checkpoint_sha256":str(source_row.checkpoint_sha256),
        "source_prediction_sha256":str(source_row.prediction_sha256),
        "selector_sha256":str(source_row.selector_sha256),
        "prior_contract_sha256":str(source_row.prior_contract_sha256),
        "seq_len":SEQ_LEN,
        "growth":GROWTH,
        "dense_layers":DENSE_LAYERS,
        "fusion_width":FUSION_WIDTH,
        "dropout":DROPOUT,
        "lr":LR,
        "weight_decay":WEIGHT_DECAY,
        "original_max_epochs":int(source_row.max_epochs),
        "extended_max_epochs":int(source_row.max_epochs)*4,
        "patience":int(source_row.patience),
        "budget_multipliers":BUDGET_MULTIPLIERS,
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode("utf-8")).hexdigest()

def evaluate_checkpoint(state_path, model, val_loader, test_loader, val_frame, test_frame, target_scaler):
    st=safe_torch_load(state_path,DEVICE)
    model.load_state_dict(st["model_state"])

    vpos,vraw=predict_loader(model,val_loader,target_scaler)
    vrows=val_frame.iloc[vpos]
    vmet=regression_metrics(vrows["target"].to_numpy(float),vraw)

    t0=time.perf_counter()
    tpos,traw=predict_loader(model,test_loader,target_scaler)
    infer=time.perf_counter()-t0
    trows=test_frame.iloc[tpos].copy()
    tmet=regression_metrics(trows["target"].to_numpy(float),traw)

    pred=trows[[
        "vehicle_id","segment_id","compound_segment","time_value","elapsed_s","target","soc_phy"
    ]].copy()
    pred["prediction"]=traw
    pred["error"]=traw-trows["target"].to_numpy(float)
    pred["abs_error"]=np.abs(pred["error"])
    return vmet,tmet,pred,infer,st

def train_extended_direct(source_row):
    domain=str(source_row.domain); fraction=float(source_row.fraction); seed=int(source_row.seed)
    set_all_seeds(seed)
    pack=build_condition_pack(source_row)

    train_frame=pack["train_frame"]; val_frame=pack["val_frame"]; test_frame=pack["test_frame"]
    labeled=pack["labeled"]
    target_scaler=TargetScaler().fit(labeled["target"].to_numpy(float))

    train_ds=CausalSequenceDataset(train_frame,pack["X_train"],labeled["seq_pos"].astype(int).to_numpy(),target_scaler,SEQ_LEN)
    val_ds=CausalSequenceDataset(val_frame,pack["X_val"],val_frame["seq_pos"].astype(int).to_numpy(),target_scaler,SEQ_LEN)
    test_ds=CausalSequenceDataset(test_frame,pack["X_test"],test_frame["seq_pos"].astype(int).to_numpy(),target_scaler,SEQ_LEN)

    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=AMP_ENABLED)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)
    test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=AMP_ENABLED)

    input_dim=len(DOMAIN_FEATURES[domain])+1
    model=DenseRecurrentNet(input_dim).to(DEVICE)
    assert count_params(model)==int(source_row.n_params),(domain,fraction,seed,count_params(model),source_row.n_params)

    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    criterion=nn.MSELoss()
    scaler=make_grad_scaler()

    cap1=int(source_row.max_epochs); cap2=2*cap1; cap4=4*cap1
    patience=int(source_row.patience)
    fp=extended_contract_fingerprint(source_row)

    tag=f"{domain}_fraction{fraction}_DirectPlusPrior_seed{seed}"
    best_path=CKPT_DIR/f"{tag}_global_best.pt"
    resume_path=CKPT_DIR/f"{tag}_resume.pt"
    history_path=OUT/f"history_{tag}.csv"
    milestone_paths={
        1:CKPT_DIR/f"{tag}_best_at_1x.pt",
        2:CKPT_DIR/f"{tag}_best_at_2x.pt",
        4:CKPT_DIR/f"{tag}_best_at_4x.pt",
    }

    best_val=np.inf; best_epoch=0; stale=0; history=[]; start_epoch=1; elapsed_before=0.0

    if resume_path.exists():
        st=safe_torch_load(resume_path,DEVICE)
        assert st["contract_fingerprint"]==fp
        model.load_state_dict(st["latest_model_state"])
        optimizer.load_state_dict(st["optimizer_state"])
        if scaler is not None and st.get("scaler_state"):
            scaler.load_state_dict(st["scaler_state"])
        start_epoch=int(st["epoch"])+1
        best_val=float(st["best_val"]); best_epoch=int(st["best_epoch"]); stale=int(st["stale"])
        history=list(st["history"]); elapsed_before=float(st.get("elapsed_seconds",0.0))
        restore_rng_state(st.get("rng_state"))
        print("RESUME",tag,"epoch",start_epoch)

    t0=time.perf_counter()
    stop_reason="4x_cap"
    terminal_resume = bool(
        history and (
            start_epoch > cap4 or
            (start_epoch > cap1 and stale >= patience)
        )
    )
    if terminal_resume and start_epoch <= cap4:
        stop_reason="patience_after_1x"

    epoch_iter = [] if terminal_resume else range(start_epoch,cap4+1)

    for epoch in epoch_iter:
        model.train(); losses=[]
        for xb,yb,_ in train_loader:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                pred=model(xb); loss=criterion(pred,yb)
            if not torch.isfinite(loss):
                raise FloatingPointError((tag,epoch,"non-finite loss"))
            if scaler is None or not AMP_ENABLED:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); optimizer.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))

        vpos,vraw=predict_loader(model,val_loader,target_scaler)
        vrows=val_frame.iloc[vpos]
        val_mae=regression_metrics(vrows["target"].to_numpy(float),vraw)["MAE"]
        history.append({"epoch":int(epoch),"train_scaled_target_MSE":float(np.mean(losses)),"val_SOC_MAE":float(val_mae)})

        if val_mae < best_val - MIN_DELTA:
            best_val=float(val_mae); best_epoch=int(epoch); stale=0
            torch.save({
                "model_state":model.state_dict(),
                "best_epoch":best_epoch,
                "best_val_SOC_MAE":best_val,
                "contract_fingerprint":fp,
                "source_run_key":[domain,fraction,DIRECT,seed],
                "n_params":count_params(model),
            },best_path)
        else:
            stale+=1

        if epoch in [cap1,cap2]:
            assert best_path.exists()
            shutil.copy2(best_path,milestone_paths[1 if epoch==cap1 else 2])

        if epoch % RESUME_SAVE_EVERY==0 or epoch in [cap1,cap2,cap4] or (epoch>cap1 and stale>=patience):
            pd.DataFrame(history).to_csv(history_path,index=False)
            torch.save({
                "latest_model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "scaler_state":scaler.state_dict() if scaler is not None else {},
                "epoch":int(epoch),"best_val":float(best_val),"best_epoch":int(best_epoch),
                "stale":int(stale),"history":history,"rng_state":capture_rng_state(),
                "elapsed_seconds":float(elapsed_before+(time.perf_counter()-t0)),
                "contract_fingerprint":fp,
            },resume_path)

        # The original run was capped at cap1. Before cap1, reproduce it exactly; do not
        # introduce a different early-stop decision into this sensitivity experiment.
        if epoch < cap1 and stale>=patience:
            raise RuntimeError((tag,"1x reproduction stopped by patience before original cap",epoch,stale,patience))

        if epoch > cap1 and stale>=patience:
            stop_reason="patience_after_1x"
            break

    final_epoch=int(history[-1]["epoch"])
    assert best_path.exists()

    # Snapshot missing later scenarios at the actual early-stop endpoint.
    if not milestone_paths[1].exists():
        raise RuntimeError((tag,"missing 1x milestone"))
    if not milestone_paths[2].exists():
        shutil.copy2(best_path,milestone_paths[2])
    shutil.copy2(best_path,milestone_paths[4])

    rows=[]
    for mult in BUDGET_MULTIPLIERS:
        vmet,tmet,pred,infer,st=evaluate_checkpoint(
            milestone_paths[mult],model,val_loader,test_loader,val_frame,test_frame,target_scaler
        )
        pred["domain"]=domain; pred["fraction"]=fraction; pred["model"]=DIRECT; pred["seed"]=seed
        pred["budget_multiplier"]=mult
        pred_path=PRED_DIR/f"{EXPERIMENT}_{tag}_{mult}x.parquet"
        pred.to_parquet(pred_path,index=False)
        rows.append({
            "domain":domain,"fraction":fraction,"model":DIRECT,"seed":seed,
            "budget_multiplier":int(mult),
            "source_original_max_epochs":cap1,
            "scenario_max_epochs":int(mult*cap1),
            "actual_training_final_epoch":final_epoch,
            "best_epoch":int(st["best_epoch"]),
            "stopped_after_1x_by_patience":bool(stop_reason=="patience_after_1x"),
            "stop_reason":stop_reason,
            "patience":patience,
            "n_params":count_params(model),
            "n_train_labels":len(train_ds),"n_validation":len(val_ds),"n_test":len(test_ds),
            "prior_mode":pack["prior_mode"],
            "contract_fingerprint":fp,
            "checkpoint":str(milestone_paths[mult]),
            "checkpoint_sha256":sha256_file(milestone_paths[mult]),
            "prediction_file":str(pred_path),
            "prediction_sha256":sha256_file(pred_path),
            "val_MAE":vmet["MAE"],"val_RMSE":vmet["RMSE"],"val_P95":vmet["P95"],"val_Bias":vmet["Bias"],
            "test_MAE":tmet["MAE"],"test_RMSE":tmet["RMSE"],"test_P95":tmet["P95"],"test_Bias":tmet["Bias"],
            "inference_seconds":infer,
        })
    return rows

# %% Code cell 8 (notebook cell 9)
# ---------- Train/resume all 31 cap-hit Direct conditions ----------

RUNS_INCREMENTAL=OUT/f"{EXPERIMENT}_EXTENDED_RUNS_INCREMENTAL.csv"

done=set()
if RUNS_INCREMENTAL.exists():
    old=pd.read_csv(RUNS_INCREMENTAL)
    for r in old.itertuples(index=False):
        key=(str(r.domain),float(r.fraction),int(r.seed),int(r.budget_multiplier))
        try:
            pp=Path(str(r.prediction_file)); cp=Path(str(r.checkpoint))
            if pp.exists() and cp.exists() and sha256_file(pp)==str(r.prediction_sha256) and sha256_file(cp)==str(r.checkpoint_sha256):
                done.add(key)
        except Exception:
            pass

for src in cap_plan.itertuples(index=False):
    condition_keys=[(str(src.domain),float(src.fraction),int(src.seed),m) for m in BUDGET_MULTIPLIERS]
    if all(k in done for k in condition_keys):
        print("SKIP completed",condition_keys[0][:3])
        continue

    print("\nEXTEND",src.domain,"fraction",src.fraction,"seed",src.seed,
          "budget",src.max_epochs,"->",int(src.max_epochs)*4)
    rows=train_extended_direct(src)

    cur=pd.read_csv(RUNS_INCREMENTAL) if RUNS_INCREMENTAL.exists() else pd.DataFrame()
    cur=pd.concat([cur,pd.DataFrame(rows)],ignore_index=True)
    cur=cur.drop_duplicates(["domain","fraction","model","seed","budget_multiplier"],keep="last")
    cur.to_csv(RUNS_INCREMENTAL,index=False)

    for rr in rows:
        done.add((rr["domain"],float(rr["fraction"]),int(rr["seed"]),int(rr["budget_multiplier"])))

    gc.collect()
    if DEVICE.type=="cuda":
        torch.cuda.empty_cache()

extended_runs=pd.read_csv(RUNS_INCREMENTAL)
assert len(extended_runs)==31*3,(len(extended_runs),31*3)
extended_runs.to_csv(OUT/f"{EXPERIMENT}_EXTENDED_RUNS.csv",index=False)

print("Extended run snapshots complete:",len(extended_runs))

# %% Code cell 9 (notebook cell 10)
# ---------- Mandatory 1x reproduction audit ----------

PHYS=["vehicle_id","segment_id","compound_segment","time_value","elapsed_s","target"]
audit=[]

for src in cap_plan.itertuples(index=False):
    rr=extended_runs[
        (extended_runs.domain.astype(str)==str(src.domain)) &
        np.isclose(extended_runs.fraction.astype(float),float(src.fraction)) &
        (extended_runs.seed.astype(int)==int(src.seed)) &
        (extended_runs.budget_multiplier.astype(int)==1)
    ]
    assert len(rr)==1
    rr=rr.iloc[0]

    old=pd.read_parquet(Path(str(src.prediction_file))).copy()
    new=pd.read_parquet(Path(str(rr.prediction_file))).copy()
    assert not old.duplicated(PHYS).any()
    assert not new.duplicated(PHYS).any()

    q=old[PHYS+["prediction"]].merge(
        new[PHYS+["prediction"]],
        on=PHYS,how="inner",validate="one_to_one",suffixes=("_source","_retrain")
    )
    assert len(q)==len(old)==len(new),(src.domain,src.fraction,src.seed,len(q),len(old),len(new))

    diff=np.abs(q.prediction_retrain.to_numpy(float)-q.prediction_source.to_numpy(float))
    mae_diff=abs(float(rr.test_MAE)-float(src.test_MAE))
    passed=bool(diff.max()<=REPRO_MAX_ABS_PRED_TOL and mae_diff<=REPRO_MAE_TOL)

    audit.append({
        "domain":src.domain,"fraction":float(src.fraction),"seed":int(src.seed),
        "source_best_epoch":int(src.best_epoch),"retrain_1x_best_epoch":int(rr.best_epoch),
        "source_test_MAE":float(src.test_MAE),"retrain_1x_test_MAE":float(rr.test_MAE),
        "abs_test_MAE_difference":mae_diff,
        "max_abs_prediction_difference":float(diff.max()),
        "mean_abs_prediction_difference":float(diff.mean()),
        "reproduction_pass":passed,
    })

repro=pd.DataFrame(audit)
repro.to_csv(OUT/f"{EXPERIMENT}_1X_REPRODUCTION_AUDIT.csv",index=False)
display(repro)

assert repro.reproduction_pass.all(), (
    "1x reproduction gate failed; do not interpret 2x/4x sensitivity until fixed",
    repro[~repro.reproduction_pass].to_dict("records")
)
print("1x reproduction gate: PASS",len(repro),"/",len(repro))

# %% Code cell 10 (notebook cell 11)
# ---------- Per-run effect of more Direct training ----------

wide=extended_runs.pivot_table(
    index=["domain","fraction","seed"],
    columns="budget_multiplier",
    values=["test_MAE","test_RMSE","test_P95","best_epoch"],
    aggfunc="first",
).reset_index()

wide.columns=[
    "_".join(str(x) for x in c if str(x)!="") if isinstance(c,tuple) else str(c)
    for c in wide.columns
]

for mult in [2,4]:
    wide[f"MAE_change_{mult}x_minus_1x"]=wide[f"test_MAE_{mult}"]-wide["test_MAE_1"]
    wide[f"MAE_improvement_pct_{mult}x_vs_1x"]=100.0*(wide["test_MAE_1"]-wide[f"test_MAE_{mult}"])/wide["test_MAE_1"]
    wide[f"improved_{mult}x"]=wide[f"test_MAE_{mult}"]<wide["test_MAE_1"]

wide.to_csv(OUT/f"{EXPERIMENT}_CAP_RUN_IMPROVEMENT.csv",index=False)
display(wide)

print("2x improved runs:",int(wide["improved_2x"].sum()),"/",len(wide))
print("4x improved runs:",int(wide["improved_4x"].sum()),"/",len(wide))

# %% Code cell 11 (notebook cell 12)
# ---------- Construct 2x/4x full-curve run tables and corrected seed averages ----------

source_seedavg=pd.read_parquet(SRC_SEEDAVG).copy()
assert not source_seedavg.duplicated([
    "domain","fraction","model","vehicle_id","segment_id","compound_segment","time_value","elapsed_s","target"
]).any()

METRIC_COLS=["val_MAE","val_RMSE","val_P95","val_Bias","test_MAE","test_RMSE","test_P95","test_Bias"]

def scenario_runs(mult):
    out=source_runs.copy()
    repl=extended_runs[extended_runs.budget_multiplier.astype(int)==int(mult)].copy()
    key=["domain","fraction","model","seed"]
    m=out.merge(repl[key+METRIC_COLS+["best_epoch","actual_training_final_epoch","scenario_max_epochs"]],
                on=key,how="left",suffixes=("","_ext"),validate="one_to_one")
    has=m["test_MAE_ext"].notna()
    for c in METRIC_COLS:
        m.loc[has,c]=m.loc[has,f"{c}_ext"]
    m.loc[has,"best_epoch"]=m.loc[has,"best_epoch_ext"].astype(int)
    m.loc[has,"final_epoch"]=m.loc[has,"actual_training_final_epoch"].astype(int)
    m.loc[has,"max_epochs"]=m.loc[has,"scenario_max_epochs"].astype(int)
    m["extended_budget_replacement"]=has
    drop=[c for c in m.columns if c.endswith("_ext") or c in ["actual_training_final_epoch","scenario_max_epochs"]]
    return m.drop(columns=drop)

def scenario_seedavg(mult):
    delta_parts=[]
    repl=extended_runs[extended_runs.budget_multiplier.astype(int)==int(mult)].copy()
    for rr in repl.itertuples(index=False):
        src=cap_plan[
            (cap_plan.domain.astype(str)==str(rr.domain)) &
            np.isclose(cap_plan.fraction.astype(float),float(rr.fraction)) &
            (cap_plan.seed.astype(int)==int(rr.seed))
        ]
        assert len(src)==1
        src=src.iloc[0]
        old=pd.read_parquet(Path(str(src.prediction_file)))[PHYS+["prediction"]].copy()
        new=pd.read_parquet(Path(str(rr.prediction_file)))[PHYS+["prediction"]].copy()
        d=old.merge(new,on=PHYS,how="inner",validate="one_to_one",suffixes=("_old","_new"))
        assert len(d)==len(old)==len(new)
        d["prediction_delta"]=(d.prediction_new-d.prediction_old)/len(SEEDS)
        d["domain"]=str(rr.domain); d["fraction"]=float(rr.fraction); d["model"]=DIRECT
        delta_parts.append(d[["domain","fraction","model"]+PHYS+["prediction_delta"]])

    delta=pd.concat(delta_parts,ignore_index=True)
    key=["domain","fraction","model"]+PHYS
    delta=delta.groupby(key,as_index=False,dropna=False)["prediction_delta"].sum()

    base=source_seedavg.copy()
    base["_order"]=np.arange(len(base))
    q=base.merge(delta,on=key,how="left",validate="one_to_one")
    q["prediction_delta"]=q["prediction_delta"].fillna(0.0)
    q["prediction"]=q["prediction"].to_numpy(float)+q["prediction_delta"].to_numpy(float)
    q["abs_error"]=np.abs(q["prediction"].to_numpy(float)-q["target"].to_numpy(float))
    q=q.sort_values("_order").drop(columns=["_order","prediction_delta"]).reset_index(drop=True)
    assert len(q)==len(base)
    return q,delta

scenario_run_tables={}
scenario_seedavgs={}

for mult in PRIMARY_SCENARIOS:
    sr=scenario_runs(mult)
    sp,delta=scenario_seedavg(mult)
    scenario_run_tables[mult]=sr
    scenario_seedavgs[mult]=sp
    sr.to_csv(OUT/f"{EXPERIMENT}_{mult}X_FULL_CURVE_RUNS.csv",index=False)
    sp.to_parquet(OUT/f"{EXPERIMENT}_{mult}X_SEED_AVERAGED_PREDICTIONS.parquet",index=False)
    delta.to_parquet(OUT/f"{EXPERIMENT}_{mult}X_SEEDAVG_DELTA_AUDIT.parquet",index=False)
    print(mult,"x changed physical seed-average rows:",len(delta))

# %% Code cell 12 (notebook cell 13)
# ---------- Recompute curve summaries, AULEC and primary 5-domain bootstrap ----------

def build_curve(run_table):
    curve=(run_table.groupby(["domain","fraction","model"],as_index=False)
           .agg(
               MAE_mean=("test_MAE","mean"),MAE_std=("test_MAE","std"),
               RMSE_mean=("test_RMSE","mean"),RMSE_std=("test_RMSE","std"),
               P95_mean=("test_P95","mean"),Bias_mean=("test_Bias","mean"),
           ))
    curve=curve.merge(budget_summary,on=["domain","fraction"],how="left",validate="many_to_one")
    return curve

def descriptive_aulec(curve):
    rows=[]
    for domain in ALL_DOMAINS:
        b=budget_summary[budget_summary.domain.astype(str)==domain].sort_values("fraction")
        x_nom=b.actual_nominal_fraction_mean.to_numpy(float)
        x_eff=b.effective_fit_fraction_mean.to_numpy(float)
        for model in MODELS:
            q=curve[(curve.domain.astype(str)==domain)&(curve.model.astype(str)==model)].sort_values("fraction")
            y=q.MAE_mean.to_numpy(float)
            assert len(y)==7
            rows.append({"domain":domain,"model":model,
                         "AULEC_log_nominal":normalized_log_auc(x_nom,y),
                         "AULEC_log_effective":normalized_log_auc(x_eff,y)})
    return pd.DataFrame(rows)

def bootstrap_domain_aulec(seedavg,domain,draws=BOOTSTRAP_DRAWS,seed=7301):
    seg=(seedavg[seedavg.domain.astype(str)==domain]
         .groupby(["fraction","model","compound_segment"],as_index=False)["abs_error"].mean())
    segments=sorted(seg.compound_segment.unique().tolist())
    fractions=np.asarray(FRACTIONS,float)
    b=budget_summary[budget_summary.domain.astype(str)==domain].sort_values("fraction")
    x_nom=b.actual_nominal_fraction_mean.to_numpy(float)
    x_eff=b.effective_fit_fraction_mean.to_numpy(float)
    mats={}
    for model in MODELS:
        p=(seg[seg.model.astype(str)==model]
           .pivot(index="compound_segment",columns="fraction",values="abs_error")
           .reindex(index=segments,columns=fractions))
        assert not p.isna().any().any(),(domain,model)
        mats[model]=p.to_numpy(float)
    D=mats[DIRECT]; P=mats[PR]
    de=normalized_log_auc(x_eff,D.mean(axis=0)); pe=normalized_log_auc(x_eff,P.mean(axis=0))
    dn=normalized_log_auc(x_nom,D.mean(axis=0)); pn=normalized_log_auc(x_nom,P.mean(axis=0))
    rng=np.random.default_rng(seed); n=len(segments)
    be=np.empty(draws); bn=np.empty(draws)
    for i in range(draws):
        ix=rng.integers(0,n,size=n)
        dc=D[ix].mean(axis=0); pc=P[ix].mean(axis=0)
        be[i]=normalized_log_auc(x_eff,pc)-normalized_log_auc(x_eff,dc)
        bn[i]=normalized_log_auc(x_nom,pc)-normalized_log_auc(x_nom,dc)
    return {
        "domain":domain,"n_segments":n,
        "Direct_effective_AULEC_segment_balanced":de,
        "PR_effective_AULEC_segment_balanced":pe,
        "delta_PR_minus_Direct_effective_AULEC":pe-de,
        "effective_CI95_low":float(np.percentile(be,2.5)),
        "effective_CI95_high":float(np.percentile(be,97.5)),
        "effective_two_sided_bootstrap_p":bootstrap_two_sided_p(be),
        "P_PR_lower_effective_AULEC":float(np.mean(be<0)),
        "Direct_nominal_AULEC_segment_balanced":dn,
        "PR_nominal_AULEC_segment_balanced":pn,
        "delta_PR_minus_Direct_nominal_AULEC":pn-dn,
        "nominal_CI95_low":float(np.percentile(bn,2.5)),
        "nominal_CI95_high":float(np.percentile(bn,97.5)),
        "nominal_two_sided_bootstrap_p":bootstrap_two_sided_p(bn),
    }

all_boot=[]
all_curve=[]
all_aulec=[]

for mult in PRIMARY_SCENARIOS:
    curve=build_curve(scenario_run_tables[mult]); curve["budget_scenario"]=f"{mult}x"
    a=descriptive_aulec(curve); a["budget_scenario"]=f"{mult}x"
    rows=[bootstrap_domain_aulec(scenario_seedavgs[mult],d,seed=7301+i*101) for i,d in enumerate(ALL_DOMAINS)]
    b=pd.DataFrame(rows)
    b["Holm_p_effective_AULEC"]=holm_adjust(b.effective_two_sided_bootstrap_p.to_numpy(float))
    b["effective_AULEC_supported_after_Holm"]=(
        (b.effective_CI95_high<0)&(b.Holm_p_effective_AULEC<0.05)
    )
    b["budget_scenario"]=f"{mult}x"

    all_curve.append(curve); all_aulec.append(a); all_boot.append(b)

curve_sens=pd.concat(all_curve,ignore_index=True)
aulec_sens=pd.concat(all_aulec,ignore_index=True)
boot_sens=pd.concat(all_boot,ignore_index=True)

curve_sens.to_csv(OUT/f"{EXPERIMENT}_CURVE_SUMMARY.csv",index=False)
aulec_sens.to_csv(OUT/f"{EXPERIMENT}_DESCRIPTIVE_AULEC.csv",index=False)
boot_sens.to_csv(OUT/f"{EXPERIMENT}_PRIMARY_EFFECTIVE_AULEC_BOOTSTRAP.csv",index=False)

display(boot_sens)

# %% Code cell 13 (notebook cell 14)
# ---------- Pointwise Direct-vs-PR impact and comparison to original R2C ----------

effects=[]
for mult in PRIMARY_SCENARIOS:
    c=curve_sens[curve_sens.budget_scenario==f"{mult}x"].copy()
    d=c[c.model==DIRECT][["domain","fraction","MAE_mean","MAE_std"]].rename(
        columns={"MAE_mean":"Direct_MAE","MAE_std":"Direct_MAE_std"})
    p=c[c.model==PR][["domain","fraction","MAE_mean","MAE_std"]].rename(
        columns={"MAE_mean":"PR_MAE","MAE_std":"PR_MAE_std"})
    e=d.merge(p,on=["domain","fraction"],validate="one_to_one")
    e["PR_improvement_pct"]=100.0*(e.Direct_MAE-e.PR_MAE)/e.Direct_MAE
    e["PR_lower_MAE"]=e.PR_MAE<e.Direct_MAE
    e["budget_scenario"]=f"{mult}x"
    effects.append(e)

effects=pd.concat(effects,ignore_index=True)
effects.to_csv(OUT/f"{EXPERIMENT}_POINTWISE_FACTORIZATION_EFFECTS.csv",index=False)

base=source_primary[[
    "domain","Direct_effective_AULEC_segment_balanced","PR_effective_AULEC_segment_balanced",
    "delta_PR_minus_Direct_effective_AULEC","effective_CI95_low","effective_CI95_high",
    "Holm_p_effective_AULEC","effective_AULEC_supported_after_Holm"
]].copy()
base["budget_scenario"]="original_1x"

comparison=pd.concat([base,boot_sens[base.columns]],ignore_index=True)
comparison.to_csv(OUT/f"{EXPERIMENT}_ORIGINAL_VS_EXTENDED_AULEC.csv",index=False)

display(effects)
display(comparison)

# %% Code cell 14 (notebook cell 15)
# ---------- Secondary vehicle-cluster-aware AULEC sensitivity (R4-compatible logic) ----------

def exact_vehicle_signflip_p(vehicle_deltas):
    import itertools
    d=np.asarray(vehicle_deltas,float)
    d=d[np.isfinite(d)]
    n=len(d)
    if n==0: return np.nan
    obs=abs(float(np.mean(d)))
    if n<=20:
        vals=[]
        for signs in itertools.product((-1.0,1.0),repeat=n):
            vals.append(abs(float(np.mean(d*np.asarray(signs,float)))))
        vals=np.asarray(vals,float)
        return float(np.mean(vals>=obs-1e-15))
    rng=np.random.default_rng(88991+n)
    B=100000
    vals=np.empty(B)
    for i in range(B):
        vals[i]=abs(float(np.mean(d*rng.choice([-1.0,1.0],size=n))))
    return float((np.sum(vals>=obs-1e-15)+1)/(B+1))

def cluster_domain_aulec(seedavg,domain,draws,seed):
    q=seedavg[(seedavg.domain.astype(str)==domain)&seedavg.model.isin(MODELS)].copy()
    q["vehicle_id"]=q.vehicle_id.astype(str)
    s=(q.groupby(["vehicle_id","compound_segment","fraction","model"],as_index=False)["abs_error"].mean())
    b=budget_summary[budget_summary.domain.astype(str)==domain].sort_values("fraction")
    x=b.effective_fit_fraction_mean.to_numpy(float)
    fracs=np.asarray(FRACTIONS,float)
    vehicles=sorted(s.vehicle_id.unique().tolist())
    per_v={}; vehicle_deltas=[]
    for v in vehicles:
        sv=s[s.vehicle_id==v]
        segs=sorted(sv.compound_segment.unique().tolist())
        mats={}
        for model in MODELS:
            p=(sv[sv.model==model].pivot(index="compound_segment",columns="fraction",values="abs_error")
               .reindex(index=segs,columns=fracs))
            assert not p.isna().any().any(),(domain,v,model)
            mats[model]=p.to_numpy(float)
        D,P=mats[DIRECT],mats[PR]
        per_v[v]=(D,P)
        vehicle_deltas.append(normalized_log_auc(x,P.mean(axis=0))-normalized_log_auc(x,D.mean(axis=0)))

    rng=np.random.default_rng(seed); nv=len(vehicles)
    boot=np.empty(draws,float)
    for bi in range(draws):
        sampled_v=rng.integers(0,nv,size=nv)
        vals=[]
        for vi in sampled_v:
            D,P=per_v[vehicles[int(vi)]]
            ix=rng.integers(0,len(D),size=len(D))
            vals.append(normalized_log_auc(x,P[ix].mean(axis=0))-normalized_log_auc(x,D[ix].mean(axis=0)))
        boot[bi]=float(np.mean(vals))

    return {
        "domain":domain,"n_vehicles":nv,
        "vehicle_balanced_delta_PR_minus_Direct_AULEC":float(np.mean(vehicle_deltas)),
        "CI95_low":float(np.percentile(boot,2.5)),
        "CI95_high":float(np.percentile(boot,97.5)),
        "hierarchical_bootstrap_p":bootstrap_two_sided_p(boot),
        "exact_vehicle_signflip_p":exact_vehicle_signflip_p(vehicle_deltas),
        "population_inference_eligible":bool(nv>=MIN_VEHICLES_FOR_POPULATION_INFERENCE),
    }

cluster_all=[]
for mult in PRIMARY_SCENARIOS:
    rows=[cluster_domain_aulec(scenario_seedavgs[mult],d,CLUSTER_BOOTSTRAP_DRAWS,91001+i*101)
          for i,d in enumerate(ALL_DOMAINS)]
    c=pd.DataFrame(rows)
    c["Holm_p_hierarchical_full_family"]=holm_adjust(c.hierarchical_bootstrap_p.to_numpy(float))
    c["PR_supported_population_cluster_bootstrap"]=(
        c.population_inference_eligible&(c.CI95_high<0)&(c.Holm_p_hierarchical_full_family<0.05)
    )
    c["Direct_supported_population_cluster_bootstrap"]=(
        c.population_inference_eligible&(c.CI95_low>0)&(c.Holm_p_hierarchical_full_family<0.05)
    )
    c["budget_scenario"]=f"{mult}x"
    cluster_all.append(c)

cluster_sens=pd.concat(cluster_all,ignore_index=True)
cluster_sens.to_csv(OUT/f"{EXPERIMENT}_CLUSTER_AWARE_AULEC_SENSITIVITY.csv",index=False)
display(cluster_sens)

# %% Code cell 15 (notebook cell 16)
# ---------- Final fail-closed manifest and manuscript notes ----------

assert len(extended_runs)==93
assert repro.reproduction_pass.all()
assert len(boot_sens)==10
assert len(cluster_sens)==10
assert set(boot_sens.budget_scenario)=={"2x","4x"}

summary={}
for mult in PRIMARY_SCENARIOS:
    q=boot_sens[boot_sens.budget_scenario==f"{mult}x"]
    cq=cluster_sens[cluster_sens.budget_scenario==f"{mult}x"]
    summary[f"{mult}x"]={
        "segment_bootstrap_PR_supported_domains_after_Holm":int(q.effective_AULEC_supported_after_Holm.sum()),
        "population_eligible_domains":int(cq.population_inference_eligible.sum()),
        "cluster_bootstrap_PR_supported_population_domains":int(cq.PR_supported_population_cluster_bootstrap.sum()),
        "cluster_bootstrap_Direct_supported_population_domains":int(cq.Direct_supported_population_cluster_bootstrap.sum()),
        "direct_cap_runs_improved_vs_1x":int(wide[f"improved_{mult}x"].sum()),
        "direct_cap_runs_total":int(len(wide)),
    }

manifest={
    **preflight,
    "status":"R5A_EXTENDED_BUDGET_DIRECT_SENSITIVITY_COMPLETE",
    "one_x_reproduction_passed":True,
    "one_x_reproduction_conditions":int(len(repro)),
    "extended_snapshot_rows":int(len(extended_runs)),
    "source_direct_cap_conditions":int(len(cap_plan)),
    "bootstrap_draws":BOOTSTRAP_DRAWS,
    "cluster_bootstrap_draws":CLUSTER_BOOTSTRAP_DRAWS,
    "holm_family_size_per_scenario":5,
    "results":summary,
    "all_results_reported":True,
}
with open(OUT/f"{EXPERIMENT}_MANIFEST.json","w") as f:
    json.dump(manifest,f,indent=2)

notes=[
    "# R5A Extended-Budget Direct Sensitivity","",
    "## Purpose","",
    "This analysis tests whether the corrected R2C Prior–Residual label-efficiency advantage can be explained by Direct+Prior runs reaching the original epoch ceiling.",
    "",
    "## Protocol","",
    f"- Only the {len(cap_plan)} Direct+Prior runs that hit the original R2C epoch cap are retrained.",
    "- Architecture, features, retained-label selector, structured prior, train/validation/test split, optimizer, learning rate, seed and patience are unchanged.",
    "- Fixed 2× and 4× maximum-epoch budgets are both reported; no test-driven budget selection is performed.",
    "- PriorResidual+Prior is never retrained.",
    "- A mandatory 1× reproduction gate must pass before extended-budget outcomes are interpreted.",
    "",
    "## Results","",
]
for mult in PRIMARY_SCENARIOS:
    q=boot_sens[boot_sens.budget_scenario==f"{mult}x"]
    cq=cluster_sens[cluster_sens.budget_scenario==f"{mult}x"]
    notes += [
        f"### {mult}× Direct budget","",
        f"- Cap-hit Direct runs with lower test MAE than their reproduced 1× run: {int(wide[f'improved_{mult}x'].sum())}/{len(wide)}.",
        f"- R2C effective-AULEC domains supporting Prior–Residual after Holm: {int(q.effective_AULEC_supported_after_Holm.sum())}/5.",
        f"- Population-eligible domains in the cluster-aware sensitivity: {int(cq.population_inference_eligible.sum())}/5.",
        f"- Prior–Residual supported in population-eligible domains: {int(cq.PR_supported_population_cluster_bootstrap.sum())}/{int(cq.population_inference_eligible.sum())}.",
        f"- Direct supported in population-eligible domains: {int(cq.Direct_supported_population_cluster_bootstrap.sum())}/{int(cq.population_inference_eligible.sum())}.",
        "",
    ]
notes += [
    "## Interpretation boundary","",
    "This is a training-budget sensitivity analysis, not a new model-selection stage. It is intended to test optimization-budget confounding in the R2C factorization claim."
]
(OUT/f"{EXPERIMENT}_MANUSCRIPT_NOTES.md").write_text("\n".join(notes),encoding="utf-8")

print(json.dumps(manifest,indent=2))
print("\n".join(notes))

# %% [markdown] Expected output directory
