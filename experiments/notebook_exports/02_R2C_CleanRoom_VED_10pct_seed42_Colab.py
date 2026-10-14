# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/primary/02_R2C_CleanRoom_VED_10pct_seed42_Colab.ipynb
# Notebook SHA256: c5d5ee2ff3d2df333660c6e4db617603f8f074d720c066bdee54e79116085ebf
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 02 — R2C Clean-Room Pilot: VED, 10% labels, seed 42

# %% Code cell 1 (notebook cell 2)
# Environment and EXACT experiment configuration.
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, random, time, gc, contextlib, warnings, platform, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
ROOT = Path('/content/drive/MyDrive/NissanLeaf')
OUTPUTS = ROOT / 'outputs'
CANON = OUTPUTS / 'model_screening_10' / 'canonical'
REV1_OUT = ROOT / 'Revizyon1' / 'outputs'
REV2_OUT = ROOT / 'Revizyon2' / 'outputs'
STAGING = ROOT / 'Revizyon4_GitHub_Reproduction_STAGING'
assert STAGING.is_dir(), STAGING
run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = STAGING / 'runs' / f'02_R2C_VED_10pct_seed42_{run_stamp}'
OUT.mkdir(parents=True, exist_ok=False)
CKPT_DIR = OUT / 'checkpoints'
PRED_DIR = OUT / 'seed_predictions'
for directory in (CKPT_DIR, PRED_DIR): directory.mkdir()

# The original source's experiment ID is preserved for a compatible contract fingerprint.
EXPERIMENT = 'R2C_CORRECTED_SPARSE_PRIOR_CURVE'
DEV_DOMAINS = ['VED','KU','SCUT','TUM']
ALL_DOMAINS = DEV_DOMAINS + ['EVE']
DOMAIN = 'VED'
FRACTION = 0.10
SEED = 42
FRACTIONS = [0.01,0.02,0.05,0.10,0.20,0.50,1.00]
SEEDS = [42,43,44,45,46]
DIRECT_STD = 'Direct+Prior'
PR_STD = 'PriorResidual+Prior'
MODELS_STD = [DIRECT_STD, PR_STD]
SEQ_LEN = 16
DENSE_LAYERS, GROWTH, FUSION_WIDTH = 3,19,64
EVE_HIDDEN, EVE_HEAD_HIDDEN = 64,32
DROPOUT, LR, WEIGHT_DECAY = 0.10,1e-3,1e-5
MIN_DELTA, GRAD_CLIP, RESUME_SAVE_EVERY = 1e-6,5.0,10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ENABLED = DEVICE.type == 'cuda'
BATCH_SIZE = 512 if AMP_ENABLED else 128
assert DEVICE.type == 'cuda', ('Please switch to a GPU Colab runtime for parity with the original R2C CUDA/AMP setup.', DEVICE)
print('SOURCE domain:', DOMAIN, 'fraction:', FRACTION, 'seed:', SEED)
print('CLEAN OUTPUT:', OUT)
print('DEVICE:', DEVICE, 'PyTorch:', torch.__version__, 'Python:', platform.python_version())

# %% [markdown] A. Frozen contracts and reference values (read-only)

# %% Code cell 2 (notebook cell 4)
def find_unique_file(root, filename):
    matches = list(Path(root).rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename!r} under {root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def stable_hash_u01(s):
    return (
        int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:12], 16)
        / float(16 ** 12)
    )

def regression_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    e = pred - y
    ae = np.abs(e)
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "P95": float(np.percentile(ae, 95)),
        "Bias": float(e.mean()),
    }

def exact_sparse_subset(df, fraction, seed):
    if float(fraction) >= 0.999999:
        return df.copy()

    pieces = []

    for _, g in df.groupby("vehicle_id", sort=False):
        k = max(1, int(round(float(fraction) * len(g))))

        keys = (
            g["vehicle_id"].astype(str)
            + "|"
            + g["segment_id"].astype(str)
            + "|"
            + g["time_value"].astype(str)
            + "|"
            + str(seed)
        )

        order = keys.map(stable_hash_u01).sort_values().index
        pieces.append(g.loc[order[:k]].copy())

    return pd.concat(pieces, ignore_index=False)

def normalized_log_auc(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.any(x <= 0):
        raise ValueError("AULEC x values must be >0.")

    lx = np.log10(x)
    width = float(lx[-1] - lx[0])

    if width <= 0:
        raise ValueError("AULEC needs >1 distinct x value.")

    return float(np.trapz(y, lx) / width)

# %% Code cell 3 (notebook cell 5)
# Load exact original VED feature mask, label selector contract and frozen VED budget.
with open(OUTPUTS / 'deep_baselines_11B_100epoch_vs_10K' / '11B_manifest.json') as f:
    man11b = json.load(f)
assert man11b.get('strict_causality') is True
assert int(man11b['sequence_length']) == SEQ_LEN
EXCLUDED_TOKENS = [str(x).lower() for x in man11b['feature_rule']['excluded_tokens']]

frozen_budget = pd.read_csv(
    REV2_OUT / 'R2C_corrected_sparse_prior_curve' / 
    'R2C_CORRECTED_SPARSE_PRIOR_CURVE_FROZEN_DOMAIN_TRAINING_BUDGET.csv'
)
rr = frozen_budget.loc[frozen_budget.domain.eq(DOMAIN)]
assert len(rr)==1
DOMAIN_BUDGET = {DOMAIN:{'max_epochs':int(rr.iloc[0]['max_epochs']), 'patience':int(rr.iloc[0]['patience'])}}
assert DOMAIN_BUDGET[DOMAIN] == {'max_epochs':200,'patience':20}

p0_selector_audit = pd.read_csv(
    REV2_OUT / 'P0_sparse_prior_calibration_audit_v2' / 'P0_SELECTOR_MANIFEST_SHA256.csv'
)
assert len(p0_selector_audit.loc[(p0_selector_audit.domain==DOMAIN)&
    np.isclose(p0_selector_audit.fraction.astype(float),FRACTION)&
    (p0_selector_audit.seed.astype(int)==SEED)]) == 1

p = CANON / f'{DOMAIN}_canonical.parquet'
m = CANON / f'{DOMAIN}_metadata.json'
assert p.exists() and m.exists()
with open(m) as f: meta = json.load(f)
d = pd.read_parquet(p).copy()
base_features = [c for c in meta['feature_columns'] if not any(tok in c.lower() for tok in EXCLUDED_TOKENS)]
features = base_features + ['soc_phy']
assert 'soc_phy' not in base_features and len(features)==43, features
required={'vehicle_id','segment_id','time_value','elapsed_s','split','target','soc0','coord','soc_phy',*meta['feature_columns']}
missing=sorted(required-set(d.columns))
assert not missing,missing
d=d[d['split'].isin(['train','val','test'])].copy()
d['vehicle_id']=d['vehicle_id'].astype(str)
d['segment_id']=d['segment_id'].astype(str)
d['compound_segment']=d['vehicle_id']+'||'+d['segment_id']
d['soc_phy_frozen']=d['soc_phy'].to_numpy(float)
for col in ('target','soc0','coord','soc_phy_frozen'):
    assert np.isfinite(d[col].to_numpy(float)).all(), col
assert d.groupby('compound_segment')['split'].nunique().max()==1, 'Segment leakage across splits'
assert d.groupby('compound_segment')['soc0'].agg(lambda x:np.ptp(x.to_numpy(float))).max() < 1e-8, 'soc0 must be segment-constant'
assert set(d['split'])=={'train','val','test'}
DOMAIN_DATA={DOMAIN:d}
DOMAIN_FEATURES={DOMAIN:features}
DOMAIN_META={DOMAIN:meta}
DOMAIN_ARCH={DOMAIN:'DenseRecurrentNet'}
print('VED rows by split:',d['split'].value_counts().to_dict())
print('Feature count:',len(features),'| raw metadata features:',len(meta['feature_columns']))
print('Original frozen budget:',DOMAIN_BUDGET)
print('Original selector SHA256:',p0_selector_audit.loc[(p0_selector_audit.domain==DOMAIN)&
    np.isclose(p0_selector_audit.fraction.astype(float),FRACTION)&
    (p0_selector_audit.seed.astype(int)==SEED),'selector_sha256'].iloc[0])

# %% [markdown] B. Exact original R2C preprocessing, selector and train-only prior

# %% Code cell 4 (notebook cell 7)
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

def prepare_split_frame(domain_df, split):
    z = domain_df[domain_df["split"] == split].copy()

    z = z.sort_values(
        ["compound_segment", "time_value"],
        kind="mergesort",
    ).reset_index(drop=True)

    z["seq_pos"] = np.arange(len(z), dtype=np.int64)
    return z

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

# %% Code cell 5 (notebook cell 8)
# Corrected supervision-matched prior calibration contracts.

GROUP_COLS = {
    "VED": [],
    "KU": ["vehicle_id"],
    "SCUT": ["vehicle_id"],
    "TUM": ["vehicle_id"],
    "EVE": [],
}

CALIBRATION_RULES = {
    "VED":  {"kind": "abs",        "threshold": 1e-2},
    "KU":   {"kind": "abs",        "threshold": 1e-4},
    "SCUT": {"kind": "abs",        "threshold": 1e-4},
    "TUM":  {"kind": "pos",        "threshold": 1e-2},
    "EVE":  {"kind": "abs_strict", "threshold": 1e-8},
}

EPS = 1e-12
PRIOR_CALIBRATION_VERSION = "P0_supervision_matched_v1"

def fraction_key(f):
    return round(float(f), 8)

def identity_string(df):
    z = (
        df["vehicle_id"].astype(str)
        + "|"
        + df["segment_id"].astype(str)
        + "|"
        + df["time_value"].astype(str)
    ).sort_values()
    return "\n".join(z.tolist())

def identity_sha256(df):
    return hashlib.sha256(
        identity_string(df).encode("utf-8")
    ).hexdigest()

def infer_frozen_beta(g):
    x = g["coord"].to_numpy(float)
    delta = (
        g["soc_phy_frozen"].to_numpy(float)
        - g["soc0"].to_numpy(float)
    )
    m = np.isfinite(x) & np.isfinite(delta) & (np.abs(x) > 1e-10)
    if not np.any(m):
        return np.nan
    return float(np.median(delta[m] / x[m]))

def fit_beta_allowed(g, domain):
    x = g["coord"].to_numpy(float)
    y = (
        g["target"].to_numpy(float)
        - g["soc0"].to_numpy(float)
    )

    rule = CALIBRATION_RULES[domain]
    finite = np.isfinite(x) & np.isfinite(y)

    if rule["kind"] == "pos":
        eligible = x >= rule["threshold"]
    elif rule["kind"] == "abs_strict":
        eligible = np.abs(x) > rule["threshold"]
    else:
        eligible = np.abs(x) >= rule["threshold"]

    m = finite & eligible
    x = x[m]
    y = y[m]

    if len(x) == 0 or float(np.dot(x, x)) <= EPS:
        return np.nan, int(len(x))

    return float(np.dot(x, y) / np.dot(x, x)), int(len(x))

def normalized_group_key(key):
    if isinstance(key, tuple):
        return "||".join(map(str, key))
    return str(key)

def fit_condition_prior(domain, fraction, seed):
    base = DOMAIN_DATA[domain]
    train = base[base["split"] == "train"].copy()

    selected = exact_sparse_subset(
        train,
        fraction,
        seed,
    ).copy()

    selector_sha = identity_sha256(selected)

    # Exact selector identity must match the P0 audit.
    q = p0_selector_audit[
        (p0_selector_audit["domain"].astype(str) == str(domain))
        & np.isclose(p0_selector_audit["fraction"].astype(float), float(fraction))
        & (p0_selector_audit["seed"].astype(int) == int(seed))
    ]
    assert len(q) == 1, (domain, fraction, seed, len(q))
    assert selector_sha == str(q["selector_sha256"].iloc[0]), (
        "selector hash mismatch", domain, fraction, seed
    )

    group_cols = GROUP_COLS[domain]
    beta_map = {}
    n_info_map = {}
    frozen_map = {}

    if group_cols:
        for key, g in selected.groupby(group_cols, sort=False):
            k = normalized_group_key(key)
            b, n_info = fit_beta_allowed(g, domain)

            if not np.isfinite(b):
                raise RuntimeError(
                    f"{domain} fraction={fraction} seed={seed} group={k}: "
                    "no informative retained rows for corrected prior calibration"
                )

            beta_map[k] = float(b)
            n_info_map[k] = int(n_info)

        train_groups = train.groupby(group_cols, sort=False)
        for key, g in train_groups:
            k = normalized_group_key(key)
            frozen_map[k] = infer_frozen_beta(g)

        expected_groups = set(
            train[group_cols[0]].astype(str).unique().tolist()
            if len(group_cols) == 1
            else train[group_cols].astype(str).agg("||".join, axis=1).unique().tolist()
        )
        assert set(beta_map) == expected_groups, (
            domain, fraction, seed,
            sorted(expected_groups - set(beta_map)),
        )
    else:
        b, n_info = fit_beta_allowed(selected, domain)
        if not np.isfinite(b):
            raise RuntimeError(
                f"{domain} fraction={fraction} seed={seed}: "
                "no informative retained rows for corrected prior calibration"
            )
        beta_map["__POOLED__"] = float(b)
        n_info_map["__POOLED__"] = int(n_info)
        frozen_map["__POOLED__"] = infer_frozen_beta(train)

    payload = {
        "version": PRIOR_CALIBRATION_VERSION,
        "domain": domain,
        "fraction": float(fraction),
        "seed": int(seed),
        "selector_sha256": selector_sha,
        "group_cols": group_cols,
        "calibration_rule": CALIBRATION_RULES[domain],
        "beta_map": {
            k: float(beta_map[k])
            for k in sorted(beta_map)
        },
    }

    contract_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "selector_sha256": selector_sha,
        "beta_map": beta_map,
        "n_info_map": n_info_map,
        "frozen_map": frozen_map,
        "prior_contract_sha256": contract_sha,
        "payload": payload,
    }

def apply_condition_prior(df, domain, beta_map):
    z = df.copy()
    group_cols = GROUP_COLS[domain]

    if group_cols:
        if len(group_cols) == 1:
            keys = z[group_cols[0]].astype(str)
        else:
            keys = z[group_cols].astype(str).agg("||".join, axis=1)

        bvec = keys.map(beta_map)

        if bvec.isna().any():
            missing = sorted(keys[bvec.isna()].unique().tolist())
            raise RuntimeError(
                f"{domain}: no corrected prior slope for groups {missing}"
            )

        beta_vec = bvec.to_numpy(float)
    else:
        beta_vec = np.full(
            len(z),
            float(beta_map["__POOLED__"]),
            dtype=float,
        )

    z["soc_phy"] = (
        z["soc0"].to_numpy(float)
        + beta_vec * z["coord"].to_numpy(float)
    )
    z["pr_target"] = (
        z["target"].to_numpy(float)
        - z["soc_phy"].to_numpy(float)
    )

    assert np.isfinite(z["soc_phy"].to_numpy(float)).all()
    assert np.isfinite(z["pr_target"].to_numpy(float)).all()

    return z

# %% Code cell 6 (notebook cell 9)
# SINGLE requested condition: one selector and one train-only prior, shared by both models.
PRIOR_CONTRACT_LOOKUP={}
prior_contract=fit_condition_prior(DOMAIN,FRACTION,SEED)
PRIOR_CONTRACT_LOOKUP[(DOMAIN,fraction_key(FRACTION),SEED)]=prior_contract
assert prior_contract['selector_sha256']=='1e01a4c470e006b20052e7d8a794ac5a7380e25d752241d7bb7551f29f98fde6', 'Selector deviates from archived R2C/P0'
assert abs(prior_contract['beta_map']['__POOLED__']-2.235709423425393)<1e-9, 'Recomputed prior beta differs from original R2C'
assert prior_contract['prior_contract_sha256']=='ce03dce063e7d0ff2070f241a92ddac3c78516f9a3a2699c5d5aee2790952377', 'Prior contract mismatch'
(OUT / 'prior_contract.json').write_text(json.dumps(prior_contract['payload'],sort_keys=True,indent=2))
print('Fresh train-only beta:',prior_contract['beta_map'])
print('Informative labeled train rows:',prior_contract['n_info_map'])

# %% Code cell 7 (notebook cell 10)
def build_condition_pack(domain, fraction, seed):
    """Build corrected prior, residual target and train-only preprocessing for one condition."""
    key = (domain, fraction_key(fraction), int(seed))
    c = PRIOR_CONTRACT_LOOKUP[key]

    corrected = apply_condition_prior(
        DOMAIN_DATA[domain],
        domain,
        c["beta_map"],
    )

    train_frame = prepare_split_frame(corrected, "train")
    val_frame = prepare_split_frame(corrected, "val")
    test_frame = prepare_split_frame(corrected, "test")

    # Selector is reapplied after stable split sorting; identity must remain identical.
    labeled = exact_sparse_subset(
        train_frame,
        fraction,
        seed,
    ).copy()

    assert identity_sha256(labeled) == c["selector_sha256"]

    features = DOMAIN_FEATURES[domain]

    pre = TrainOnlyPreprocessor().fit(
        train_frame[features].to_numpy(float)
    )

    return {
        "train_frame": train_frame,
        "val_frame": val_frame,
        "test_frame": test_frame,
        "labeled": labeled,
        "X_train": pre.transform(
            train_frame[features].to_numpy(float)
        ),
        "X_val": pre.transform(
            val_frame[features].to_numpy(float)
        ),
        "X_test": pre.transform(
            test_frame[features].to_numpy(float)
        ),
        "preprocessor": pre,
        "selector_sha256": c["selector_sha256"],
        "prior_contract_sha256": c["prior_contract_sha256"],
        "beta_map": c["beta_map"],
    }

# %% [markdown] C. Unchanged architecture, RNG and training functions

# %% Code cell 8 (notebook cell 12)
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


class FrozenFamilyGRU(nn.Module):
    # Exact R2A EVE architecture.
    def __init__(self, input_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=EVE_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(EVE_HIDDEN)

        self.head = nn.Sequential(
            nn.Linear(EVE_HIDDEN, EVE_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EVE_HEAD_HIDDEN, 1),
        )

    def forward(self, x):
        h, _ = self.gru(x)
        z = self.norm(h[:, -1, :])
        return self.head(z).squeeze(-1)


def build_model(domain, input_dim):
    if domain in DEV_DOMAINS:
        return DenseRecurrentNet(input_dim)
    if domain == "EVE":
        return FrozenFamilyGRU(input_dim)
    raise ValueError(domain)

def count_params(model):
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

# %% Code cell 9 (notebook cell 13)
model_a=build_model(DOMAIN,len(features)+1)
model_b=build_model(DOMAIN,len(features)+1)
assert count_params(model_a)==count_params(model_b)==23197
print('Model capacity matched:', count_params(model_a),'parameters per model')
del model_a,model_b

# %% Code cell 10 (notebook cell 14)
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def _cpu_byte_rng_tensor(x):
    # Resume checkpoints may be loaded with map_location=DEVICE.
    # If DEVICE is CUDA, saved CPU RNG tensors are moved to CUDA too,
    # but torch.set_rng_state() strictly requires a CPU ByteTensor.
    if torch.is_tensor(x):
        return x.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(x, dtype=torch.uint8, device="cpu").contiguous()

def restore_rng_state(state):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch.set_rng_state(
        _cpu_byte_rng_tensor(state["torch_cpu"])
    )

    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [
            _cpu_byte_rng_tensor(s)
            for s in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)

def safe_torch_load(path, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)

def make_grad_scaler():
    if not AMP_ENABLED:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)

def autocast_context():
    if not AMP_ENABLED:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)

def target_col_for_model(model_std):
    if model_std == DIRECT_STD:
        return "target"
    if model_std == PR_STD:
        return "pr_target"
    raise ValueError(model_std)

def reconstruct_soc(frame_rows, raw_model_output, model_std):
    raw_model_output = np.asarray(raw_model_output, float)

    if model_std == DIRECT_STD:
        return raw_model_output

    if model_std == PR_STD:
        return (
            frame_rows["soc_phy"].to_numpy(float)
            + raw_model_output
        )

    raise ValueError(model_std)

@torch.no_grad()
def predict_loader(model, loader, target_scaler):
    model.eval()

    pred_scaled = []
    positions = []

    for xb, _, pos in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        p = model(xb)

        pred_scaled.append(p.detach().cpu().numpy())
        positions.append(np.asarray(pos))

    pred_scaled = np.concatenate(pred_scaled)
    positions = np.concatenate(positions).astype(int)

    return positions, target_scaler.inverse(pred_scaled)

# %% Code cell 11 (notebook cell 15)
def run_contract_fingerprint(domain, fraction, model_std, seed):
    key = (domain, fraction_key(fraction), int(seed))
    prior_contract = PRIOR_CONTRACT_LOOKUP[key]

    payload = {
        "experiment": EXPERIMENT,
        "domain": domain,
        "fraction": float(fraction),
        "model": model_std,
        "seed": int(seed),
        "architecture": DOMAIN_ARCH[domain],
        "features": DOMAIN_FEATURES[domain],
        "seq_len": SEQ_LEN,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": int(DOMAIN_BUDGET[domain]["max_epochs"]),
        "patience": int(DOMAIN_BUDGET[domain]["patience"]),
        "target_mode": (
            "direct"
            if model_std == DIRECT_STD
            else "prior_residual"
        ),
        "structured_prior_input": "soc_phy",
        "prior_calibration_version": PRIOR_CALIBRATION_VERSION,
        "prior_contract_sha256": prior_contract["prior_contract_sha256"],
        "selector_sha256": prior_contract["selector_sha256"],
        "budget_rule": "R2C corrected frozen domain curve budget",
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def train_one(domain, fraction, model_std, seed):
    set_all_seeds(seed)

    pack = build_condition_pack(
        domain,
        fraction,
        seed,
    )

    train_frame = pack["train_frame"]
    val_frame = pack["val_frame"]
    test_frame = pack["test_frame"]
    labeled = pack["labeled"]

    target_col = target_col_for_model(model_std)

    target_scaler = TargetScaler().fit(
        labeled[target_col].to_numpy(float)
    )

    train_positions = labeled["seq_pos"].astype(int).to_numpy()
    val_positions = val_frame["seq_pos"].astype(int).to_numpy()
    test_positions = test_frame["seq_pos"].astype(int).to_numpy()

    train_ds = CausalSequenceDataset(
        train_frame,
        pack["X_train"],
        train_positions,
        target_scaler,
        target_col,
        SEQ_LEN,
    )
    val_ds = CausalSequenceDataset(
        val_frame,
        pack["X_val"],
        val_positions,
        target_scaler,
        target_col,
        SEQ_LEN,
    )
    test_ds = CausalSequenceDataset(
        test_frame,
        pack["X_test"],
        test_positions,
        target_scaler,
        target_col,
        SEQ_LEN,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=AMP_ENABLED,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=AMP_ENABLED,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=AMP_ENABLED,
    )

    input_dim = len(DOMAIN_FEATURES[domain]) + 1
    model = build_model(domain, input_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()
    scaler = make_grad_scaler()

    max_epochs = int(DOMAIN_BUDGET[domain]["max_epochs"])
    patience = int(DOMAIN_BUDGET[domain]["patience"])

    tag = f"{domain}_fraction{fraction}_{model_std}_seed{seed}"
    safe_tag = (
        tag.replace("+", "Plus")
        .replace(" ", "_")
        .replace("/", "_")
    )

    fp = run_contract_fingerprint(
        domain, fraction, model_std, seed
    )

    best_path = CKPT_DIR / f"{safe_tag}_best.pt"
    resume_path = CKPT_DIR / f"{safe_tag}_resume.pt"
    history_path = OUT / f"history_{safe_tag}.csv"

    best_val = np.inf
    best_epoch = 0
    stale = 0
    history = []
    start_epoch = 1
    elapsed_before = 0.0

    if resume_path.exists():
        state = safe_torch_load(resume_path, DEVICE)

        assert state["contract_fingerprint"] == fp

        model.load_state_dict(state["latest_model_state"])
        optimizer.load_state_dict(state["optimizer_state"])

        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])

        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        history = list(state["history"])
        elapsed_before = float(state.get("elapsed_seconds", 0.0))

        restore_rng_state(state.get("rng_state"))
        print("RESUME", safe_tag, "epoch", start_epoch)

    t0 = time.perf_counter()

    terminal_resume = bool(
        history
        and (stale >= patience or start_epoch > max_epochs)
    )

    epoch_iter = (
        []
        if terminal_resume
        else range(start_epoch, max_epochs + 1)
    )

    for epoch in epoch_iter:
        model.train()
        losses = []

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                pred = model(xb)
                loss = criterion(pred, yb)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{safe_tag}: non-finite loss"
                )

            if scaler is None or not AMP_ENABLED:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )
                scaler.step(optimizer)
                scaler.update()

            losses.append(float(loss.detach().cpu()))

        val_pos, val_raw = predict_loader(
            model,
            val_loader,
            target_scaler,
        )

        val_rows = val_frame.iloc[val_pos]
        val_soc = reconstruct_soc(
            val_rows,
            val_raw,
            model_std,
        )

        val_metrics = regression_metrics(
            val_rows["target"].to_numpy(float),
            val_soc,
        )
        val_mae = float(val_metrics["MAE"])

        history.append({
            "epoch": int(epoch),
            "train_scaled_target_MSE": float(np.mean(losses)),
            "val_SOC_MAE": val_mae,
        })

        if val_mae < best_val - MIN_DELTA:
            best_val = val_mae
            best_epoch = int(epoch)
            stale = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "domain": domain,
                    "fraction": float(fraction),
                    "model": model_std,
                    "seed": int(seed),
                    "input_dim": int(input_dim),
                    "n_params": count_params(model),
                    "target_col": target_col,
                    "best_epoch": best_epoch,
                    "best_val_SOC_MAE": best_val,
                    "target_scaler": {
                        "mean": target_scaler.mean_,
                        "std": target_scaler.std_,
                    },
                    "feature_columns": DOMAIN_FEATURES[domain],
                    "contract_fingerprint": fp,
                    "selector_sha256": pack["selector_sha256"],
                    "prior_contract_sha256": pack["prior_contract_sha256"],
                    "beta_map": pack["beta_map"],
                },
                best_path,
            )
        else:
            stale += 1

        if (
            epoch % RESUME_SAVE_EVERY == 0
            or stale >= patience
            or epoch == max_epochs
        ):
            pd.DataFrame(history).to_csv(
                history_path,
                index=False,
            )

            torch.save(
                {
                    "latest_model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": (
                        scaler.state_dict()
                        if scaler is not None
                        else {}
                    ),
                    "epoch": int(epoch),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "stale": int(stale),
                    "history": history,
                    "rng_state": capture_rng_state(),
                    "elapsed_seconds": float(
                        elapsed_before
                        + (time.perf_counter() - t0)
                    ),
                    "contract_fingerprint": fp,
                },
                resume_path,
            )

        if stale >= patience:
            break

    assert best_path.exists(), safe_tag
    final_epoch = int(history[-1]["epoch"])

    best_state = safe_torch_load(best_path, DEVICE)
    assert best_state["contract_fingerprint"] == fp
    model.load_state_dict(best_state["model_state"])

    val_pos, val_raw = predict_loader(
        model,
        val_loader,
        target_scaler,
    )
    val_rows = val_frame.iloc[val_pos]
    val_soc = reconstruct_soc(
        val_rows,
        val_raw,
        model_std,
    )
    val_metrics = regression_metrics(
        val_rows["target"].to_numpy(float),
        val_soc,
    )

    infer_t0 = time.perf_counter()

    test_pos, test_raw = predict_loader(
        model,
        test_loader,
        target_scaler,
    )

    inference_seconds = time.perf_counter() - infer_t0

    test_rows = test_frame.iloc[test_pos].copy()
    test_soc = reconstruct_soc(
        test_rows,
        test_raw,
        model_std,
    )
    y_test = test_rows["target"].to_numpy(float)

    test_metrics = regression_metrics(
        y_test,
        test_soc,
    )

    pred_df = test_rows[
        [
            "vehicle_id",
            "segment_id",
            "compound_segment",
            "time_value",
            "elapsed_s",
            "target",
            "soc_phy",
        ]
    ].copy()

    pred_df["domain"] = domain
    pred_df["fraction"] = float(fraction)
    pred_df["model"] = model_std
    pred_df["seed"] = int(seed)
    pred_df["prediction"] = test_soc
    pred_df["error"] = test_soc - y_test
    pred_df["abs_error"] = np.abs(pred_df["error"])

    pred_path = (
        PRED_DIR
        / f"{EXPERIMENT}_{safe_tag}.parquet"
    )

    pred_df.to_parquet(pred_path, index=False)

    stopped_by_patience = bool(
        stale >= patience and final_epoch < max_epochs
    )
    hit_epoch_cap = bool(
        final_epoch >= max_epochs and stale < patience
    )

    return {
        "domain": domain,
        "fraction": float(fraction),
        "model": model_std,
        "seed": int(seed),
        "source_experiment": "R2C_corrected_sparse_prior_training",
        "selector_sha256": pack["selector_sha256"],
        "prior_contract_sha256": pack["prior_contract_sha256"],
        "beta_map_json": json.dumps(pack["beta_map"], sort_keys=True),
        "reused_prior_endpoint": False,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "prediction_file": str(pred_path),
        "prediction_sha256": sha256_file(pred_path),
        "resume_checkpoint": str(resume_path),
        "best_epoch": int(best_state["best_epoch"]),
        "final_epoch": int(final_epoch),
        "stopped_by_patience": stopped_by_patience,
        "hit_epoch_cap": hit_epoch_cap,
        "n_params": int(best_state["n_params"]),
        "n_train_labels": int(len(train_ds)),
        "n_validation": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "max_epochs": max_epochs,
        "patience": patience,
        "train_seconds": float(
            elapsed_before + (time.perf_counter() - t0)
        ),
        "inference_seconds": float(inference_seconds),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

# %% [markdown] D. Launch 2 genuinely fresh trainings

# %% Code cell 12 (notebook cell 17)
RUNS_INCREMENTAL = OUT / 'fresh_pair_runs.csv'
results=[]
for name in MODELS_STD:
    print('\nSTART FRESH TRAINING:', name, '|', DOMAIN, FRACTION, SEED, flush=True)
    item=train_one(DOMAIN,FRACTION,name,SEED)
    item['clean_room_run']=True
    results.append(item)
    pd.DataFrame(results).to_csv(RUNS_INCREMENTAL,index=False)
    print(name,'done: best epoch',item['best_epoch'],
          'test MAE=',round(item['test_MAE'],6),
          'test RMSE=',round(item['test_RMSE'],6),flush=True)
    gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
assert len(results)==2 and {x['model'] for x in results}==set(MODELS_STD)
print('BOTH independent trainings completed.')

# %% [markdown] E. Independent pairing and archived-reference comparison (read-only)

# %% Code cell 13 (notebook cell 19)
REF = REV2_OUT / 'R2C_corrected_sparse_prior_curve' / 'R2C_CORRECTED_SPARSE_PRIOR_CURVE_ALL_CURVE_RUNS.csv'
assert REF.is_file(),REF
old = pd.read_csv(REF)
old = old.loc[(old.domain==DOMAIN)&np.isclose(old.fraction.astype(float),FRACTION)&(old.seed.astype(int)==SEED)&old.model.isin(MODELS_STD)].copy()
assert len(old)==2 and old.model.nunique()==2
fresh = pd.read_csv(OUT/'fresh_pair_runs.csv')
assert len(fresh)==2
cols=['model','best_epoch','final_epoch','max_epochs','patience','n_params','n_train_labels','n_validation','n_test','val_MAE','test_MAE','test_RMSE']
comparison = fresh[cols].merge(old[cols],on='model',suffixes=('_fresh','_archived'),validate='one_to_one')
comparison['delta_test_MAE']=comparison['test_MAE_fresh']-comparison['test_MAE_archived']
comparison['abs_delta_test_MAE']=comparison['delta_test_MAE'].abs()
comparison['delta_test_RMSE']=comparison['test_RMSE_fresh']-comparison['test_RMSE_archived']
comparison['mae_within_0p05_pp']=comparison['abs_delta_test_MAE']<=0.05
for k in ('n_params','n_train_labels','n_validation','n_test','max_epochs','patience'):
    assert (comparison[f'{k}_fresh']==comparison[f'{k}_archived']).all(),f'{k}: contract mismatch'

pred_by_model={}
for r in fresh.itertuples(index=False):
    z=pd.read_parquet(r.prediction_file).copy()
    assert len(z)==int(r.n_test)
    key=['vehicle_id','segment_id','time_value']
    assert not z.duplicated(key).any()
    assert np.isfinite(z[['target','prediction']]).all().all()
    pred_by_model[r.model]=z.set_index(key).sort_index()
left=pred_by_model[DIRECT_STD]
right=pred_by_model[PR_STD]
assert left.index.equals(right.index), 'Model test-row population mismatch'
assert np.allclose(left['target'],right['target'],atol=0,rtol=0)
assert np.allclose(left['soc_phy'],right['soc_phy'],atol=0,rtol=0)
assert fresh['selector_sha256'].nunique()==1 and fresh['prior_contract_sha256'].nunique()==1
assert set(fresh['selector_sha256'])=={prior_contract['selector_sha256']}
assert set(fresh['prior_contract_sha256'])=={prior_contract['prior_contract_sha256']}
assert not fresh['reused_prior_endpoint'].astype(bool).any()
comparison.to_csv(OUT/'fresh_vs_archived_comparison.csv',index=False)

# Diagnostic whether the PR improvement direction is reproduced in this fresh pair.
new_map=dict(zip(fresh.model,fresh.test_MAE))
old_map=dict(zip(old.model,old.test_MAE))
delta_fresh=float(new_map[PR_STD]-new_map[DIRECT_STD])
delta_archived=float(old_map[PR_STD]-old_map[DIRECT_STD])
assert np.isfinite(delta_fresh)
verdict={
 'source':'R2C corrected frozen VED canonical only',
 'condition':{'domain':DOMAIN,'fraction':FRACTION,'seed':SEED},
 'n_test_paired_rows':len(left),
 'selector_and_train_only_prior':'PASS',
 'pairing_and_training_contract':'PASS',
 'fresh_PR_minus_direct_MAE_pp':delta_fresh,
 'archived_PR_minus_direct_MAE_pp':delta_archived,
 'both_improve_same_direction': bool(delta_fresh < 0 and delta_archived < 0),
 'both_MAE_within_0p05_pp':bool(comparison['mae_within_0p05_pp'].all()),
 'exact_bitwise_reproduction':False,
 'full_350_run_reproduction':False,
 'github_release_ready':False
}
(OUT/'pair_verdict.json').write_text(json.dumps(verdict,indent=2))
print(comparison[['model','test_MAE_archived','test_MAE_fresh','delta_test_MAE','mae_within_0p05_pp']].to_string(index=False))
print('Fresh PR-Direct MAE difference:',delta_fresh,'| Archived:',delta_archived)
print('Verdict:',json.dumps(verdict,indent=2))

# %% [markdown] F. Final immutable-ish run manifest

# %% Code cell 14 (notebook cell 21)
source_r2c = ROOT/'Revizyon2'/'R2C_CorrectedSparsePrior'/'R2C_CorrectedSparsePrior_LabelFractionCurve_AULEC_Colab.ipynb'
manifest={
  'stage':'02_R2C_clean_pair',
  'status':'FRESH_TRAINED_TWO_MODELS_NOT_FULL_REPRODUCTION',
  'created_utc':run_stamp,
  'run_folder':str(OUT),
  'domain':DOMAIN,'fraction':FRACTION,'seed':SEED,
  'original_code_protocol':'R2C-C corrected prior residual source (unchanged training/model implementations)',
  'canonical_input_path':str(p),
  'canonical_input_sha256':sha256_file(p),
  'canonical_metadata_path':str(m),
  'canonical_metadata_sha256':sha256_file(m),
  'reference_training_budget_sha256':sha256_file(REV2_OUT/'R2C_corrected_sparse_prior_curve'/'R2C_CORRECTED_SPARSE_PRIOR_CURVE_FROZEN_DOMAIN_TRAINING_BUDGET.csv'),
  'source_r2c_code_sha256':sha256_file(source_r2c) if source_r2c.is_file() else None,
  'new_training_runs':results,
  'verdict':verdict,
  'environment':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
  'no_prior_checkpoints_reused':True,
  'isolated_output_folder':True
}
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
print('DONE:', OUT)
print('Share the new folder/link or the run_manifest.json and fresh_vs_archived_comparison.csv.')
