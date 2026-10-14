# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/R2A_EVE_InfoMatched_GRU_Colab.ipynb
# Notebook SHA256: b84688dc5b74cad9f82ce5bcb9b839eb18df0f7c5150701014d9c3843ea950a1
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R2A — EVE Information-Matched GRU Fairness Audit

# %% Code cell 1 (notebook cell 2)

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import hashlib
import random
import time
import gc
import contextlib
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
OUTPUTS = ROOT / "outputs"

TRAINVAL_DIR = OUTPUTS / "eve_12B_v4_frozen_trainval"
TEST_DIR = OUTPUTS / "eve_12C_one_shot_confirmatory"

TRAINVAL_PARQUET = TRAINVAL_DIR / "12B_v4_EVE_canonical_trainval.parquet"
TRAINVAL_META = TRAINVAL_DIR / "12B_v4_EVE_canonical_metadata.json"
FREEZE_MANIFEST = TRAINVAL_DIR / "12B_v4_FREEZE_MANIFEST.json"
SPARSE_AUDIT_REF = TRAINVAL_DIR / "12B_v4_sparse_selector_audit.csv"
DEEP_RUNS_REF = TRAINVAL_DIR / "deep" / "12B_v4_deep_runs.csv"

TEST_PARQUET = TEST_DIR / "12C_EVE_confirmatory_canonical.parquet"
TEST_AUDIT = TEST_DIR / "12C_canonicalization_audit.json"
FINAL_12C_MANIFEST = TEST_DIR / "12C_FINAL_MANIFEST.json"

OUT = ROOT / "Revizyon2" / "outputs" / "R2A_eve_info_matched_gru"
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "seed_predictions"
FIG_DIR = OUT / "figures"

for p in [OUT, CKPT_DIR, PRED_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R2A_EVE_INFO_MATCHED_GRU"

DIRECT_MODEL = "EVE_GRU_PriorInput_Direct"
PR_MODEL = "EVE_GRU_PriorInput_PR"
RUN_MODELS = [DIRECT_MODEL, PR_MODEL]

MODEL_SPECS = {
    DIRECT_MODEL: {"target_mode": "direct"},
    PR_MODEL: {"target_mode": "residual"},
}

LABEL_FRACTIONS = [1.0, 0.10]
SEEDS = [42, 43, 44, 45, 46]

SEQ_LEN = 16
HIDDEN = 64
HEAD_HIDDEN = 32
DROPOUT = 0.10

LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-6
GRAD_CLIP = 5.0
RESUME_SAVE_EVERY = 10
BOOTSTRAP_DRAWS = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
BATCH_SIZE = 512 if AMP_ENABLED else 128

EXPECTED_RUNS = (
    len(LABEL_FRACTIONS)
    * len(RUN_MODELS)
    * len(SEEDS)
)
assert EXPECTED_RUNS == 20

print("Experiment:", EXPERIMENT)
print("Device:", DEVICE)
print("Expected runs:", EXPECTED_RUNS)
print("Output:", OUT)

# %% Code cell 2 (notebook cell 3)

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

for p in [
    TRAINVAL_PARQUET, TRAINVAL_META, FREEZE_MANIFEST,
    SPARSE_AUDIT_REF, DEEP_RUNS_REF,
    TEST_PARQUET, TEST_AUDIT, FINAL_12C_MANIFEST,
]:
    assert p.exists(), p

with open(TRAINVAL_META) as f:
    meta = json.load(f)

with open(FREEZE_MANIFEST) as f:
    freeze_manifest = json.load(f)

with open(TEST_AUDIT) as f:
    test_audit = json.load(f)

with open(FINAL_12C_MANIFEST) as f:
    final12c = json.load(f)

deep_ref = pd.read_csv(DEEP_RUNS_REF)
sparse_ref = pd.read_csv(SPARSE_AUDIT_REF)

assert freeze_manifest["dataset"] == "EVE Pilot Renault ZOE Q90"
assert freeze_manifest["target"] == "dashboard/user-facing BMS-reported State of Charge"
assert freeze_manifest["prior"] == "SIGNED_AH"
assert freeze_manifest["real_state_of_charge_predictor"] is False
assert freeze_manifest["deep_baselines"]["sequence_length"] == SEQ_LEN
assert freeze_manifest["deep_baselines"]["hidden"] == HIDDEN
assert freeze_manifest["deep_baselines"]["dropout"] == DROPOUT
assert freeze_manifest["deep_baselines"]["optimizer"] == "AdamW"
assert np.isclose(freeze_manifest["deep_baselines"]["lr"], LR)
assert np.isclose(freeze_manifest["deep_baselines"]["weight_decay"], WEIGHT_DECAY)
assert freeze_manifest["deep_baselines"]["max_epochs"] == MAX_EPOCHS
assert freeze_manifest["deep_baselines"]["patience"] == PATIENCE
assert list(freeze_manifest["deep_baselines"]["seeds"]) == SEEDS
assert list(freeze_manifest["supervision_fractions"]) == LABEL_FRACTIONS

assert test_audit["scored_canonical_rows"] == 47576
assert test_audit["segments"] == 548
assert test_audit["real_state_of_charge_predictor"] is False

assert final12c["confirmatory_target_opened"] is True
assert final12c["model_retraining_in_12C"] is False

print("Frozen EVE protocol checks: PASS")
print("Train-only prior beta:", meta["beta_pp_per_ah_train_only"])
print("Historical 12C scored rows:", test_audit["scored_canonical_rows"])

# %% Code cell 3 (notebook cell 4)

trainval = pd.read_parquet(TRAINVAL_PARQUET).copy()
test = pd.read_parquet(TEST_PARQUET).copy()

BASE_FEATURES = list(meta["feature_columns"])

assert "soc_phy" not in BASE_FEATURES
FEATURES = BASE_FEATURES + ["soc_phy"]

required_common = {
    "vehicle_id",
    "segment_id",
    "time_value",
    "elapsed_s",
    "target",
    "soc_phy",
    *BASE_FEATURES,
}

missing_tv = sorted(required_common - set(trainval.columns))
missing_test = sorted(required_common - set(test.columns))

assert not missing_tv, missing_tv
assert not missing_test, missing_test
assert "split" in trainval.columns

trainval = trainval[
    trainval["split"].isin(["train", "val"])
].copy()

# 12C canonical is already the frozen scored test surface.
if "split" in test.columns:
    if (test["split"] == "test").any():
        test = test[test["split"] == "test"].copy()

assert int((trainval["split"] == "train").sum()) == int(meta["canonical_train_rows"]) == 180878
assert int((trainval["split"] == "val").sum()) == int(meta["canonical_val_rows"]) == 45023
assert len(test) == int(test_audit["scored_canonical_rows"]) == 47576

for d in [trainval, test]:
    d["vehicle_id"] = d["vehicle_id"].astype(str)
    d["segment_id"] = d["segment_id"].astype(str)
    d["compound_segment"] = (
        d["vehicle_id"] + "||" + d["segment_id"]
    )
    d["pr_target"] = (
        d["target"].to_numpy(float)
        - d["soc_phy"].to_numpy(float)
    )

assert np.isfinite(trainval["target"].to_numpy(float)).all()
assert np.isfinite(trainval["soc_phy"].to_numpy(float)).all()
assert np.isfinite(test["target"].to_numpy(float)).all()
assert np.isfinite(test["soc_phy"].to_numpy(float)).all()
assert test["compound_segment"].nunique() == 548

print("Input features:", FEATURES)
print("Train:", int((trainval["split"] == "train").sum()))
print("Val:", int((trainval["split"] == "val").sum()))
print("Test:", len(test))
print("Test segments:", test["compound_segment"].nunique())

# %% Code cell 4 (notebook cell 5)

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

def stable_hash_u01(s):
    return (
        int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:12], 16)
        / float(16 ** 12)
    )

def exact_sparse_subset(df, fraction, seed):
    if fraction >= 0.999999:
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

    return pd.concat(pieces, ignore_index=True)

train_raw = trainval[trainval["split"] == "train"].copy()

selector_rows = []

for fraction in LABEL_FRACTIONS:
    for seed in SEEDS:
        z = exact_sparse_subset(train_raw, fraction, seed)

        ref = sparse_ref[
            np.isclose(sparse_ref["fraction"].astype(float), fraction)
            & (sparse_ref["seed"].astype(int) == seed)
        ]
        assert len(ref) == 1
        expected_n = int(ref.iloc[0]["n_selected"])

        assert len(z) == expected_n, (
            fraction, seed, len(z), expected_n
        )

        selector_rows.append({
            "fraction": float(fraction),
            "seed": int(seed),
            "n_selected": int(len(z)),
            "expected_12B_selected": expected_n,
            "count_match": True,
        })

selector_audit = pd.DataFrame(selector_rows)
selector_audit.to_csv(
    OUT / f"{EXPERIMENT}_sparse_selector_audit.csv",
    index=False,
)

display(selector_audit)

# %% Code cell 5 (notebook cell 6)

class TrainOnlyPreprocessor:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        Xi = np.where(
            np.isfinite(X),
            X,
            med[None, :],
        )

        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)

        std = np.where(
            np.isfinite(std) & (std > 1e-8),
            std,
            1.0,
        )

        self.median_ = med.astype(np.float32)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        X = np.where(
            np.isfinite(X),
            X,
            self.median_[None, :],
        )

        X = (
            X - self.mean_[None, :]
        ) / self.std_[None, :]

        return X.astype(np.float32)


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
            (np.asarray(y, float) - self.mean_)
            / self.std_
        ).astype(np.float32)

    def inverse(self, y):
        return np.asarray(y, float) * self.std_ + self.mean_


def prepare_frame(df):
    z = df.copy()

    z = z.sort_values(
        ["compound_segment", "time_value"],
        kind="stable",
    ).reset_index(drop=True)

    z["seq_pos"] = np.arange(len(z), dtype=int)
    return z


train_frame = prepare_frame(
    trainval[trainval["split"] == "train"]
)
val_frame = prepare_frame(
    trainval[trainval["split"] == "val"]
)
test_frame = prepare_frame(test)

# Critical leakage control:
# preprocessing is fit on EVE TRAIN only, including the structured prior input.
preprocessor = TrainOnlyPreprocessor().fit(
    train_frame[FEATURES].to_numpy(float)
)

X_train = preprocessor.transform(
    train_frame[FEATURES].to_numpy(float)
)
X_val = preprocessor.transform(
    val_frame[FEATURES].to_numpy(float)
)
X_test = preprocessor.transform(
    test_frame[FEATURES].to_numpy(float)
)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

print("Train-only preprocessing: PASS")

# %% Code cell 6 (notebook cell 7)

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
            self.frame.groupby(
                "compound_segment",
                sort=False,
            )["seq_pos"]
            .min()
            .astype(int)
            .to_dict()
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        pos = int(self.positions[i])
        row = self.frame.iloc[pos]

        seg0 = int(
            self.seg_start[row["compound_segment"]]
        )

        start = max(
            seg0,
            pos - self.seq_len + 1,
        )

        x = self.X[start:pos + 1]
        n = len(x)
        pad = self.seq_len - n

        # Last channel is a causal validity mask.
        out = np.zeros(
            (
                self.seq_len,
                self.X.shape[1] + 1,
            ),
            dtype=np.float32,
        )

        if n:
            out[pad:, :-1] = x
            out[pad:, -1] = 1.0

        y_raw = float(row[self.target_col])
        y_scaled = float(
            self.target_scaler.transform([y_raw])[0]
        )

        return (
            torch.from_numpy(out),
            torch.tensor(y_scaled, dtype=torch.float32),
            pos,
        )

# %% Code cell 7 (notebook cell 8)

class FrozenFamilyGRU(nn.Module):
    '''
    EVE frozen GRU family:
      GRU(hidden=64) -> LayerNorm -> Linear(64,32) -> ReLU
      -> Dropout(0.1) -> Linear(32,1)

    The historical direct GRU had:
      11 telemetry features + causal mask = input_dim 12
      17,217 trainable parameters.

    R2A adds soc_phy to BOTH new models:
      12 matched information features + causal mask = input_dim 13
      17,409 parameters for BOTH models.
    '''
    def __init__(self, input_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(HIDDEN)

        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, x):
        h, _ = self.gru(x)
        z = self.norm(h[:, -1, :])
        return self.head(z).squeeze(-1)


def count_params(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


historical_input_dim = len(BASE_FEATURES) + 1  # mask
new_input_dim = len(FEATURES) + 1             # mask

historical_param_check = count_params(
    FrozenFamilyGRU(historical_input_dim)
)
new_direct_params = count_params(
    FrozenFamilyGRU(new_input_dim)
)
new_pr_params = count_params(
    FrozenFamilyGRU(new_input_dim)
)

gru_ref_params = deep_ref[
    deep_ref["model"] == "GRU"
]["n_params"].unique()

assert len(gru_ref_params) == 1
assert int(gru_ref_params[0]) == 17217
assert historical_param_check == 17217
assert new_direct_params == new_pr_params == 17409

parameter_audit = pd.DataFrame([{
    "historical_GRU_params": 17217,
    "historical_input_dim_with_mask": historical_input_dim,
    "new_direct_params": new_direct_params,
    "new_PR_params": new_pr_params,
    "new_input_dim_with_mask": new_input_dim,
    "direct_and_PR_exact_same_capacity": True,
    "exact_same_input_information": True,
    "structured_prior_input": True,
    "parameter_increase_vs_historical_GRU_pct": (
        100.0 * (new_direct_params - 17217) / 17217
    ),
    "within_5pct_historical_budget": bool(
        abs(new_direct_params - 17217) / 17217 <= 0.05
    ),
}])

parameter_audit.to_csv(
    OUT / f"{EXPERIMENT}_parameter_budget_audit.csv",
    index=False,
)

display(parameter_audit)

# %% Code cell 8 (notebook cell 9)

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


def restore_rng_state(state):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def safe_torch_load(path, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def make_grad_scaler():
    if not AMP_ENABLED:
        try:
            return torch.amp.GradScaler(
                "cuda",
                enabled=False,
            )
        except Exception:
            return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


def autocast_context():
    if not AMP_ENABLED:
        return contextlib.nullcontext()

    try:
        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(
            enabled=True,
        )


def target_col_for_model(model_name):
    mode = MODEL_SPECS[model_name]["target_mode"]

    if mode == "direct":
        return "target"

    if mode == "residual":
        return "pr_target"

    raise ValueError(mode)


def reconstruct_soc(frame_rows, raw_model_output, model_name):
    mode = MODEL_SPECS[model_name]["target_mode"]
    raw_model_output = np.asarray(raw_model_output, float)

    if mode == "direct":
        return raw_model_output

    if mode == "residual":
        return (
            frame_rows["soc_phy"].to_numpy(float)
            + raw_model_output
        )

    raise ValueError(mode)


@torch.no_grad()
def predict_loader(model, loader, target_scaler):
    model.eval()

    pred_scaled = []
    positions = []

    for xb, _, pos in loader:
        xb = xb.to(
            DEVICE,
            non_blocking=True,
        )

        p = model(xb)

        pred_scaled.append(
            p.detach().cpu().numpy()
        )
        positions.append(np.asarray(pos))

    pred_scaled = np.concatenate(pred_scaled)
    positions = np.concatenate(positions).astype(int)

    raw_pred = target_scaler.inverse(pred_scaled)
    return positions, raw_pred


trainval_sha = sha256_file(TRAINVAL_PARQUET)
test_sha = sha256_file(TEST_PARQUET)

def contract_fingerprint(fraction, model_name, seed):
    payload = {
        "experiment": EXPERIMENT,
        "dataset": "EVE Pilot Renault ZOE Q90",
        "fraction": float(fraction),
        "model": model_name,
        "target_mode": MODEL_SPECS[model_name]["target_mode"],
        "seed": int(seed),
        "features": FEATURES,
        "sequence_length": SEQ_LEN,
        "hidden": HIDDEN,
        "head_hidden": HEAD_HIDDEN,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "trainval_sha256": trainval_sha,
        "test_sha256": test_sha,
        "prior_beta_pp_per_ah": float(meta["beta_pp_per_ah_train_only"]),
        "evaluation_role": "post_confirmatory_eve_information_matched_fairness_audit",
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

# %% Code cell 9 (notebook cell 10)

preflight = {
    "experiment": EXPERIMENT,
    "status": "FROZEN_BEFORE_R2A_OUTCOME_INSPECTION",
    "dataset": "EVE Pilot Renault ZOE Q90",
    "target": "dashboard/user-facing BMS-reported State of Charge",
    "primary_question": (
        "Does prior-residual target factorization improve EVE BMS-reported SoC "
        "tracking when Direct and PR GRUs receive exactly the same telemetry + soc_phy prior?"
    ),
    "models": {
        DIRECT_MODEL: "[X, soc_phy] -> BMS SoC",
        PR_MODEL: "[X, soc_phy] -> BMS SoC - soc_phy; reconstruct soc_phy + residual",
    },
    "information_match": {
        "same_architecture": True,
        "same_parameter_count": True,
        "same_telemetry_features": True,
        "same_structured_prior_input": "soc_phy",
        "same_sequence_length": True,
        "same_sparse_label_subset": True,
        "same_optimizer": True,
        "same_max_epochs": True,
        "same_patience": True,
        "same_seeds": True,
        "same_train_validation_test_rows": True,
        "only_intended_difference": (
            "direct target versus prior-residual target and reconstruction"
        ),
    },
    "fractions": LABEL_FRACTIONS,
    "seeds": SEEDS,
    "expected_runs": EXPECTED_RUNS,
    "sequence_length": SEQ_LEN,
    "historical_GRU_parameters": 17217,
    "new_matched_GRU_parameters": new_direct_params,
    "new_parameter_increase_vs_historical_pct": float(
        100.0 * (new_direct_params - 17217) / 17217
    ),
    "train_rows": int(len(train_frame)),
    "validation_rows": int(len(val_frame)),
    "test_rows": int(len(test_frame)),
    "test_segments": int(test_frame["compound_segment"].nunique()),
    "test_was_already_opened_in_12C": True,
    "historical_test_exposure": True,
    "confirmatory": False,
    "evaluation_role": "post-confirmatory EVE information-matched fairness audit",
    "no_rescue_rule": (
        "No architecture, feature, prior, learning-rate, dropout, sequence length, "
        "epoch budget, patience, seed set, sparse selector, or analysis threshold "
        "will be changed after inspecting R2A EVE test outcomes."
    ),
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_unit": "compound acquisition segment",
    "multiplicity_family": (
        "two primary information-matched factorization contrasts: full and 10%"
    ),
    "EVE_test_target_used": True,
    "EVE_test_target_use_role": "post_confirmatory_fairness_audit_only",
}

with open(OUT / f"{EXPERIMENT}_PREFLIGHT.json", "w") as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% Code cell 10 (notebook cell 11)

def train_one(fraction, model_name, seed):
    set_all_seeds(seed)

    target_col = target_col_for_model(model_name)

    labeled = exact_sparse_subset(
        train_frame,
        fraction,
        seed,
    ).copy()

    train_positions = (
        labeled["seq_pos"].astype(int).to_numpy()
    )
    val_positions = (
        val_frame["seq_pos"].astype(int).to_numpy()
    )
    test_positions = (
        test_frame["seq_pos"].astype(int).to_numpy()
    )

    target_scaler = TargetScaler().fit(
        labeled[target_col].to_numpy(float)
    )

    train_ds = CausalSequenceDataset(
        train_frame,
        X_train,
        train_positions,
        target_scaler,
        target_col,
        SEQ_LEN,
    )

    val_ds = CausalSequenceDataset(
        val_frame,
        X_val,
        val_positions,
        target_scaler,
        target_col,
        SEQ_LEN,
    )

    test_ds = CausalSequenceDataset(
        test_frame,
        X_test,
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

    input_dim = len(FEATURES) + 1  # + causal mask
    model = FrozenFamilyGRU(input_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()
    scaler = make_grad_scaler()

    tag = (
        f"EVE_fraction{fraction}_"
        f"{model_name}_seed{seed}"
    )

    fp = contract_fingerprint(
        fraction,
        model_name,
        seed,
    )

    best_path = CKPT_DIR / f"{tag}_best.pt"
    resume_path = CKPT_DIR / f"{tag}_resume.pt"
    history_path = OUT / f"history_{tag}.csv"

    best_val = np.inf
    best_epoch = 0
    stale = 0
    history = []
    start_epoch = 1
    elapsed_before = 0.0

    if resume_path.exists():
        state = safe_torch_load(resume_path, DEVICE)
        assert state["contract_fingerprint"] == fp

        model.load_state_dict(
            state["latest_model_state"]
        )
        optimizer.load_state_dict(
            state["optimizer_state"]
        )

        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(
                state["scaler_state"]
            )

        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        history = list(state["history"])
        elapsed_before = float(
            state.get("elapsed_seconds", 0.0)
        )

        restore_rng_state(
            state.get("rng_state")
        )

        print("RESUME", tag, "epoch", start_epoch)

    t0 = time.perf_counter()

    terminal_resume = bool(
        history
        and (
            stale >= PATIENCE
            or start_epoch > MAX_EPOCHS
        )
    )

    epoch_iter = (
        []
        if terminal_resume
        else range(start_epoch, MAX_EPOCHS + 1)
    )

    for epoch in epoch_iter:
        model.train()
        losses = []

        for xb, yb, _ in train_loader:
            xb = xb.to(
                DEVICE,
                non_blocking=True,
            )
            yb = yb.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                pred = model(xb)
                loss = criterion(pred, yb)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{tag}: non-finite loss"
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

            losses.append(
                float(loss.detach().cpu())
            )

        val_pos, val_raw = predict_loader(
            model,
            val_loader,
            target_scaler,
        )

        val_rows = val_frame.iloc[val_pos]
        val_soc = reconstruct_soc(
            val_rows,
            val_raw,
            model_name,
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
                    "fraction": float(fraction),
                    "model_name": model_name,
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
                    "feature_columns": FEATURES,
                    "structured_prior_as_neural_input": True,
                    "contract_fingerprint": fp,
                },
                best_path,
            )
        else:
            stale += 1

        if (
            epoch % RESUME_SAVE_EVERY == 0
            or stale >= PATIENCE
            or epoch == MAX_EPOCHS
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

        if stale >= PATIENCE:
            break

    assert best_path.exists(), tag
    final_epoch = int(history[-1]["epoch"])

    best_state = safe_torch_load(
        best_path,
        DEVICE,
    )
    assert best_state["contract_fingerprint"] == fp
    model.load_state_dict(
        best_state["model_state"]
    )

    val_pos, val_raw = predict_loader(
        model,
        val_loader,
        target_scaler,
    )
    val_rows = val_frame.iloc[val_pos]
    val_soc = reconstruct_soc(
        val_rows,
        val_raw,
        model_name,
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
        model_name,
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

    pred_df["fraction"] = float(fraction)
    pred_df["model"] = model_name
    pred_df["seed"] = int(seed)
    pred_df["raw_network_output"] = test_raw
    pred_df["prediction"] = test_soc
    pred_df["error"] = test_soc - y_test
    pred_df["abs_error"] = np.abs(pred_df["error"])

    pred_path = (
        PRED_DIR
        / f"{EXPERIMENT}_fraction{fraction}_{model_name}_seed{seed}.parquet"
    )
    pred_df.to_parquet(
        pred_path,
        index=False,
    )

    stopped_by_patience = bool(
        stale >= PATIENCE
        and final_epoch < MAX_EPOCHS
    )
    hit_epoch_cap = bool(
        final_epoch >= MAX_EPOCHS
        and stale < PATIENCE
    )

    return {
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "prediction_file": str(pred_path),
        "prediction_sha256": sha256_file(pred_path),
        "resume_checkpoint": str(resume_path),
        "target_mode": MODEL_SPECS[model_name]["target_mode"],
        "best_epoch": int(best_state["best_epoch"]),
        "final_epoch": int(final_epoch),
        "stopped_by_patience": stopped_by_patience,
        "hit_epoch_cap": hit_epoch_cap,
        "n_params": int(best_state["n_params"]),
        "n_train_labels": int(len(train_ds)),
        "n_validation": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "train_seconds": float(
            elapsed_before
            + (time.perf_counter() - t0)
        ),
        "inference_seconds": float(inference_seconds),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

# %% Code cell 11 (notebook cell 12)

RUNS_INCREMENTAL = OUT / f"{EXPERIMENT}_runs_incremental.csv"
completed = set()

if RUNS_INCREMENTAL.exists():
    old = pd.read_csv(RUNS_INCREMENTAL)

    for r in old.itertuples(index=False):
        try:
            key = (
                float(r.fraction),
                str(r.model),
                int(r.seed),
            )

            ckpt = Path(str(r.checkpoint))
            pred_file = Path(str(r.prediction_file))

            if not ckpt.exists() or not pred_file.exists():
                continue

            expected_fp = contract_fingerprint(
                float(r.fraction),
                str(r.model),
                int(r.seed),
            )

            state = safe_torch_load(
                ckpt,
                "cpu",
            )

            if (
                state.get("contract_fingerprint") == expected_fp
                and sha256_file(ckpt) == str(r.checkpoint_sha256)
                and sha256_file(pred_file) == str(r.prediction_sha256)
            ):
                completed.add(key)
        except Exception:
            pass


for fraction in LABEL_FRACTIONS:
    for model_name in RUN_MODELS:
        for seed in SEEDS:
            key = (
                float(fraction),
                model_name,
                int(seed),
            )

            if key in completed:
                print("SKIP completed:", key)
                continue

            print(
                "\nTRAIN",
                "fraction=", fraction,
                "model=", model_name,
                "seed=", seed,
            )

            result = train_one(
                fraction,
                model_name,
                seed,
            )

            row = {
                "fraction": float(fraction),
                "model": model_name,
                "seed": int(seed),
                "evaluation_role": (
                    "post_confirmatory_eve_information_matched_fairness_audit"
                ),
                **result,
            }

            cur = (
                pd.read_csv(RUNS_INCREMENTAL)
                if RUNS_INCREMENTAL.exists()
                else pd.DataFrame()
            )

            cur = pd.concat(
                [cur, pd.DataFrame([row])],
                ignore_index=True,
            )

            cur = cur.drop_duplicates(
                ["fraction", "model", "seed"],
                keep="last",
            )

            cur.to_csv(
                RUNS_INCREMENTAL,
                index=False,
            )

            resume_ckpt = Path(
                str(result["resume_checkpoint"])
            )
            if resume_ckpt.exists():
                resume_ckpt.unlink()

            completed.add(key)

            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


runs = pd.read_csv(RUNS_INCREMENTAL)
assert len(runs) == EXPECTED_RUNS, (
    len(runs),
    EXPECTED_RUNS,
)

RUNS_FINAL = OUT / f"{EXPERIMENT}_runs.csv"
runs.to_csv(RUNS_FINAL, index=False)

display(
    runs.sort_values(
        ["fraction", "model", "seed"]
    )
)

# %% Code cell 12 (notebook cell 13)

prediction_parts = []

for r in runs.itertuples(index=False):
    p = pd.read_parquet(
        Path(str(r.prediction_file))
    )
    assert len(p) == int(r.n_test)
    assert p["seed"].nunique() == 1
    prediction_parts.append(p)

all_seed_pred = pd.concat(
    prediction_parts,
    ignore_index=True,
)

group_key = [
    "fraction",
    "model",
    "vehicle_id",
    "segment_id",
    "compound_segment",
    "time_value",
    "elapsed_s",
    "target",
    "soc_phy",
]

coverage = (
    all_seed_pred
    .groupby(group_key, dropna=False)["seed"]
    .nunique()
    .reset_index(name="n_seeds")
)

assert (coverage["n_seeds"] == len(SEEDS)).all()

seed_avg = (
    all_seed_pred
    .groupby(
        group_key,
        as_index=False,
        dropna=False,
    )["prediction"]
    .mean()
)

SEED_AVG_PATH = (
    OUT / f"{EXPERIMENT}_seed_averaged_predictions.parquet"
)
seed_avg.to_parquet(
    SEED_AVG_PATH,
    index=False,
)

summary = (
    runs.groupby(
        ["fraction", "model", "target_mode"],
        as_index=False,
    )
    .agg(
        MAE_mean=("test_MAE", "mean"),
        MAE_std=("test_MAE", "std"),
        RMSE_mean=("test_RMSE", "mean"),
        P95_mean=("test_P95", "mean"),
        Bias_mean=("test_Bias", "mean"),
        val_MAE_mean=("val_MAE", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        n_params=("n_params", "first"),
        n_train_labels=("n_train_labels", "first"),
        n_test=("n_test", "first"),
        train_seconds_mean=("train_seconds", "mean"),
    )
)

summary.to_csv(
    OUT / f"{EXPERIMENT}_summary.csv",
    index=False,
)

display(summary)

# %% Code cell 13 (notebook cell 14)

effect_rows = []

for fraction in LABEL_FRACTIONS:
    direct = summary[
        np.isclose(summary["fraction"], fraction)
        & (summary["model"] == DIRECT_MODEL)
    ].iloc[0]

    pr = summary[
        np.isclose(summary["fraction"], fraction)
        & (summary["model"] == PR_MODEL)
    ].iloc[0]

    direct_mae = float(direct["MAE_mean"])
    pr_mae = float(pr["MAE_mean"])

    effect_rows.append({
        "fraction": float(fraction),
        "information_matched_direct_MAE": direct_mae,
        "information_matched_PR_MAE": pr_mae,
        "delta_PR_minus_direct_MAE": pr_mae - direct_mae,
        "factorization_effect_pct": (
            100.0 * (direct_mae - pr_mae) / direct_mae
        ),
        "PR_better_than_information_matched_direct": bool(
            pr_mae < direct_mae
        ),
    })

effects = pd.DataFrame(effect_rows)
effects.to_csv(
    OUT / f"{EXPERIMENT}_factorization_effects.csv",
    index=False,
)

display(effects)

# %% Code cell 14 (notebook cell 15)

def paired_segment_table(pr_rows, direct_rows):
    keys = [
        "compound_segment",
        "time_value",
        "target",
    ]

    a = pr_rows[
        keys + ["prediction"]
    ].rename(
        columns={"prediction": "pred_pr"}
    )

    b = direct_rows[
        keys + ["prediction"]
    ].rename(
        columns={"prediction": "pred_direct"}
    )

    m = a.merge(
        b,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    assert len(m) == len(a) == len(b)

    m["ae_pr"] = np.abs(
        m["pred_pr"] - m["target"]
    )
    m["ae_direct"] = np.abs(
        m["pred_direct"] - m["target"]
    )

    return (
        m.groupby(
            "compound_segment",
            as_index=False,
        )
        .agg(
            PR_segment_MAE=("ae_pr", "mean"),
            Direct_segment_MAE=("ae_direct", "mean"),
            n_rows=("ae_pr", "size"),
        )
    )


def paired_segment_bootstrap(
    seg,
    draws=BOOTSTRAP_DRAWS,
    seed=4201,
):
    diff = (
        seg["PR_segment_MAE"].to_numpy(float)
        - seg["Direct_segment_MAE"].to_numpy(float)
    )

    rng = np.random.default_rng(seed)
    n = len(diff)
    assert n >= 2

    samples = np.empty(draws, dtype=float)

    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diff[idx]))

    p_left = (
        np.sum(samples <= 0) + 1.0
    ) / (draws + 1.0)
    p_right = (
        np.sum(samples >= 0) + 1.0
    ) / (draws + 1.0)

    return {
        "n_segments": int(n),
        "PR_segment_balanced_MAE": float(
            seg["PR_segment_MAE"].mean()
        ),
        "Direct_segment_balanced_MAE": float(
            seg["Direct_segment_MAE"].mean()
        ),
        "delta_PR_minus_Direct": float(
            np.mean(diff)
        ),
        "CI95_low": float(
            np.percentile(samples, 2.5)
        ),
        "CI95_high": float(
            np.percentile(samples, 97.5)
        ),
        "P_PR_lower_MAE": float(
            np.mean(samples < 0)
        ),
        "two_sided_bootstrap_p": float(
            min(
                1.0,
                2.0 * min(p_left, p_right),
            )
        ),
        "n_boot": int(draws),
    }


bootstrap_rows = []

for fraction in LABEL_FRACTIONS:
    pr_rows = seed_avg[
        np.isclose(seed_avg["fraction"], fraction)
        & (seed_avg["model"] == PR_MODEL)
    ].copy()

    direct_rows = seed_avg[
        np.isclose(seed_avg["fraction"], fraction)
        & (seed_avg["model"] == DIRECT_MODEL)
    ].copy()

    assert len(pr_rows) == len(test_frame)
    assert len(direct_rows) == len(test_frame)

    seg = paired_segment_table(
        pr_rows,
        direct_rows,
    )

    result = paired_segment_bootstrap(
        seg,
        seed=4201 + int(round(fraction * 100)),
    )

    bootstrap_rows.append({
        "fraction": float(fraction),
        **result,
    })

bootstrap = pd.DataFrame(bootstrap_rows)


def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    m = len(p)

    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted_sorted[rank] = min(1.0, running)

    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted


bootstrap["Holm_p"] = holm_adjust(
    bootstrap["two_sided_bootstrap_p"].to_numpy(float)
)

bootstrap["CI_supports_PR"] = (
    bootstrap["CI95_high"] < 0
)
bootstrap["Holm_significant_0p05"] = (
    bootstrap["Holm_p"] < 0.05
)
bootstrap["factorization_supported_after_Holm"] = (
    bootstrap["CI_supports_PR"]
    & bootstrap["Holm_significant_0p05"]
)

bootstrap.to_csv(
    OUT / f"{EXPERIMENT}_paired_segment_bootstrap.csv",
    index=False,
)

display(bootstrap)

# %% Code cell 15 (notebook cell 16)

# Seed-paired diagnostic: each direct and PR run uses the same seed,
# architecture, inputs, sparse-label subset, and budget.

seed_pair = runs[
    [
        "fraction", "model", "seed",
        "test_MAE", "test_RMSE", "test_P95", "test_Bias",
    ]
].pivot(
    index=["fraction", "seed"],
    columns="model",
    values=["test_MAE", "test_RMSE", "test_P95", "test_Bias"],
).reset_index()

# Flatten MultiIndex columns.
seed_pair.columns = [
    (
        f"{a}__{b}"
        if b
        else str(a)
    )
    for a, b in seed_pair.columns
]

seed_pair["delta_MAE_PR_minus_direct"] = (
    seed_pair[f"test_MAE__{PR_MODEL}"]
    - seed_pair[f"test_MAE__{DIRECT_MODEL}"]
)

seed_pair["PR_lower_MAE"] = (
    seed_pair["delta_MAE_PR_minus_direct"] < 0
)

seed_pair.to_csv(
    OUT / f"{EXPERIMENT}_seed_paired_diagnostic.csv",
    index=False,
)

display(seed_pair)

# %% Code cell 16 (notebook cell 17)

decision = effects.merge(
    bootstrap[
        [
            "fraction",
            "PR_segment_balanced_MAE",
            "Direct_segment_balanced_MAE",
            "delta_PR_minus_Direct",
            "CI95_low",
            "CI95_high",
            "two_sided_bootstrap_p",
            "Holm_p",
            "factorization_supported_after_Holm",
        ]
    ],
    on="fraction",
    validate="one_to_one",
)

seed_support = (
    seed_pair.groupby("fraction", as_index=False)
    .agg(
        seeds_PR_lower_MAE=("PR_lower_MAE", "sum"),
        n_seeds=("PR_lower_MAE", "size"),
        mean_seed_paired_delta_MAE=(
            "delta_MAE_PR_minus_direct",
            "mean",
        ),
    )
)

decision = decision.merge(
    seed_support,
    on="fraction",
    validate="one_to_one",
)

decision.to_csv(
    OUT / f"{EXPERIMENT}_decision_table.csv",
    index=False,
)

display(decision)

print(
    "Row-level PR better:",
    int(
        decision[
            "PR_better_than_information_matched_direct"
        ].sum()
    ),
    "/",
    len(decision),
)

print(
    "Segment bootstrap + Holm supports PR:",
    int(
        decision[
            "factorization_supported_after_Holm"
        ].sum()
    ),
    "/",
    len(decision),
)

# %% Code cell 17 (notebook cell 18)

# Optional context only: historical 12C headline results.
# These are not part of the R2A primary inferential family.

final_table = pd.read_csv(
    TEST_DIR / "FINAL_EVE_CONFIRMATORY_TABLE.csv"
)

final_table.to_csv(
    OUT / f"{EXPERIMENT}_historical_12C_context.csv",
    index=False,
)

display(final_table)

# %% Code cell 18 (notebook cell 19)

# Final integrity checks and manifest.

assert len(runs) == EXPECTED_RUNS
assert set(runs["model"]) == set(RUN_MODELS)
assert set(runs["seed"]) == set(SEEDS)

assert np.isfinite(
    runs[
        [
            "val_MAE",
            "test_MAE",
            "test_RMSE",
            "test_P95",
            "test_Bias",
        ]
    ].to_numpy(float)
).all()

for r in runs.itertuples(index=False):
    ckpt = Path(str(r.checkpoint))
    pred_file = Path(str(r.prediction_file))

    assert ckpt.exists()
    assert pred_file.exists()

    assert sha256_file(ckpt) == str(r.checkpoint_sha256)
    assert sha256_file(pred_file) == str(r.prediction_sha256)

assert selector_audit["count_match"].all()
assert parameter_audit["direct_and_PR_exact_same_capacity"].all()
assert parameter_audit["exact_same_input_information"].all()
assert parameter_audit["within_5pct_historical_budget"].all()
assert len(bootstrap) == 2

manifest = {
    "experiment": EXPERIMENT,
    "status": "POST_CONFIRMATORY_EVE_INFO_MATCHED_FAIRNESS_AUDIT_COMPLETE",
    "dataset": "EVE Pilot Renault ZOE Q90",
    "target": "dashboard/user-facing BMS-reported State of Charge",
    "primary_model": PR_MODEL,
    "ablation_model": DIRECT_MODEL,
    "fractions": LABEL_FRACTIONS,
    "seeds": SEEDS,
    "n_runs": int(len(runs)),
    "expected_runs": EXPECTED_RUNS,
    "sequence_length": SEQ_LEN,
    "hidden": HIDDEN,
    "head_hidden": HEAD_HIDDEN,
    "dropout": DROPOUT,
    "optimizer": "AdamW",
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "base_feature_columns": BASE_FEATURES,
    "structured_prior_as_neural_input": True,
    "structured_prior_input_column": "soc_phy",
    "prior_beta_pp_per_ah_train_only": float(
        meta["beta_pp_per_ah_train_only"]
    ),
    "direct_target": "BMS-reported SoC",
    "residual_target": "BMS_SOC - soc_phy",
    "prediction_reconstruction_for_PR": (
        "soc_phy + predicted_residual"
    ),
    "exact_information_match_between_new_models": True,
    "exact_architecture_match_between_new_models": True,
    "exact_parameter_count_match_between_new_models": True,
    "only_intended_primary_difference": (
        "direct target versus prior-residual target/reconstruction"
    ),
    "historical_GRU_params": 17217,
    "new_matched_GRU_params": new_direct_params,
    "new_parameter_increase_vs_historical_pct": float(
        100.0 * (new_direct_params - 17217) / 17217
    ),
    "train_rows": int(len(train_frame)),
    "validation_rows": int(len(val_frame)),
    "test_rows": int(len(test_frame)),
    "test_segments": int(test_frame["compound_segment"].nunique()),
    "sparse_selector": (
        "exact per-vehicle deterministic MD5 subset, matched to frozen 12B counts"
    ),
    "test_was_already_opened_in_12C": True,
    "historical_test_exposure": True,
    "evaluation_role": (
        "post-confirmatory EVE information-matched fairness audit"
    ),
    "confirmatory": False,
    "no_posthoc_architecture_tuning": True,
    "no_rescue_after_outcome_inspection": True,
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_unit": "compound acquisition segment",
    "multiplicity_correction": (
        "Holm across two primary EVE factorization contrasts: full and 10%"
    ),
    "trainval_sha256": trainval_sha,
    "test_sha256": test_sha,
}

with open(
    OUT / f"{EXPERIMENT}_MANIFEST.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )

print(json.dumps(manifest, indent=2))

# %% [markdown] Primary interpretation
