# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/R1A_InfoMatched_GRU_Factorization_Colab.ipynb
# Notebook SHA256: 677cb0453fb557b4ff33058051ddac5f6c3e47842a55a71ccffacce6e418cfea
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R1A — Information-Matched Prior-Residual Factorization Audit (GRU)

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

BASE_10 = OUTPUTS / "model_screening_10"
CANON = BASE_10 / "canonical"

OUT = ROOT / "Revizyon1" / "outputs" / "R1A_info_matched_gru"
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "seed_predictions"
FIG_DIR = OUT / "figures"

for p in [OUT, CKPT_DIR, PRED_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R1A_INFO_MATCHED_GRU"
FAMILY = "GRU"
DIRECT_MODEL = "Dense_GRU_PriorInput_Direct"
PR_MODEL = "Dense_GRU_PriorInput_PR"
FROZEN_PR_MODEL = "Dense_PR_GRU"
FROZEN_VANILLA_MODEL = "GRU"

DOMAINS = ["VED", "KU", "SCUT", "TUM"]
DOMAIN_RUN_ORDER = ["VED", "SCUT", "TUM", "KU"]

MODEL_SPECS = {
    DIRECT_MODEL: {
        "target_mode": "direct",
    },
    PR_MODEL: {
        "target_mode": "residual",
    },
}

RUN_MODELS = [DIRECT_MODEL, PR_MODEL]

LABEL_FRACTIONS = [1.0, 0.10]
SEEDS = [42, 43, 44, 45, 46]

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
BOOTSTRAP_DRAWS = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
BATCH_SIZE = 512 if AMP_ENABLED else 128

EXPECTED_RUNS = (
    len(DOMAINS)
    * len(LABEL_FRACTIONS)
    * len(RUN_MODELS)
    * len(SEEDS)
)

assert EXPECTED_RUNS == 80

print("Experiment:", EXPERIMENT)
print("DEVICE:", DEVICE)
print("Output:", OUT)
print("New models:", RUN_MODELS)
print("Expected runs:", EXPECTED_RUNS)

# %% [markdown] 1. Locate frozen reference artifacts

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


FINAL_BASELINE_SUMMARY = find_unique_file(
    OUTPUTS,
    "FINAL_temporal_baseline_summary_11B_plus_11C_v2.csv",
)

FINAL_BASELINE_PRED = find_unique_file(
    OUTPUTS,
    "FINAL_temporal_baseline_seed_averaged_predictions_11B_plus_11C_v2.parquet",
)

SUMMARY_13B = find_unique_file(
    OUTPUTS,
    "14B_DENSE_PR_GRU_summary.csv",
)

PRED_13B = find_unique_file(
    OUTPUTS,
    "14B_DENSE_PR_GRU_seed_averaged_predictions.parquet",
)

K10_SUMMARY = find_unique_file(
    OUTPUTS,
    "10K_summary.csv",
)

K10_PRED = find_unique_file(
    OUTPUTS,
    "10K_seed_averaged_predictions.parquet",
)

MANIFEST_11B = find_unique_file(
    OUTPUTS,
    "11B_manifest.json",
)

MANIFEST_11C = find_unique_file(
    OUTPUTS,
    "11C_v2_manifest.json",
)

MANIFEST_10K = find_unique_file(
    OUTPUTS,
    "10K_manifest.json",
)

baseline_summary = pd.read_csv(FINAL_BASELINE_SUMMARY)
summary_13b = pd.read_csv(SUMMARY_13B)
k10_summary_all = pd.read_csv(K10_SUMMARY)

with open(MANIFEST_11B) as f:
    man11b = json.load(f)

with open(MANIFEST_11C) as f:
    man11c = json.load(f)

with open(MANIFEST_10K) as f:
    man10k = json.load(f)

assert man11b["strict_causality"] is True
assert int(man11b["sequence_length"]) == SEQ_LEN
assert man11b["architecture_tuning"] is False
assert man11b["feature_rule"]["soc_phy_given_to_deep_baselines"] is False
assert man11b["feature_rule"]["target_or_residual_given_to_deep_baselines"] is False

assert man11c["architecture_changed_from_11B"] is False
assert man11c["features_changed_from_11B"] is False
assert man11c["sequence_length_changed_from_11B"] is False

assert man10k["target"] == "production BMS-reported SoC"

EXCLUDED_TOKENS = [
    str(x).lower()
    for x in man11b["feature_rule"]["excluded_tokens"]
]


def get_training_budget(domain, fraction):
    q = baseline_summary[
        (baseline_summary["domain"] == domain)
        & np.isclose(
            baseline_summary["fraction"].astype(float),
            float(fraction),
        )
        & (baseline_summary["model"] == FAMILY)
    ]

    assert len(q) == 1, (domain, fraction, FAMILY, len(q))

    source = str(q.iloc[0]["source_experiment"])

    if source == "11B_100epoch":
        return 100, 10, source

    if source == "11C_v2_200epoch":
        return 200, 20, source

    raise RuntimeError(
        f"Unexpected final {FAMILY} source_experiment: {source}"
    )


print("Frozen baseline:", FINAL_BASELINE_SUMMARY)
print("Frozen 14B PR reference:", SUMMARY_13B)
print("Frozen 10K:", K10_SUMMARY)
print("Excluded feature tokens:", EXCLUDED_TOKENS)

# %% [markdown] 2. Load canonical domains and reproduce the final deep feature contract

# %% Code cell 3 (notebook cell 6)
DOMAIN_DATA = {}
DOMAIN_FEATURES = {}
DOMAIN_META = {}
DOMAIN_FINGERPRINTS = {}

feature_audit_rows = []

for domain in DOMAINS:
    p = CANON / f"{domain}_canonical.parquet"
    m = CANON / f"{domain}_metadata.json"

    assert p.exists(), p
    assert m.exists(), m

    d = pd.read_parquet(p).copy()

    with open(m) as f:
        meta = json.load(f)

    raw_features = list(meta["feature_columns"])

    base_features = [
        c
        for c in raw_features
        if not any(
            token in c.lower()
            for token in EXCLUDED_TOKENS
        )
    ]

    # R1A fairness change: both new GRU variants receive the
    # same structured prior p_t explicitly as a causal neural input.
    assert "soc_phy" not in base_features
    features = base_features + ["soc_phy"]

    excluded_features = [
        c for c in raw_features
        if c not in base_features
    ]

    required = {
        "vehicle_id",
        "segment_id",
        "time_value",
        "elapsed_s",
        "split",
        "target",
        "soc_phy",
        *raw_features,
    }

    missing = sorted(required - set(d.columns))
    assert not missing, (domain, missing)

    d = d[
        d["split"].isin(["train", "val", "test"])
    ].copy()

    actual_counts = {
        k: int(v)
        for k, v in d["split"].value_counts().to_dict().items()
    }

    expected_counts = {
        k: int(v)
        for k, v in meta["split_counts"].items()
    }

    assert actual_counts == expected_counts, (
        domain,
        actual_counts,
        expected_counts,
    )

    assert np.isfinite(d["target"].to_numpy(float)).all()
    assert np.isfinite(d["soc_phy"].to_numpy(float)).all()

    d["vehicle_id"] = d["vehicle_id"].astype(str)
    d["segment_id"] = d["segment_id"].astype(str)

    d["compound_segment"] = (
        d["vehicle_id"]
        + "||"
        + d["segment_id"]
    )

    d["pr_target"] = (
        d["target"].to_numpy(float)
        - d["soc_phy"].to_numpy(float)
    )

    assert "soc_phy" in features
    assert features[-1] == "soc_phy"

    canonical_sha = sha256_file(p)

    fp_payload = {
        "domain": domain,
        "canonical_sha256": canonical_sha,
        "features": features,
        "excluded_features": excluded_features,
        "n_rows": int(len(d)),
        "split_counts": expected_counts,
        "n_segments": int(d["compound_segment"].nunique()),
    }

    fp = hashlib.sha256(
        json.dumps(
            fp_payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    DOMAIN_DATA[domain] = d
    DOMAIN_FEATURES[domain] = features
    DOMAIN_META[domain] = meta
    DOMAIN_FINGERPRINTS[domain] = fp

    for c in raw_features:
        feature_audit_rows.append({
            "domain": domain,
            "feature": c,
            "used": bool(c in features),
            "excluded_by_11B_rule": bool(c not in features),
        })

    feature_audit_rows.append({
        "domain": domain,
        "feature": "soc_phy",
        "used": True,
        "excluded_by_11B_rule": True,
        "R1A_role": "explicit structured-prior input for both new models",
    })

    print(
        domain,
        "rows=", len(d),
        "R1A_input_features=", len(features),
        "excluded=", excluded_features,
    )

feature_audit = pd.DataFrame(feature_audit_rows)
feature_audit.to_csv(
    OUT / f"{EXPERIMENT}_feature_audit.csv",
    index=False,
)

# %% [markdown] 3. Exact 10K sparse-label selector

# %% Code cell 4 (notebook cell 8)
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
        int(
            hashlib.md5(
                str(s).encode("utf-8")
            ).hexdigest()[:12],
            16,
        )
        / float(16 ** 12)
    )


def exact_sparse_subset(df, fraction, seed):
    if fraction >= 0.999999:
        return df.copy()

    pieces = []

    for _, g in df.groupby("vehicle_id"):
        k = max(
            1,
            int(round(float(fraction) * len(g))),
        )

        keys = (
            g["vehicle_id"].astype(str)
            + "|"
            + g["segment_id"].astype(str)
            + "|"
            + g["time_value"].astype(str)
            + "|"
            + str(seed)
        )

        order = (
            keys
            .map(stable_hash_u01)
            .sort_values()
            .index
        )

        pieces.append(
            g.loc[order[:k]].copy()
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


selector_rows = []

for domain in DOMAINS:
    train = DOMAIN_DATA[domain]
    train = train[train["split"] == "train"].copy()

    for fraction in LABEL_FRACTIONS:
        ref = baseline_summary[
            (baseline_summary["domain"] == domain)
            & np.isclose(
                baseline_summary["fraction"].astype(float),
                float(fraction),
            )
            & (baseline_summary["model"] == FAMILY)
        ]

        assert len(ref) == 1

        expected_n = int(
            ref.iloc[0]["n_train_labels"]
        )

        for seed in SEEDS:
            z = exact_sparse_subset(
                train,
                fraction,
                seed,
            )

            assert len(z) == expected_n, (
                domain,
                fraction,
                seed,
                len(z),
                expected_n,
            )

            selector_rows.append({
                "domain": domain,
                "fraction": float(fraction),
                "seed": int(seed),
                "n_selected": int(len(z)),
                "expected_reference_n": int(expected_n),
                "count_match": True,
            })

selector_audit = pd.DataFrame(selector_rows)

selector_audit.to_csv(
    OUT / f"{EXPERIMENT}_sparse_selector_audit.csv",
    index=False,
)

display(selector_audit)

# %% [markdown] 4. Train-only preprocessing and strictly causal sequence endpoints

# %% Code cell 5 (notebook cell 10)
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

        if (
            not np.isfinite(self.std_)
            or self.std_ < 1e-8
        ):
            self.std_ = 1.0

        return self

    def transform(self, y):
        return (
            (
                np.asarray(y, float)
                - self.mean_
            )
            / self.std_
        ).astype(np.float32)

    def inverse(self, y):
        return (
            np.asarray(y, float)
            * self.std_
            + self.mean_
        )


def prepare_split_frame(domain_df, split):
    z = domain_df[
        domain_df["split"] == split
    ].copy()

    z = z.sort_values(
        [
            "compound_segment",
            "time_value",
        ],
        kind="stable",
    ).reset_index(drop=True)

    z["seq_pos"] = np.arange(
        len(z),
        dtype=int,
    )

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

        self.positions = np.asarray(
            endpoint_positions,
            dtype=int,
        )

        self.target_scaler = target_scaler
        self.target_col = target_col
        self.seq_len = int(seq_len)

        self.seg_start = (
            self.frame
            .groupby(
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
            self.seg_start[
                row["compound_segment"]
            ]
        )

        start = max(
            seg0,
            pos - self.seq_len + 1,
        )

        x = self.X[start : pos + 1]

        n = len(x)
        pad = self.seq_len - n

        # Final channel is the causal-validity mask.
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
            self.target_scaler
            .transform([y_raw])[0]
        )

        return (
            torch.from_numpy(out),
            torch.tensor(
                y_scaled,
                dtype=torch.float32,
            ),
            pos,
        )


DOMAIN_PREP = {}

for domain in DOMAINS:
    d = DOMAIN_DATA[domain]
    features = DOMAIN_FEATURES[domain]

    train_frame = prepare_split_frame(d, "train")
    val_frame = prepare_split_frame(d, "val")
    test_frame = prepare_split_frame(d, "test")

    prep = TrainOnlyPreprocessor().fit(
        train_frame[features].to_numpy(float)
    )

    X_train = prep.transform(
        train_frame[features].to_numpy(float)
    )
    X_val = prep.transform(
        val_frame[features].to_numpy(float)
    )
    X_test = prep.transform(
        test_frame[features].to_numpy(float)
    )

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_test).all()

    DOMAIN_PREP[domain] = {
        "train_frame": train_frame,
        "val_frame": val_frame,
        "test_frame": test_frame,
        "preprocessor": prep,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
    }

print("Train-only preprocessing complete.")

# %% [markdown] 5. Information-matched Dense-GRU architecture

# %% Code cell 6 (notebook cell 12)
class DenseRecurrentNet(nn.Module):
    """
    DenseNet-style recurrent feature reuse.

    Layer 1 sees X.
    Layer 2 sees [X, H1].
    Layer 3 sees [X, H1, H2].

    Direct and prior-residual variants share this\n    exact architecture, parameter count, and prior input.
    """
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = int(input_dim)

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

        self.feature_dropout = nn.Dropout(
            DROPOUT
        )

        self.dense_fusion = nn.Linear(
            DENSE_LAYERS * GROWTH,
            FUSION_WIDTH,
        )

        # Feature-level residual projection from
        # the current raw/padded endpoint vector.
        self.input_skip = nn.Linear(
            input_dim,
            FUSION_WIDTH,
        )

        self.fusion_norm = nn.LayerNorm(
            FUSION_WIDTH
        )

        self.head = nn.Sequential(
            nn.Linear(
                FUSION_WIDTH,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h1, _ = self.rnn1(x)
        h1_for_next = self.feature_dropout(h1)

        x2 = torch.cat(
            [x, h1_for_next],
            dim=-1,
        )

        h2, _ = self.rnn2(x2)
        h2_for_next = self.feature_dropout(h2)

        x3 = torch.cat(
            [
                x,
                h1_for_next,
                h2_for_next,
            ],
            dim=-1,
        )

        h3, _ = self.rnn3(x3)

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

        fused = self.fusion_norm(fused)
        fused = torch.relu(fused)

        return (
            self.head(fused)
            .squeeze(-1)
        )


def build_model(input_dim):
    return DenseRecurrentNet(input_dim)


def count_params(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

# %% [markdown] 6. Parameter-budget and exact information-match audit

# %% Code cell 7 (notebook cell 14)
parameter_rows = []

for domain in DOMAINS:
    input_dim = len(DOMAIN_FEATURES[domain]) + 1

    ref = baseline_summary[
        (baseline_summary["domain"] == domain)
        & (baseline_summary["model"] == FAMILY)
    ]

    ref_counts = sorted(
        set(ref["n_params"].astype(int))
    )

    assert len(ref_counts) == 1

    baseline_params = ref_counts[0]

    model = build_model(input_dim)
    dense_params = count_params(model)

    ratio = dense_params / baseline_params

    # Fixed in advance: close-capacity comparison.
    assert 0.95 <= ratio <= 1.05, (
        domain,
        FAMILY,
        dense_params,
        baseline_params,
        ratio,
    )

    with torch.no_grad():
        x = torch.zeros(
            4,
            SEQ_LEN,
            input_dim,
        )

        y = model(x)

        assert tuple(y.shape) == (4,)
        assert torch.isfinite(y).all()

    parameter_rows.append({
        "domain": domain,
        "family": FAMILY,
        "input_dim_with_mask": int(input_dim),
        "frozen_baseline_params": int(baseline_params),
        "dense_params": int(dense_params),
        "dense_to_baseline_ratio": float(ratio),
        "within_5pct": True,
        "direct_and_residual_exact_same_capacity": True,
        "direct_and_residual_exact_same_input_information": True,
        "structured_prior_input": True,
    })

    del model

parameter_audit = pd.DataFrame(parameter_rows)

parameter_audit.to_csv(
    OUT / f"{EXPERIMENT}_parameter_budget_audit.csv",
    index=False,
)

display(parameter_audit)

# %% [markdown] 7. Reproducibility, target reconstruction, and safe resume

# %% Code cell 8 (notebook cell 16)
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

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


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


def reconstruct_soc(
    frame_rows,
    raw_model_output,
    model_name,
):
    mode = MODEL_SPECS[model_name]["target_mode"]

    raw_model_output = np.asarray(
        raw_model_output,
        float,
    )

    if mode == "direct":
        return raw_model_output

    if mode == "residual":
        return (
            frame_rows["soc_phy"].to_numpy(float)
            + raw_model_output
        )

    raise ValueError(mode)


def run_contract_fingerprint(
    domain,
    fraction,
    model_name,
    seed,
):
    max_epochs, patience, source = (
        get_training_budget(
            domain,
            fraction,
        )
    )

    payload = {
        "experiment": EXPERIMENT,
        "family": FAMILY,
        "domain": domain,
        "fraction": float(fraction),
        "model_name": model_name,
        "target_mode": MODEL_SPECS[model_name]["target_mode"],
        "seed": int(seed),
        "features": DOMAIN_FEATURES[domain],
        "sequence_length": SEQ_LEN,
        "dense_layers": DENSE_LAYERS,
        "growth": GROWTH,
        "fusion_width": FUSION_WIDTH,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "matched_budget_source": source,
        "domain_fingerprint": DOMAIN_FINGERPRINTS[domain],
    }

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@torch.no_grad()
def predict_loader(
    model,
    loader,
    target_scaler,
):
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

        positions.append(
            np.asarray(pos)
        )

    pred_scaled = np.concatenate(pred_scaled)
    positions = np.concatenate(positions).astype(int)

    raw_pred = target_scaler.inverse(
        pred_scaled
    )

    return positions, raw_pred

# %% [markdown] 8. Preflight / no-rescue contract

# %% Code cell 9 (notebook cell 18)
budget_rows = []

for domain in DOMAINS:
    for fraction in LABEL_FRACTIONS:
        max_epochs, patience, source = get_training_budget(
            domain,
            fraction,
        )

        budget_rows.append({
            "domain": domain,
            "fraction": float(fraction),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "matched_source": source,
        })

budget_audit = pd.DataFrame(budget_rows)

budget_audit.to_csv(
    OUT / f"{EXPERIMENT}_training_budget_audit.csv",
    index=False,
)


preflight = {
    "experiment": EXPERIMENT,
    "status": "FROZEN_BEFORE_R1A_OUTCOME_INSPECTION",
    "family": FAMILY,
    "primary_model": PR_MODEL,
    "ablation_model": DIRECT_MODEL,
    "frozen_comparators": [
        FROZEN_VANILLA_MODEL,
        FROZEN_PR_MODEL,
        "SC_RS_SFPEF_10K",
    ],
    "primary_question": (
        f"Does {PR_MODEL} improve over {DIRECT_MODEL} when both models "
        "receive the exact same causal telemetry + structured prior input, "
        "thereby isolating target-space prior-residual factorization?"
    ),
    "secondary_question": (
        f"How does the information-matched {PR_MODEL} compare with the "
        f"frozen no-prior-input {FROZEN_PR_MODEL} reference?"
    ),
    "information_match": {
        "same_architecture": True,
        "same_parameter_count": True,
        "same_base_features": True,
        "same_structured_prior_input": "soc_phy",
        "same_sequence_length": True,
        "same_sparse_label_subset": True,
        "same_optimizer_and_budget": True,
        "only_intended_difference": "direct target vs residual target + reconstruction",
    },
    "no_rescue_rule": (
        "No layer count, growth width, fusion width, feature, prior, "
        "learning-rate, dropout, sequence length, architecture, or "
        "threshold will be changed after inspecting R1A test outcomes."
    ),
    "historical_test_exposure": True,
    "confirmatory": False,
    "evaluation_role": "post-confirmatory development fairness audit",
    "EVE_12C_target_used": False,
}

with open(
    OUT / f"{EXPERIMENT}_PREFLIGHT.json",
    "w",
) as f:
    json.dump(
        preflight,
        f,
        indent=2,
    )

display(budget_audit)
print(json.dumps(preflight, indent=2))

# %% [markdown] 9. Train and score one frozen information-matched run

# %% Code cell 10 (notebook cell 20)
def train_one(
    domain,
    fraction,
    model_name,
    seed,
):
    set_all_seeds(seed)

    target_col = target_col_for_model(model_name)

    max_epochs, patience, budget_source = (
        get_training_budget(
            domain,
            fraction,
        )
    )

    pack = DOMAIN_PREP[domain]

    train_frame = pack["train_frame"]
    val_frame = pack["val_frame"]
    test_frame = pack["test_frame"]

    labeled = exact_sparse_subset(
        train_frame,
        fraction,
        seed,
    ).copy()

    train_positions = (
        labeled["seq_pos"]
        .astype(int)
        .to_numpy()
    )

    target_scaler = TargetScaler().fit(
        labeled[target_col].to_numpy(float)
    )

    val_positions = (
        val_frame["seq_pos"]
        .astype(int)
        .to_numpy()
    )

    test_positions = (
        test_frame["seq_pos"]
        .astype(int)
        .to_numpy()
    )

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

    model = build_model(input_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()
    scaler = make_grad_scaler()

    tag = (
        f"{domain}_fraction{fraction}_"
        f"{model_name}_seed{seed}"
    )

    contract_fp = run_contract_fingerprint(
        domain,
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
        state = safe_torch_load(
            resume_path,
            DEVICE,
        )

        assert state["contract_fingerprint"] == contract_fp

        model.load_state_dict(
            state["latest_model_state"]
        )

        optimizer.load_state_dict(
            state["optimizer_state"]
        )

        if (
            scaler is not None
            and state.get("scaler_state")
        ):
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
            stale >= patience
            or start_epoch > max_epochs
        )
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
                    "domain": domain,
                    "fraction": float(fraction),
                    "model_name": model_name,
                    "family": FAMILY,
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
                    "structured_prior_as_neural_input": True,
                    "contract_fingerprint": contract_fp,
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
                    "contract_fingerprint": contract_fp,
                },
                resume_path,
            )

        if stale >= patience:
            break

    assert best_path.exists(), tag

    final_epoch = int(history[-1]["epoch"])

    best_state = safe_torch_load(
        best_path,
        DEVICE,
    )

    assert best_state["contract_fingerprint"] == contract_fp

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

    inference_seconds = (
        time.perf_counter() - infer_t0
    )

    test_rows = (
        test_frame.iloc[test_pos]
        .copy()
    )

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

    pred_df["domain"] = domain
    pred_df["fraction"] = float(fraction)
    pred_df["model"] = model_name
    pred_df["seed"] = int(seed)
    pred_df["raw_network_output"] = test_raw
    pred_df["prediction"] = test_soc
    pred_df["error"] = test_soc - y_test
    pred_df["abs_error"] = np.abs(pred_df["error"])

    pred_path = (
        PRED_DIR
        / (
            f"{EXPERIMENT}_{domain}_"
            f"fraction{fraction}_"
            f"{model_name}_seed{seed}.parquet"
        )
    )

    pred_df.to_parquet(
        pred_path,
        index=False,
    )

    stopped_by_patience = bool(
        stale >= patience
        and final_epoch < max_epochs
    )

    hit_epoch_cap = bool(
        final_epoch >= max_epochs
        and stale < patience
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
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "matched_budget_source": budget_source,
        "train_seconds": float(
            elapsed_before
            + (time.perf_counter() - t0)
        ),
        "inference_seconds": float(inference_seconds),
        **{
            f"val_{k}": v
            for k, v in val_metrics.items()
        },
        **{
            f"test_{k}": v
            for k, v in test_metrics.items()
        },
    }

# %% [markdown] 10. Run all 80 frozen information-matched GRU experiments with safe resume

# %% Code cell 11 (notebook cell 22)
RUNS_INCREMENTAL = (
    OUT / f"{EXPERIMENT}_runs_incremental.csv"
)

completed = set()

if RUNS_INCREMENTAL.exists():
    old = pd.read_csv(RUNS_INCREMENTAL)

    for r in old.itertuples(index=False):
        try:
            key = (
                str(r.domain),
                float(r.fraction),
                str(r.model),
                int(r.seed),
            )

            ckpt = Path(str(r.checkpoint))
            pred = Path(str(r.prediction_file))

            if not ckpt.exists() or not pred.exists():
                continue

            expected_fp = run_contract_fingerprint(
                str(r.domain),
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
                and sha256_file(pred) == str(r.prediction_sha256)
            ):
                completed.add(key)

        except Exception:
            pass


for domain in DOMAIN_RUN_ORDER:
    for fraction in LABEL_FRACTIONS:
        for model_name in RUN_MODELS:
            for seed in SEEDS:
                key = (
                    domain,
                    float(fraction),
                    model_name,
                    int(seed),
                )

                if key in completed:
                    print("SKIP completed:", key)
                    continue

                print(
                    "\nTRAIN",
                    domain,
                    "fraction=", fraction,
                    "model=", model_name,
                    "seed=", seed,
                )

                result = train_one(
                    domain,
                    fraction,
                    model_name,
                    seed,
                )

                row = {
                    "domain": domain,
                    "fraction": float(fraction),
                    "model": model_name,
                    "seed": int(seed),
                    "evaluation_role": (
                        "post_confirmatory_development_test"
                    ),
                    **result,
                }

                cur = (
                    pd.read_csv(RUNS_INCREMENTAL)
                    if RUNS_INCREMENTAL.exists()
                    else pd.DataFrame()
                )

                cur = pd.concat(
                    [
                        cur,
                        pd.DataFrame([row]),
                    ],
                    ignore_index=True,
                )

                cur = cur.drop_duplicates(
                    [
                        "domain",
                        "fraction",
                        "model",
                        "seed",
                    ],
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

assert set(runs["domain"]) == set(DOMAINS)
assert set(runs["model"]) == set(RUN_MODELS)
assert set(runs["seed"]) == set(SEEDS)

RUNS_FINAL = OUT / f"{EXPERIMENT}_runs.csv"

runs.to_csv(
    RUNS_FINAL,
    index=False,
)

display(
    runs.sort_values(
        [
            "domain",
            "fraction",
            "model",
            "seed",
        ]
    )
)

# %% [markdown] 11. Five-seed summary and exact seed-averaged predictions

# %% Code cell 12 (notebook cell 24)
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
    "domain",
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
    .groupby(
        group_key,
        dropna=False,
    )["seed"]
    .nunique()
    .reset_index(name="n_seeds")
)

assert (
    coverage["n_seeds"] == len(SEEDS)
).all()

seed_avg = (
    all_seed_pred
    .groupby(
        group_key,
        as_index=False,
        dropna=False,
    )["prediction"]
    .mean()
)

seed_avg.to_parquet(
    OUT / f"{EXPERIMENT}_seed_averaged_predictions.parquet",
    index=False,
)

summary = (
    runs
    .groupby(
        [
            "domain",
            "fraction",
            "model",
            "target_mode",
        ],
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
        final_epoch_mean=("final_epoch", "mean"),
        seeds_hitting_epoch_cap=("hit_epoch_cap", "sum"),
        seeds_stopped_by_patience=("stopped_by_patience", "sum"),
        n_params=("n_params", "first"),
        n_train_labels=("n_train_labels", "first"),
        n_test=("n_test", "first"),
        train_seconds_mean=("train_seconds", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
    )
)

summary.to_csv(
    OUT / f"{EXPERIMENT}_summary.csv",
    index=False,
)

display(summary)

# %% [markdown] 12. Compare against frozen no-prior GRU, frozen 14B PR-GRU, and 10K

# %% Code cell 13 (notebook cell 26)
vanilla_ref = baseline_summary[
    baseline_summary["model"] == FAMILY
][
    [
        "domain",
        "fraction",
        "MAE_mean",
        "MAE_std",
        "n_params",
        "source_experiment",
    ]
].copy()

vanilla_ref = vanilla_ref.rename(
    columns={
        "MAE_mean": "vanilla_MAE",
        "MAE_std": "vanilla_MAE_std",
        "n_params": "vanilla_params",
        "source_experiment": "vanilla_source",
    }
)

pr_ref = summary_13b[
    summary_13b["model"] == FROZEN_PR_MODEL
][
    [
        "domain",
        "fraction",
        "MAE_mean",
        "MAE_std",
        "n_params",
    ]
].copy()

pr_ref = pr_ref.rename(
    columns={
        "MAE_mean": "frozen_PR_MAE",
        "MAE_std": "frozen_PR_MAE_std",
        "n_params": "frozen_PR_params",
    }
)

k10 = k10_summary_all[
    k10_summary_all["model"] == "SC_RS_SFPEF"
][
    [
        "domain",
        "fraction",
        "MAE_mean",
    ]
].copy()

k10 = k10.rename(
    columns={
        "MAE_mean": "10K_MAE",
    }
)

comparison = (
    summary
    .merge(
        vanilla_ref,
        on=["domain", "fraction"],
        how="left",
        validate="many_to_one",
    )
    .merge(
        pr_ref,
        on=["domain", "fraction"],
        how="left",
        validate="many_to_one",
    )
    .merge(
        k10,
        on=["domain", "fraction"],
        how="left",
        validate="many_to_one",
    )
)

comparison["improvement_vs_vanilla_pct"] = (
    100.0
    * (
        comparison["vanilla_MAE"]
        - comparison["MAE_mean"]
    )
    / comparison["vanilla_MAE"]
)

comparison["improvement_vs_frozen_PR_pct"] = (
    100.0
    * (
        comparison["frozen_PR_MAE"]
        - comparison["MAE_mean"]
    )
    / comparison["frozen_PR_MAE"]
)

comparison["improvement_vs_10K_pct"] = (
    100.0
    * (
        comparison["10K_MAE"]
        - comparison["MAE_mean"]
    )
    / comparison["10K_MAE"]
)

comparison.to_csv(
    OUT / f"{EXPERIMENT}_comparison_vs_frozen_references.csv",
    index=False,
)

display(comparison)

# %% [markdown] 13. Primary factorization effect under exact information matching

# %% Code cell 14 (notebook cell 28)
effect_rows = []

for domain in DOMAINS:
    for fraction in LABEL_FRACTIONS:
        q = comparison[
            (comparison["domain"] == domain)
            & np.isclose(
                comparison["fraction"].astype(float),
                float(fraction),
            )
        ]

        d = q[q["model"] == DIRECT_MODEL]
        p = q[q["model"] == PR_MODEL]

        assert len(d) == 1
        assert len(p) == 1

        direct_prior = float(d.iloc[0]["MAE_mean"])
        pr_prior = float(p.iloc[0]["MAE_mean"])
        frozen_pr = float(p.iloc[0]["frozen_PR_MAE"])
        vanilla = float(p.iloc[0]["vanilla_MAE"])

        effect_rows.append({
            "domain": domain,
            "fraction": float(fraction),
            "vanilla_GRU_MAE": vanilla,
            "frozen_14B_PR_GRU_MAE": frozen_pr,
            "information_matched_direct_MAE": direct_prior,
            "information_matched_PR_MAE": pr_prior,

            "factorization_effect_pct": (
                100.0
                * (direct_prior - pr_prior)
                / direct_prior
            ),

            "explicit_prior_input_effect_within_PR_pct": (
                100.0
                * (frozen_pr - pr_prior)
                / frozen_pr
            ),

            "PR_better_than_information_matched_direct": bool(
                pr_prior < direct_prior
            ),

            "information_matched_PR_better_than_frozen_PR": bool(
                pr_prior < frozen_pr
            ),
        })

effects = pd.DataFrame(effect_rows)

effects.to_csv(
    OUT / f"{EXPERIMENT}_factorization_effects.csv",
    index=False,
)

display(effects)

# %% [markdown] 14. Full → 10% sparse degradation

# %% Code cell 15 (notebook cell 30)
degradation_rows = []

for model_name, source in [
    (PR_MODEL, "new"),
    (DIRECT_MODEL, "new"),
    (FROZEN_PR_MODEL, "frozen_pr"),
    (FROZEN_VANILLA_MODEL, "vanilla"),
    ("10K", "10k"),
]:
    for domain in DOMAINS:
        if source == "new":
            q = summary[
                (summary["domain"] == domain)
                & (summary["model"] == model_name)
            ]

            full = float(
                q[np.isclose(q["fraction"], 1.0)]
                .iloc[0]["MAE_mean"]
            )

            sparse = float(
                q[np.isclose(q["fraction"], 0.1)]
                .iloc[0]["MAE_mean"]
            )

        elif source == "frozen_pr":
            q = pr_ref[pr_ref["domain"] == domain]

            full = float(
                q[np.isclose(q["fraction"], 1.0)]
                .iloc[0]["frozen_PR_MAE"]
            )

            sparse = float(
                q[np.isclose(q["fraction"], 0.1)]
                .iloc[0]["frozen_PR_MAE"]
            )

        elif source == "vanilla":
            q = vanilla_ref[
                vanilla_ref["domain"] == domain
            ]

            full = float(
                q[np.isclose(q["fraction"], 1.0)]
                .iloc[0]["vanilla_MAE"]
            )

            sparse = float(
                q[np.isclose(q["fraction"], 0.1)]
                .iloc[0]["vanilla_MAE"]
            )

        else:
            q = k10[k10["domain"] == domain]

            full = float(
                q[np.isclose(q["fraction"], 1.0)]
                .iloc[0]["10K_MAE"]
            )

            sparse = float(
                q[np.isclose(q["fraction"], 0.1)]
                .iloc[0]["10K_MAE"]
            )

        degradation_rows.append({
            "domain": domain,
            "model": model_name,
            "full_MAE": full,
            "sparse_MAE": sparse,
            "full_to_sparse_pct": (
                100.0
                * (sparse - full)
                / full
            ),
        })

degradation = pd.DataFrame(degradation_rows)

degradation.to_csv(
    OUT / f"{EXPERIMENT}_full_to_sparse_degradation.csv",
    index=False,
)

display(
    degradation.sort_values(
        ["domain", "full_to_sparse_pct"]
    )
)

# %% [markdown] 15. Segment-balanced paired bootstrap for the information-matched PR model

# %% Code cell 16 (notebook cell 32)
baseline_pred = pd.read_parquet(
    FINAL_BASELINE_PRED
)

pr13_pred = pd.read_parquet(
    PRED_13B
)

k10_pred_all = pd.read_parquet(
    K10_PRED
)

k10_pred = k10_pred_all[
    k10_pred_all["model"] == "SC_RS_SFPEF"
].copy()


def paired_segment_table(a_rows, b_rows):
    keys = [
        "compound_segment",
        "time_value",
        "target",
    ]

    a = a_rows[
        keys + ["prediction"]
    ].rename(
        columns={"prediction": "pred_a"}
    )

    b = b_rows[
        keys + ["prediction"]
    ].rename(
        columns={"prediction": "pred_b"}
    )

    m = a.merge(
        b,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    assert len(m) == len(a) == len(b)

    m["ae_a"] = np.abs(
        m["pred_a"] - m["target"]
    )

    m["ae_b"] = np.abs(
        m["pred_b"] - m["target"]
    )

    return (
        m.groupby(
            "compound_segment",
            as_index=False,
        )
        .agg(
            A_segment_MAE=("ae_a", "mean"),
            B_segment_MAE=("ae_b", "mean"),
            n_rows=("ae_a", "size"),
        )
    )


def paired_segment_bootstrap(
    seg,
    comparator,
    draws=BOOTSTRAP_DRAWS,
    seed=1501,
):
    diff = (
        seg["A_segment_MAE"].to_numpy(float)
        - seg["B_segment_MAE"].to_numpy(float)
    )

    rng = np.random.default_rng(seed)

    n = len(diff)
    assert n >= 2

    samples = np.empty(
        draws,
        dtype=float,
    )

    for i in range(draws):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        samples[i] = float(
            np.mean(diff[idx])
        )

    p_left = (
        np.sum(samples <= 0) + 1.0
    ) / (draws + 1.0)

    p_right = (
        np.sum(samples >= 0) + 1.0
    ) / (draws + 1.0)

    return {
        "comparator": comparator,
        "n_segments": int(n),
        "Dense_PR_segment_balanced_MAE": float(
            seg["A_segment_MAE"].mean()
        ),
        "comparator_segment_balanced_MAE": float(
            seg["B_segment_MAE"].mean()
        ),
        "delta_Dense_PR_minus_comparator": float(
            np.mean(diff)
        ),
        "CI95_low": float(
            np.percentile(samples, 2.5)
        ),
        "CI95_high": float(
            np.percentile(samples, 97.5)
        ),
        "P_Dense_PR_lower_MAE": float(
            np.mean(samples < 0)
        ),
        "two_sided_bootstrap_p": float(
            min(
                1.0,
                2.0 * min(
                    p_left,
                    p_right,
                ),
            )
        ),
        "n_boot": int(draws),
    }


bootstrap_rows = []

for domain in DOMAINS:
    for fraction in LABEL_FRACTIONS:
        primary = seed_avg[
            (seed_avg["domain"] == domain)
            & np.isclose(seed_avg["fraction"], fraction)
            & (seed_avg["model"] == PR_MODEL)
        ].copy()

        dense_direct = seed_avg[
            (seed_avg["domain"] == domain)
            & np.isclose(seed_avg["fraction"], fraction)
            & (seed_avg["model"] == DIRECT_MODEL)
        ].copy()

        frozen_pr_rows = pr13_pred[
            (pr13_pred["domain"] == domain)
            & np.isclose(pr13_pred["fraction"], fraction)
            & (pr13_pred["model"] == FROZEN_PR_MODEL)
        ].copy()

        vanilla_rows = baseline_pred[
            (baseline_pred["domain"] == domain)
            & np.isclose(baseline_pred["fraction"], fraction)
            & (baseline_pred["model"] == FAMILY)
        ].copy()

        k10_rows = k10_pred[
            (k10_pred["domain"] == domain)
            & np.isclose(k10_pred["fraction"], fraction)
        ].copy()

        expected_n = int(
            (
                DOMAIN_DATA[domain]["split"]
                == "test"
            ).sum()
        )

        for x in [
            primary,
            dense_direct,
            frozen_pr_rows,
            vanilla_rows,
            k10_rows,
        ]:
            assert len(x) == expected_n, (
                domain,
                fraction,
                len(x),
                expected_n,
            )

        comparators = [
            (DIRECT_MODEL, dense_direct, 0),
            (FROZEN_PR_MODEL, frozen_pr_rows, 1000),
            (FAMILY, vanilla_rows, 2000),
            ("10K_SC_RS_SFPEF", k10_rows, 3000),
        ]

        for comp_name, comp_rows, offset in comparators:
            seg = paired_segment_table(
                primary,
                comp_rows,
            )

            result = paired_segment_bootstrap(
                seg,
                comp_name,
                seed=(
                    1501
                    + DOMAINS.index(domain) * 100
                    + int(fraction * 10)
                    + offset
                ),
            )

            bootstrap_rows.append({
                "domain": domain,
                "fraction": float(fraction),
                "primary_model": PR_MODEL,
                **result,
            })

bootstrap = pd.DataFrame(bootstrap_rows)

bootstrap.to_csv(
    OUT / f"{EXPERIMENT}_paired_segment_bootstrap.csv",
    index=False,
)

display(bootstrap)

# %% [markdown] 16. Compact decision table for the fairness question

# %% Code cell 17 (notebook cell 34)
decision_rows = []

for domain in DOMAINS:
    for fraction in LABEL_FRACTIONS:
        e = effects[
            (effects["domain"] == domain)
            & np.isclose(
                effects["fraction"],
                fraction,
            )
        ].iloc[0]

        b = bootstrap[
            (bootstrap["domain"] == domain)
            & np.isclose(
                bootstrap["fraction"],
                fraction,
            )
        ]

        def ci_for(comp):
            z = b[
                b["comparator"] == comp
            ].iloc[0]

            return (
                float(z["CI95_low"]),
                float(z["CI95_high"]),
                float(z["two_sided_bootstrap_p"]),
            )

        ci_direct = ci_for(DIRECT_MODEL)
        ci_frozen_pr = ci_for(FROZEN_PR_MODEL)

        decision_rows.append({
            "domain": domain,
            "fraction": float(fraction),
            "information_matched_PR_MAE": float(
                e["information_matched_PR_MAE"]
            ),
            "information_matched_direct_MAE": float(
                e["information_matched_direct_MAE"]
            ),
            "factorization_effect_pct": float(
                e["factorization_effect_pct"]
            ),
            "PR_better_than_information_matched_direct": bool(
                e["PR_better_than_information_matched_direct"]
            ),
            "bootstrap_CI_vs_information_matched_direct_excludes_zero_in_PR_favor": bool(
                ci_direct[1] < 0
            ),
            "bootstrap_p_vs_information_matched_direct": float(
                ci_direct[2]
            ),
            "information_matched_PR_better_than_frozen_14B_PR": bool(
                e["information_matched_PR_better_than_frozen_PR"]
            ),
            "bootstrap_CI_vs_frozen_14B_PR_excludes_zero_in_new_PR_favor": bool(
                ci_frozen_pr[1] < 0
            ),
        })

decision = pd.DataFrame(decision_rows)


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


decision["Holm_p_vs_information_matched_direct"] = holm_adjust(
    decision["bootstrap_p_vs_information_matched_direct"].to_numpy(float)
)

decision["factorization_supported_after_Holm_0p05"] = (
    decision[
        "bootstrap_CI_vs_information_matched_direct_excludes_zero_in_PR_favor"
    ]
    & (
        decision["Holm_p_vs_information_matched_direct"] < 0.05
    )
)

decision.to_csv(
    OUT / f"{EXPERIMENT}_decision_table.csv",
    index=False,
)

display(decision)

print(
    "Information-matched PR beats information-matched direct:",
    int(
        decision[
            "PR_better_than_information_matched_direct"
        ].sum()
    ),
    "/",
    len(decision),
)

print(
    "Paired bootstrap CI supports PR over direct:",
    int(
        decision[
            "bootstrap_CI_vs_information_matched_direct_excludes_zero_in_PR_favor"
        ].sum()
    ),
    "/",
    len(decision),
)

print(
    "Factorization advantage survives Holm correction:",
    int(
        decision[
            "factorization_supported_after_Holm_0p05"
        ].sum()
    ),
    "/",
    len(decision),
)

# %% [markdown] 17. Final integrity audit and R1A manifest

# %% Code cell 18 (notebook cell 36)
runs = pd.read_csv(RUNS_FINAL)

assert len(runs) == EXPECTED_RUNS

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
    pred = Path(str(r.prediction_file))

    assert ckpt.exists()
    assert pred.exists()

    assert sha256_file(ckpt) == str(r.checkpoint_sha256)
    assert sha256_file(pred) == str(r.prediction_sha256)

assert parameter_audit["within_5pct"].all()
assert selector_audit["count_match"].all()

assert len(bootstrap) == (
    len(DOMAINS)
    * len(LABEL_FRACTIONS)
    * 4
)


manifest = {
    "experiment": EXPERIMENT,
    "status": "POST_CONFIRMATORY_DEVELOPMENT_FAIRNESS_AUDIT_COMPLETE",
    "family": FAMILY,
    "primary_model": PR_MODEL,
    "ablation_model": DIRECT_MODEL,
    "frozen_reference_models": [
        FROZEN_VANILLA_MODEL,
        FROZEN_PR_MODEL,
        "SC_RS_SFPEF_10K",
    ],
    "domains": DOMAINS,
    "fractions": LABEL_FRACTIONS,
    "seeds": SEEDS,
    "n_runs": int(len(runs)),
    "expected_runs": int(EXPECTED_RUNS),
    "sequence_length": SEQ_LEN,
    "dense_layers": DENSE_LAYERS,
    "growth": GROWTH,
    "fusion_width": FUSION_WIDTH,
    "dense_connectivity": (
        "layer l receives raw X plus all earlier recurrent feature maps"
    ),
    "feature_level_residual_projection": True,
    "structured_prior_as_neural_input": True,
    "structured_prior_input_column": "soc_phy",
    "direct_target": "BMS-reported SoC",
    "residual_target": "BMS_SOC - structured_prior",
    "prediction_reconstruction_for_PR": (
        "structured_prior + predicted_residual"
    ),
    "exact_information_match_between_new_models": True,
    "exact_architecture_match_between_new_models": True,
    "exact_parameter_count_match_between_new_models": True,
    "only_intended_primary_difference": (
        "direct target vs prior-residual target/reconstruction"
    ),
    "dense_parameter_budget_within_5pct_of_frozen_baseline": True,
    "optimizer": "AdamW",
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "sparse_selector": (
        "exact 10K per-vehicle MD5 deterministic subset"
    ),
    "historical_test_exposure": True,
    "evaluation_role": (
        "post-confirmatory development fairness audit"
    ),
    "confirmatory": False,
    "EVE_12C_confirmatory_target_used": False,
    "no_posthoc_architecture_tuning": True,
    "no_rescue_after_outcome_inspection": True,
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_unit": "compound acquisition segment",
    "multiplicity_correction": "Holm across 8 primary information-matched factorization contrasts",
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

# %% [markdown] Interpretation rule
