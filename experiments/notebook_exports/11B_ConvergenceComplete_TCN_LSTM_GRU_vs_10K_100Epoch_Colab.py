# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/11B_ConvergenceComplete_TCN_LSTM_GRU_vs_10K_100Epoch_Colab.ipynb
# Notebook SHA256: 921961c84ac80429b264093409e1a981f4e83a42d13acf6f8b0dad0565705961
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 11B — Convergence-Complete Strict-Causal Deep Temporal Baselines vs 10K

# %% Code cell 1 (notebook cell 2)
# Optional: use a GPU runtime in Colab.
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow scikit-learn

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import os, gc, json, math, time, random, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
BASE = ROOT / "outputs" / "model_screening_10"
CANON = BASE / "canonical"

REF_10K = ROOT / "outputs" / "sc_rs_sfpef_10K_final"
SUMMARY_10K = REF_10K / "10K_summary.csv"
PRED_10K = REF_10K / "10K_seed_averaged_predictions.parquet"

OUT = ROOT / "outputs" / "deep_baselines_11B_100epoch_vs_10K"
CKPT = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

for p in [CANON, SUMMARY_10K, PRED_10K]:
    assert p.exists(), f"Missing required input: {p}"

DOMAINS = ["VED", "KU", "SCUT", "TUM"]
LABEL_FRACTIONS = [1.0, 0.10]
MODELS = ["TCN", "LSTM", "GRU"]
FINAL_SEEDS = [42, 43, 44, 45, 46]

# Fixed before looking at 11B test results.
SEQ_LEN = 16
HIDDEN = 64
TCN_CHANNELS = 64
TCN_DILATIONS = [1, 2, 4]
TCN_KERNEL = 3
DROPOUT = 0.10

MAX_EPOCHS = 100
PATIENCE = 10
ORIGINAL_11A_MAX_EPOCHS = 20
ORIGINAL_11A_PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE_GPU = 512
BATCH_SIZE_CPU = 128
GRAD_CLIP = 5.0

# Numerical optimization only; no architecture/feature/sequence-length tuning.
MIN_DELTA = 1e-5

BOOTSTRAP_DRAWS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = BATCH_SIZE_GPU if DEVICE.type == "cuda" else BATCH_SIZE_CPU
AMP_ENABLED = DEVICE.type == "cuda"

print("Device:", DEVICE)
print("Batch size:", BATCH_SIZE)
print("Output:", OUT)
print("Fixed sequence length:", SEQ_LEN)

# %% [markdown] 1. Reproduce 10K's exact sparse-label selector

# %% Code cell 2 (notebook cell 4)
def stable_hash_u01(s):
    return (
        int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:12], 16)
        / float(16 ** 12)
    )

def exact_sparse_subset(df, fraction, seed):
    if fraction >= 0.999999:
        return df.copy()

    pieces = []
    for veh, g in df.groupby("vehicle_id", sort=False):
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

# %% [markdown] 2. Load canonical benchmark and construct a conservative external-baseline feature set

# %% Code cell 3 (notebook cell 6)
EXCLUDE_TOKENS = [
    "soc_phy",
    "physics",
    "target",
    "residual",
    "ground_truth",
    "true_soc",
    "label",
    "future",
    "endpoint",
]

def external_baseline_features(meta_features):
    keep = []
    removed = []
    for c in meta_features:
        low = str(c).lower()
        if any(tok in low for tok in EXCLUDE_TOKENS):
            removed.append(c)
        else:
            keep.append(c)
    return keep, removed

DOMAIN_DATA = {}
DOMAIN_FEATURES = {}
FEATURE_AUDIT = []

for domain in DOMAINS:
    dp = CANON / f"{domain}_canonical.parquet"
    mp = CANON / f"{domain}_metadata.json"

    d = pd.read_parquet(dp)
    with open(mp) as f:
        meta = json.load(f)

    d = d[d["split"].isin(["train", "val", "test"])].copy()
    d["compound_segment"] = (
        d["vehicle_id"].astype(str)
        + "||"
        + d["segment_id"].astype(str)
    )
    d["row_uid"] = (
        d["vehicle_id"].astype(str)
        + "||"
        + d["segment_id"].astype(str)
        + "||"
        + d["time_value"].astype(str)
    )

    features, removed = external_baseline_features(
        list(meta["feature_columns"])
    )
    assert len(features) > 0, domain

    DOMAIN_DATA[domain] = d
    DOMAIN_FEATURES[domain] = features

    for c in features:
        FEATURE_AUDIT.append({
            "domain": domain,
            "feature": c,
            "included": True,
            "reason": "causal predictor feature",
        })
    for c in removed:
        FEATURE_AUDIT.append({
            "domain": domain,
            "feature": c,
            "included": False,
            "reason": "excluded structured-prior/target/future-like field",
        })

    print(
        domain,
        "rows=", len(d),
        "features=", len(features),
        "removed=", len(removed),
        "segments=", d["compound_segment"].nunique(),
    )

feature_audit = pd.DataFrame(FEATURE_AUDIT)
feature_audit.to_csv(OUT / "11B_feature_audit.csv", index=False)
display(feature_audit)

# %% [markdown] 3. Train-only preprocessing

# %% Code cell 4 (notebook cell 8)
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
        X = (X - self.mean_[None, :]) / self.std_[None, :]
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
        return ((np.asarray(y, float) - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, y):
        return np.asarray(y, float) * self.std_ + self.mean_

# %% [markdown] 4. Strict-causal sequence builder

# %% Code cell 5 (notebook cell 10)
def prepare_split_frame(df, split):
    z = df[df["split"] == split].copy()
    z = z.sort_values(
        ["compound_segment", "time_value"],
        kind="stable",
    ).reset_index(drop=True)

    z["seq_pos"] = np.arange(len(z), dtype=int)

    # start position of each segment inside this split frame
    seg_start = {}
    for seg, g in z.groupby("compound_segment", sort=False):
        seg_start[seg] = int(g.index.min())

    return z, seg_start


class CausalSequenceDataset(Dataset):
    def __init__(
        self,
        frame,
        X_scaled,
        endpoint_row_uids,
        target_scaler,
        seq_len,
    ):
        self.frame = frame
        self.X = X_scaled
        self.seq_len = int(seq_len)
        self.target_scaler = target_scaler

        self.uid_to_pos = pd.Series(
            frame["seq_pos"].to_numpy(),
            index=frame["row_uid"].astype(str),
        ).to_dict()

        self.seg_start = (
            frame.groupby("compound_segment", sort=False)["seq_pos"]
            .min()
            .astype(int)
            .to_dict()
        )

        positions = []
        for uid in endpoint_row_uids:
            uid = str(uid)
            if uid in self.uid_to_pos:
                positions.append(int(self.uid_to_pos[uid]))

        self.positions = np.asarray(positions, dtype=int)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        pos = int(self.positions[i])
        row = self.frame.iloc[pos]
        seg = row["compound_segment"]
        seg0 = int(self.seg_start[seg])

        start = max(seg0, pos - self.seq_len + 1)
        x = self.X[start : pos + 1]

        n = len(x)
        pad = self.seq_len - n

        # Standardized zero = train mean. Mask distinguishes padding.
        out = np.zeros((self.seq_len, self.X.shape[1] + 1), dtype=np.float32)
        if n:
            out[pad:, :-1] = x
            out[pad:, -1] = 1.0

        y = float(row["target"])
        y_scaled = float(self.target_scaler.transform([y])[0])

        return (
            torch.from_numpy(out),
            torch.tensor(y_scaled, dtype=torch.float32),
            pos,
        )


def make_all_endpoint_uids(frame):
    return frame["row_uid"].astype(str).tolist()

# %% [markdown] 5. Fixed architectures

# %% Code cell 6 (notebook cell 12)
class RNNRegressor(nn.Module):
    def __init__(self, input_dim, kind="LSTM", hidden=64):
        super().__init__()
        if kind == "LSTM":
            self.rnn = nn.LSTM(
                input_dim,
                hidden,
                num_layers=1,
                batch_first=True,
            )
        elif kind == "GRU":
            self.rnn = nn.GRU(
                input_dim,
                hidden,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError(kind)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        o, _ = self.rnn(x)
        h = o[:, -1, :]
        return self.head(h).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.c2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        residual = self.skip(x)
        y = self.drop(self.act(self.c1(x)))
        y = self.drop(self.act(self.c2(y)))
        return self.act(y + residual)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        blocks = []
        in_ch = input_dim
        for dilation in TCN_DILATIONS:
            blocks.append(
                TCNBlock(
                    in_ch,
                    TCN_CHANNELS,
                    TCN_KERNEL,
                    dilation,
                    DROPOUT,
                )
            )
            in_ch = TCN_CHANNELS

        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(TCN_CHANNELS),
            nn.Linear(TCN_CHANNELS, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # [B, T, F] -> [B, F, T]
        y = self.tcn(x.transpose(1, 2))
        h = y[:, :, -1]
        return self.head(h).squeeze(-1)


def build_model(model_name, input_dim):
    if model_name == "TCN":
        return TCNRegressor(input_dim)
    if model_name in {"LSTM", "GRU"}:
        return RNNRegressor(input_dim, kind=model_name, hidden=HIDDEN)
    raise ValueError(model_name)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% [markdown] 6. Training, early stopping, prediction, metrics

# %% Code cell 7 (notebook cell 14)
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


@torch.no_grad()
def predict_loader(model, loader, target_scaler):
    model.eval()
    pred_scaled = []
    positions = []

    for xb, _, pos in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        p = model(xb).detach().cpu().numpy()
        pred_scaled.append(p)
        positions.append(np.asarray(pos))

    ps = np.concatenate(pred_scaled)
    positions = np.concatenate(positions).astype(int)
    pred = target_scaler.inverse(ps)
    return positions, pred


def evaluate_loader(model, loader, frame, target_scaler):
    pos, pred = predict_loader(model, loader, target_scaler)
    y = frame.iloc[pos]["target"].to_numpy(float)
    return regression_metrics(y, pred), pos, pred


def train_one(
    model_name,
    domain,
    fraction,
    seed,
    train_frame,
    val_frame,
    test_frame,
    X_train,
    X_val,
    X_test,
    train_endpoint_uids,
    target_scaler,
):
    set_all_seeds(seed)

    train_ds = CausalSequenceDataset(
        train_frame, X_train, train_endpoint_uids,
        target_scaler, SEQ_LEN,
    )
    val_ds = CausalSequenceDataset(
        val_frame, X_val, make_all_endpoint_uids(val_frame),
        target_scaler, SEQ_LEN,
    )
    test_ds = CausalSequenceDataset(
        test_frame, X_test, make_all_endpoint_uids(test_frame),
        target_scaler, SEQ_LEN,
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

    input_dim = X_train.shape[1] + 1
    model = build_model(model_name, input_dim).to(DEVICE)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED,
    )

    best_val = np.inf
    best_epoch = 0
    stale = 0

    ckpt_path = CKPT / (
        f"{domain}_{fraction}_{model_name}_seed{seed}.pt"
    )

    history = []
    t0 = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=AMP_ENABLED,
            ):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

        val_metrics, _, _ = evaluate_loader(
            model, val_loader, val_frame, target_scaler
        )
        val_mae = val_metrics["MAE"]

        history.append({
            "epoch": epoch,
            "train_mse_scaled": float(np.mean(train_losses)),
            "val_MAE": val_mae,
        })

        if val_mae < best_val - MIN_DELTA:
            best_val = val_mae
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_mae": best_val,
                    "best_epoch": best_epoch,
                    "n_params": n_params,
                },
                ckpt_path,
            )
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    final_epoch = int(history[-1]["epoch"]) if history else 0
    stopped_by_patience = bool(final_epoch < MAX_EPOCHS)
    hit_epoch_cap = bool(final_epoch >= MAX_EPOCHS)

    train_seconds = time.perf_counter() - t0

    state = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(state["model_state"])

    val_metrics, _, _ = evaluate_loader(
        model, val_loader, val_frame, target_scaler
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    test_metrics, test_pos, test_pred = evaluate_loader(
        model, test_loader, test_frame, target_scaler
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t1

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(
        OUT / (
            f"history_{domain}_{fraction}_{model_name}_seed{seed}.csv"
        ),
        index=False,
    )

    return {
        "model": model,
        "best_epoch": int(state["best_epoch"]),
        "final_epoch": int(final_epoch),
        "stopped_by_patience": bool(stopped_by_patience),
        "hit_epoch_cap": bool(hit_epoch_cap),
        "best_val_MAE": float(state["best_val_mae"]),
        "n_params": int(state["n_params"]),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_pos": test_pos,
        "test_pred": test_pred,
        "n_train_endpoints": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

# %% [markdown] 7. Sequence coverage audit

# %% Code cell 8 (notebook cell 16)
coverage_rows = []

for domain in DOMAINS:
    d = DOMAIN_DATA[domain]
    for split in ["train", "val", "test"]:
        z = d[d["split"] == split].copy()
        sizes = z.groupby("compound_segment").size().to_numpy(int)

        coverage_rows.append({
            "domain": domain,
            "split": split,
            "n_rows": int(len(z)),
            "n_segments": int(len(sizes)),
            "median_rows_per_segment": float(np.median(sizes)),
            "mean_rows_per_segment": float(np.mean(sizes)),
            "fraction_segments_ge_SEQ_LEN": float(np.mean(sizes >= SEQ_LEN)),
            "SEQ_LEN": SEQ_LEN,
        })

sequence_coverage = pd.DataFrame(coverage_rows)
sequence_coverage.to_csv(
    OUT / "11B_sequence_coverage.csv",
    index=False,
)
display(sequence_coverage)

# %% [markdown] 8. Run all 3 baselines × 4 domains × 2 label fractions × 5 seeds

# %% Code cell 9 (notebook cell 18)
RUNS_PATH = OUT / "11B_runs_incremental.csv"
PREDS_DIR = OUT / "seed_predictions"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

if RUNS_PATH.exists():
    runs_done = pd.read_csv(RUNS_PATH)
else:
    runs_done = pd.DataFrame()

def run_is_done(domain, fraction, model, seed):
    if not len(runs_done):
        return False
    q = runs_done[
        (runs_done["domain"] == domain)
        & np.isclose(runs_done["fraction"].astype(float), float(fraction))
        & (runs_done["model"] == model)
        & (runs_done["seed"].astype(int) == int(seed))
    ]
    pred_path = PREDS_DIR / f"{domain}_{fraction}_{model}_seed{seed}.parquet"
    return len(q) == 1 and pred_path.exists()


for domain in DOMAINS:
    print("\n" + "=" * 80)
    print("DOMAIN:", domain)

    d = DOMAIN_DATA[domain].copy()
    features = DOMAIN_FEATURES[domain]

    train_frame, _ = prepare_split_frame(d, "train")
    val_frame, _ = prepare_split_frame(d, "val")
    test_frame, _ = prepare_split_frame(d, "test")

    # Predictor preprocessing is identical across fractions/seeds.
    prep = TrainOnlyPreprocessor().fit(
        train_frame[features].to_numpy(float)
    )
    X_train = prep.transform(train_frame[features].to_numpy(float))
    X_val = prep.transform(val_frame[features].to_numpy(float))
    X_test = prep.transform(test_frame[features].to_numpy(float))

    for fraction in LABEL_FRACTIONS:
        for seed in FINAL_SEEDS:
            # Exact 10K sparse-label subset.
            labeled_train = exact_sparse_subset(
                train_frame,
                fraction,
                seed,
            )

            target_scaler = TargetScaler().fit(
                labeled_train["target"].to_numpy(float)
            )

            train_endpoint_uids = (
                labeled_train["row_uid"].astype(str).tolist()
            )

            for model_name in MODELS:
                if run_is_done(domain, fraction, model_name, seed):
                    print("SKIP completed:", domain, fraction, model_name, seed)
                    continue

                print(
                    "RUN:",
                    domain,
                    fraction,
                    model_name,
                    "seed", seed,
                    "labels", len(labeled_train),
                )

                result = train_one(
                    model_name=model_name,
                    domain=domain,
                    fraction=fraction,
                    seed=seed,
                    train_frame=train_frame,
                    val_frame=val_frame,
                    test_frame=test_frame,
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                    train_endpoint_uids=train_endpoint_uids,
                    target_scaler=target_scaler,
                )

                row = {
                    "domain": domain,
                    "fraction": fraction,
                    "model": model_name,
                    "seed": seed,
                    "SEQ_LEN": SEQ_LEN,
                    "n_features": len(features),
                    "n_train_labels": result["n_train_endpoints"],
                    "n_val": result["n_val"],
                    "n_test": result["n_test"],
                    "best_epoch": result["best_epoch"],
                    "final_epoch": result["final_epoch"],
                    "stopped_by_patience": result["stopped_by_patience"],
                    "hit_epoch_cap": result["hit_epoch_cap"],
                    "best_val_MAE": result["best_val_MAE"],
                    "n_params": result["n_params"],
                    "train_seconds": result["train_seconds"],
                    "inference_seconds": result["inference_seconds"],
                    "evaluation_role": "development_test_not_confirmatory",
                    **result["test_metrics"],
                }

                if RUNS_PATH.exists():
                    current = pd.read_csv(RUNS_PATH)
                else:
                    current = pd.DataFrame()

                current = pd.concat(
                    [current, pd.DataFrame([row])],
                    ignore_index=True,
                )
                current.to_csv(RUNS_PATH, index=False)

                pos = result["test_pos"]
                pp = test_frame.iloc[pos][[
                    "vehicle_id",
                    "segment_id",
                    "compound_segment",
                    "time_value",
                    "elapsed_s",
                    "target",
                    "row_uid",
                ]].copy()
                pp["domain"] = domain
                pp["fraction"] = fraction
                pp["model"] = model_name
                pp["seed"] = seed
                pp["prediction"] = result["test_pred"]

                pp.to_parquet(
                    PREDS_DIR / (
                        f"{domain}_{fraction}_{model_name}_seed{seed}.parquet"
                    ),
                    index=False,
                )

                del result
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

# Reload final incremental table.
runs = pd.read_csv(RUNS_PATH)

expected = len(DOMAINS) * len(LABEL_FRACTIONS) * len(MODELS) * len(FINAL_SEEDS)
assert len(runs.drop_duplicates(
    ["domain", "fraction", "model", "seed"]
)) == expected, (
    "Not all runs completed.",
    len(runs),
    expected,
)

runs = runs.drop_duplicates(
    ["domain", "fraction", "model", "seed"],
    keep="last",
).sort_values(
    ["domain", "fraction", "model", "seed"]
)
runs.to_csv(OUT / "11B_runs.csv", index=False)

display(runs)

# %% [markdown] 9. Five-seed summaries

# %% Code cell 10 (notebook cell 20)
summary = (
    runs.groupby(
        ["domain", "fraction", "model"],
        as_index=False,
    )
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        P95_mean=("P95", "mean"),
        Bias_mean=("Bias", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        final_epoch_mean=("final_epoch", "mean"),
        seeds_hitting_epoch_cap=("hit_epoch_cap", "sum"),
        seeds_stopped_by_patience=("stopped_by_patience", "sum"),
        n_params=("n_params", "first"),
        train_seconds_mean=("train_seconds", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
        n_train_labels=("n_train_labels", "first"),
        n_test=("n_test", "first"),
    )
)

summary["evaluation_role"] = "development_test_not_confirmatory"
summary.to_csv(OUT / "11B_deep_baseline_summary.csv", index=False)
display(summary)

# %% [markdown] Convergence audit

# %% Code cell 11 (notebook cell 22)
convergence_audit = (
    runs.groupby(
        ["domain", "fraction", "model"],
        as_index=False,
    )
    .agg(
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_max=("best_epoch", "max"),
        final_epoch_mean=("final_epoch", "mean"),
        final_epoch_max=("final_epoch", "max"),
        seeds_hitting_epoch_100=("hit_epoch_cap", "sum"),
        seeds_stopped_by_patience=("stopped_by_patience", "sum"),
        best_val_MAE_mean=("best_val_MAE", "mean"),
        best_val_MAE_std=("best_val_MAE", "std"),
    )
)

convergence_audit["all_five_seeds_avoided_epoch_cap"] = (
    convergence_audit["seeds_hitting_epoch_100"] == 0
)

convergence_audit["optimization_budget_status"] = np.where(
    convergence_audit["seeds_hitting_epoch_100"] == 0,
    "CONVERGENCE_BUDGET_ADEQUATE",
    "REVIEW_RUNS_HIT_100_EPOCH_CAP",
)

convergence_audit.to_csv(
    OUT / "11B_convergence_audit.csv",
    index=False,
)

display(convergence_audit)

# %% [markdown] 10. Average predictions across seeds

# %% Code cell 12 (notebook cell 24)
pred_parts = []

for domain in DOMAINS:
    for fraction in LABEL_FRACTIONS:
        for model_name in MODELS:
            q = []
            for seed in FINAL_SEEDS:
                p = PREDS_DIR / f"{domain}_{fraction}_{model_name}_seed{seed}.parquet"
                assert p.exists(), p
                q.append(pd.read_parquet(p))

            x = pd.concat(q, ignore_index=True)

            avg = (
                x.groupby(
                    [
                        "domain", "fraction", "model",
                        "vehicle_id", "segment_id",
                        "compound_segment", "time_value",
                        "elapsed_s", "target", "row_uid",
                    ],
                    as_index=False,
                )["prediction"]
                .mean()
            )
            pred_parts.append(avg)

deep_predictions = pd.concat(pred_parts, ignore_index=True)
deep_predictions.to_parquet(
    OUT / "11B_seed_averaged_predictions.parquet",
    index=False,
)

print("Seed-averaged deep predictions:", len(deep_predictions))

# %% [markdown] 11. Direct comparison to locked 10K

# %% Code cell 13 (notebook cell 26)
s10k = pd.read_csv(SUMMARY_10K)
s10k = s10k[s10k["model"] == "SC_RS_SFPEF"].copy()

cmp = summary.merge(
    s10k[[
        "domain", "fraction", "MAE_mean",
        "RMSE_mean", "P95_mean", "Bias_mean",
    ]].rename(columns={
        "MAE_mean": "10K_MAE",
        "RMSE_mean": "10K_RMSE",
        "P95_mean": "10K_P95",
        "Bias_mean": "10K_Bias",
    }),
    on=["domain", "fraction"],
    how="left",
    validate="many_to_one",
)

cmp["delta_baseline_minus_10K_MAE"] = (
    cmp["MAE_mean"] - cmp["10K_MAE"]
)
cmp["10K_improvement_over_baseline_pct"] = (
    100.0
    * (cmp["MAE_mean"] - cmp["10K_MAE"])
    / cmp["MAE_mean"]
)

cmp["lower_MAE"] = np.where(
    cmp["10K_MAE"] < cmp["MAE_mean"],
    "SC_RS_SFPEF",
    np.where(
        cmp["10K_MAE"] > cmp["MAE_mean"],
        cmp["model"],
        "Tie",
    ),
)

cmp.to_csv(OUT / "11B_vs_10K.csv", index=False)
display(cmp[[
    "domain", "fraction", "model",
    "MAE_mean", "10K_MAE",
    "delta_baseline_minus_10K_MAE",
    "10K_improvement_over_baseline_pct",
    "lower_MAE",
]])

# %% [markdown] 12. Segment-paired bootstrap: deep baseline minus 10K

# %% Code cell 14 (notebook cell 28)
pred10k = pd.read_parquet(PRED_10K)
pred10k = pred10k[pred10k["model"] == "SC_RS_SFPEF"].copy()

# Stable join keys.
join_keys = [
    "domain", "fraction", "vehicle_id", "segment_id",
    "compound_segment", "time_value", "elapsed_s", "target",
]

def paired_segment_bootstrap_delta(
    merged,
    draws=1000,
    seed=202611,
):
    # Per-row absolute errors.
    merged = merged.copy()
    merged["ae_deep"] = np.abs(
        merged["pred_deep"].to_numpy(float)
        - merged["target"].to_numpy(float)
    )
    merged["ae_10k"] = np.abs(
        merged["pred_10k"].to_numpy(float)
        - merged["target"].to_numpy(float)
    )

    seg = (
        merged.groupby("compound_segment", as_index=False)
        .agg(
            ae_deep=("ae_deep", "mean"),
            ae_10k=("ae_10k", "mean"),
        )
    )

    delta = seg["ae_deep"].to_numpy(float) - seg["ae_10k"].to_numpy(float)

    rng = np.random.default_rng(seed)
    n = len(delta)
    boot = np.empty(draws, dtype=float)

    for b in range(draws):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(delta[idx].mean())

    return {
        "delta_MAE_deep_minus_10K": float(delta.mean()),
        "CI2.5": float(np.percentile(boot, 2.5)),
        "CI97.5": float(np.percentile(boot, 97.5)),
        "P_10K_better": float(np.mean(boot > 0)),
        "n_segments": int(n),
    }


boot_rows = []

for domain in DOMAINS:
    for fraction in LABEL_FRACTIONS:
        for model_name in MODELS:
            a = deep_predictions[
                (deep_predictions["domain"] == domain)
                & np.isclose(
                    deep_predictions["fraction"].astype(float),
                    float(fraction),
                )
                & (deep_predictions["model"] == model_name)
            ].copy()

            b = pred10k[
                (pred10k["domain"] == domain)
                & np.isclose(
                    pred10k["fraction"].astype(float),
                    float(fraction),
                )
            ].copy()

            m = a.merge(
                b[join_keys + ["prediction"]].rename(
                    columns={"prediction": "pred_10k"}
                ),
                on=join_keys,
                how="inner",
                validate="one_to_one",
            ).rename(columns={"prediction": "pred_deep"})

            assert len(m) == len(a), (
                domain, fraction, model_name,
                len(m), len(a),
            )

            sc = paired_segment_bootstrap_delta(
                m,
                draws=BOOTSTRAP_DRAWS,
                seed=202611,
            )

            boot_rows.append({
                "domain": domain,
                "fraction": fraction,
                "model": model_name,
                **sc,
            })

bootstrap = pd.DataFrame(boot_rows)
bootstrap.to_csv(
    OUT / "11B_paired_segment_bootstrap_vs_10K.csv",
    index=False,
)
display(bootstrap)

# %% [markdown] 13. Compact decision table

# %% Code cell 15 (notebook cell 30)
decision_rows = []

for fraction in LABEL_FRACTIONS:
    for model_name in MODELS:
        q = cmp[
            np.isclose(cmp["fraction"].astype(float), float(fraction))
            & (cmp["model"] == model_name)
        ]

        decision_rows.append({
            "fraction": fraction,
            "baseline": model_name,
            "mean_baseline_MAE_across_domains": float(q["MAE_mean"].mean()),
            "mean_10K_MAE_across_domains": float(q["10K_MAE"].mean()),
            "mean_10K_improvement_over_baseline_pct": float(
                q["10K_improvement_over_baseline_pct"].mean()
            ),
            "domains_10K_lower_MAE": int(
                (q["10K_MAE"] < q["MAE_mean"]).sum()
            ),
            "domains_baseline_lower_MAE": int(
                (q["10K_MAE"] > q["MAE_mean"]).sum()
            ),
            "n_domains": int(len(q)),
            "evaluation_role": "development_test_not_confirmatory",
        })

decision_table = pd.DataFrame(decision_rows)
decision_table.to_csv(
    OUT / "11B_decision_table.csv",
    index=False,
)
display(decision_table)

# %% [markdown] 11A → 11B convergence diagnostic

# %% Code cell 16 (notebook cell 32)
OLD_11A = ROOT / "outputs" / "deep_baselines_11A_vs_10K" / "11A_deep_baseline_summary.csv"

if OLD_11A.exists():
    old11a = pd.read_csv(OLD_11A)

    budget_cmp = summary.merge(
        old11a[[
            "domain", "fraction", "model",
            "MAE_mean", "best_epoch_mean",
        ]].rename(columns={
            "MAE_mean": "11A_MAE_20epoch",
            "best_epoch_mean": "11A_best_epoch_mean",
        }),
        on=["domain", "fraction", "model"],
        how="left",
        validate="one_to_one",
    )

    budget_cmp["11B_MAE_100epoch"] = budget_cmp["MAE_mean"]
    budget_cmp["MAE_change_11B_minus_11A"] = (
        budget_cmp["11B_MAE_100epoch"]
        - budget_cmp["11A_MAE_20epoch"]
    )
    budget_cmp["relative_MAE_change_pct"] = (
        100.0
        * budget_cmp["MAE_change_11B_minus_11A"]
        / budget_cmp["11A_MAE_20epoch"]
    )

    budget_cmp.to_csv(
        OUT / "11B_vs_11A_training_budget_diagnostic.csv",
        index=False,
    )

    display(budget_cmp[[
        "domain", "fraction", "model",
        "11A_MAE_20epoch",
        "11B_MAE_100epoch",
        "MAE_change_11B_minus_11A",
        "relative_MAE_change_pct",
        "11A_best_epoch_mean",
        "best_epoch_mean",
    ]])
else:
    print("11A summary not found; skipping 20-vs-100 epoch diagnostic.")

# %% [markdown] 14. Figures

# %% Code cell 17 (notebook cell 34)
for fraction in LABEL_FRACTIONS:
    q = cmp[np.isclose(cmp["fraction"].astype(float), fraction)].copy()

    # MAE comparison
    pivot = q.pivot(
        index="domain",
        columns="model",
        values="MAE_mean",
    )
    k = (
        q[["domain", "10K_MAE"]]
        .drop_duplicates()
        .set_index("domain")["10K_MAE"]
    )
    pivot["SC_RS_SFPEF"] = k

    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("MAE")
    ax.set_title(f"Deep temporal baselines vs 10K — fraction={fraction}")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUT / f"11B_MAE_fraction_{fraction}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.show()

    # 10K improvement relative to each baseline
    fig, ax = plt.subplots(figsize=(11, 5))
    for model_name in MODELS:
        z = q[q["model"] == model_name].set_index("domain")
        ax.plot(
            z.index,
            z["10K_improvement_over_baseline_pct"],
            marker="o",
            label=model_name,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("10K improvement over baseline (%)")
    ax.set_title(f"10K relative to temporal baselines — fraction={fraction}")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUT / f"11B_10K_relative_improvement_fraction_{fraction}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.show()

# %% [markdown] 15. Manifest

# %% Code cell 18 (notebook cell 36)
manifest = {
    "experiment": "11B_CAUSAL_DEEP_BASELINES_VS_10K",
    "baselines": ["TCN", "LSTM", "GRU"],
    "proposed_model": "SC-RS-SFPEF",
    "proposed_model_source": str(REF_10K),
    "target": "production BMS-reported SoC",
    "ground_truth_boundary": "not independently measured electrochemical SoC",
    "domains": DOMAINS,
    "label_fractions": LABEL_FRACTIONS,
    "seeds": FINAL_SEEDS,
    "sparse_selector": "exact 10K per-vehicle MD5 deterministic subset",
    "strict_causality": True,
    "sequence_definition": (
        "last up to 16 predictor rows from the same segment, "
        "ending at the current prediction row; no future rows"
    ),
    "sequence_length": SEQ_LEN,
    "feature_rule": {
        "source": "canonical metadata feature_columns",
        "excluded_tokens": EXCLUDE_TOKENS,
        "soc_phy_given_to_deep_baselines": False,
        "target_or_residual_given_to_deep_baselines": False,
    },
    "preprocessing": (
        "train-only median imputation and train-only mean/std normalization"
    ),
    "architecture_tuning": False,
    "fixed_architecture": {
        "LSTM": {"layers": 1, "hidden": HIDDEN},
        "GRU": {"layers": 1, "hidden": HIDDEN},
        "TCN": {
            "channels": TCN_CHANNELS,
            "kernel": TCN_KERNEL,
            "dilations": TCN_DILATIONS,
            "dropout": DROPOUT,
        },
    },
    "optimization": {
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "checkpoint_metric": "validation MAE",
    },
    "test_tuning": False,
    "historical_test_exposure": True,
    "current_test_role": "development benchmark only; not confirmatory",
    "confirmatory_evaluation_required": True,
}

with open(OUT / "11B_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))

# %% [markdown] Interpretation after execution
