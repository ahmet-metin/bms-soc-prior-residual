# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/primary/R2C_CorrectedSparsePrior_LabelFractionCurve_AULEC_Colab.ipynb
# Notebook SHA256: ab1b2313af5177e68a244df1a5cd24a688690d63c468a17c80d3cd4873c9386d
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R2C-C — Corrected Label-Fraction Curve with Supervision-Matched Prior Calibration

# %% [markdown] Correction governance

# %% Code cell 1 (notebook cell 3)

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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
OUTPUTS = ROOT / "outputs"
REV1_OUT = ROOT / "Revizyon1" / "outputs"
REV2_OUT = ROOT / "Revizyon2" / "outputs"

CANON = OUTPUTS / "model_screening_10" / "canonical"

OUT = REV2_OUT / "R2C_corrected_sparse_prior_curve"
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "seed_predictions"
FIG_DIR = OUT / "figures"

for p in [OUT, CKPT_DIR, PRED_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R2C_CORRECTED_SPARSE_PRIOR_CURVE"

DEV_DOMAINS = ["VED", "KU", "SCUT", "TUM"]
ALL_DOMAINS = DEV_DOMAINS + ["EVE"]

# Small domains first for partial-result visibility.
DOMAIN_RUN_ORDER = ["VED", "SCUT", "TUM", "KU", "EVE"]

FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
ENDPOINT_FRACTIONS = [0.10, 1.00]
FULL_PRIOR_REUSE_TOL_PCT = 1e-9
NEW_INTERMEDIATE_FRACTIONS = [0.01, 0.02, 0.05, 0.20, 0.50]
SEEDS = [42, 43, 44, 45, 46]

DIRECT_STD = "Direct+Prior"
PR_STD = "PriorResidual+Prior"
MODELS_STD = [DIRECT_STD, PR_STD]

DEV_SOURCE_MODEL = {
    DIRECT_STD: "Dense_GRU_PriorInput_Direct",
    PR_STD: "Dense_GRU_PriorInput_PR",
}
EVE_SOURCE_MODEL = {
    DIRECT_STD: "EVE_GRU_PriorInput_Direct",
    PR_STD: "EVE_GRU_PriorInput_PR",
}

SEQ_LEN = 16

# Development R1A architecture
DENSE_LAYERS = 3
GROWTH = 19
FUSION_WIDTH = 64

# EVE R2A architecture
EVE_HIDDEN = 64
EVE_HEAD_HIDDEN = 32

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

print("Experiment:", EXPERIMENT)
print("Device:", DEVICE)
print("Fractions:", FRACTIONS)
print("Output:", OUT)

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

# Prior endpoint runs and manifests.

R1A_RUNS = find_unique_file(
    REV1_OUT,
    "R1A_INFO_MATCHED_GRU_runs_incremental.csv",
)
R1A_MANIFEST = find_unique_file(
    REV1_OUT,
    "R1A_INFO_MATCHED_GRU_MANIFEST.json",
)

R2A_RUNS = find_unique_file(
    REV2_OUT,
    "R2A_EVE_INFO_MATCHED_GRU_runs.csv",
)
R2A_MANIFEST = find_unique_file(
    REV2_OUT,
    "R2A_EVE_INFO_MATCHED_GRU_MANIFEST.json",
)

R2B_MANIFEST = find_unique_file(
    REV2_OUT,
    "R2B_EFFECTIVE_SUPERVISION_BUDGET_MANIFEST.json",
)

MANIFEST_11B = find_unique_file(
    OUTPUTS,
    "11B_manifest.json",
)

r1a_runs = pd.read_csv(R1A_RUNS)
r2a_runs = pd.read_csv(R2A_RUNS)

with open(R1A_MANIFEST) as f:
    r1a_manifest = json.load(f)

with open(R2A_MANIFEST) as f:
    r2a_manifest = json.load(f)

with open(R2B_MANIFEST) as f:
    r2b_manifest = json.load(f)

with open(MANIFEST_11B) as f:
    man11b = json.load(f)

assert r1a_manifest["status"] == "POST_CONFIRMATORY_DEVELOPMENT_FAIRNESS_AUDIT_COMPLETE"
assert r2a_manifest["status"] == "POST_CONFIRMATORY_EVE_INFO_MATCHED_FAIRNESS_AUDIT_COMPLETE"
assert r2b_manifest["status"] == "R2B_EFFECTIVE_SUPERVISION_BUDGET_COMPLETE"

assert r1a_manifest["exact_information_match_between_new_models"] is True
assert r2a_manifest["exact_information_match_between_new_models"] is True

assert r2b_manifest["anchor_label_overlap_is_double_counted"] is False
assert r2b_manifest["test_targets_count_as_supervision"] is False

assert man11b["strict_causality"] is True
assert int(man11b["sequence_length"]) == SEQ_LEN

EXCLUDED_TOKENS = [
    str(x).lower()
    for x in man11b["feature_rule"]["excluded_tokens"]
]

print("Prior endpoint integrity checks: PASS")


P0_AUDIT_DIR = REV2_OUT / "P0_sparse_prior_calibration_audit_v2"
P0_FULL_GATE = P0_AUDIT_DIR / "P0_FULL_FRACTION_REPRODUCTION_GATE.csv"
P0_SELECTOR_AUDIT_FILE = P0_AUDIT_DIR / "P0_SELECTOR_MANIFEST_SHA256.csv"

assert P0_FULL_GATE.exists(), P0_FULL_GATE
assert P0_SELECTOR_AUDIT_FILE.exists(), P0_SELECTOR_AUDIT_FILE

p0_full_gate = pd.read_csv(P0_FULL_GATE)
p0_selector_audit = pd.read_csv(P0_SELECTOR_AUDIT_FILE)

assert set(p0_full_gate["domain"]) == set(ALL_DOMAINS)
assert len(p0_selector_audit) == len(ALL_DOMAINS) * len(FRACTIONS) * len(SEEDS)

print("P0 sparse-prior audit references: PASS")
display(p0_full_gate)

# %% Code cell 4 (notebook cell 6)

# Freeze one curve-training budget per domain from PRE-EXISTING endpoint budgets.
#
# For each development domain, use the larger max_epochs/patience seen at the
# frozen 10% / 100% R1A endpoints. EVE uses the R2A endpoint budget.

budget_rows = []

for domain in DEV_DOMAINS:
    q = r1a_runs[
        (r1a_runs["domain"] == domain)
        & r1a_runs["fraction"].astype(float).isin(ENDPOINT_FRACTIONS)
    ].copy()

    assert len(q) == 20, (domain, len(q))

    budget_rows.append({
        "domain": domain,
        "max_epochs": int(q["max_epochs"].max()),
        "patience": int(q["patience"].max()),
        "source": "max frozen R1A 10%/100% endpoint budget",
    })

q = r2a_runs[
    r2a_runs["fraction"].astype(float).isin(ENDPOINT_FRACTIONS)
].copy()

assert len(q) == 20, len(q)

budget_rows.append({
    "domain": "EVE",
    "max_epochs": int(q["max_epochs"].max()),
    "patience": int(q["patience"].max()),
    "source": "max frozen R2A 10%/100% endpoint budget",
})

DOMAIN_BUDGET = (
    pd.DataFrame(budget_rows)
    .set_index("domain")
    .to_dict(orient="index")
)

budget_table = pd.DataFrame(budget_rows)
budget_table.to_csv(
    OUT / f"{EXPERIMENT}_FROZEN_DOMAIN_TRAINING_BUDGET.csv",
    index=False,
)

display(budget_table)

# %% Code cell 5 (notebook cell 7)
# Determine which pre-existing endpoints are scientifically reusable after the P0 correction.
#
# Rules:
# - NO sparse endpoint is reused because its prior must be refit from sparse retained targets.
# - 100% may be reused only if:
#     (a) source endpoint is complete,
#     (b) training budget matches,
#     (c) prediction files are valid,
#     (d) P0 full-fraction prior reproduction is within numerical tolerance.
# - TUM fails criterion (d) on the current canonical training universe and is retrained at 100%.

endpoint_rows = []

full_gate_map = {
    str(r.domain): float(r.max_rel_diff_pct)
    for r in p0_full_gate.itertuples(index=False)
}

for domain in ALL_DOMAINS:
    source_runs = r1a_runs if domain in DEV_DOMAINS else r2a_runs
    model_map = DEV_SOURCE_MODEL if domain in DEV_DOMAINS else EVE_SOURCE_MODEL

    for fraction in ENDPOINT_FRACTIONS:
        q = source_runs[
            np.isclose(source_runs["fraction"].astype(float), fraction)
        ].copy()

        if domain in DEV_DOMAINS:
            q = q[q["domain"] == domain].copy()

        expected_model_names = set(model_map.values())
        q = q[q["model"].isin(expected_model_names)].copy()

        expected_budget = DOMAIN_BUDGET[domain]

        complete = (
            len(q) == 10
            and set(q["seed"].astype(int)) == set(SEEDS)
            and set(q["model"]) == expected_model_names
        )

        budget_match = bool(
            complete
            and (q["max_epochs"].astype(int) == int(expected_budget["max_epochs"])).all()
            and (q["patience"].astype(int) == int(expected_budget["patience"])).all()
        )

        files_valid = False
        if complete:
            files_valid = True
            for r in q.itertuples(index=False):
                p = Path(str(r.prediction_file))
                if not p.exists():
                    files_valid = False
                    break
                if "prediction_sha256" in q.columns:
                    if sha256_file(p) != str(r.prediction_sha256):
                        files_valid = False
                        break

        full_prior_reproduced = bool(
            np.isclose(float(fraction), 1.0)
            and full_gate_map[domain] <= FULL_PRIOR_REUSE_TOL_PCT
        )

        sparse_refit_required = not np.isclose(float(fraction), 1.0)

        reusable = bool(
            complete
            and budget_match
            and files_valid
            and full_prior_reproduced
            and not sparse_refit_required
        )

        if sparse_refit_required:
            reason = "sparse_prior_must_be_refit"
        elif not full_prior_reproduced:
            reason = "full_canonical_prior_not_reproduced_on_current_train_universe"
        elif not complete:
            reason = "source_endpoint_incomplete"
        elif not budget_match:
            reason = "source_budget_mismatch"
        elif not files_valid:
            reason = "source_prediction_file_invalid"
        else:
            reason = "verified_full_endpoint_reuse"

        endpoint_rows.append({
            "domain": domain,
            "fraction": float(fraction),
            "complete_10_runs": bool(complete),
            "budget_match": bool(budget_match),
            "prediction_files_valid": bool(files_valid),
            "P0_full_prior_max_rel_diff_pct": float(full_gate_map[domain]),
            "full_prior_reproduced": bool(full_prior_reproduced),
            "sparse_refit_required": bool(sparse_refit_required),
            "reusable": reusable,
            "reuse_reason": reason,
            "frozen_max_epochs": int(expected_budget["max_epochs"]),
            "frozen_patience": int(expected_budget["patience"]),
        })

endpoint_reuse = pd.DataFrame(endpoint_rows)

endpoint_reuse.to_csv(
    OUT / f"{EXPERIMENT}_ENDPOINT_REUSE_AUDIT.csv",
    index=False,
)

display(endpoint_reuse)

# %% Code cell 6 (notebook cell 8)
# Freeze the corrected R2C plan before corrected outcomes are inspected.

plan_rows = []

for domain in ALL_DOMAINS:
    for fraction in FRACTIONS:
        if fraction in ENDPOINT_FRACTIONS:
            r = endpoint_reuse[
                (endpoint_reuse["domain"] == domain)
                & np.isclose(endpoint_reuse["fraction"], fraction)
            ].iloc[0]
            action = "reuse_verified_full_endpoint" if bool(r["reusable"]) else "train_corrected_R2C"
        else:
            action = "train_corrected_R2C"

        for model in MODELS_STD:
            for seed in SEEDS:
                plan_rows.append({
                    "domain": domain,
                    "fraction": float(fraction),
                    "model": model,
                    "seed": int(seed),
                    "action": action,
                    "max_epochs": int(DOMAIN_BUDGET[domain]["max_epochs"]),
                    "patience": int(DOMAIN_BUDGET[domain]["patience"]),
                })

train_plan = pd.DataFrame(plan_rows)

n_total_curve_runs = len(train_plan)
n_reused = int((train_plan["action"] == "reuse_verified_full_endpoint").sum())
n_new = int((train_plan["action"] == "train_corrected_R2C").sum())

assert n_total_curve_runs == (
    len(ALL_DOMAINS) * len(FRACTIONS) * len(MODELS_STD) * len(SEEDS)
)

train_plan.to_csv(
    OUT / f"{EXPERIMENT}_FROZEN_TRAIN_PLAN.csv",
    index=False,
)

preflight = {
    "experiment": EXPERIMENT,
    "status": "CORRECTED_R2C_PLAN_FROZEN_BEFORE_CORRECTED_OUTCOME_INSPECTION",
    "target": "BMS-reported SoC",
    "fractions": FRACTIONS,
    "domains": ALL_DOMAINS,
    "models": MODELS_STD,
    "seeds": SEEDS,
    "total_curve_runs": int(n_total_curve_runs),
    "verified_full_endpoint_reuse_runs": int(n_reused),
    "new_corrected_training_runs": int(n_new),
    "prior_correction": (
        "every target-dependent prior slope is refit per domain/fraction/seed "
        "using only retained training target rows allowed in that condition"
    ),
    "mandatory_segment_start_BMS_anchors": (
        "permitted and counted in effective supervision; they define soc0 and "
        "normally contribute zero-coordinate equations to the slope fit"
    ),
    "validation_targets_used_for_prior_fit": False,
    "test_targets_used_for_prior_fit": False,
    "sparse_endpoint_reuse_allowed": False,
    "TUM_full_endpoint_reuse_allowed": False,
    "full_endpoint_reuse_requires_P0_reproduction_gate": True,
    "full_prior_reuse_tolerance_pct": FULL_PRIOR_REUSE_TOL_PCT,
    "domain_budget_rule": (
        "larger max_epochs/patience from pre-existing 10% and 100% information-matched endpoints"
    ),
    "primary_x_axis": "effective unique BMS fit-observation fraction",
    "secondary_x_axis": "nominal retained training-target fraction",
    "effective_budget_definition": (
        "set union of selected TRAIN target-label rows and one genuine segment-start BMS anchor per TRAIN compound segment"
    ),
    "anchor_label_overlap_double_counted": False,
    "primary_summary_metric": "normalized log-effective-observation AULEC; lower is better",
    "secondary_summary_metric": "normalized log-nominal-label AULEC; lower is better",
    "primary_inference": (
        "paired compound-segment bootstrap of effective-observation AULEC difference, "
        "PriorResidual+Prior minus Direct+Prior; Holm over five domains"
    ),
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "EVE_test_was_previously_opened": True,
    "confirmatory": False,
    "evaluation_role": "corrective post-confirmatory supervision-matched label-efficiency audit",
}

with open(OUT / f"{EXPERIMENT}_PREFLIGHT.json", "w") as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% Code cell 7 (notebook cell 9)
# Load canonical data and reconstruct the exact information-matched feature contracts.
# The frozen soc_phy is preserved only for audit / verified full-endpoint reuse.
# Corrected condition-specific soc_phy is created later from soc0 + beta*coord.

DOMAIN_DATA = {}
DOMAIN_FEATURES = {}
DOMAIN_META = {}
DOMAIN_ARCH = {}

feature_rows = []

for domain in DEV_DOMAINS:
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
        if not any(token in c.lower() for token in EXCLUDED_TOKENS)
    ]

    assert "soc_phy" not in base_features
    features = base_features + ["soc_phy"]

    required = {
        "vehicle_id",
        "segment_id",
        "time_value",
        "elapsed_s",
        "split",
        "target",
        "soc0",
        "coord",
        "soc_phy",
        *raw_features,
    }

    missing = sorted(required - set(d.columns))
    assert not missing, (domain, missing)

    d = d[d["split"].isin(["train", "val", "test"])].copy()
    d["vehicle_id"] = d["vehicle_id"].astype(str)
    d["segment_id"] = d["segment_id"].astype(str)
    d["compound_segment"] = d["vehicle_id"] + "||" + d["segment_id"]
    d["soc_phy_frozen"] = d["soc_phy"].to_numpy(float)

    assert np.isfinite(d["target"].to_numpy(float)).all()
    assert np.isfinite(d["soc0"].to_numpy(float)).all()
    assert np.isfinite(d["coord"].to_numpy(float)).all()
    assert np.isfinite(d["soc_phy_frozen"].to_numpy(float)).all()

    DOMAIN_DATA[domain] = d
    DOMAIN_FEATURES[domain] = features
    DOMAIN_META[domain] = meta
    DOMAIN_ARCH[domain] = "DenseRecurrentNet"

    for c in features:
        feature_rows.append({
            "domain": domain,
            "architecture": DOMAIN_ARCH[domain],
            "feature": c,
            "used": True,
        })

# EVE
EVE_TRAINVAL = (
    OUTPUTS
    / "eve_12B_v4_frozen_trainval"
    / "12B_v4_EVE_canonical_trainval.parquet"
)
EVE_META = (
    OUTPUTS
    / "eve_12B_v4_frozen_trainval"
    / "12B_v4_EVE_canonical_metadata.json"
)
EVE_TEST = (
    OUTPUTS
    / "eve_12C_one_shot_confirmatory"
    / "12C_EVE_confirmatory_canonical.parquet"
)

assert EVE_TRAINVAL.exists()
assert EVE_META.exists()
assert EVE_TEST.exists()

eve_tv = pd.read_parquet(EVE_TRAINVAL).copy()
eve_test = pd.read_parquet(EVE_TEST).copy()

with open(EVE_META) as f:
    eve_meta = json.load(f)

eve_features = list(eve_meta["feature_columns"]) + ["soc_phy"]
assert len(set(eve_features)) == len(eve_features)

for d in [eve_tv, eve_test]:
    required = {
        "vehicle_id", "segment_id", "time_value", "elapsed_s",
        "target", "soc0", "coord", "soc_phy",
    }
    missing = sorted(required - set(d.columns))
    assert not missing, ("EVE", missing)

    d["vehicle_id"] = d["vehicle_id"].astype(str)
    d["segment_id"] = d["segment_id"].astype(str)
    d["compound_segment"] = d["vehicle_id"] + "||" + d["segment_id"]
    d["soc_phy_frozen"] = d["soc_phy"].to_numpy(float)

eve_tv = eve_tv[eve_tv["split"].isin(["train", "val"])].copy()

if "split" in eve_test.columns and (eve_test["split"] == "test").any():
    eve_test = eve_test[eve_test["split"] == "test"].copy()

eve_test["split"] = "test"
eve_all = pd.concat([eve_tv, eve_test], ignore_index=True)

DOMAIN_DATA["EVE"] = eve_all
DOMAIN_FEATURES["EVE"] = eve_features
DOMAIN_META["EVE"] = eve_meta
DOMAIN_ARCH["EVE"] = "FrozenFamilyGRU"

for c in eve_features:
    feature_rows.append({
        "domain": "EVE",
        "architecture": "FrozenFamilyGRU",
        "feature": c,
        "used": True,
    })

feature_audit = pd.DataFrame(feature_rows)
feature_audit.to_csv(
    OUT / f"{EXPERIMENT}_FEATURE_AUDIT.csv",
    index=False,
)

for domain in ALL_DOMAINS:
    print(
        domain,
        DOMAIN_ARCH[domain],
        DOMAIN_DATA[domain]["split"].value_counts().to_dict(),
        "features=", len(DOMAIN_FEATURES[domain]),
    )

# %% Code cell 8 (notebook cell 10)

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

# %% Code cell 9 (notebook cell 11)
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

# Precompute only compact prior/selector contracts; no model outcome is inspected.
PRIOR_CONTRACT_LOOKUP = {}
prior_manifest_rows = []

for domain in ALL_DOMAINS:
    train = DOMAIN_DATA[domain][
        DOMAIN_DATA[domain]["split"] == "train"
    ].copy()

    for fraction in FRACTIONS:
        for seed in SEEDS:
            c = fit_condition_prior(
                domain,
                fraction,
                seed,
            )

            PRIOR_CONTRACT_LOOKUP[
                (domain, fraction_key(fraction), int(seed))
            ] = c

            for group, beta in c["beta_map"].items():
                frozen = float(c["frozen_map"].get(group, np.nan))
                prior_manifest_rows.append({
                    "domain": domain,
                    "fraction": float(fraction),
                    "seed": int(seed),
                    "group": group,
                    "selector_sha256": c["selector_sha256"],
                    "prior_contract_sha256": c["prior_contract_sha256"],
                    "n_informative_retained_targets": int(c["n_info_map"][group]),
                    "beta_corrected": float(beta),
                    "beta_frozen": frozen,
                    "beta_diff": float(beta - frozen) if np.isfinite(frozen) else np.nan,
                    "beta_rel_diff_pct": (
                        100.0 * (beta - frozen) / abs(frozen)
                        if np.isfinite(frozen) and abs(frozen) > EPS
                        else np.nan
                    ),
                })

corrected_prior_manifest = pd.DataFrame(prior_manifest_rows)
corrected_prior_manifest.to_csv(
    OUT / f"{EXPERIMENT}_PRIOR_CALIBRATION_BY_CONDITION.csv",
    index=False,
)

print(
    "Corrected prior contracts:",
    len(PRIOR_CONTRACT_LOOKUP),
    "conditions"
)
display(corrected_prior_manifest.head(30))

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

# %% Code cell 10 (notebook cell 12)

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

parameter_rows = []

for domain in ALL_DOMAINS:
    input_dim = len(DOMAIN_FEATURES[domain]) + 1
    a = build_model(domain, input_dim)
    b = build_model(domain, input_dim)

    n_a = count_params(a)
    n_b = count_params(b)

    assert n_a == n_b

    parameter_rows.append({
        "domain": domain,
        "architecture": DOMAIN_ARCH[domain],
        "input_dim_including_mask": input_dim,
        "direct_params": n_a,
        "PR_params": n_b,
        "exact_capacity_match": True,
    })

parameter_audit = pd.DataFrame(parameter_rows)
parameter_audit.to_csv(
    OUT / f"{EXPERIMENT}_PARAMETER_AUDIT.csv",
    index=False,
)

display(parameter_audit)

# %% Code cell 11 (notebook cell 13)
# Split-only audit.
# Feature preprocessing is now condition-specific because corrected soc_phy depends on fraction/seed.

split_audit_rows = []

for domain in ALL_DOMAINS:
    d = DOMAIN_DATA[domain]

    for split in ["train", "val", "test"]:
        z = prepare_split_frame(d, split)

        split_audit_rows.append({
            "domain": domain,
            "split": split,
            "rows": int(len(z)),
            "vehicles": int(z["vehicle_id"].nunique()),
            "compound_segments": int(z["compound_segment"].nunique()),
        })

split_audit = pd.DataFrame(split_audit_rows)
split_audit.to_csv(
    OUT / f"{EXPERIMENT}_SPLIT_AUDIT.csv",
    index=False,
)

display(split_audit)

# %% Code cell 12 (notebook cell 14)

# Effective budget for EVERY fraction/seed, using the exact R2B union rule.

budget_rows = []

for domain in ALL_DOMAINS:
    train = prepare_split_frame(DOMAIN_DATA[domain], "train")

    # seq_pos is the exact local canonical row identity in this sorted train frame.
    anchors = (
        train
        .sort_values(
            ["compound_segment", "elapsed_s", "time_value"],
            kind="mergesort",
        )
        .drop_duplicates("compound_segment", keep="first")
    )

    anchor_ids = set(anchors["seq_pos"].astype(int).tolist())
    n_train = int(len(train))

    assert len(anchor_ids) == train["compound_segment"].nunique()

    for fraction in FRACTIONS:
        for seed in SEEDS:
            selected = exact_sparse_subset(
                train,
                fraction,
                seed,
            )

            label_ids = set(
                selected["seq_pos"].astype(int).tolist()
            )

            overlap = label_ids.intersection(anchor_ids)
            union = label_ids.union(anchor_ids)

            budget_rows.append({
                "domain": domain,
                "fraction": float(fraction),
                "seed": int(seed),
                "n_train_rows": n_train,
                "n_selected_labels": int(len(label_ids)),
                "actual_nominal_fraction": float(len(label_ids) / n_train),
                "n_train_anchors": int(len(anchor_ids)),
                "n_anchor_label_overlap": int(len(overlap)),
                "n_unique_fit_observations": int(len(union)),
                "effective_fit_fraction": float(len(union) / n_train),
                "anchor_overhead_pp": float(
                    100.0 * (len(union) - len(label_ids)) / n_train
                ),
            })

budget_seed = pd.DataFrame(budget_rows)

budget_seed.to_csv(
    OUT / f"{EXPERIMENT}_EFFECTIVE_BUDGET_BY_SEED.csv",
    index=False,
)

budget_summary = (
    budget_seed
    .groupby(["domain", "fraction"], as_index=False)
    .agg(
        actual_nominal_fraction_mean=("actual_nominal_fraction", "mean"),
        actual_nominal_fraction_min=("actual_nominal_fraction", "min"),
        actual_nominal_fraction_max=("actual_nominal_fraction", "max"),
        effective_fit_fraction_mean=("effective_fit_fraction", "mean"),
        effective_fit_fraction_min=("effective_fit_fraction", "min"),
        effective_fit_fraction_max=("effective_fit_fraction", "max"),
        anchor_overhead_pp_mean=("anchor_overhead_pp", "mean"),
        n_selected_labels_mean=("n_selected_labels", "mean"),
        n_unique_fit_observations_mean=("n_unique_fit_observations", "mean"),
    )
)

budget_summary.to_csv(
    OUT / f"{EXPERIMENT}_EFFECTIVE_BUDGET_SUMMARY.csv",
    index=False,
)

display(budget_summary)

# %% Code cell 13 (notebook cell 15)

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

# %% Code cell 14 (notebook cell 16)
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

# %% Code cell 15 (notebook cell 17)
from pathlib import Path
import shutil

tag = "EVE_fraction0.2_DirectPlusPrior_seed44"

resume_path = CKPT_DIR / f"{tag}_resume.pt"
best_path = CKPT_DIR / f"{tag}_best.pt"

for p in [resume_path, best_path]:
    if p.exists():
        backup = p.with_name(p.stem + "_STALE_BACKUP" + p.suffix)
        shutil.move(str(p), str(backup))
        print("Moved stale checkpoint:", p.name, "->", backup.name)

# %% Code cell 16 (notebook cell 18)

# Train only plan rows that are not safely reusable.

RUNS_INCREMENTAL = OUT / f"{EXPERIMENT}_NEW_RUNS_INCREMENTAL.csv"

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

            state = safe_torch_load(ckpt, "cpu")

            if (
                state.get("contract_fingerprint") == expected_fp
                and sha256_file(ckpt) == str(r.checkpoint_sha256)
                and sha256_file(pred) == str(r.prediction_sha256)
            ):
                completed.add(key)

        except Exception:
            pass


to_train = train_plan[
    train_plan["action"] == "train_corrected_R2C"
].copy()

# deterministic low-cost-first order
domain_rank = {d: i for i, d in enumerate(DOMAIN_RUN_ORDER)}
fraction_rank = {f: i for i, f in enumerate(FRACTIONS)}

to_train["_domain_rank"] = to_train["domain"].map(domain_rank)
to_train["_fraction_rank"] = to_train["fraction"].map(fraction_rank)

to_train = (
    to_train
    .sort_values(
        ["_domain_rank", "_fraction_rank", "model", "seed"]
    )
    .drop(columns=["_domain_rank", "_fraction_rank"])
)

print("New corrected R2C runs planned:", len(to_train))
print("Already valid completed:", len(completed))

for row in to_train.itertuples(index=False):
    key = (
        str(row.domain),
        float(row.fraction),
        str(row.model),
        int(row.seed),
    )

    if key in completed:
        print("SKIP completed:", key)
        continue

    print(
        "\nTRAIN",
        row.domain,
        "fraction=", row.fraction,
        "model=", row.model,
        "seed=", row.seed,
    )

    result = train_one(
        row.domain,
        float(row.fraction),
        row.model,
        int(row.seed),
    )

    cur = (
        pd.read_csv(RUNS_INCREMENTAL)
        if RUNS_INCREMENTAL.exists()
        else pd.DataFrame()
    )

    cur = pd.concat(
        [cur, pd.DataFrame([result])],
        ignore_index=True,
    )

    cur = cur.drop_duplicates(
        ["domain", "fraction", "model", "seed"],
        keep="last",
    )

    cur.to_csv(RUNS_INCREMENTAL, index=False)

    resume_path = Path(str(result["resume_checkpoint"]))
    if resume_path.exists():
        resume_path.unlink()

    completed.add(key)

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

new_runs = (
    pd.read_csv(RUNS_INCREMENTAL)
    if RUNS_INCREMENTAL.exists()
    else pd.DataFrame()
)

assert len(new_runs) == n_new, (
    len(new_runs),
    n_new,
)

new_runs.to_csv(
    OUT / f"{EXPERIMENT}_NEW_RUNS.csv",
    index=False,
)

print("New training complete:", len(new_runs), "/", n_new)

# %% Code cell 17 (notebook cell 19)

# Standardize and import reusable R1A/R2A endpoint runs.

def standardize_prior_run(row, domain, model_std, source_name):
    return {
        "domain": domain,
        "fraction": float(row.fraction),
        "model": model_std,
        "seed": int(row.seed),
        "source_experiment": source_name,
        "selector_sha256": "",
        "prior_contract_sha256": "",
        "beta_map_json": "",
        "reused_prior_endpoint": True,
        "checkpoint": str(row.checkpoint),
        "checkpoint_sha256": str(row.checkpoint_sha256),
        "prediction_file": str(row.prediction_file),
        "prediction_sha256": str(row.prediction_sha256),
        "resume_checkpoint": str(getattr(row, "resume_checkpoint", "")),
        "best_epoch": int(row.best_epoch),
        "final_epoch": int(row.final_epoch),
        "stopped_by_patience": bool(row.stopped_by_patience),
        "hit_epoch_cap": bool(row.hit_epoch_cap),
        "n_params": int(row.n_params),
        "n_train_labels": int(row.n_train_labels),
        "n_validation": int(row.n_validation),
        "n_test": int(row.n_test),
        "max_epochs": int(row.max_epochs),
        "patience": int(row.patience),
        "train_seconds": float(row.train_seconds),
        "inference_seconds": float(row.inference_seconds),
        "val_MAE": float(row.val_MAE),
        "val_RMSE": float(row.val_RMSE),
        "val_P95": float(row.val_P95),
        "val_Bias": float(row.val_Bias),
        "test_MAE": float(row.test_MAE),
        "test_RMSE": float(row.test_RMSE),
        "test_P95": float(row.test_P95),
        "test_Bias": float(row.test_Bias),
    }

reused_rows = []

for rr in endpoint_reuse.itertuples(index=False):
    if not bool(rr.reusable):
        continue

    domain = str(rr.domain)
    fraction = float(rr.fraction)

    source_runs = r1a_runs if domain in DEV_DOMAINS else r2a_runs
    model_map = DEV_SOURCE_MODEL if domain in DEV_DOMAINS else EVE_SOURCE_MODEL
    source_name = "R1A_verified_full_endpoint" if domain in DEV_DOMAINS else "R2A_verified_full_endpoint"

    for model_std, source_model in model_map.items():
        q = source_runs[
            np.isclose(source_runs["fraction"].astype(float), fraction)
            & (source_runs["model"] == source_model)
        ].copy()

        if domain in DEV_DOMAINS:
            q = q[q["domain"] == domain].copy()

        assert len(q) == 5

        for row in q.itertuples(index=False):
            reused_rows.append(
                standardize_prior_run(
                    row,
                    domain,
                    model_std,
                    source_name,
                )
            )

reused_runs = pd.DataFrame(reused_rows)

all_runs = pd.concat(
    [new_runs, reused_runs],
    ignore_index=True,
)

all_runs = all_runs.drop_duplicates(
    ["domain", "fraction", "model", "seed"],
    keep="last",
)

assert len(all_runs) == n_total_curve_runs, (
    len(all_runs),
    n_total_curve_runs,
)

assert set(all_runs["domain"]) == set(ALL_DOMAINS)
assert set(all_runs["model"]) == set(MODELS_STD)
assert set(all_runs["seed"].astype(int)) == set(SEEDS)

all_runs.to_csv(
    OUT / f"{EXPERIMENT}_ALL_CURVE_RUNS.csv",
    index=False,
)

display(
    all_runs[
        [
            "domain", "fraction", "model", "seed",
            "source_experiment", "n_train_labels",
            "test_MAE", "test_RMSE", "test_P95",
        ]
    ].sort_values(
        ["domain", "fraction", "model", "seed"]
    )
)

# %% Code cell 18 (notebook cell 20)
# Load all seed predictions into a standardized corrected-R2C frame.
#
# IMPORTANT: corrected soc_phy is seed-specific because beta is fit from the retained target
# subset for that seed. Therefore soc_phy MUST NOT be part of the cross-seed row identity.

prediction_parts = []

for r in all_runs.itertuples(index=False):
    p = Path(str(r.prediction_file))
    assert p.exists(), p
    assert sha256_file(p) == str(r.prediction_sha256)

    z = pd.read_parquet(p).copy()

    required = {
        "vehicle_id",
        "segment_id",
        "compound_segment",
        "time_value",
        "elapsed_s",
        "target",
        "soc_phy",
        "prediction",
    }

    missing = sorted(required - set(z.columns))
    assert not missing, (r.domain, r.fraction, r.model, r.seed, missing)

    z["domain"] = str(r.domain)
    z["fraction"] = float(r.fraction)
    z["model"] = str(r.model)
    z["seed"] = int(r.seed)

    prediction_parts.append(
        z[
            [
                "domain",
                "fraction",
                "model",
                "seed",
                "vehicle_id",
                "segment_id",
                "compound_segment",
                "time_value",
                "elapsed_s",
                "target",
                "soc_phy",
                "prediction",
            ]
        ]
    )

all_seed_pred = pd.concat(
    prediction_parts,
    ignore_index=True,
)

row_identity = [
    "domain",
    "fraction",
    "model",
    "vehicle_id",
    "segment_id",
    "compound_segment",
    "time_value",
    "elapsed_s",
    "target",
]

coverage = (
    all_seed_pred
    .groupby(
        row_identity,
        dropna=False,
    )["seed"]
    .nunique()
    .reset_index(name="n_seeds")
)

assert (coverage["n_seeds"] == len(SEEDS)).all(), (
    coverage["n_seeds"].value_counts().to_dict()
)

seed_avg_pred = (
    all_seed_pred
    .groupby(
        row_identity,
        as_index=False,
        dropna=False,
    )
    .agg(
        prediction=("prediction", "mean"),
        soc_phy_mean=("soc_phy", "mean"),
        soc_phy_std=("soc_phy", "std"),
    )
)

seed_avg_pred["soc_phy_std"] = seed_avg_pred["soc_phy_std"].fillna(0.0)

seed_avg_pred["abs_error"] = np.abs(
    seed_avg_pred["prediction"].to_numpy(float)
    - seed_avg_pred["target"].to_numpy(float)
)

seed_avg_pred.to_parquet(
    OUT / f"{EXPERIMENT}_SEED_AVERAGED_PREDICTIONS.parquet",
    index=False,
)

all_seed_pred.to_parquet(
    OUT / f"{EXPERIMENT}_ALL_SEED_PREDICTIONS.parquet",
    index=False,
)

print("Seed-averaged predictions:", len(seed_avg_pred))

# %% Code cell 19 (notebook cell 21)

# Main curve summary: five-seed mean ± SD plus effective budget.

curve = (
    all_runs
    .groupby(["domain", "fraction", "model"], as_index=False)
    .agg(
        MAE_mean=("test_MAE", "mean"),
        MAE_std=("test_MAE", "std"),
        RMSE_mean=("test_RMSE", "mean"),
        RMSE_std=("test_RMSE", "std"),
        P95_mean=("test_P95", "mean"),
        Bias_mean=("test_Bias", "mean"),
        n_train_labels_mean=("n_train_labels", "mean"),
        n_params=("n_params", "first"),
        n_test=("n_test", "first"),
        reused_runs=("reused_prior_endpoint", "sum"),
    )
)

curve = curve.merge(
    budget_summary,
    on=["domain", "fraction"],
    how="left",
    validate="many_to_one",
)

assert curve["effective_fit_fraction_mean"].notna().all()

curve.to_csv(
    OUT / f"{EXPERIMENT}_CURVE_SUMMARY.csv",
    index=False,
)

display(curve)

# %% Code cell 20 (notebook cell 22)

# Pointwise factorization effects along the curve.

direct = curve[
    curve["model"] == DIRECT_STD
].copy()

pr = curve[
    curve["model"] == PR_STD
].copy()

effects = direct[
    [
        "domain",
        "fraction",
        "MAE_mean",
        "MAE_std",
        "actual_nominal_fraction_mean",
        "effective_fit_fraction_mean",
    ]
].rename(
    columns={
        "MAE_mean": "Direct_MAE",
        "MAE_std": "Direct_MAE_std",
    }
).merge(
    pr[
        [
            "domain",
            "fraction",
            "MAE_mean",
            "MAE_std",
        ]
    ].rename(
        columns={
            "MAE_mean": "PR_MAE",
            "MAE_std": "PR_MAE_std",
        }
    ),
    on=["domain", "fraction"],
    validate="one_to_one",
)

effects["delta_PR_minus_Direct_MAE"] = (
    effects["PR_MAE"] - effects["Direct_MAE"]
)

effects["PR_improvement_pct"] = (
    100.0
    * (effects["Direct_MAE"] - effects["PR_MAE"])
    / effects["Direct_MAE"]
)

effects["PR_lower_MAE"] = (
    effects["PR_MAE"] < effects["Direct_MAE"]
)

effects.to_csv(
    OUT / f"{EXPERIMENT}_POINTWISE_FACTORIZATION_EFFECTS.csv",
    index=False,
)

display(effects)

# %% Code cell 21 (notebook cell 23)

# Descriptive AULEC from the five-seed mean MAE curve.

aulec_rows = []

for domain in ALL_DOMAINS:
    b = budget_summary[
        budget_summary["domain"] == domain
    ].sort_values("fraction")

    x_nom = b["actual_nominal_fraction_mean"].to_numpy(float)
    x_eff = b["effective_fit_fraction_mean"].to_numpy(float)

    for model in MODELS_STD:
        q = curve[
            (curve["domain"] == domain)
            & (curve["model"] == model)
        ].sort_values("fraction")

        y = q["MAE_mean"].to_numpy(float)

        assert len(y) == len(FRACTIONS)

        a_nom = normalized_log_auc(x_nom, y)
        a_eff = normalized_log_auc(x_eff, y)

        aulec_rows.append({
            "domain": domain,
            "model": model,
            "AULEC_log_nominal": a_nom,
            "AULEC_log_effective": a_eff,
        })

aulec = pd.DataFrame(aulec_rows)

aulec_wide = (
    aulec
    .pivot(
        index="domain",
        columns="model",
        values=["AULEC_log_nominal", "AULEC_log_effective"],
    )
    .reset_index()
)

aulec_wide.columns = [
    "__".join([str(x) for x in c if str(x)])
    if isinstance(c, tuple)
    else str(c)
    for c in aulec_wide.columns
]

aulec_wide["effective_AULEC_improvement_pct"] = (
    100.0
    * (
        aulec_wide[f"AULEC_log_effective__{DIRECT_STD}"]
        - aulec_wide[f"AULEC_log_effective__{PR_STD}"]
    )
    / aulec_wide[f"AULEC_log_effective__{DIRECT_STD}"]
)

aulec_wide["nominal_AULEC_improvement_pct"] = (
    100.0
    * (
        aulec_wide[f"AULEC_log_nominal__{DIRECT_STD}"]
        - aulec_wide[f"AULEC_log_nominal__{PR_STD}"]
    )
    / aulec_wide[f"AULEC_log_nominal__{DIRECT_STD}"]
)

aulec_wide.to_csv(
    OUT / f"{EXPERIMENT}_AULEC_SUMMARY.csv",
    index=False,
)

display(aulec_wide)

# %% Code cell 22 (notebook cell 24)

# Label-efficiency knee:
# smallest fraction that stays within 5% / 10% of the model's own full-label MAE.

knee_rows = []

for domain in ALL_DOMAINS:
    for model in MODELS_STD:
        q = curve[
            (curve["domain"] == domain)
            & (curve["model"] == model)
        ].sort_values("fraction")

        full = float(
            q[np.isclose(q["fraction"], 1.0)]["MAE_mean"].iloc[0]
        )

        for tolerance_pct in [5, 10]:
            threshold = full * (1.0 + tolerance_pct / 100.0)

            eligible = q[
                q["MAE_mean"] <= threshold
            ].sort_values("fraction")

            if len(eligible):
                r = eligible.iloc[0]
                nominal_fraction = float(r["fraction"])
                effective_fraction = float(r["effective_fit_fraction_mean"])
                mae = float(r["MAE_mean"])
            else:
                nominal_fraction = np.nan
                effective_fraction = np.nan
                mae = np.nan

            knee_rows.append({
                "domain": domain,
                "model": model,
                "tolerance_from_full_pct": int(tolerance_pct),
                "full_label_MAE": full,
                "minimum_nominal_fraction": nominal_fraction,
                "minimum_effective_fit_fraction": effective_fraction,
                "MAE_at_knee": mae,
            })

knees = pd.DataFrame(knee_rows)

knees.to_csv(
    OUT / f"{EXPERIMENT}_LABEL_EFFICIENCY_KNEES.csv",
    index=False,
)

display(knees)

# %% Code cell 23 (notebook cell 25)

# Primary inference:
# paired compound-segment bootstrap of the ENTIRE effective-budget AULEC.
#
# One segment resample is applied consistently across all seven fractions and both models.
# The primary family contains exactly 5 tests (one per domain), Holm corrected.

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


def bootstrap_domain_aulec(domain, draws=BOOTSTRAP_DRAWS, seed=7301):
    seg = (
        seed_avg_pred[
            seed_avg_pred["domain"] == domain
        ]
        .groupby(
            ["fraction", "model", "compound_segment"],
            as_index=False,
        )["abs_error"]
        .mean()
    )

    segments = sorted(seg["compound_segment"].unique().tolist())
    fractions = np.asarray(FRACTIONS, float)

    b = (
        budget_summary[
            budget_summary["domain"] == domain
        ]
        .sort_values("fraction")
    )

    x_nom = b["actual_nominal_fraction_mean"].to_numpy(float)
    x_eff = b["effective_fit_fraction_mean"].to_numpy(float)

    mats = {}

    for model in MODELS_STD:
        p = (
            seg[seg["model"] == model]
            .pivot(
                index="compound_segment",
                columns="fraction",
                values="abs_error",
            )
            .reindex(index=segments, columns=fractions)
        )

        assert not p.isna().any().any()
        mats[model] = p.to_numpy(float)

    direct_mat = mats[DIRECT_STD]
    pr_mat = mats[PR_STD]

    point_direct_eff = normalized_log_auc(
        x_eff,
        direct_mat.mean(axis=0),
    )
    point_pr_eff = normalized_log_auc(
        x_eff,
        pr_mat.mean(axis=0),
    )

    point_direct_nom = normalized_log_auc(
        x_nom,
        direct_mat.mean(axis=0),
    )
    point_pr_nom = normalized_log_auc(
        x_nom,
        pr_mat.mean(axis=0),
    )

    rng = np.random.default_rng(seed)
    n_seg = len(segments)

    delta_eff = np.empty(draws, dtype=float)
    delta_nom = np.empty(draws, dtype=float)

    for i in range(draws):
        idx = rng.integers(0, n_seg, size=n_seg)

        d_curve = direct_mat[idx].mean(axis=0)
        p_curve = pr_mat[idx].mean(axis=0)

        delta_eff[i] = (
            normalized_log_auc(x_eff, p_curve)
            - normalized_log_auc(x_eff, d_curve)
        )

        delta_nom[i] = (
            normalized_log_auc(x_nom, p_curve)
            - normalized_log_auc(x_nom, d_curve)
        )

    def two_sided_sign_p(samples):
        p_left = (np.sum(samples <= 0) + 1.0) / (len(samples) + 1.0)
        p_right = (np.sum(samples >= 0) + 1.0) / (len(samples) + 1.0)
        return float(min(1.0, 2.0 * min(p_left, p_right)))

    return {
        "domain": domain,
        "n_segments": int(n_seg),

        "Direct_effective_AULEC_segment_balanced": float(point_direct_eff),
        "PR_effective_AULEC_segment_balanced": float(point_pr_eff),
        "delta_PR_minus_Direct_effective_AULEC": float(
            point_pr_eff - point_direct_eff
        ),
        "effective_CI95_low": float(np.percentile(delta_eff, 2.5)),
        "effective_CI95_high": float(np.percentile(delta_eff, 97.5)),
        "effective_two_sided_bootstrap_p": two_sided_sign_p(delta_eff),
        "P_PR_lower_effective_AULEC": float(np.mean(delta_eff < 0)),

        "Direct_nominal_AULEC_segment_balanced": float(point_direct_nom),
        "PR_nominal_AULEC_segment_balanced": float(point_pr_nom),
        "delta_PR_minus_Direct_nominal_AULEC": float(
            point_pr_nom - point_direct_nom
        ),
        "nominal_CI95_low": float(np.percentile(delta_nom, 2.5)),
        "nominal_CI95_high": float(np.percentile(delta_nom, 97.5)),
        "nominal_two_sided_bootstrap_p": two_sided_sign_p(delta_nom),

        "n_boot": int(draws),
    }


boot_rows = []

for i, domain in enumerate(ALL_DOMAINS):
    boot_rows.append(
        bootstrap_domain_aulec(
            domain,
            draws=BOOTSTRAP_DRAWS,
            seed=7301 + i * 101,
        )
    )

bootstrap_aulec = pd.DataFrame(boot_rows)

bootstrap_aulec["Holm_p_effective_AULEC"] = holm_adjust(
    bootstrap_aulec[
        "effective_two_sided_bootstrap_p"
    ].to_numpy(float)
)

bootstrap_aulec["effective_AULEC_supported_after_Holm"] = (
    (bootstrap_aulec["effective_CI95_high"] < 0)
    & (bootstrap_aulec["Holm_p_effective_AULEC"] < 0.05)
)

bootstrap_aulec.to_csv(
    OUT / f"{EXPERIMENT}_PRIMARY_EFFECTIVE_AULEC_BOOTSTRAP.csv",
    index=False,
)

display(bootstrap_aulec)

# %% Code cell 24 (notebook cell 26)

# Figures — one figure per domain, no multi-panel figures.

for domain in ALL_DOMAINS:
    q = curve[curve["domain"] == domain].copy()

    # Nominal label fraction
    plt.figure(figsize=(7, 5))

    for model in MODELS_STD:
        z = q[q["model"] == model].sort_values("fraction")

        plt.errorbar(
            100.0 * z["actual_nominal_fraction_mean"],
            z["MAE_mean"],
            yerr=z["MAE_std"],
            marker="o",
            capsize=3,
            label=model,
        )

    plt.xscale("log")
    plt.xlabel("Retained training-target rows (%)")
    plt.ylabel("Test MAE (percentage points)")
    plt.title(f"{domain} — nominal training-target fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{domain}_nominal_label_fraction_curve.png",
        dpi=220,
    )
    plt.show()

    # Effective unique fit observation fraction
    plt.figure(figsize=(7, 5))

    for model in MODELS_STD:
        z = q[q["model"] == model].sort_values("fraction")

        plt.errorbar(
            100.0 * z["effective_fit_fraction_mean"],
            z["MAE_mean"],
            yerr=z["MAE_std"],
            marker="o",
            capsize=3,
            label=model,
        )

    plt.xscale("log")
    plt.xlabel("Effective unique BMS fit observations (% of train rows)")
    plt.ylabel("Test MAE (percentage points)")
    plt.title(f"{domain} — effective supervision budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{domain}_effective_observation_curve.png",
        dpi=220,
    )
    plt.show()

# %% Code cell 25 (notebook cell 27)

# Compact manuscript-ready tables.

manuscript_curve = effects[
    [
        "domain",
        "fraction",
        "actual_nominal_fraction_mean",
        "effective_fit_fraction_mean",
        "Direct_MAE",
        "Direct_MAE_std",
        "PR_MAE",
        "PR_MAE_std",
        "PR_improvement_pct",
        "PR_lower_MAE",
    ]
].copy()

manuscript_curve["nominal_pct"] = (
    100.0 * manuscript_curve["actual_nominal_fraction_mean"]
)
manuscript_curve["effective_pct"] = (
    100.0 * manuscript_curve["effective_fit_fraction_mean"]
)

manuscript_curve.to_csv(
    OUT / f"{EXPERIMENT}_MANUSCRIPT_CURVE_TABLE.csv",
    index=False,
)

manuscript_aulec = bootstrap_aulec[
    [
        "domain",
        "n_segments",
        "Direct_effective_AULEC_segment_balanced",
        "PR_effective_AULEC_segment_balanced",
        "delta_PR_minus_Direct_effective_AULEC",
        "effective_CI95_low",
        "effective_CI95_high",
        "Holm_p_effective_AULEC",
        "effective_AULEC_supported_after_Holm",
    ]
].copy()

manuscript_aulec.to_csv(
    OUT / f"{EXPERIMENT}_MANUSCRIPT_AULEC_TABLE.csv",
    index=False,
)

display(manuscript_curve)
display(manuscript_aulec)

# %% Code cell 26 (notebook cell 28)

# Final integrity manifest.

assert len(curve) == len(ALL_DOMAINS) * len(FRACTIONS) * len(MODELS_STD)
assert len(effects) == len(ALL_DOMAINS) * len(FRACTIONS)
assert len(bootstrap_aulec) == len(ALL_DOMAINS)

assert parameter_audit["exact_capacity_match"].all()

assert np.isfinite(
    curve[
        [
            "MAE_mean",
            "RMSE_mean",
            "P95_mean",
            "effective_fit_fraction_mean",
        ]
    ].to_numpy(float)
).all()

# Budget must be non-decreasing with nominal fraction in every domain.
for domain in ALL_DOMAINS:
    b = (
        budget_summary[
            budget_summary["domain"] == domain
        ]
        .sort_values("fraction")
    )

    eff = b["effective_fit_fraction_mean"].to_numpy(float)
    assert np.all(np.diff(eff) >= -1e-12), (domain, eff)

manifest = {
    **preflight,
    "status": "R2C_CORRECTED_SPARSE_PRIOR_CURVE_COMPLETE",
    "new_training_runs_completed": int(len(new_runs)),
    "reused_endpoint_runs_completed": int(len(reused_runs)),
    "all_curve_runs": int(len(all_runs)),
    "curve_conditions": int(len(effects)),
    "AULEC_domains": int(len(bootstrap_aulec)),
    "primary_effective_AULEC_supported_domains_after_Holm": int(
        bootstrap_aulec["effective_AULEC_supported_after_Holm"].sum()
    ),
    "training_target_fraction_curve": FRACTIONS,
    "effective_budget_is_anchor_union_corrected": True,
    "prior_is_refit_per_domain_fraction_seed": True,
    "prior_fit_uses_only_retained_training_targets": True,
    "validation_targets_used_for_prior_fit": False,
    "test_targets_used_for_prior_fit": False,
    "sparse_endpoint_reuse_allowed": False,
    "TUM_full_retrained": True,
    "seed_specific_prior_handled_in_seed_averaging": True,
    "validation_targets_not_in_effective_fit_fraction": True,
    "test_targets_not_in_supervision_budget": True,
    "EVE_historical_test_exposure": True,
    "confirmatory": False,
    "no_retuning": True,
}

with open(OUT / f"{EXPERIMENT}_MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))

# %% Code cell 27 (notebook cell 29)
# Auto-generated manuscript notes from corrected outcomes.

notes = []

notes.append("# Corrected R2C Manuscript Notes")
notes.append("")
notes.append(
    "R2C-C evaluates information-matched Direct+Prior and PriorResidual+Prior models "
    "at nominal training-target fractions of 1%, 2%, 5%, 10%, 20%, 50%, and 100%."
)
notes.append("")
notes.append(
    "For every domain/fraction/seed condition, all target-dependent prior slopes are "
    "re-estimated exclusively from retained training target rows permitted in that condition. "
    "Validation and test targets are never used for prior calibration."
)
notes.append("")
notes.append(
    "Because segment-start BMS anchors are genuine target observations, the primary "
    "label-efficiency analysis uses the effective fit-observation fraction "
    "(selected training labels UNION required training anchors), with overlap counted once."
)
notes.append("")
notes.append(
    "TUM is retrained at all fractions including 100% because the current canonical training "
    "universe does not reproduce the historical full-training distance-prior slope. "
    "For VED, KU, SCUT and EVE, a 100% endpoint may be reused only after the P0 full-prior "
    "reproduction gate verifies numerical equivalence."
)
notes.append("")
notes.append("## Effective-observation AULEC")
notes.append("")
notes.append("| Domain | Direct AULEC | PR AULEC | PR-Direct | 95% CI | Holm p | Supported |")
notes.append("|---|---:|---:|---:|---:|---:|---:|")

for r in bootstrap_aulec.itertuples(index=False):
    notes.append(
        f"| {r.domain} | "
        f"{r.Direct_effective_AULEC_segment_balanced:.4f} | "
        f"{r.PR_effective_AULEC_segment_balanced:.4f} | "
        f"{r.delta_PR_minus_Direct_effective_AULEC:.4f} | "
        f"[{r.effective_CI95_low:.4f}, {r.effective_CI95_high:.4f}] | "
        f"{r.Holm_p_effective_AULEC:.4g} | "
        f"{bool(r.effective_AULEC_supported_after_Holm)} |"
    )

notes.append("")
notes.append(
    "**Interpretation boundary:** this is a corrective post-confirmatory analysis. "
    "It repairs the supervision accounting of the structured prior and does not restore "
    "confirmatory status to the previously opened EVE test."
)

notes_path = OUT / f"{EXPERIMENT}_MANUSCRIPT_NOTES.md"
notes_path.write_text("\n".join(notes), encoding="utf-8")

print(notes_path.read_text(encoding="utf-8"))

# %% [markdown] Main outputs
