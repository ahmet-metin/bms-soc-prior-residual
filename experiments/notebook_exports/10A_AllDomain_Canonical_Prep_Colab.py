# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/10A_AllDomain_Canonical_Prep_Colab.ipynb
# Notebook SHA256: 2c009ad47cc38dfbd9408aeb7a85d40efcd2cb3ba26da72ca81c87e4e06499a5
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 10A — All-Domain Canonical Data Preparation

# %% Code cell 1 (notebook cell 2)
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow scikit-learn joblib requests

import gc
import json
import math
import hashlib
import random
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# %% Code cell 2 (notebook cell 3)
from google.colab import drive
drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
OUT = ROOT / "outputs" / "model_screening_10"
CANON = OUT / "canonical"
CANON.mkdir(parents=True, exist_ok=True)

print("Canonical output folder:", CANON)

# %% [markdown] Shared helpers

# %% Code cell 3 (notebook cell 5)
STATS = ["last", "mean", "std", "min", "max", "delta"]

def fit_zero_intercept(x, y, min_abs_x=1e-4, min_n=20):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(x) >= float(min_abs_x))
    )

    x = x[keep]
    y = y[keep]

    if len(x) < min_n:
        raise RuntimeError(f"Too few informative physics anchors: {len(x)}")

    den = float(np.dot(x, x))
    if den <= 0:
        raise RuntimeError("Physics calibration denominator is zero.")

    beta = float(np.dot(x, y) / den)
    return beta, len(x)


def assign_chronological_session_split(df, vehicle_col="vehicle_id", segment_col="session_id",
                                       time_col="timestamp", target_col="soc_pct"):
    sess = (
        df.groupby([vehicle_col, segment_col], as_index=False)
        .agg(
            start_time=(time_col, "min"),
            anchors=(target_col, "size"),
        )
    )

    def assign_one(g):
        g = (
            g.sort_values(["start_time", segment_col])
            .reset_index(drop=True)
            .copy()
        )

        cnt = g["anchors"].to_numpy(float)
        total = float(cnt.sum())

        if len(g) < 5 or total < 20:
            g["split"] = "insufficient"
            return g

        cum = np.cumsum(cnt)

        c1 = np.arange(1, len(g) - 2)
        cut1 = int(c1[np.argmin(np.abs(cum[c1 - 1] - 0.60 * total))])

        c2 = np.arange(cut1 + 1, len(g))
        cut2 = int(c2[np.argmin(np.abs(cum[c2 - 1] - 0.80 * total))])

        g["split"] = "test"
        g.loc[: cut1 - 1, "split"] = "train"
        g.loc[cut1 : cut2 - 1, "split"] = "val"

        return g

    split_assign = (
        sess.groupby(vehicle_col, group_keys=False)
        .apply(assign_one)
        .reset_index(drop=True)
    )

    out = df.merge(
        split_assign[[vehicle_col, segment_col, "split"]],
        on=[vehicle_col, segment_col],
        how="left",
        validate="many_to_one",
    )

    return out, split_assign


def save_canonical(df, domain, feature_cols, coordinate_kind, source_note):
    out = df.copy()

    required = [
        "domain",
        "vehicle_id",
        "segment_id",
        "time_value",
        "split",
        "target",
        "soc0",
        "coord",
        "soc_phy",
        "physics_delta",
        "elapsed_s",
    ]

    missing = [c for c in required if c not in out.columns]
    assert not missing, missing

    assert set(out["split"].unique()) <= {"train", "val", "test"}
    assert len(out) > 0

    path = CANON / f"{domain}_canonical.parquet"
    out.to_parquet(path, index=False)

    meta = {
        "domain": domain,
        "coordinate_kind": coordinate_kind,
        "target": "BMS-reported SoC",
        "feature_columns": feature_cols,
        "n_rows": int(len(out)),
        "n_vehicles": int(out["vehicle_id"].nunique()),
        "n_segments": int(out["segment_id"].nunique()),
        "split_counts": {
            str(k): int(v)
            for k, v in out["split"].value_counts().to_dict().items()
        },
        "source_note": source_note,
    }

    with open(CANON / f"{domain}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        domain,
        "rows=", len(out),
        "vehicles=", out["vehicle_id"].nunique(),
        "segments=", out["segment_id"].nunique(),
    )

    return path, meta

# %% [markdown] A) VED canonical table

# %% Code cell 4 (notebook cell 7)
VED_FEATURE_CACHE = ROOT / "outputs" / "cache" / "endpoint_features_c1da8d4d47.parquet"
VED_MANIFEST = ROOT / "outputs" / "run_manifest_c1da8d4d47.json"

assert VED_FEATURE_CACHE.exists()
assert VED_MANIFEST.exists()

ved = pd.read_parquet(VED_FEATURE_CACHE)

with open(VED_MANIFEST) as f:
    ved_manifest = json.load(f)

BETA_POOL = float(ved_manifest["beta_pp_per_Ah"])

VED_DEV_VEHICLES = [10, 455]

BASE_SIGNALS = [
    "current_a_aligned",
    "voltage_v_aligned",
    "speed_kmh_aligned",
    "oat_c_aligned",
    "soc_phy",
    "q_abs_ah",
    "elapsed_s",
    "soc0",
]

ALL_48 = [
    f"{signal}__{stat}"
    for signal in BASE_SIGNALS
    for stat in STATS
]

assert not [c for c in ALL_48 if c not in ved.columns]

NON_POOLED = [
    c for c in ALL_48
    if not c.startswith("soc_phy__")
]

ved = ved[ved["veh_id"].isin(VED_DEV_VEHICLES)].copy()

ved["q_signed_ah_endpoint"] = (
    ved["soc_phy_at_endpoint"].to_numpy(float)
    - ved["soc0__last"].to_numpy(float)
) / BETA_POOL

train_anchor = ved[
    (ved["split"] == "train")
    & (ved["soc_mask"] > 0.5)
    & np.isfinite(ved["soc_target"])
    & np.isfinite(ved["q_signed_ah_endpoint"])
].copy()

VED_BETA, ved_beta_n = fit_zero_intercept(
    train_anchor["q_signed_ah_endpoint"],
    train_anchor["soc_target"] - train_anchor["soc0__last"],
    min_abs_x=0.01,
    min_n=20,
)

ved["soc_phy_source"] = (
    ved["soc0__last"].to_numpy(float)
    + VED_BETA * ved["q_signed_ah_endpoint"].to_numpy(float)
)

VED_DIRECT_FEATURES = NON_POOLED + ["q_signed_ah_endpoint"]

mask = (
    (ved["soc_mask"] > 0.5)
    & np.isfinite(ved["soc_target"])
    & np.isfinite(ved["soc_phy_source"])
)

for c in VED_DIRECT_FEATURES:
    mask &= np.isfinite(pd.to_numeric(ved[c], errors="coerce"))

v = ved[mask].copy()

canon_ved = pd.DataFrame({
    "domain": "VED",
    "vehicle_id": v["veh_id"].astype(str),
    "segment_id": v["segment_id"].astype(str),
    "time_value": v["endpoint_t_abs_s"].astype(float),
    "split": v["split"].astype(str),
    "target": v["soc_target"].astype(float),
    "soc0": v["soc0__last"].astype(float),
    "coord": v["q_signed_ah_endpoint"].astype(float),
    "soc_phy": v["soc_phy_source"].astype(float),
    "physics_delta": (
        v["soc_phy_source"].astype(float)
        - v["soc0__last"].astype(float)
    ),
    "elapsed_s": v["elapsed_s__last"].astype(float),
})

VED_X = []

for c in VED_DIRECT_FEATURES:
    new = f"x__{c}"
    canon_ved[new] = pd.to_numeric(v[c], errors="coerce").to_numpy(float)
    VED_X.append(new)

save_canonical(
    canon_ved,
    "VED",
    VED_X,
    "signed_Ah",
    (
        "VED Leaf 10+455; genuine BMS SoC-change anchors; "
        f"train-only pooled beta={VED_BETA:.8f} pp/Ah from {ved_beta_n} anchors"
    ),
)

# %% [markdown] B) KU Leuven canonical table

# %% Code cell 5 (notebook cell 9)
EXT_CACHE = (
    ROOT
    / "dataset"
    / "external_validation"
    / "external_endpoint_features_v5_scut_schema_time_multisheet.parquet"
)

assert EXT_CACHE.exists()

ext = pd.read_parquet(EXT_CACHE)

ku = ext[ext["dataset"] == "KULEUVEN"].copy()

PRIMARY_SIGNALS = [
    "current_a",
    "voltage_v",
    "speed_kmh",
    "q_abs_ah",
    "elapsed_s",
    "soc0",
]

KU_DIRECT_RAW = [
    f"{s}__{st}"
    for s in PRIMARY_SIGNALS
    for st in STATS
] + ["q_signed_ah"]

missing = [c for c in KU_DIRECT_RAW if c not in ku.columns]
assert not missing, missing

ku, ku_split = assign_chronological_session_split(
    ku,
    vehicle_col="vehicle_id",
    segment_col="session_id",
    time_col="timestamp",
    target_col="soc_pct",
)

ku = ku[ku["split"].isin(["train", "val", "test"])].copy()

beta_bank = {}
beta_rows = []

for veh, g in ku[ku["split"] == "train"].groupby("vehicle_id"):
    beta, n = fit_zero_intercept(
        g["q_signed_ah"],
        g["soc_pct"] - g["soc0"],
        min_abs_x=1e-4,
        min_n=20,
    )

    beta_bank[veh] = beta
    beta_rows.append({
        "vehicle_id": veh,
        "beta_pp_per_Ah": beta,
        "n_train_anchors": n,
    })

ku["beta_vehicle"] = ku["vehicle_id"].map(beta_bank)

assert ku["beta_vehicle"].notna().all()

ku["soc_phy_source"] = (
    ku["soc0"].to_numpy(float)
    + ku["beta_vehicle"].to_numpy(float)
    * ku["q_signed_ah"].to_numpy(float)
)

mask = (
    np.isfinite(ku["soc_pct"])
    & np.isfinite(ku["soc_phy_source"])
)

for c in KU_DIRECT_RAW:
    mask &= np.isfinite(pd.to_numeric(ku[c], errors="coerce"))

k = ku[mask].copy()

canon_ku = pd.DataFrame({
    "domain": "KU",
    "vehicle_id": k["vehicle_id"].astype(str),
    "segment_id": k["session_id"].astype(str),
    "time_value": pd.to_datetime(k["timestamp"], utc=True).astype("int64") / 1e9,
    "split": k["split"].astype(str),
    "target": k["soc_pct"].astype(float),
    "soc0": k["soc0"].astype(float),
    "coord": k["q_signed_ah"].astype(float),
    "soc_phy": k["soc_phy_source"].astype(float),
    "physics_delta": (
        k["soc_phy_source"].astype(float)
        - k["soc0"].astype(float)
    ),
    "elapsed_s": k["elapsed_s"].astype(float),
})

KU_X = []

for c in KU_DIRECT_RAW:
    new = f"x__{c}"
    canon_ku[new] = pd.to_numeric(k[c], errors="coerce").to_numpy(float)
    KU_X.append(new)

pd.DataFrame(beta_rows).to_csv(
    OUT / "KU" / "canonical_beta_audit.csv",
    index=False,
)

save_canonical(
    canon_ku,
    "KU",
    KU_X,
    "signed_Ah",
    "KU Leuven BEV1+BEV2; vehicle-specific train-only beta; temperature excluded from primary features",
)

# %% [markdown] C) corrected SCUT canonical table

# %% Code cell 6 (notebook cell 11)
SCUT_CACHE_ROOT = (
    ROOT
    / "dataset"
    / "external_validation"
    / "feature_cache_7B"
)

SCUT_FILES = sorted(
    SCUT_CACHE_ROOT.glob(
        "SCUT_*_7B_scut_vehicle7_drivecode_fix_v1.parquet"
    )
)

assert len(SCUT_FILES) == 7, SCUT_FILES

scut = pd.concat(
    [pd.read_parquet(p) for p in SCUT_FILES],
    ignore_index=True,
)

if "dataset" not in scut.columns:
    scut["dataset"] = "SCUT"

SCUT_DIRECT_RAW = [
    f"{s}__{st}"
    for s in PRIMARY_SIGNALS
    for st in STATS
] + ["q_signed_ah"]

missing = [c for c in SCUT_DIRECT_RAW if c not in scut.columns]
assert not missing, missing

scut, scut_split = assign_chronological_session_split(
    scut,
    vehicle_col="vehicle_id",
    segment_col="session_id",
    time_col="timestamp",
    target_col="soc_pct",
)

scut = scut[scut["split"].isin(["train", "val", "test"])].copy()

beta_bank = {}
beta_rows = []

for veh, g in scut[scut["split"] == "train"].groupby("vehicle_id"):
    beta, n = fit_zero_intercept(
        g["q_signed_ah"],
        g["soc_pct"] - g["soc0"],
        min_abs_x=1e-4,
        min_n=20,
    )

    beta_bank[veh] = beta
    beta_rows.append({
        "vehicle_id": veh,
        "beta_pp_per_Ah": beta,
        "n_train_anchors": n,
    })

scut["beta_vehicle"] = scut["vehicle_id"].map(beta_bank)
assert scut["beta_vehicle"].notna().all()

scut["soc_phy_source"] = (
    scut["soc0"].to_numpy(float)
    + scut["beta_vehicle"].to_numpy(float)
    * scut["q_signed_ah"].to_numpy(float)
)

mask = (
    np.isfinite(scut["soc_pct"])
    & np.isfinite(scut["soc_phy_source"])
)

for c in SCUT_DIRECT_RAW:
    mask &= np.isfinite(pd.to_numeric(scut[c], errors="coerce"))

s = scut[mask].copy()

canon_scut = pd.DataFrame({
    "domain": "SCUT",
    "vehicle_id": s["vehicle_id"].astype(str),
    "segment_id": s["session_id"].astype(str),
    "time_value": pd.to_datetime(s["timestamp"], utc=True).astype("int64") / 1e9,
    "split": s["split"].astype(str),
    "target": s["soc_pct"].astype(float),
    "soc0": s["soc0"].astype(float),
    "coord": s["q_signed_ah"].astype(float),
    "soc_phy": s["soc_phy_source"].astype(float),
    "physics_delta": (
        s["soc_phy_source"].astype(float)
        - s["soc0"].astype(float)
    ),
    "elapsed_s": s["elapsed_s"].astype(float),
})

SCUT_X = []

for c in SCUT_DIRECT_RAW:
    new = f"x__{c}"
    canon_scut[new] = pd.to_numeric(s[c], errors="coerce").to_numpy(float)
    SCUT_X.append(new)

pd.DataFrame(beta_rows).to_csv(
    OUT / "SCUT" / "canonical_beta_audit.csv",
    index=False,
)

save_canonical(
    canon_scut,
    "SCUT",
    SCUT_X,
    "signed_Ah",
    (
        "SCUT vehicles 1-7 from corrected 7B caches; corrected driving-state mapping; "
        "vehicle-specific train-only beta"
    ),
)

# %% [markdown] D) TUM/FTM canonical table

# %% Code cell 7 (notebook cell 13)
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow scikit-learn joblib requests

import os
import gc
import json
import math
import shutil
import hashlib
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

FINAL_SEEDS = [42, 43, 44, 45, 46]
TUNING_SEED = 777
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

REPO_URL = "https://github.com/TUMFTM/electric-vehicle-uds-dataset.git"
REPO_ZIP = "https://github.com/TUMFTM/electric-vehicle-uds-dataset/archive/refs/heads/main.zip"

EXPECTED_SIGNAL_IDS = {
    "ambient_air_temp": 15,
    "vehicle_speed": 4,
    "hv_battery_voltage": 1200,
    "hv_temp_min": 1208,
    "hv_temp_max": 1209,
    "cell_c_rate": 1288,
    "hv_soc": 900,
}

# Required for the all-seven primary analysis.
PRIMARY_RAW_SIGNALS = [
    "hv_soc",
    "hv_battery_voltage",
    "vehicle_speed",
]

# Read only for audit / possible later sensitivity analysis.
AUDIT_RAW_SIGNALS = [
    "cell_c_rate",
    "ambient_air_temp",
    "hv_temp_min",
    "hv_temp_max",
]

REQUESTED_SIGNALS = list(dict.fromkeys(
    PRIMARY_RAW_SIGNALS + AUDIT_RAW_SIGNALS
))

MAX_AGE_S = {
    "hv_battery_voltage": 3.0,
    "vehicle_speed": 3.0,
    "distance_cum_km": 3.0,
    "ambient_air_temp": 30.0,
    "hv_temp_min": 30.0,
    "hv_temp_max": 30.0,
}

SPEED_INTEGRATION_MAX_GAP_S = 2.0

MOVING_SPEED_KMH = 1.0
DRIVING_MEMORY_S = 60.0
SEGMENT_GAP_S = 20.0
MIN_SEGMENT_DURATION_S = 30.0
MIN_GENUINE_ANCHORS_PER_SEGMENT = 3

HISTORY_S = 120.0
STAT_NAMES = ["last", "mean", "std", "min", "max", "delta"]

N_TREES_TUNE = 100
N_TREES_FINAL = 150

ET_GRID = [
    {"min_samples_leaf": 1, "max_features": 0.5},
    {"min_samples_leaf": 2, "max_features": 0.5},
    {"min_samples_leaf": 4, "max_features": 0.5},
    {"min_samples_leaf": 8, "max_features": 0.5},
    {"min_samples_leaf": 2, "max_features": 0.8},
    {"min_samples_leaf": 4, "max_features": 0.8},
]

LABEL_FRACTIONS = [1.0, 0.10]

print("Notebook 9 v2 protocol initialized.")

# %% [markdown] 1) Drive and clean v2 output folder

# %% Code cell 8 (notebook cell 15)
ROOT = Path("/content/drive/MyDrive/NissanLeaf")
OUT9 = ROOT / "outputs" / "model_screening_10" / "TUM" / "prep_audit"
OUT9.mkdir(parents=True, exist_ok=True)

LOCAL_REPO = Path("/content/tum_ev_uds_dataset")
LOCAL_ZIP = Path("/content/tum_ev_uds_dataset.zip")

print("TUM screening prep audit:", OUT9)

# %% [markdown] 2) Download repository and verify the seven parquet files

# %% Code cell 9 (notebook cell 17)
def run_cmd(cmd, cwd=None, check=True):
    print("$", " ".join(cmd))

    p = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if p.stdout:
        print(p.stdout[-5000:])

    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}"
        )

    return p


def parquet_is_lfs_pointer(path):
    path = Path(path)

    if not path.exists():
        return False

    if path.stat().st_size > 4096:
        return False

    try:
        txt = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return False

    return txt.startswith(
        "version https://git-lfs.github.com/spec/v1"
    )


def validate_raw_parquets(repo_dir):
    parquet_dir = Path(repo_dir) / "data" / "uds_data"
    files = sorted(parquet_dir.glob("*.parquet"))

    if not files:
        return False, []

    bad = []

    for p in files:
        if parquet_is_lfs_pointer(p):
            bad.append((p.name, "Git-LFS pointer"))
            continue

        try:
            pf = pq.ParquetFile(p)
            _ = pf.schema_arrow
        except Exception as e:
            bad.append((p.name, f"unreadable parquet: {e}"))

    return len(bad) == 0, bad


if LOCAL_REPO.exists():
    ok, bad = validate_raw_parquets(LOCAL_REPO)

    if not ok:
        print("Removing incomplete existing repository:", bad)
        shutil.rmtree(LOCAL_REPO, ignore_errors=True)


if not LOCAL_REPO.exists():
    run_cmd(
        [
            "bash",
            "-lc",
            "apt-get -qq update && apt-get -qq install -y git-lfs unzip",
        ]
    )

    run_cmd(["git", "lfs", "install"])

    clone = run_cmd(
        [
            "git",
            "clone",
            "--depth",
            "1",
            REPO_URL,
            str(LOCAL_REPO),
        ],
        check=False,
    )

    if clone.returncode == 0:
        run_cmd(
            [
                "git",
                "lfs",
                "pull",
                "-I",
                "data/uds_data/*.parquet",
            ],
            cwd=str(LOCAL_REPO),
            check=False,
        )


ok, bad = validate_raw_parquets(LOCAL_REPO)

if not ok:
    print("Git-LFS route incomplete; using repository ZIP.", bad)

    shutil.rmtree(LOCAL_REPO, ignore_errors=True)

    if LOCAL_ZIP.exists():
        LOCAL_ZIP.unlink()

    run_cmd(
        [
            "wget",
            "-q",
            "--show-progress",
            "-O",
            str(LOCAL_ZIP),
            REPO_ZIP,
        ]
    )

    extract_dir = Path("/content/tum_zip_extract")
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            "unzip",
            "-q",
            str(LOCAL_ZIP),
            "-d",
            str(extract_dir),
        ]
    )

    candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
    assert len(candidates) == 1, candidates

    shutil.move(str(candidates[0]), str(LOCAL_REPO))


ok, bad = validate_raw_parquets(LOCAL_REPO)

assert ok, f"TUM parquet download failed: {bad}"

PARQUET_DIR = LOCAL_REPO / "data" / "uds_data"
PARQUET_FILES = sorted(PARQUET_DIR.glob("*.parquet"))

assert len(PARQUET_FILES) == 7, (
    f"Expected 7 vehicle parquet files, found {len(PARQUET_FILES)}"
)

print("Readable vehicle parquet files:")
for p in PARQUET_FILES:
    print(" ", p.name, f"{p.stat().st_size / (1024**2):.1f} MB")

# %% [markdown] 3) Repository metadata and signal dictionary audit

# %% Code cell 10 (notebook cell 19)
OVERVIEW_PATH = LOCAL_REPO / "data" / "value_overview.csv"
LICENSE_PATH = LOCAL_REPO / "LICENSE"

assert OVERVIEW_PATH.exists()
assert LICENSE_PATH.exists()

overview = pd.read_csv(OVERVIEW_PATH)

required_cols = {
    "value_id",
    "variable_name",
    "unit",
}

assert required_cols.issubset(set(overview.columns)), overview.columns

signal_dictionary_rows = []

for name, expected_id in EXPECTED_SIGNAL_IDS.items():
    r = overview.loc[
        overview["variable_name"] == name
    ]

    assert len(r) == 1, (
        f"Signal dictionary entry is not unique: {name}",
        r,
    )

    r = r.iloc[0]

    signal_dictionary_rows.append(
        {
            "variable_name": name,
            "expected_value_id": int(expected_id),
            "actual_value_id": int(r["value_id"]),
            "id_match": int(r["value_id"]) == int(expected_id),
            "unit": r["unit"],
            "sampling_interval_ms": (
                r["sampling_interval_ms"]
                if "sampling_interval_ms" in overview.columns
                else np.nan
            ),
        }
    )

signal_dictionary_audit = pd.DataFrame(signal_dictionary_rows)
assert signal_dictionary_audit["id_match"].all()

signal_dictionary_audit.to_csv(
    OUT9 / "tum_signal_dictionary_audit.csv",
    index=False,
)

display(signal_dictionary_audit)

license_text = LICENSE_PATH.read_text(
    encoding="utf-8",
    errors="ignore",
)

print("License preview:")
print(license_text[:1000])

# %% [markdown] 4) Schema-adaptive raw reader

# %% Code cell 11 (notebook cell 21)
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

# %% [markdown] 5) Native-speed distance integration

# %% Code cell 12 (notebook cell 23)
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

# %% [markdown] 6) Align common raw signals to native BMS SoC timestamps

# %% Code cell 13 (notebook cell 25)
def merge_latest_signal(
    base,
    signal_df,
    signal_name,
    max_age_s,
):
    left = (
        base[["time"]]
        .copy()
        .sort_values("time")
    )

    right = (
        signal_df.copy()
        .rename(
            columns={
                "value": signal_name,
                "time": f"{signal_name}__time",
            }
        )
        .sort_values(f"{signal_name}__time")
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="time",
        right_on=f"{signal_name}__time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=float(max_age_s)),
    )

    merged[f"{signal_name}__age_s"] = (
        merged["time"]
        - merged[f"{signal_name}__time"]
    ).dt.total_seconds()

    return merged.drop(
        columns=[f"{signal_name}__time"]
    )


def align_vehicle_to_soc(
    series_dict,
    vehicle_id,
):
    missing_primary = [
        s
        for s in PRIMARY_RAW_SIGNALS
        if s not in series_dict
        or len(series_dict[s]) == 0
    ]

    if missing_primary:
        raise RuntimeError(
            f"{vehicle_id}: missing required raw signals {missing_primary}. "
            f"Available: {sorted(series_dict.keys())}"
        )

    distance_series = speed_to_cumulative_distance(
        series_dict["vehicle_speed"]
    )

    soc = (
        series_dict["hv_soc"]
        .rename(columns={"value": "hv_soc"})
        .copy()
    )

    base = (
        soc[["time", "hv_soc"]]
        .sort_values("time")
        .reset_index(drop=True)
    )

    align_sources = {
        "hv_battery_voltage": series_dict["hv_battery_voltage"],
        "vehicle_speed": series_dict["vehicle_speed"],
        "distance_cum_km": distance_series,
    }

    for s in [
        "ambient_air_temp",
        "hv_temp_min",
        "hv_temp_max",
    ]:
        if s in series_dict and len(series_dict[s]):
            align_sources[s] = series_dict[s]

    for signal, signal_df in align_sources.items():
        aligned = merge_latest_signal(
            base,
            signal_df,
            signal,
            MAX_AGE_S[signal],
        )

        base = base.merge(
            aligned,
            on="time",
            how="left",
            validate="one_to_one",
        )

    base["vehicle_id"] = str(vehicle_id)

    base.loc[
        ~base["hv_soc"].between(0.0, 100.0),
        "hv_soc",
    ] = np.nan

    base.loc[
        ~base["vehicle_speed"].between(0.0, 260.0),
        "vehicle_speed",
    ] = np.nan

    base.loc[
        ~base["hv_battery_voltage"].between(100.0, 1000.0),
        "hv_battery_voltage",
    ] = np.nan

    return base

# %% [markdown] 7) Driving segmentation

# %% Code cell 14 (notebook cell 27)
def add_driving_segments(d):
    d = (
        d.sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    speed = d["vehicle_speed"].fillna(0.0)

    tmp = pd.DataFrame(
        {
            "time": d["time"],
            "speed": speed,
        }
    ).set_index("time")

    recent_speed_max = (
        tmp["speed"]
        .rolling(
            f"{int(DRIVING_MEMORY_S)}s",
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    d["driving_context"] = (
        recent_speed_max > MOVING_SPEED_KMH
    )

    dt = (
        d["time"]
        .diff()
        .dt.total_seconds()
    )

    new_seg = (
        d["driving_context"]
        & (
            ~d["driving_context"].shift(fill_value=False)
            | (dt > SEGMENT_GAP_S)
        )
    )

    seg_num = new_seg.cumsum()

    d["segment_local"] = np.where(
        d["driving_context"],
        seg_num,
        np.nan,
    )

    d = d[
        d["driving_context"]
        & d["segment_local"].notna()
    ].copy()

    d["segment_id"] = (
        d["vehicle_id"].astype(str)
        + "__"
        + d["segment_local"].astype(int).astype(str)
    )

    pieces = []

    for sid, g in d.groupby("segment_id", sort=False):
        g = (
            g.sort_values("time")
            .reset_index(drop=True)
            .copy()
        )

        duration = float(
            (
                g["time"].iloc[-1]
                - g["time"].iloc[0]
            ).total_seconds()
        )

        soc = g["hv_soc"]

        genuine = (
            soc.notna()
            & soc.shift().notna()
            & ((soc - soc.shift()).abs() > 1e-6)
        )

        if (
            duration < MIN_SEGMENT_DURATION_S
            or int(genuine.sum()) < MIN_GENUINE_ANCHORS_PER_SEGMENT
        ):
            continue

        g["segment_duration_s"] = duration
        pieces.append(g)

    if not pieces:
        return pd.DataFrame()

    return pd.concat(
        pieces,
        ignore_index=True,
    )

# %% [markdown] 8) Read all seven vehicles and save actual raw-signal availability

# %% Code cell 15 (notebook cell 29)
aligned_frames = []
schema_rows = []
availability_rows = []

for parquet_path in PARQUET_FILES:
    vehicle_id = parquet_path.stem

    print("\n==============================")
    print("Vehicle:", vehicle_id)

    series_dict, schema_info = read_requested_signal_series(
        parquet_path,
        overview,
    )

    for signal in REQUESTED_SIGNALS:
        n_rows = (
            len(series_dict[signal])
            if signal in series_dict
            else 0
        )

        availability_rows.append(
            {
                "vehicle_id": vehicle_id,
                "signal": signal,
                "available_raw": n_rows > 0,
                "n_raw_rows": int(n_rows),
            }
        )

    schema_rows.append(
        {
            "vehicle_id": vehicle_id,
            **schema_info,
            "signals_found": ",".join(
                sorted(series_dict.keys())
            ),
        }
    )

    print("Signals found:", sorted(series_dict.keys()))

    aligned = align_vehicle_to_soc(
        series_dict,
        vehicle_id,
    )

    segmented = add_driving_segments(
        aligned
    )

    print("Aligned SoC rows:", len(aligned))
    print("Retained driving rows:", len(segmented))
    print(
        "Retained segments:",
        segmented["segment_id"].nunique()
        if len(segmented)
        else 0,
    )

    if len(segmented):
        aligned_frames.append(segmented)

    del series_dict
    del aligned
    del segmented
    gc.collect()


schema_audit = pd.DataFrame(schema_rows)

raw_signal_availability = pd.DataFrame(
    availability_rows
)

schema_audit.to_csv(
    OUT9 / "tum_parquet_schema_audit.csv",
    index=False,
)

raw_signal_availability.to_csv(
    OUT9 / "tum_raw_signal_availability.csv",
    index=False,
)

display(schema_audit)

availability_pivot = (
    raw_signal_availability.pivot(
        index="vehicle_id",
        columns="signal",
        values="available_raw",
    )
)

display(availability_pivot)

for signal in PRIMARY_RAW_SIGNALS:
    assert raw_signal_availability.loc[
        raw_signal_availability["signal"] == signal,
        "available_raw",
    ].all(), (
        f"Primary common signal is not available on all seven vehicles: {signal}"
    )

cell_c_rate_support = raw_signal_availability.loc[
    raw_signal_availability["signal"] == "cell_c_rate",
    ["vehicle_id", "available_raw", "n_raw_rows"],
].copy()

print("Raw cell_c_rate availability:")
display(cell_c_rate_support)


raw = pd.concat(
    aligned_frames,
    ignore_index=True,
)

assert raw["vehicle_id"].nunique() == 7, (
    f"Expected 7 retained vehicles; got {raw['vehicle_id'].nunique()}"
)

print("Total retained rows:", f"{len(raw):,}")
print("Vehicles:", raw["vehicle_id"].nunique())
print("Segments:", raw["segment_id"].nunique())

# %% [markdown] 9) Segment-relative distance and causal features

# %% Code cell 16 (notebook cell 31)
def rolling_time_features(
    g,
    signal_col,
    history_s,
):
    x = (
        g[["time", signal_col]]
        .set_index("time")[signal_col]
    )

    r = x.rolling(
        f"{int(history_s)}s",
        min_periods=1,
    )

    out = pd.DataFrame(index=g.index)

    out[f"{signal_col}__last"] = (
        g[signal_col].to_numpy(float)
    )

    out[f"{signal_col}__mean"] = (
        r.mean().to_numpy()
    )

    out[f"{signal_col}__std"] = (
        r.std(ddof=0)
        .fillna(0.0)
        .to_numpy()
    )

    out[f"{signal_col}__min"] = (
        r.min().to_numpy()
    )

    out[f"{signal_col}__max"] = (
        r.max().to_numpy()
    )

    times = (
        g["time"]
        .astype("int64")
        .to_numpy()
    )

    values = g[signal_col].to_numpy(float)

    left_times = (
        times
        - int(history_s * 1e9)
    )

    left_idx = np.searchsorted(
        times,
        left_times,
        side="left",
    )

    oldest = values[left_idx]

    out[f"{signal_col}__delta"] = (
        values - oldest
    )

    return out


def build_segment_features(g):
    g = (
        g.sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    t_s = (
        g["time"]
        .astype("int64")
        .to_numpy()
        / 1e9
    )

    g["elapsed_s"] = t_s - t_s[0]

    d0 = float(g["distance_cum_km"].iloc[0])

    g["distance_km"] = (
        g["distance_cum_km"].to_numpy(float)
        - d0
    )

    # Numerical safety only.
    g["distance_km"] = np.maximum(
        g["distance_km"],
        0.0,
    )

    first_valid_soc = (
        g["hv_soc"]
        .dropna()
        .iloc[0]
    )

    g["soc0"] = float(first_valid_soc)

    feature_signals = [
        "hv_battery_voltage",
        "vehicle_speed",
        "distance_km",
        "elapsed_s",
        "soc0",
    ]

    feats = [
        rolling_time_features(
            g,
            s,
            HISTORY_S,
        )
        for s in feature_signals
    ]

    g = pd.concat(
        [
            g,
            pd.concat(feats, axis=1),
        ],
        axis=1,
    )

    g["soc_anchor"] = (
        g["hv_soc"].notna()
        & g["hv_soc"].shift().notna()
        & (
            (
                g["hv_soc"]
                - g["hv_soc"].shift()
            ).abs()
            > 1e-6
        )
    )

    g["soc_target"] = g["hv_soc"]

    return g


feature_frames = [
    build_segment_features(g)
    for _, g in raw.groupby(
        "segment_id",
        sort=False,
    )
]

feat = pd.concat(
    feature_frames,
    ignore_index=True,
)


DIRECT_FEATURES = [
    f"{s}__{stat}"
    for s in [
        "hv_battery_voltage",
        "vehicle_speed",
        "distance_km",
        "elapsed_s",
        "soc0",
    ]
    for stat in STAT_NAMES
]

print("Direct feature count:", len(DIRECT_FEATURES))
print("Genuine anchors:", int(feat["soc_anchor"].sum()))

# %% [markdown] 10) Chronological ~60/20/20 split by anchor mass per vehicle

# %% Code cell 17 (notebook cell 33)
def split_vehicle_segments_by_anchor_mass(vehicle_df):
    seg = (
        vehicle_df.groupby("segment_id")
        .agg(
            start_time=("time", "min"),
            n_anchor=("soc_anchor", "sum"),
        )
        .reset_index()
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    seg = seg[seg["n_anchor"] > 0].copy()

    assert len(seg) >= 3, (
        vehicle_df["vehicle_id"].iloc[0],
        len(seg),
    )

    total = float(seg["n_anchor"].sum())

    frac_before = (
        (
            seg["n_anchor"].cumsum()
            - seg["n_anchor"]
        )
        / total
    )

    seg["split"] = np.where(
        frac_before < 0.60,
        "train",
        np.where(
            frac_before < 0.80,
            "val",
            "test",
        ),
    )

    if set(seg["split"]) != {"train", "val", "test"}:
        n = len(seg)

        i1 = max(1, int(round(0.60 * n)))
        i2 = max(i1 + 1, int(round(0.80 * n)))
        i2 = min(i2, n - 1)

        seg["split"] = "test"
        seg.loc[: i1 - 1, "split"] = "train"
        seg.loc[i1 : i2 - 1, "split"] = "val"

    return seg[
        [
            "segment_id",
            "split",
            "start_time",
            "n_anchor",
        ]
    ]


split_maps = []

for veh, g in feat.groupby("vehicle_id"):
    sm = split_vehicle_segments_by_anchor_mass(g)
    sm["vehicle_id"] = veh
    split_maps.append(sm)

split_map = pd.concat(
    split_maps,
    ignore_index=True,
)

feat = feat.merge(
    split_map[
        [
            "vehicle_id",
            "segment_id",
            "split",
        ]
    ],
    on=[
        "vehicle_id",
        "segment_id",
    ],
    how="inner",
    validate="many_to_one",
)


split_audit = (
    feat[feat["soc_anchor"]]
    .groupby(
        [
            "vehicle_id",
            "split",
        ]
    )
    .agg(
        anchors=("soc_target", "size"),
        segments=("segment_id", "nunique"),
        first_time=("time", "min"),
        last_time=("time", "max"),
    )
    .reset_index()
)

split_audit.to_csv(
    OUT9 / "tum_split_audit.csv",
    index=False,
)

display(split_audit)

# %% [markdown] 11) Train-only distance-prior calibration

# %% Code cell 18 (notebook cell 35)
def fit_gamma_zero_intercept(a):
    x = a["distance_km"].to_numpy(float)

    y = (
        a["soc_target"].to_numpy(float)
        - a["soc0"].to_numpy(float)
    )

    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.01)
    )

    x = x[keep]
    y = y[keep]

    if len(x) < 20:
        raise RuntimeError(
            f"Too few training anchors for distance-prior gamma: {len(x)}"
        )

    denom = float(np.dot(x, x))

    if denom <= 0:
        raise RuntimeError(
            "Distance-prior gamma denominator is zero."
        )

    gamma = float(
        np.dot(x, y)
        / denom
    )

    return gamma, len(x)


gamma_rows = []

for veh, g in feat.groupby("vehicle_id"):
    a = g[
        (g["split"] == "train")
        & g["soc_anchor"]
    ].copy()

    gamma, n = fit_gamma_zero_intercept(a)

    gamma_rows.append(
        {
            "vehicle_id": veh,
            "gamma_pp_per_km": gamma,
            "n_train_anchors": n,
        }
    )


gamma_table = pd.DataFrame(gamma_rows)

gamma_map = dict(
    zip(
        gamma_table["vehicle_id"],
        gamma_table["gamma_pp_per_km"],
    )
)

feat["gamma_vehicle"] = (
    feat["vehicle_id"].map(gamma_map)
)

feat["soc_phy"] = (
    feat["soc0"]
    + feat["gamma_vehicle"]
    * feat["distance_km"]
)

feat["residual_target"] = (
    feat["soc_target"]
    - feat["soc_phy"]
)

gamma_table.to_csv(
    OUT9 / "tum_train_only_distance_gamma.csv",
    index=False,
)

display(gamma_table)

# %% [markdown] 12) Final genuine-anchor table and leakage checks

# %% Code cell 19 (notebook cell 37)
PHYSICS_AWARE_FEATURES = (
    DIRECT_FEATURES
    + ["soc_phy"]
)

anchor_df = feat[
    feat["soc_anchor"]
    & np.isfinite(feat["soc_target"])
].copy()


def all_finite(df, cols):
    return np.isfinite(
        df[cols].to_numpy(float)
    ).all(axis=1)


keep = (
    all_finite(
        anchor_df,
        DIRECT_FEATURES,
    )
    & np.isfinite(anchor_df["soc_phy"])
)

anchor_df = (
    anchor_df[keep]
    .reset_index(drop=True)
)

# Primary models must not contain target-derived contemporaneous SoC.
for c in DIRECT_FEATURES:
    assert "hv_soc" not in c.lower()
    assert "soc_target" not in c.lower()
    assert "residual_target" not in c.lower()

# Explicitly ensure the missing/optional C-rate cannot silently enter the primary model.
assert all(
    "cell_c_rate" not in c
    for c in DIRECT_FEATURES
)

assert anchor_df["vehicle_id"].nunique() == 7
assert set(anchor_df["split"].unique()) == {
    "train",
    "val",
    "test",
}

print("Usable genuine anchors:", len(anchor_df))
display(
    anchor_df.groupby(
        [
            "vehicle_id",
            "split",
        ]
    ).size().rename("anchors").reset_index()
)

# %% [markdown] Export TUM canonical table

# %% Code cell 20 (notebook cell 39)
t = anchor_df.copy()

TUM_X = []
canon_tum = pd.DataFrame({
    "domain": "TUM",
    "vehicle_id": t["vehicle_id"].astype(str),
    "segment_id": t["segment_id"].astype(str),
    "time_value": pd.to_datetime(t["time"], utc=True).astype("int64") / 1e9,
    "split": t["split"].astype(str),
    "target": t["soc_target"].astype(float),
    "soc0": t["soc0"].astype(float),
    "coord": t["distance_km"].astype(float),
    "soc_phy": t["soc_phy"].astype(float),
    "physics_delta": (
        t["soc_phy"].astype(float)
        - t["soc0"].astype(float)
    ),
    "elapsed_s": t["elapsed_s"].astype(float),
})

for c in DIRECT_FEATURES:
    new = f"x__{c}"
    canon_tum[new] = pd.to_numeric(t[c], errors="coerce").to_numpy(float)
    TUM_X.append(new)

save_canonical(
    canon_tum,
    "TUM",
    TUM_X,
    "distance_km",
    (
        "TUM/FTM seven-vehicle temporal validation; native-speed-integrated distance prior; "
        "vehicle-specific train-only pp/km slope; not Coulomb counting"
    ),
)

# %% [markdown] Final cross-domain audit

# %% Code cell 21 (notebook cell 41)
audit_rows = []

for domain in ["VED", "KU", "SCUT", "TUM"]:
    p = CANON / f"{domain}_canonical.parquet"
    m = CANON / f"{domain}_metadata.json"

    assert p.exists(), p
    assert m.exists(), m

    d = pd.read_parquet(p)

    for split, g in d.groupby("split"):
        audit_rows.append({
            "domain": domain,
            "split": split,
            "rows": len(g),
            "vehicles": g["vehicle_id"].nunique(),
            "segments": g["segment_id"].nunique(),
            "target_min": float(g["target"].min()),
            "target_max": float(g["target"].max()),
            "physics_MAE": float(
                np.mean(
                    np.abs(
                        g["soc_phy"].to_numpy(float)
                        - g["target"].to_numpy(float)
                    )
                )
            ),
        })

audit = pd.DataFrame(audit_rows)
audit.to_csv(
    OUT / "master" / "10A_canonical_domain_audit.csv",
    index=False,
)

display(audit)

print("10A canonical preparation completed.")
