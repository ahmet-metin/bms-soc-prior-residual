# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/9_TUMFTM_External_Validation_Colab_v2.ipynb
# Notebook SHA256: 4b199d7ed28d61f291b83ab286eac212333a981b24a04dceadca90f30674b117
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 9 v2 — TUM/FTM External Validation

# %% Code cell 1 (notebook cell 2)
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

# %% Code cell 2 (notebook cell 4)
from google.colab import drive

drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
OUT9 = ROOT / "outputs" / "external_validation_9_tum_ftm_v2"
OUT9.mkdir(parents=True, exist_ok=True)

LOCAL_REPO = Path("/content/tum_ev_uds_dataset")
LOCAL_ZIP = Path("/content/tum_ev_uds_dataset.zip")

print("Outputs:", OUT9)

# %% [markdown] 2) Download repository and verify the seven parquet files

# %% Code cell 3 (notebook cell 6)
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

# %% Code cell 4 (notebook cell 8)
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

# %% Code cell 5 (notebook cell 10)
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

# %% Code cell 6 (notebook cell 12)
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

# %% Code cell 7 (notebook cell 14)
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

# %% Code cell 8 (notebook cell 16)
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

# %% Code cell 9 (notebook cell 18)
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

# %% Code cell 10 (notebook cell 20)
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

# %% Code cell 11 (notebook cell 22)
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

# %% Code cell 12 (notebook cell 24)
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

# %% Code cell 13 (notebook cell 26)
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

# %% [markdown] 13) Metrics, ExtraTrees, and sparse-supervision helpers

# %% Code cell 14 (notebook cell 28)
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


def make_et(hp, seed, n_trees):
    return ExtraTreesRegressor(
        n_estimators=int(n_trees),
        min_samples_leaf=int(hp["min_samples_leaf"]),
        max_features=float(hp["max_features"]),
        random_state=int(seed),
        n_jobs=-1,
    )


def stable_vehicle_seed(vehicle_id):
    return (
        int(
            hashlib.sha1(
                str(vehicle_id).encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        # [NOTEBOOK MAGIC, MANUALLY SET UP] % 100000
    )


def sample_ml_supervision(
    train_df,
    fraction,
    seed,
):
    if fraction >= 0.999999:
        return train_df.copy()

    pieces = []

    for veh, g in train_df.groupby("vehicle_id"):
        n = len(g)

        k = max(
            1,
            int(round(float(fraction) * n)),
        )

        k = min(k, n)

        pieces.append(
            g.sample(
                n=k,
                random_state=int(seed) + stable_vehicle_seed(veh),
                replace=False,
            )
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def model_spec(model_name):
    if model_name == "Direct":
        return {
            "features": DIRECT_FEATURES,
            "target": "soc_target",
            "prediction_mode": "direct",
        }

    if model_name == "Direct+Physics":
        return {
            "features": PHYSICS_AWARE_FEATURES,
            "target": "soc_target",
            "prediction_mode": "direct",
        }

    if model_name == "Residual":
        return {
            "features": PHYSICS_AWARE_FEATURES,
            "target": "residual_target",
            "prediction_mode": "residual",
        }

    raise ValueError(model_name)


MODEL_NAMES = [
    "Direct",
    "Direct+Physics",
    "Residual",
]

# %% [markdown] 14) Validation-only model hyperparameter selection

# %% Code cell 15 (notebook cell 30)
train_all = anchor_df[
    anchor_df["split"] == "train"
].copy()

val_all = anchor_df[
    anchor_df["split"] == "val"
].copy()

test_all = anchor_df[
    anchor_df["split"] == "test"
].copy()


hp_rows = []
selected_hp = {}


for fraction in LABEL_FRACTIONS:
    train_sub = sample_ml_supervision(
        train_all,
        fraction,
        TUNING_SEED,
    )

    print(
        "\nLabel fraction:",
        fraction,
        "ML train anchors:",
        len(train_sub),
    )

    for model_name in MODEL_NAMES:
        spec = model_spec(model_name)

        candidate_rows = []

        for hp in ET_GRID:
            model = make_et(
                hp,
                TUNING_SEED,
                N_TREES_TUNE,
            )

            model.fit(
                train_sub[spec["features"]],
                train_sub[spec["target"]],
            )

            raw_pred = model.predict(
                val_all[spec["features"]]
            )

            if spec["prediction_mode"] == "residual":
                pred = (
                    val_all["soc_phy"].to_numpy(float)
                    + raw_pred
                )
            else:
                pred = raw_pred

            sc = regression_metrics(
                val_all["soc_target"],
                pred,
            )

            candidate_rows.append(
                {
                    "label_fraction": fraction,
                    "model": model_name,
                    **hp,
                    "val_MAE": sc["MAE"],
                    "val_RMSE": sc["RMSE"],
                }
            )

        cand = pd.DataFrame(
            candidate_rows
        ).sort_values(
            [
                "val_MAE",
                "min_samples_leaf",
                "max_features",
            ]
        )

        best = cand.iloc[0]

        selected_hp[
            (
                float(fraction),
                model_name,
            )
        ] = {
            "min_samples_leaf": int(
                best["min_samples_leaf"]
            ),
            "max_features": float(
                best["max_features"]
            ),
        }

        hp_rows.extend(candidate_rows)

        print(
            model_name,
            selected_hp[(float(fraction), model_name)],
            "val MAE=",
            float(best["val_MAE"]),
        )


hp_audit = pd.DataFrame(hp_rows)

hp_audit.to_csv(
    OUT9 / "tum_validation_hp_grid.csv",
    index=False,
)

with open(
    OUT9 / "tum_selected_hyperparameters.json",
    "w",
) as f:
    json.dump(
        {
            f"{fraction}_{model}": hp
            for (fraction, model), hp in selected_hp.items()
        },
        f,
        indent=2,
    )

# %% [markdown] 15) Frozen five-seed test evaluation

# %% Code cell 16 (notebook cell 32)
test_rows = []
prediction_frames = []

physics_sc = regression_metrics(
    test_all["soc_target"],
    test_all["soc_phy"],
)


for fraction in LABEL_FRACTIONS:
    for seed in FINAL_SEEDS:
        test_rows.append(
            {
                "label_fraction": fraction,
                "seed": seed,
                "model": "Physics",
                "n_train_labels": len(train_all),
                "n_test_anchors": len(test_all),
                **physics_sc,
            }
        )

    for seed in FINAL_SEEDS:
        train_sub = sample_ml_supervision(
            train_all,
            fraction,
            seed,
        )

        for model_name in MODEL_NAMES:
            spec = model_spec(model_name)

            hp = selected_hp[
                (
                    float(fraction),
                    model_name,
                )
            ]

            model = make_et(
                hp,
                seed,
                N_TREES_FINAL,
            )

            model.fit(
                train_sub[spec["features"]],
                train_sub[spec["target"]],
            )

            raw_pred = model.predict(
                test_all[spec["features"]]
            )

            if spec["prediction_mode"] == "residual":
                pred = (
                    test_all["soc_phy"].to_numpy(float)
                    + raw_pred
                )
            else:
                pred = raw_pred

            sc = regression_metrics(
                test_all["soc_target"],
                pred,
            )

            test_rows.append(
                {
                    "label_fraction": fraction,
                    "seed": seed,
                    "model": model_name,
                    "n_train_labels": len(train_sub),
                    "n_test_anchors": len(test_all),
                    **sc,
                }
            )

            p = test_all[
                [
                    "vehicle_id",
                    "segment_id",
                    "time",
                    "soc_target",
                    "soc_phy",
                ]
            ].copy()

            p["prediction"] = pred
            p["label_fraction"] = fraction
            p["seed"] = seed
            p["model"] = model_name

            prediction_frames.append(p)

            print(
                fraction,
                seed,
                model_name,
                "MAE",
                sc["MAE"],
            )


test_runs = pd.DataFrame(test_rows)

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

test_runs.to_csv(
    OUT9 / "tum_test_runs.csv",
    index=False,
)

predictions.to_parquet(
    OUT9 / "tum_test_predictions.parquet",
    index=False,
)

display(test_runs.head(20))

# %% [markdown] 16) Headline, formulation, and label-efficiency summaries

# %% Code cell 17 (notebook cell 34)
headline = (
    test_runs.groupby(
        [
            "label_fraction",
            "model",
        ],
        as_index=False,
    )
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        P95_mean=("P95", "mean"),
        Bias_mean=("Bias", "mean"),
        n_train_labels=("n_train_labels", "first"),
        n_test_anchors=("n_test_anchors", "first"),
    )
)

headline.to_csv(
    OUT9 / "tum_headline_summary.csv",
    index=False,
)

display(headline)


comparison_rows = []

for fraction in LABEL_FRACTIONS:
    sub = test_runs[
        test_runs["label_fraction"] == fraction
    ].copy()

    physics_mae = float(
        sub.loc[
            sub["model"] == "Physics",
            "MAE",
        ].mean()
    )

    seed_mae = {
        model: (
            sub[sub["model"] == model]
            .set_index("seed")["MAE"]
            .sort_index()
        )
        for model in MODEL_NAMES
    }

    for model_name in MODEL_NAMES:
        model_mae = float(
            seed_mae[model_name].mean()
        )

        comparison_rows.append(
            {
                "label_fraction": fraction,
                "comparison": model_name,
                "physics_MAE": physics_mae,
                "model_MAE": model_mae,
                "improvement_vs_physics_pct": (
                    100.0
                    * (physics_mae - model_mae)
                    / physics_mae
                ),
                "wins_5_seed": np.nan,
            }
        )

    comparison_rows.extend(
        [
            {
                "label_fraction": fraction,
                "comparison": "Residual_vs_Direct",
                "physics_MAE": physics_mae,
                "model_MAE": float(
                    seed_mae["Residual"].mean()
                ),
                "improvement_vs_physics_pct": np.nan,
                "wins_5_seed": int(
                    (
                        seed_mae["Residual"]
                        < seed_mae["Direct"]
                    ).sum()
                ),
            },
            {
                "label_fraction": fraction,
                "comparison": "Residual_vs_Direct+Physics",
                "physics_MAE": physics_mae,
                "model_MAE": float(
                    seed_mae["Residual"].mean()
                ),
                "improvement_vs_physics_pct": np.nan,
                "wins_5_seed": int(
                    (
                        seed_mae["Residual"]
                        < seed_mae["Direct+Physics"]
                    ).sum()
                ),
            },
        ]
    )


formulation_comparison = pd.DataFrame(
    comparison_rows
)

formulation_comparison.to_csv(
    OUT9 / "tum_formulation_comparison.csv",
    index=False,
)

display(formulation_comparison)


label_eff_rows = []

for model_name in MODEL_NAMES:
    full = headline[
        (headline["label_fraction"] == 1.0)
        & (headline["model"] == model_name)
    ].iloc[0]

    sparse = headline[
        (headline["label_fraction"] == 0.10)
        & (headline["model"] == model_name)
    ].iloc[0]

    label_eff_rows.append(
        {
            "model": model_name,
            "full_MAE": float(full["MAE_mean"]),
            "10pct_MAE": float(sparse["MAE_mean"]),
            "degradation_pct": (
                100.0
                * (
                    float(sparse["MAE_mean"])
                    - float(full["MAE_mean"])
                )
                / float(full["MAE_mean"])
            ),
        }
    )


label_efficiency = pd.DataFrame(label_eff_rows)

label_efficiency.to_csv(
    OUT9 / "tum_label_efficiency.csv",
    index=False,
)

display(label_efficiency)

# %% [markdown] 17) Per-vehicle results

# %% Code cell 18 (notebook cell 36)
vehicle_rows = []

for (
    fraction,
    seed,
    model_name,
), g in predictions.groupby(
    [
        "label_fraction",
        "seed",
        "model",
    ]
):
    for veh, h in g.groupby("vehicle_id"):
        sc = regression_metrics(
            h["soc_target"],
            h["prediction"],
        )

        vehicle_rows.append(
            {
                "label_fraction": fraction,
                "seed": seed,
                "model": model_name,
                "vehicle_id": veh,
                "n_test_anchors": len(h),
                **sc,
            }
        )


for fraction in LABEL_FRACTIONS:
    for seed in FINAL_SEEDS:
        for veh, h in test_all.groupby("vehicle_id"):
            sc = regression_metrics(
                h["soc_target"],
                h["soc_phy"],
            )

            vehicle_rows.append(
                {
                    "label_fraction": fraction,
                    "seed": seed,
                    "model": "Physics",
                    "vehicle_id": veh,
                    "n_test_anchors": len(h),
                    **sc,
                }
            )


vehicle_runs = pd.DataFrame(vehicle_rows)

vehicle_summary = (
    vehicle_runs.groupby(
        [
            "label_fraction",
            "model",
            "vehicle_id",
        ],
        as_index=False,
    )
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        P95_mean=("P95", "mean"),
        n_test_anchors=("n_test_anchors", "first"),
    )
)

vehicle_runs.to_csv(
    OUT9 / "tum_vehicle_runs.csv",
    index=False,
)

vehicle_summary.to_csv(
    OUT9 / "tum_vehicle_summary.csv",
    index=False,
)

display(vehicle_summary)

# %% [markdown] 18) Paired segment bootstrap

# %% Code cell 19 (notebook cell 38)
ml_pred_avg = (
    predictions.groupby(
        [
            "label_fraction",
            "model",
            "vehicle_id",
            "segment_id",
            "time",
            "soc_target",
        ],
        as_index=False,
    )["prediction"]
    .mean()
)


def segment_bootstrap_mae_delta(
    df_a,
    df_b,
    n_boot=2000,
    seed=2026,
):
    key = [
        "vehicle_id",
        "segment_id",
        "time",
        "soc_target",
    ]

    a = df_a[
        key + ["prediction"]
    ].rename(
        columns={"prediction": "pred_a"}
    )

    b = df_b[
        key + ["prediction"]
    ].rename(
        columns={"prediction": "pred_b"}
    )

    m = a.merge(
        b,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    assert len(m) > 0, "No paired test predictions."

    m["ae_a"] = np.abs(
        m["pred_a"] - m["soc_target"]
    )

    m["ae_b"] = np.abs(
        m["pred_b"] - m["soc_target"]
    )

    seg_stats = (
        m.groupby(
            [
                "vehicle_id",
                "segment_id",
            ]
        )
        .agg(
            sum_ae_a=("ae_a", "sum"),
            sum_ae_b=("ae_b", "sum"),
            n=("ae_a", "size"),
        )
        .reset_index()
    )

    nseg = len(seg_stats)

    assert nseg >= 2, (
        f"Too few test segments for bootstrap: {nseg}"
    )

    rng = np.random.default_rng(seed)

    draws = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            nseg,
            size=nseg,
        )

        s = seg_stats.iloc[idx]

        mae_a = (
            s["sum_ae_a"].sum()
            / s["n"].sum()
        )

        mae_b = (
            s["sum_ae_b"].sum()
            / s["n"].sum()
        )

        draws[i] = mae_a - mae_b

    point = float(
        (
            seg_stats["sum_ae_a"].sum()
            - seg_stats["sum_ae_b"].sum()
        )
        / seg_stats["n"].sum()
    )

    return {
        "delta_MAE_A_minus_B": point,
        "CI2.5": float(np.percentile(draws, 2.5)),
        "CI97.5": float(np.percentile(draws, 97.5)),
        "P_A_better": float(np.mean(draws < 0)),
        "n_segments": nseg,
    }


bootstrap_rows = []

for fraction in LABEL_FRACTIONS:
    for a_name, b_name in [
        ("Residual", "Direct"),
        ("Residual", "Direct+Physics"),
        ("Direct+Physics", "Direct"),
    ]:
        a = ml_pred_avg[
            (ml_pred_avg["label_fraction"] == fraction)
            & (ml_pred_avg["model"] == a_name)
        ]

        b = ml_pred_avg[
            (ml_pred_avg["label_fraction"] == fraction)
            & (ml_pred_avg["model"] == b_name)
        ]

        boot = segment_bootstrap_mae_delta(
            a,
            b,
            n_boot=2000,
            seed=(
                2026
                + int(fraction * 100)
                + len(a_name)
            ),
        )

        bootstrap_rows.append(
            {
                "label_fraction": fraction,
                "A": a_name,
                "B": b_name,
                **boot,
            }
        )


bootstrap_summary = pd.DataFrame(
    bootstrap_rows
)

bootstrap_summary.to_csv(
    OUT9 / "tum_segment_bootstrap_formulation_deltas.csv",
    index=False,
)

display(bootstrap_summary)

# %% [markdown] 19) Dataset diagnostics and descriptive winner

# %% Code cell 20 (notebook cell 40)
diagnostic_rows = []

for veh, g in anchor_df.groupby("vehicle_id"):
    soc_values = (
        g["soc_target"]
        .dropna()
        .to_numpy(float)
    )

    unique_sorted = np.sort(
        np.unique(soc_values)
    )

    positive_diffs = np.diff(
        unique_sorted
    )

    positive_diffs = positive_diffs[
        positive_diffs > 1e-9
    ]

    diagnostic_rows.append(
        {
            "vehicle_id": veh,
            "n_anchors": len(g),
            "n_segments": g["segment_id"].nunique(),
            "soc_min": float(np.min(soc_values)),
            "soc_max": float(np.max(soc_values)),
            "soc_unique_values": int(len(unique_sorted)),
            "soc_min_positive_resolution_pp": (
                float(np.min(positive_diffs))
                if len(positive_diffs)
                else np.nan
            ),
            "soc_all_integer": bool(
                np.allclose(
                    soc_values,
                    np.round(soc_values),
                    atol=1e-9,
                )
            ),
            "median_hv_voltage": float(
                np.nanmedian(
                    g["hv_battery_voltage"]
                )
            ),
            "median_speed_kmh": float(
                np.nanmedian(
                    g["vehicle_speed"]
                )
            ),
            "median_segment_distance_km": float(
                np.nanmedian(
                    g["distance_km"]
                )
            ),
        }
    )


dataset_diagnostics = pd.DataFrame(
    diagnostic_rows
)

dataset_diagnostics.to_csv(
    OUT9 / "tum_dataset_diagnostics.csv",
    index=False,
)

display(dataset_diagnostics)


winner_rows = []

for fraction in LABEL_FRACTIONS:
    h = headline[
        (headline["label_fraction"] == fraction)
        & headline["model"].isin(MODEL_NAMES)
    ].copy()

    best = h.sort_values("MAE_mean").iloc[0]

    winner_rows.append(
        {
            "label_fraction": fraction,
            "lowest_MAE_formulation": best["model"],
            "lowest_MAE": float(best["MAE_mean"]),
            "note": (
                "Descriptive external result; accepted regardless of formulation."
            ),
        }
    )


winner_summary = pd.DataFrame(
    winner_rows
)

winner_summary.to_csv(
    OUT9 / "tum_descriptive_formulation_winner.csv",
    index=False,
)

display(winner_summary)

# %% [markdown] 20) Figures

# %% Code cell 21 (notebook cell 42)
plt.figure(figsize=(9.5, 5.8))

for model_name in [
    "Physics",
    "Direct",
    "Direct+Physics",
    "Residual",
]:
    g = headline[
        headline["model"] == model_name
    ].sort_values(
        "label_fraction",
        ascending=False,
    )

    assert len(g) == 2, (
        model_name,
        g[
            [
                "label_fraction",
                "MAE_mean",
            ]
        ],
    )

    plt.plot(
        ["Full", "10%"],
        g["MAE_mean"].to_numpy(),
        marker="o",
        label=model_name,
    )

plt.ylabel(
    "BMS-reported SoC MAE [percentage points]"
)

plt.title(
    "TUM/FTM External Validation v2 — Formulation and Label Efficiency"
)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT9 / "tum_formulation_label_efficiency.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()


full_vehicle = vehicle_summary[
    vehicle_summary["label_fraction"] == 1.0
].copy()

pivot = full_vehicle.pivot(
    index="vehicle_id",
    columns="model",
    values="MAE_mean",
)

pivot = pivot[
    [
        c
        for c in [
            "Physics",
            "Direct",
            "Direct+Physics",
            "Residual",
        ]
        if c in pivot.columns
    ]
]

ax = pivot.plot(
    kind="bar",
    figsize=(11, 5.8),
)

ax.set_ylabel(
    "BMS-reported SoC MAE [percentage points]"
)

ax.set_title(
    "TUM/FTM External Validation v2 — Per-Vehicle Full-Label MAE"
)

ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUT9 / "tum_per_vehicle_full_label_mae.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()

# %% [markdown] 21) Integrity checks

# %% Code cell 22 (notebook cell 44)
assert anchor_df["vehicle_id"].nunique() == 7

for veh, g in anchor_df.groupby("vehicle_id"):
    assert set(g["split"].unique()) == {
        "train",
        "val",
        "test",
    }

    train_max = g.loc[
        g["split"] == "train",
        "time",
    ].max()

    val_min = g.loc[
        g["split"] == "val",
        "time",
    ].min()

    test_min = g.loc[
        g["split"] == "test",
        "time",
    ].min()

    assert train_max <= val_min <= test_min


assert (
    gamma_table["n_train_anchors"] > 0
).all()

assert np.isfinite(
    gamma_table["gamma_pp_per_km"].to_numpy(float)
).all()

assert (
    anchor_df["distance_km"] >= 0
).all()

assert np.isfinite(
    test_runs[
        [
            "MAE",
            "RMSE",
            "P95",
            "Bias",
        ]
    ].to_numpy(float)
).all()


n_full = int(
    test_runs.loc[
        (test_runs["label_fraction"] == 1.0)
        & (test_runs["model"] == "Residual"),
        "n_train_labels",
    ].iloc[0]
)

n_sparse = int(
    test_runs.loc[
        (test_runs["label_fraction"] == 0.10)
        & (test_runs["model"] == "Residual"),
        "n_train_labels",
    ].iloc[0]
)

assert n_sparse < n_full

# No contemporaneous SoC-derived variable enters a primary ML feature.
for feature in DIRECT_FEATURES:
    low = feature.lower()

    assert "hv_soc" not in low
    assert "soc_target" not in low
    assert "hv_dod" not in low
    assert "cell_c_rate" not in low

print("All Notebook 9 v2 integrity checks passed.")

# %% [markdown] 22) Experiment manifest

# %% Code cell 23 (notebook cell 46)
manifest = {
    "experiment": "TUM_FTM_external_validation_v2",
    "dataset_repository": "TUMFTM/electric-vehicle-uds-dataset",
    "dataset_license": "CC BY-NC 4.0 according to repository",
    "associated_paper": (
        "Schreiber et al., Lab-to-field gap in battery aging studies, "
        "eTransportation 27 (2026) 100518"
    ),
    "doi": "10.1016/j.etran.2025.100518",
    "vehicles_expected": 7,
    "vehicles_used": int(
        anchor_df["vehicle_id"].nunique()
    ),
    "target": "BMS-reported hv_soc",
    "supervision_policy": "genuine SoC-change anchors only",
    "primary_raw_signals": PRIMARY_RAW_SIGNALS,
    "cell_c_rate_policy": (
        "raw availability audited; excluded from primary all-seven experiment"
    ),
    "temperature_in_primary_models": False,
    "physics_coordinate": (
        "native-speed-integrated segment distance in km"
    ),
    "physics_model": (
        "SOC_phy = SOC0 + gamma_vehicle_pp_per_km * distance_km"
    ),
    "gamma_policy": (
        "per-vehicle, zero-intercept, full training genuine anchors only"
    ),
    "physics_interpretation_boundary": (
        "distance prior is a kinematic field prior, not Coulomb counting; "
        "gamma is not capacity or SoH"
    ),
    "split": (
        "chronological approximately 60/20/20 by per-vehicle segment anchor mass"
    ),
    "models": [
        "Physics",
        "Direct ExtraTrees",
        "Direct+Physics ExtraTrees",
        "Physics-Residual ExtraTrees",
    ],
    "label_fractions": LABEL_FRACTIONS,
    "sparse_protocol": (
        "10% ML training anchors within vehicle; physics gamma uses full training anchors"
    ),
    "final_seeds": FINAL_SEEDS,
    "hyperparameter_selection": "validation only",
    "inference_bootstrap": (
        "paired test driving-segment bootstrap, 2000 draws"
    ),
    "outcome_policy": (
        "Report external result regardless of which formulation wins."
    ),
    "target_warning": (
        "Target is BMS-reported SoC, not electrochemical ground truth."
    ),
    "v1_bug_fixed": (
        "v1 incorrectly required cell_c_rate in every raw parquet; "
        "CUP1 runtime showed it was absent"
    ),
}

with open(
    OUT9 / "tum_experiment_manifest.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )

print(json.dumps(manifest, indent=2))

# %% [markdown] Main outputs
