# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/12B_v4_EVE_Frozen_TrainVal_Model_Freeze_Colab.ipynb
# Notebook SHA256: e79f29ce9c7713adf5302a82e712ebc73aa5eea13c13c6f55d2d0860f58f92c8
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 12B-v4 — EVE Frozen Train/Validation Model Preparation

# %% Code cell 1 (notebook cell 2)
# Colab: GPU recommended for the temporal baselines.
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import os, gc, json, math, time, random, hashlib, warnings
import traceback, contextlib, re, pickle, shutil
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")

A12 = ROOT / "outputs" / "eve_12A_untouched_audit"
OUT = ROOT / "outputs" / "eve_12B_v4_frozen_trainval"
RAW12B = ROOT / "data" / "EVE_Pilot" / "12B_v4_channels"
CACHE = OUT / "channel_cache_trainval"
CTX_OUT = OUT / "context"
K10_OUT = OUT / "10K"
K10_BUNDLES = K10_OUT / "bundles"
DEEP_OUT = OUT / "deep"
DEEP_CKPT = DEEP_OUT / "checkpoints"

for p in [
    OUT, RAW12B, CACHE, CTX_OUT,
    K10_OUT, K10_BUNDLES,
    DEEP_OUT, DEEP_CKPT,
]:
    p.mkdir(parents=True, exist_ok=True)

# 12A inputs
MANIFEST_12A = A12 / "12A_manifest.json"
SEAL_12A = A12 / "12A_CONFIRMATORY_TEST_SEAL.json"
BLACKLIST_12A = A12 / "12A_EVE_leakage_blacklist.json"
PRIOR_GATE_12A = A12 / "12A_prior_gate_trainval_predictor_only.json"
CATALOG_12A = A12 / "12A_official_file_catalog.csv"
TARGET_TRAIN = A12 / "EVE_target_train.parquet"
TARGET_VAL = A12 / "EVE_target_validation.parquet"
SEALED_TEST = A12 / "EVE_target_CONFIRMATORY_TEST_SEALED.parquet"

# Reusable sources created before this v4 run.
V3_CACHE = (
    ROOT / "outputs" / "eve_12B_v3_frozen_trainval"
    / "channel_cache_trainval"
)
V2_CACHE = (
    ROOT / "outputs" / "eve_12B_v2_frozen_trainval"
    / "channel_cache_trainval"
)
A12_SELECTED_RAW = (
    ROOT / "data" / "EVE_Pilot" / "selected_channels"
)

# Existing development artifacts — candidate families are not invented on EVE.
PROTOCOL_10B = (
    ROOT / "outputs" / "model_screening_10"
    / "master" / "10B_selected_validation_protocols.csv"
)
PROTOCOL_10C = (
    ROOT / "outputs" / "sf_pef_v2_10C"
    / "10C_selected_protocols.csv"
)

for p in [
    MANIFEST_12A, SEAL_12A, BLACKLIST_12A,
    PRIOR_GATE_12A, CATALOG_12A,
    TARGET_TRAIN, TARGET_VAL, SEALED_TEST,
    PROTOCOL_10B, PROTOCOL_10C,
]:
    assert p.exists(), f"Missing required input: {p}"

DATAVERSE = "https://entrepot.recherche.data.gouv.fr"

TRAIN_END_EXCLUSIVE = pd.Timestamp("2022-08-10 00:00:00")
VAL_END_EXCLUSIVE = pd.Timestamp("2023-03-15 00:00:00")

LABEL_FRACTIONS = [1.0, 0.10]
FINAL_SEEDS = [42, 43, 44, 45, 46]
TUNING_SEED = 777

# Frozen acquisition / alignment rules from 12A.
BURST_BREAK_S = 300.0
ALIGN_TOLERANCE_S = 30.0
CURRENT_SIGN_DOCUMENTATION = (
    "EVE data paper describes battery-to-motor as positive; "
    "v4 does not hard-code that orientation and instead uses raw signed current "
    "with train-only beta calibration."
)

# Frozen 10K constants from 10K.
RIDGE_A = 5.0
RIDGE_B = 2.0
SLOW_A_CLIP = (-1.5, 1.5)
SLOW_B_CLIP = (-10.0, 10.0)
SLOW_HP = {"min_samples_leaf": 4, "max_features": 0.7}
OOF_FOLDS = 3
N_TREES_SLOW = 100
N_TREES_FAST_TUNE = 80
N_TREES_FAST_FINAL = 120
N_TREES_ROUTER = 120
ROUTER_HP = {"min_samples_leaf": 6, "max_features": 0.7}
EPS_ADV = 0.05
N_STABILITY_BLOCKS = 3
STABILITY_EPS = 1e-12

ROUTER_GRID_FINAL = [
    {"mode": "Routed", "kappa": 0.0, "margin": 1e9, "sf_weight": 0.00},
    {"mode": "Routed", "kappa": 0.0, "margin": 0.05, "sf_weight": 0.50},
    {"mode": "Routed", "kappa": 0.5, "margin": 0.05, "sf_weight": 0.50},
    {"mode": "Routed", "kappa": 1.0, "margin": 0.05, "sf_weight": 0.50},
    {"mode": "Routed", "kappa": 0.5, "margin": 0.10, "sf_weight": 0.75},
    {"mode": "Routed", "kappa": 1.0, "margin": 0.10, "sf_weight": 0.75},
    {"mode": "Routed", "kappa": 0.5, "margin": 0.20, "sf_weight": 1.00},
    {"mode": "Routed", "kappa": 1.0, "margin": 0.20, "sf_weight": 1.00},
]

# Frozen temporal-baseline architecture / optimization contract from 11C-v2.
DEEP_MODELS = ["TCN", "LSTM", "GRU"]
SEQ_LEN = 16
HIDDEN = 64
TCN_CHANNELS = 64
TCN_DILATIONS = [1, 2, 4]
TCN_KERNEL = 3
DROPOUT = 0.10
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MIN_DELTA = 1e-5
BATCH_SIZE_GPU = 512
BATCH_SIZE_CPU = 128
RESUME_SAVE_EVERY = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = BATCH_SIZE_GPU if DEVICE.type == "cuda" else BATCH_SIZE_CPU
AMP_ENABLED = DEVICE.type == "cuda"

np.random.seed(42)
random.seed(42)

print("Device:", DEVICE)
print("12B output:", OUT)
print("Confirmatory target path exists but will NOT be read:", SEALED_TEST)

# %% [markdown] 1. 12A preflight — confirmatory target remains sealed

# %% Code cell 2 (notebook cell 4)
with open(MANIFEST_12A) as f:
    manifest12a = json.load(f)
with open(SEAL_12A) as f:
    seal12a = json.load(f)
with open(BLACKLIST_12A) as f:
    blacklist12a = json.load(f)
with open(PRIOR_GATE_12A) as f:
    prior_gate12a = json.load(f)

assert manifest12a["confirmatory_status"] == "UNTOUCHED_NO_MODEL_SCORE_COMPUTED"
assert manifest12a["selected_prior_from_trainval_predictor_gate"] == "SIGNED_AH"
assert manifest12a["test_tuning_allowed"] is False
assert manifest12a["primary_target"]["real_state_of_charge_used_as_predictor"] is False

assert prior_gate12a["signed_ah_eligible"] is True
assert prior_gate12a["predeclared_selected_prior"] == "SIGNED_AH"

assert "Battery/Real State of Charge.tab" in (
    blacklist12a["target_like_forbidden_repository_paths"]
)
assert "Battery/State of Charge.tab" in (
    blacklist12a["target_like_forbidden_repository_paths"]
)

assert int(seal12a["n_timestamped_rows"]) == 48838
assert seal12a["start_inclusive"] == "2023-03-15T00:00:00"
assert Path(seal12a["file"]) == SEALED_TEST

# IMPORTANT:
# We deliberately do NOT hash SEALED_TEST here, because hashing would still
# open/read its bytes. The expected hash is inherited from 12A metadata only.
EXPECTED_TEST_SHA256 = seal12a["sha256"]

preflight = {
    "12A_confirmatory_status": manifest12a["confirmatory_status"],
    "prior": prior_gate12a["predeclared_selected_prior"],
    "sealed_test_rows": int(seal12a["n_timestamped_rows"]),
    "sealed_test_expected_sha256_from_12A": EXPECTED_TEST_SHA256,
    "sealed_test_opened_in_12B": False,
    "test_tuning_allowed": False,
}

with open(OUT / "12B_preflight.json", "w") as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% [markdown] 2. Frozen candidate families inherited from earlier development

# %% Code cell 3 (notebook cell 6)
protocol10b = pd.read_csv(PROTOCOL_10B)
protocol10c = pd.read_csv(PROTOCOL_10C)

required10b = {
    "fraction",
    "residual_min_samples_leaf",
    "residual_max_features",
}
required10c = {
    "fraction",
    "early_window_s",
    "slow_strength",
    "tau_pp",
}
assert required10b.issubset(protocol10b.columns)
assert required10c.issubset(protocol10c.columns)

residual_candidate_rows = []
sf_candidate_rows = []

RESIDUAL_CANDIDATES = {}
SF_CANDIDATES = {}

for fraction in LABEL_FRACTIONS:
    r = (
        protocol10b[
            np.isclose(protocol10b["fraction"].astype(float), float(fraction))
        ][[
            "residual_min_samples_leaf",
            "residual_max_features",
        ]]
        .drop_duplicates()
        .sort_values([
            "residual_min_samples_leaf",
            "residual_max_features",
        ])
    )

    RESIDUAL_CANDIDATES[float(fraction)] = [
        {
            "min_samples_leaf": int(x.residual_min_samples_leaf),
            "max_features": float(x.residual_max_features),
        }
        for x in r.itertuples(index=False)
    ]

    s = (
        protocol10c[
            np.isclose(protocol10c["fraction"].astype(float), float(fraction))
        ][[
            "early_window_s",
            "slow_strength",
            "tau_pp",
        ]]
        .drop_duplicates()
        .sort_values([
            "early_window_s",
            "slow_strength",
            "tau_pp",
        ])
    )

    SF_CANDIDATES[float(fraction)] = [
        {
            "early_window_s": float(x.early_window_s),
            "slow_strength": float(x.slow_strength),
            "tau_pp": float(x.tau_pp),
        }
        for x in s.itertuples(index=False)
    ]

    assert RESIDUAL_CANDIDATES[float(fraction)]
    assert SF_CANDIDATES[float(fraction)]

    for x in RESIDUAL_CANDIDATES[float(fraction)]:
        residual_candidate_rows.append({
            "fraction": float(fraction),
            **x,
        })
    for x in SF_CANDIDATES[float(fraction)]:
        sf_candidate_rows.append({
            "fraction": float(fraction),
            **x,
        })

pd.DataFrame(residual_candidate_rows).to_csv(
    OUT / "12B_frozen_residual_candidate_pool.csv",
    index=False,
)
pd.DataFrame(sf_candidate_rows).to_csv(
    OUT / "12B_frozen_sf_candidate_pool.csv",
    index=False,
)

MAX_EARLY_WINDOW_S = max(
    x["early_window_s"]
    for configs in SF_CANDIDATES.values()
    for x in configs
)

print("Residual candidates:", RESIDUAL_CANDIDATES)
print("SF candidates:", SF_CANDIDATES)
print("Maximum inherited early window:", MAX_EARLY_WINDOW_S)

# %% [markdown] 3. Predeclared raw physical feature contract

# %% Code cell 4 (notebook cell 8)
catalog = pd.read_csv(CATALOG_12A)

# Exact catalog contract from EVE dataset version 3.2.
# These names are taken from the official 12A Dataverse catalog.
REQUIRED_CHANNELS = {
    "current_a": {
        "path": "Battery/HV LBC current measure.tab",
        "file_id": 627211,
        "unit": "A",
    },
    "voltage_v": {
        "path": "Battery/HV LBC voltage measure.tab",
        "file_id": 627172,
        "unit": "V",
    },
    "battery_temp_c": {
        "path": "Battery/HV Battery Temp.tab",
        "file_id": 627280,
        "unit": "degC",
    },
    "speed_kmh": {
        "path": "Automotive/Speed.tab",
        "file_id": 627272,
        "unit": "km/h",
    },
    "peb_torque_nm": {
        "path": "Automotive/PEBTorque.tab",
        "file_id": 627177,
        "unit": "N.m",
    },
    "hydraulic_brake_torque_req_nm": {
        "path": "Automotive/Total Hydraulic brake wheels torque request.tab",
        "file_id": 627252,
        "unit": "N.m",
    },
    "distance_km": {
        "path": "Automotive/Total vehicle distance.tab",
        "file_id": 627237,
        "unit": "km",
    },
}

forbidden_paths = {
    str(x).lower()
    for x in blacklist12a["target_like_forbidden_repository_paths"]
}

catalog["path_lower"] = catalog["path"].astype(str).str.lower()

resolved_rows = {}

for feature, spec in REQUIRED_CHANNELS.items():
    q = catalog[
        catalog["path_lower"].eq(spec["path"].lower())
    ].copy()

    assert len(q) == 1, (
        "Required official EVE channel missing or ambiguous",
        feature,
        spec["path"],
        q[["file_id", "path"]].to_dict("records"),
    )

    row = q.iloc[0].to_dict()

    assert int(row["file_id"]) == int(spec["file_id"]), (
        "EVE file ID changed from the 12A-frozen dataset catalog",
        feature,
        spec["file_id"],
        row["file_id"],
    )

    assert str(row["path"]).lower() not in forbidden_paths
    assert "state of charge" not in str(row["path"]).lower()
    assert "soc" not in Path(str(row["path"])).stem.lower()

    md5 = str(row.get("md5", "")).strip().lower()
    assert re.fullmatch(r"[0-9a-f]{32}", md5), (
        "Missing/invalid official MD5",
        feature,
        md5,
    )

    row["declared_unit"] = spec["unit"]
    resolved_rows[feature] = row

feature_contract = pd.DataFrame([
    {
        "feature": feature,
        "repository_path": row["path"],
        "file_id": int(row["file_id"]),
        "declared_unit": row["declared_unit"],
        "filesize_bytes": int(row["filesize_bytes"]),
        "official_md5": row["md5"],
        "selection_basis": "predeclared direct physical signal; exact 12A catalog match",
    }
    for feature, row in resolved_rows.items()
])

feature_contract.to_csv(
    OUT / "12B_v4_raw_feature_contract.csv",
    index=False,
)

assert len(feature_contract) == 7
assert feature_contract["file_id"].nunique() == 7

display(feature_contract)

# %% [markdown] 4. Download and cache train+validation predictor channels

# %% Code cell 5 (notebook cell 10)
def md5_file(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalized_file_stem(path):
    name = Path(str(path)).name
    stem = re.sub(r"\.(tab|csv|tsv|txt)$", "", name, flags=re.I)
    return re.sub(r"\s+", " ", stem.strip().lower())


def find_12a_local_channel(repository_path):
    """
    Find a transformed/raw copy created by 12A in selected_channels.

    Dataverse may expose a tabular .tab source as a downloaded .csv, so matching
    is by normalized basename stem rather than extension. This fallback is used
    only for channels already downloaded by 12A.
    """
    if not A12_SELECTED_RAW.exists():
        return None

    wanted = normalized_file_stem(repository_path)

    candidates = []
    for p in A12_SELECTED_RAW.iterdir():
        if p.is_file() and normalized_file_stem(p.name) == wanted:
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous 12A local fallback for {repository_path}: {candidates}"
        )

    return None


def retry_download_original(
    file_id,
    out_dir,
    fallback_name,
    expected_md5,
    max_attempts=6,
):
    """
    Download an original Dataverse file with DNS/connection retry.

    Only successfully downloaded original bytes are checked against the official
    catalog MD5. A partially downloaded file is deleted before retrying.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / fallback_name

    if dest.exists() and dest.stat().st_size > 0:
        got = md5_file(dest)
        if got.lower() == str(expected_md5).lower():
            return dest
        dest.unlink()

    url = f"{DATAVERSE}/api/access/datafile/{int(file_id)}"
    waits = [2, 5, 10, 20, 40, 60]
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        tmp = dest.with_suffix(dest.suffix + ".partial")
        tmp.unlink(missing_ok=True)

        try:
            print(
                f"Dataverse download attempt {attempt}/{max_attempts}: "
                f"file_id={file_id}"
            )

            with requests.get(
                url,
                params={"format": "original"},
                stream=True,
                timeout=(30, 600),
            ) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            got = md5_file(tmp)
            assert got.lower() == str(expected_md5).lower(), (
                "Downloaded EVE file failed official MD5 verification",
                file_id,
                expected_md5,
                got,
            )

            tmp.replace(dest)
            return dest

        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)

            if attempt < max_attempts:
                wait_s = waits[min(attempt - 1, len(waits) - 1)]
                print(
                    f"Download failed ({type(exc).__name__}: {exc}). "
                    f"Retrying in {wait_s}s..."
                )
                time.sleep(wait_s)

    raise ConnectionError(
        f"Dataverse download failed after {max_attempts} attempts "
        f"for file_id={file_id}"
    ) from last_exc


def sniff_sep(path):
    with open(path, "rb") as f:
        sample = f.read(65536)
    text = sample.decode("utf-8", errors="ignore")
    counts = {
        "\t": text.count("\t"),
        ",": text.count(","),
        ";": text.count(";"),
    }
    return max(counts, key=counts.get)


def detect_columns(df):
    named_time = [
        c for c in df.columns
        if re.search(r"time|date|timestamp", str(c), flags=re.I)
    ]

    time_scores = []
    for c in df.columns:
        x = pd.to_datetime(
            df[c].head(min(5000, len(df))),
            errors="coerce",
            utc=True,
        )
        rate = float(x.notna().mean())
        time_scores.append(
            (rate + (0.2 if c in named_time else 0.0), rate, c)
        )

    time_scores.sort(reverse=True, key=lambda x: x[0])
    _, tr, tcol = time_scores[0]
    assert tr >= 0.70, (tcol, tr)

    value_scores = []
    for c in df.columns:
        if c == tcol:
            continue
        x = pd.to_numeric(
            df[c].head(min(5000, len(df))),
            errors="coerce",
        )
        value_scores.append((float(x.notna().mean()), c))

    value_scores.sort(reverse=True)
    vr, vcol = value_scores[0]
    assert vr >= 0.70, (vcol, vr)
    return tcol, vcol


def expected_cache_contract(feature, row):
    return {
        "feature": feature,
        "file_id": int(row["file_id"]),
        "repository_path": str(row["path"]),
        "official_md5": str(row["md5"]).lower(),
        "time_cutoff_exclusive": VAL_END_EXCLUSIVE.isoformat(),
    }


def cache_is_valid(cache_path, meta_path, expected_meta):
    if not cache_path.exists() or not meta_path.exists():
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # v2 metadata may contain exactly these fields; v3 can add provenance.
        for k, v in expected_meta.items():
            if meta.get(k) != v:
                return False

        z = pd.read_parquet(cache_path)
        if not len(z):
            return False
        if "timestamp" not in z.columns:
            return False
        if expected_meta["feature"] not in z.columns:
            return False
        if not (
            pd.to_datetime(z["timestamp"]) < VAL_END_EXCLUSIVE
        ).all():
            return False

        return True

    except Exception:
        return False


def parse_to_trainval_cache(
    source_path,
    feature,
    cache_path,
    meta_path,
    expected_meta,
    source_mode,
):
    sep = sniff_sep(source_path)

    df = pd.read_csv(
        source_path,
        sep=sep,
        low_memory=False,
    ).dropna(axis=1, how="all")

    tcol, vcol = detect_columns(df)

    z = pd.DataFrame({
        "timestamp": pd.to_datetime(
            df[tcol],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None),
        feature: pd.to_numeric(
            df[vcol],
            errors="coerce",
        ),
    })

    z = z[
        z["timestamp"].notna()
        & (z["timestamp"] < VAL_END_EXCLUSIVE)
    ].copy()

    z = (
        z.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    assert len(z) > 0
    assert z["timestamp"].is_monotonic_increasing
    assert (z["timestamp"] < VAL_END_EXCLUSIVE).all()

    finite_fraction = float(z[feature].notna().mean())
    assert finite_fraction > 0.95, (feature, finite_fraction)

    z.to_parquet(cache_path, index=False)

    meta = {
        **expected_meta,
        "source_mode": source_mode,
        "source_file": str(source_path),
        "n_trainval_rows": int(len(z)),
        "finite_fraction": finite_fraction,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    del df, z
    gc.collect()

    return meta


def build_trainval_channel_cache(feature, row):
    cache_path = CACHE / f"{feature}.parquet"
    meta_path = CACHE / f"{feature}.json"
    expected_meta = expected_cache_contract(feature, row)

    # 1) Current v3 cache.
    if cache_is_valid(cache_path, meta_path, expected_meta):
        with open(meta_path) as f:
            meta = json.load(f)
        return cache_path, meta

    cache_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)

    # 2) Reuse already-successful v3 processed cache first.
    for cache_root, source_mode in [
        (V3_CACHE, "reused_12B_v3_processed_cache"),
        (V2_CACHE, "reused_12B_v2_processed_cache"),
    ]:
        old_cache = cache_root / f"{feature}.parquet"
        old_meta = cache_root / f"{feature}.json"

        if cache_is_valid(old_cache, old_meta, expected_meta):
            shutil.copy2(old_cache, cache_path)

            zcheck = pd.read_parquet(cache_path)
            meta = {
                **expected_meta,
                "source_mode": source_mode,
                "source_file": str(old_cache),
                "n_trainval_rows": int(len(zcheck)),
                "finite_fraction": float(
                    zcheck[feature].notna().mean()
                ),
            }
            del zcheck

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            return cache_path, meta

    # 3) Prefer an already-downloaded 12A local copy when available.
    local12a = find_12a_local_channel(row["path"])

    if local12a is not None:
        meta = parse_to_trainval_cache(
            local12a,
            feature,
            cache_path,
            meta_path,
            expected_meta,
            source_mode="12A_selected_channels_local_fallback",
        )
        return cache_path, meta

    # 4) Network only when the channel does not already exist locally.
    raw_path = retry_download_original(
        row["file_id"],
        RAW12B,
        Path(str(row["path"])).name,
        row["md5"],
    )

    meta = parse_to_trainval_cache(
        raw_path,
        feature,
        cache_path,
        meta_path,
        expected_meta,
        source_mode="Dataverse_original_verified_md5",
    )

    return cache_path, meta


channel_cache = {}
channel_provenance = []

for feature, row in resolved_rows.items():
    try:
        p, meta = build_trainval_channel_cache(feature, row)
        channel_cache[feature] = p
        channel_provenance.append({
            "feature": feature,
            "repository_path": row["path"],
            "file_id": int(row["file_id"]),
            "cache_path": str(p),
            "source_mode": meta["source_mode"],
            "n_trainval_rows": meta.get("n_trainval_rows"),
            "finite_fraction": meta.get("finite_fraction"),
            "status": "OK",
        })
        print(feature, "->", p, "|", meta["source_mode"])

    except Exception as exc:
        channel_provenance.append({
            "feature": feature,
            "repository_path": row["path"],
            "file_id": int(row["file_id"]),
            "cache_path": None,
            "source_mode": None,
            "n_trainval_rows": None,
            "finite_fraction": None,
            "status": f"FAILED: {type(exc).__name__}: {exc}",
        })

        pd.DataFrame(channel_provenance).to_csv(
            OUT / "12B_v4_channel_source_provenance.csv",
            index=False,
        )
        raise


provenance_df = pd.DataFrame(channel_provenance)
provenance_df.to_csv(
    OUT / "12B_v4_channel_source_provenance.csv",
    index=False,
)

assert set(channel_cache) == set(REQUIRED_CHANNELS)
assert (provenance_df["status"] == "OK").all()

with open(OUT / "12B_v4_channel_cache_paths.json", "w") as f:
    json.dump(
        {k: str(v) for k, v in channel_cache.items()},
        f,
        indent=2,
    )

display(provenance_df)

# %% [markdown] 5. Build the signed-Ah acquisition coordinate

# %% Code cell 6 (notebook cell 12)
current = pd.read_parquet(channel_cache["current_a"])
current = current.dropna(
    subset=["timestamp", "current_a"]
).sort_values("timestamp").reset_index(drop=True)

dt = current["timestamp"].diff().dt.total_seconds()
new_seg = (
    dt.isna()
    | (dt <= 0)
    | (dt > BURST_BREAK_S)
)

current["raw_segment_id"] = new_seg.cumsum().astype(int) - 1

prev_i = current["current_a"].shift(1)
same_seg = (
    current["raw_segment_id"]
    == current["raw_segment_id"].shift(1)
)

step_ah = np.where(
    same_seg,
    0.5
    * (
        current["current_a"].to_numpy(float)
        + prev_i.fillna(current["current_a"]).to_numpy(float)
    )
    * dt.fillna(0).to_numpy(float)
    / 3600.0,
    0.0,
)

step_qabs = np.where(
    same_seg,
    0.5
    * (
        np.abs(current["current_a"].to_numpy(float))
        + np.abs(prev_i.fillna(current["current_a"]).to_numpy(float))
    )
    * dt.fillna(0).to_numpy(float)
    / 3600.0,
    0.0,
)

current["ah_cum_segment"] = (
    pd.Series(step_ah)
    .groupby(current["raw_segment_id"])
    .cumsum()
    .to_numpy(float)
)
current["qabs_cum_segment"] = (
    pd.Series(step_qabs)
    .groupby(current["raw_segment_id"])
    .cumsum()
    .to_numpy(float)
)

CURRENT_STATE = OUT / "12B_v4_current_trainval_state.parquet"
current.to_parquet(CURRENT_STATE, index=False)

current_segment_audit = (
    current.groupby("raw_segment_id", as_index=False)
    .agg(
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        n_current_rows=("timestamp", "size"),
    )
)
current_segment_audit["duration_s"] = (
    current_segment_audit["last_timestamp"]
    - current_segment_audit["first_timestamp"]
).dt.total_seconds()

current_segment_audit.to_csv(
    OUT / "12B_v4_current_segment_audit_trainval.csv",
    index=False,
)

print("Current train+val rows:", len(current))
print("Acquisition segments:", current["raw_segment_id"].nunique())

# %% [markdown] 6. Build the EVE train/validation canonical table

# %% Code cell 7 (notebook cell 14)
target_train = pd.read_parquet(TARGET_TRAIN).copy()
target_val = pd.read_parquet(TARGET_VAL).copy()

assert set(target_train.columns) >= {"timestamp", "target_soc"}
assert set(target_val.columns) >= {"timestamp", "target_soc"}

target_train["timestamp"] = pd.to_datetime(target_train["timestamp"])
target_val["timestamp"] = pd.to_datetime(target_val["timestamp"])

assert (target_train["timestamp"] < TRAIN_END_EXCLUSIVE).all()
assert (
    (target_val["timestamp"] >= TRAIN_END_EXCLUSIVE)
    & (target_val["timestamp"] < VAL_END_EXCLUSIVE)
).all()

assert np.isfinite(target_train["target_soc"].to_numpy(float)).all()
assert np.isfinite(target_val["target_soc"].to_numpy(float)).all()

target_train["split"] = "train"
target_val["split"] = "val"

targets = (
    pd.concat([target_train, target_val], ignore_index=True)
    .sort_values("timestamp", kind="stable")
    .reset_index(drop=True)
    .rename(columns={"target_soc": "target"})
)

# Preserve the actual source-current timestamp used by backward asof.
current_for_merge = current[[
    "timestamp",
    "raw_segment_id",
    "current_a",
    "ah_cum_segment",
    "qabs_cum_segment",
]].copy()

current_for_merge = current_for_merge.rename(
    columns={"timestamp": "current_source_timestamp"}
).sort_values("current_source_timestamp")

canon = pd.merge_asof(
    targets.sort_values("timestamp"),
    current_for_merge,
    left_on="timestamp",
    right_on="current_source_timestamp",
    direction="backward",
    tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
)

before = len(canon)
canon = canon.dropna(
    subset=[
        "raw_segment_id",
        "current_a",
        "current_source_timestamp",
    ]
).copy()

canon["raw_segment_id"] = canon["raw_segment_id"].astype(int)

# No acquisition segment may bridge train and validation.
segment_split_counts = (
    canon.groupby("raw_segment_id")["split"].nunique()
)
cross_split_segments = segment_split_counts[
    segment_split_counts > 1
].index.tolist()

assert len(cross_split_segments) == 0, (
    "Current acquisition segment crosses the train/validation boundary; "
    "split-local resegmentation would be required.",
    cross_split_segments[:10],
)
canon["current_alignment_age_s"] = (
    canon["timestamp"] - canon["current_source_timestamp"]
).dt.total_seconds()

assert (
    (canon["current_alignment_age_s"] >= -1e-9)
    & (canon["current_alignment_age_s"] <= ALIGN_TOLERANCE_S + 1e-9)
).all()

# CRITICAL: first physical row, not pandas groupby.first() which skips NaNs.
anchor_rows = (
    canon.sort_values(
        ["raw_segment_id", "timestamp"],
        kind="stable",
    )
    .groupby("raw_segment_id", sort=False, group_keys=False)
    .head(1)[[
        "raw_segment_id",
        "timestamp",
        "current_source_timestamp",
        "target",
        "ah_cum_segment",
        "qabs_cum_segment",
    ]]
    .rename(columns={
        "timestamp": "anchor_timestamp",
        "current_source_timestamp": "anchor_current_timestamp",
        "target": "soc0",
        "ah_cum_segment": "anchor_ah_cum",
        "qabs_cum_segment": "anchor_qabs_cum",
    })
)

assert not anchor_rows["raw_segment_id"].duplicated().any()

canon = canon.merge(
    anchor_rows,
    on="raw_segment_id",
    how="left",
    validate="many_to_one",
)

canon["elapsed_s"] = (
    canon["timestamp"] - canon["anchor_timestamp"]
).dt.total_seconds()

# v4: preserve the raw signed-current encoding.
canon["coord"] = (
    canon["ah_cum_segment"].to_numpy(float)
    - canon["anchor_ah_cum"].to_numpy(float)
)
canon["q_abs_ah"] = (
    canon["qabs_cum_segment"].to_numpy(float)
    - canon["anchor_qabs_cum"].to_numpy(float)
)

assert (canon["elapsed_s"] >= -1e-9).all()
assert (canon["q_abs_ah"] >= -1e-8).all()

# Merge remaining raw predictors causally.
for feature in [
    "voltage_v",
    "battery_temp_c",
    "speed_kmh",
    "peb_torque_nm",
    "hydraulic_brake_torque_req_nm",
    "distance_km",
]:
    z = pd.read_parquet(channel_cache[feature])[
        ["timestamp", feature]
    ].copy()

    source_col = f"{feature}__source_timestamp"
    z = z.rename(columns={"timestamp": source_col}).sort_values(source_col)

    canon = pd.merge_asof(
        canon.sort_values("timestamp"),
        z,
        left_on="timestamp",
        right_on=source_col,
        direction="backward",
        tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
    )

    age_col = f"{feature}__age_s"
    canon[age_col] = (
        canon["timestamp"] - canon[source_col]
    ).dt.total_seconds()

    finite_age = canon[age_col].dropna().to_numpy(float)
    if len(finite_age):
        assert np.nanmin(finite_age) >= -1e-9
        assert np.nanmax(finite_age) <= ALIGN_TOLERANCE_S + 1e-9

# Strict-causal distance anchor:
# take exactly the first canonical row in the segment. If its distance is
# unavailable, fail closed to a zero distance-delta feature for that segment.
first_distance_rows = (
    canon.sort_values(
        ["raw_segment_id", "timestamp"],
        kind="stable",
    )
    .groupby("raw_segment_id", sort=False, group_keys=False)
    .head(1)[["raw_segment_id", "distance_km"]]
    .rename(columns={"distance_km": "anchor_distance_km"})
)

canon = canon.merge(
    first_distance_rows,
    on="raw_segment_id",
    how="left",
    validate="many_to_one",
)

anchor_distance_valid = np.isfinite(
    canon["anchor_distance_km"].to_numpy(float)
)
distance_valid = np.isfinite(
    canon["distance_km"].to_numpy(float)
)

canon["distance_since_anchor_km"] = np.where(
    anchor_distance_valid & distance_valid,
    canon["distance_km"].to_numpy(float)
    - canon["anchor_distance_km"].to_numpy(float),
    0.0,
)

canon["distance_anchor_available"] = anchor_distance_valid.astype(np.int8)

canon["vehicle_id"] = "EVE_ZOE_Q90"
canon["segment_id"] = canon["raw_segment_id"].astype(str)
canon["compound_segment"] = (
    canon["vehicle_id"] + "||" + canon["segment_id"]
)
canon["time_value"] = canon["timestamp"].astype("int64") / 1e9
canon["row_uid"] = (
    canon["vehicle_id"]
    + "||"
    + canon["segment_id"]
    + "||"
    + canon["time_value"].astype(str)
)

FEATURE_COLUMNS = [
    "soc0",
    "current_a",
    "voltage_v",
    "battery_temp_c",
    "speed_kmh",
    "peb_torque_nm",
    "hydraulic_brake_torque_req_nm",
    "distance_since_anchor_km",
    "coord",
    "q_abs_ah",
    "elapsed_s",
]

# Hard leakage guard.
forbidden_tokens = [
    "target",
    "residual",
    "real_state",
    "real state",
    "soc_phy",
]
bad_features = [
    c for c in FEATURE_COLUMNS
    if any(tok in c.lower() for tok in forbidden_tokens)
]
assert not bad_features, bad_features

# Train-only median imputation for structured/tree models.
train_mask = canon["split"].eq("train")
feature_medians = {}

for c in FEATURE_COLUMNS:
    x = pd.to_numeric(canon.loc[train_mask, c], errors="coerce")
    med = float(x.median())
    if not np.isfinite(med):
        med = 0.0

    feature_medians[c] = med
    canon[c] = pd.to_numeric(canon[c], errors="coerce").fillna(med)

assert np.isfinite(canon[FEATURE_COLUMNS].to_numpy(float)).all()

# Prior slope is calibrated on TRAIN labels only.
# This calibration precedes both full and sparse correction experiments.
tr = canon[canon["split"] == "train"].copy()

x = tr["coord"].to_numpy(float)
y = tr["target"].to_numpy(float) - tr["soc0"].to_numpy(float)

keep = (
    np.isfinite(x)
    & np.isfinite(y)
    & (np.abs(x) > 1e-8)
)
assert keep.sum() >= 100

beta_pp_per_ah = float(
    np.dot(x[keep], y[keep])
    / np.dot(x[keep], x[keep])
)

assert np.isfinite(beta_pp_per_ah)
assert abs(beta_pp_per_ah) > 1e-8, (
    "Train-only signed-Ah calibration is numerically degenerate",
    beta_pp_per_ah,
)

train_coord = x[keep]
train_soc_delta = y[keep]
train_coord_soc_corr = float(
    np.corrcoef(train_coord, train_soc_delta)[0, 1]
)

# Also document what v3's forced sign-flip would have produced.
v3_flipped_beta_pp_per_ah = float(-beta_pp_per_ah)

sign_audit = {
    "coordinate_definition_v4": "Ah_cum - Ah_at_SOC0_anchor",
    "documentation_note": CURRENT_SIGN_DOCUMENTATION,
    "n_train_rows_used": int(keep.sum()),
    "beta_pp_per_ah_train_only": float(beta_pp_per_ah),
    "coord_soc_delta_correlation_train_only": train_coord_soc_corr,
    "v3_equivalent_flipped_coord_beta_pp_per_ah": v3_flipped_beta_pp_per_ah,
    "confirmatory_test_used": False,
    "decision": (
        "Use raw signed-Ah encoding and train-only beta; "
        "do not hard-code current polarity from metadata."
    ),
}

with open(OUT / "12B_v4_current_sign_audit_train_only.json", "w") as f:
    json.dump(sign_audit, f, indent=2)

canon["physics_delta"] = beta_pp_per_ah * canon["coord"].to_numpy(float)
canon["soc_phy"] = canon["soc0"].to_numpy(float) + canon["physics_delta"].to_numpy(float)
canon["residual_target"] = canon["target"].to_numpy(float) - canon["soc_phy"].to_numpy(float)

assert set(canon["split"].unique()) == {"train", "val"}
assert (canon["timestamp"] < VAL_END_EXCLUSIVE).all()
assert not canon["row_uid"].duplicated().any()

CANON_PATH = OUT / "12B_v4_EVE_canonical_trainval.parquet"
canon.to_parquet(CANON_PATH, index=False)

canonical_meta = {
    "dataset": "EVE_ZOE_Q90",
    "target": "dashboard BMS-reported State of Charge",
    "current_positive_means": "discharge",
    "prior": "SIGNED_AH",
    "beta_pp_per_ah_train_only": beta_pp_per_ah,
    "burst_break_s": BURST_BREAK_S,
    "alignment_tolerance_s": ALIGN_TOLERANCE_S,
    "feature_columns": FEATURE_COLUMNS,
    "feature_medians_train_only": feature_medians,
    "soc0_allowed": True,
    "real_state_of_charge_predictor": False,
    "confirmatory_test_rows_present": False,
    "target_rows_before_current_alignment": int(before),
    "canonical_rows_after_current_alignment": int(len(canon)),
    "canonical_train_rows": int((canon["split"] == "train").sum()),
    "canonical_val_rows": int((canon["split"] == "val").sum()),
    "canonical_segments": int(canon["compound_segment"].nunique()),
    "distance_anchor_policy": "exact first row; missing anchor => zero delta",
    "current_alignment_age_max_s": float(canon["current_alignment_age_s"].max()),
    "cross_split_raw_segments": int(len(cross_split_segments)),
    "current_coordinate_definition": "raw signed Ah: Ah_cum - Ah_anchor",
    "current_sign_audit_file": str(
        OUT / "12B_v4_current_sign_audit_train_only.json"
    ),
}

canonical_contract_payload = {
    "dataset": canonical_meta["dataset"],
    "target": canonical_meta["target"],
    "prior": canonical_meta["prior"],
    "beta_pp_per_ah_train_only": round(float(beta_pp_per_ah), 12),
    "feature_columns": FEATURE_COLUMNS,
    "feature_medians_train_only": {
        k: round(float(v), 12)
        for k, v in feature_medians.items()
    },
    "train_end_exclusive": TRAIN_END_EXCLUSIVE.isoformat(),
    "val_end_exclusive": VAL_END_EXCLUSIVE.isoformat(),
    "burst_break_s": float(BURST_BREAK_S),
    "alignment_tolerance_s": float(ALIGN_TOLERANCE_S),
    "coord_definition": "Ah_cum - Ah_anchor",
}

CANONICAL_FINGERPRINT = hashlib.sha256(
    json.dumps(
        canonical_contract_payload,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

canonical_meta["canonical_fingerprint_sha256"] = CANONICAL_FINGERPRINT

with open(OUT / "12B_v4_EVE_canonical_metadata.json", "w") as f:
    json.dump(canonical_meta, f, indent=2)

display(pd.DataFrame([canonical_meta]).drop(
    columns=["feature_columns", "feature_medians_train_only"]
))
print("Train-only beta pp/Ah:", beta_pp_per_ah)

# %% [markdown] 7. Strict-causal raw early-context bank for 10K

# %% Code cell 8 (notebook cell 16)
anchor_map = (
    canon[[
        "raw_segment_id",
        "compound_segment",
        "anchor_timestamp",
        "anchor_current_timestamp",
        "soc0",
        "anchor_ah_cum",
        "anchor_qabs_cum",
        "anchor_distance_km",
        "distance_anchor_available",
    ]]
    .drop_duplicates("raw_segment_id")
)

ctx_base = current.merge(
    anchor_map,
    on="raw_segment_id",
    how="inner",
    validate="many_to_one",
)

# Context timing begins at the causally matched current state used around SOC0.
ctx_base["context_elapsed_s"] = (
    ctx_base["timestamp"] - ctx_base["anchor_current_timestamp"]
).dt.total_seconds()

ctx_base = ctx_base[
    (ctx_base["context_elapsed_s"] >= -1e-9)
    & (ctx_base["context_elapsed_s"] <= MAX_EARLY_WINDOW_S + 1e-9)
].copy()

ctx_base["coord"] = (
    ctx_base["ah_cum_segment"].to_numpy(float)
    - ctx_base["anchor_ah_cum"].to_numpy(float)
)
ctx_base["q_abs_ah"] = (
    ctx_base["qabs_cum_segment"].to_numpy(float)
    - ctx_base["anchor_qabs_cum"].to_numpy(float)
)
ctx_base["physics_delta"] = beta_pp_per_ah * ctx_base["coord"]
ctx_base["soc_phy"] = ctx_base["soc0"] + ctx_base["physics_delta"]

# Causal raw predictor alignment.
for feature in [
    "voltage_v",
    "battery_temp_c",
    "speed_kmh",
    "peb_torque_nm",
    "hydraulic_brake_torque_req_nm",
    "distance_km",
]:
    z = pd.read_parquet(channel_cache[feature])[
        ["timestamp", feature]
    ].sort_values("timestamp")

    ctx_base = pd.merge_asof(
        ctx_base.sort_values("timestamp"),
        z,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
    )

# Use the canonical anchor distance; never search forward for a non-null distance.
anchor_dist = ctx_base["anchor_distance_km"].to_numpy(float)
distance_now = ctx_base["distance_km"].to_numpy(float)

ctx_base["distance_since_anchor_km"] = np.where(
    np.isfinite(anchor_dist)
    & np.isfinite(distance_now)
    & (ctx_base["distance_anchor_available"].to_numpy(int) == 1),
    distance_now - anchor_dist,
    0.0,
)

ctx_base["elapsed_s"] = ctx_base["context_elapsed_s"]
ctx_base["time_value"] = ctx_base["timestamp"].astype("int64") / 1e9

RAW_CONTEXT_SIGNALS = [
    "current_a",
    "voltage_v",
    "battery_temp_c",
    "speed_kmh",
    "peb_torque_nm",
    "hydraulic_brake_torque_req_nm",
    "distance_since_anchor_km",
    "coord",
    "q_abs_ah",
    "elapsed_s",
]

def six_stats(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(float)
    finite = vals[np.isfinite(vals)]

    if not len(finite):
        return {
            "last": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "delta": 0.0,
        }

    return {
        "last": float(finite[-1]),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "delta": float(finite[-1] - finite[0]),
    }


def summarize_context(window_s):
    rows = []

    for seg, g in ctx_base.groupby("compound_segment", sort=False):
        h = g[
            g["context_elapsed_s"] <= float(window_s) + 1e-9
        ].sort_values("timestamp")

        if not len(h):
            continue

        tv = h["time_value"].to_numpy(float)
        gaps = np.diff(tv)
        gaps = gaps[np.isfinite(gaps) & (gaps > 0)]

        row = {
            "compound_segment": seg,
            "ctx_rows": float(len(h)),
            "ctx_duration_s": float(
                h["context_elapsed_s"].max()
                - h["context_elapsed_s"].min()
            ),
            "ctx_median_gap_s": float(np.median(gaps)) if len(gaps) else 0.0,
            "ctx_p90_gap_s": float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
            "ctx_sampling_hz": float(
                (len(h) - 1) / max(tv[-1] - tv[0], 1e-9)
            ) if len(tv) >= 2 else 0.0,
            "ctx_soc0": float(h["soc0"].iloc[0]),
            "ctx_phy_last": float(h["soc_phy"].iloc[-1]),
            "ctx_phy_delta_last": float(h["physics_delta"].iloc[-1]),
        }

        for c in RAW_CONTEXT_SIGNALS:
            st = six_stats(h[c])
            for stat, val in st.items():
                row[f"raw_{stat}__{c}"] = val

            vals = pd.to_numeric(h[c], errors="coerce")
            row[f"raw_finite_frac__{c}"] = float(
                np.mean(np.isfinite(vals.to_numpy(float)))
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    assert len(out)
    assert not out["compound_segment"].duplicated().any()
    return out


RAW_CONTEXT_BANK = {}

unique_windows = sorted({
    float(x["early_window_s"])
    for configs in SF_CANDIDATES.values()
    for x in configs
})

context_audit_rows = []

for w in unique_windows:
    ctx = summarize_context(w)
    p = CTX_OUT / f"12B_v4_EVE_raw_context_{int(w)}s.parquet"
    ctx.to_parquet(p, index=False)
    RAW_CONTEXT_BANK[float(w)] = ctx

    needed = set(canon["compound_segment"])
    got = set(ctx["compound_segment"])
    coverage = len(needed & got) / max(len(needed), 1)

    context_audit_rows.append({
        "early_window_s": float(w),
        "canonical_segments": len(needed),
        "context_segments": len(got),
        "coverage": coverage,
        "n_context_features": len(ctx.columns) - 1,
    })

    assert coverage >= 0.95, (w, coverage)

context_audit = pd.DataFrame(context_audit_rows)
context_audit.to_csv(
    OUT / "12B_v4_raw_context_coverage.csv",
    index=False,
)
display(context_audit)

# %% [markdown] 8. Shared deterministic supervision selector and metrics

# %% Code cell 9 (notebook cell 18)
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

    for veh, g in df.groupby("vehicle_id"):
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

        order = keys.map(
            stable_hash_u01
        ).sort_values().index

        pieces.append(
            g.loc[order[:k]].copy()
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def tree_prediction_std(model, X):
    pred = np.vstack([
        t.predict(X)
        for t in model.estimators_
    ])
    return np.std(pred, axis=0, ddof=0)


train_df = canon[canon["split"] == "train"].copy()
val_df = canon[canon["split"] == "val"].copy()

sparse_audit = []
for fraction in LABEL_FRACTIONS:
    for seed in FINAL_SEEDS:
        q = exact_sparse_subset(
            train_df,
            fraction,
            seed,
        )
        sparse_audit.append({
            "fraction": fraction,
            "seed": seed,
            "n_train_total": len(train_df),
            "n_selected": len(q),
            "selected_fraction": len(q) / len(train_df),
        })

pd.DataFrame(sparse_audit).to_csv(
    OUT / "12B_v4_sparse_selector_audit.csv",
    index=False,
)

# %% [markdown] 9. Frozen 10K SC-RS-SFPEF implementation

# %% Code cell 10 (notebook cell 20)
def make_et(hp, seed, n_trees):
    return ExtraTreesRegressor(
        n_estimators=int(n_trees),
        min_samples_leaf=int(hp["min_samples_leaf"]),
        max_features=float(hp["max_features"]),
        random_state=int(seed),
        n_jobs=-1,
    )


def fit_residual(train_rows, feature_cols, hp, seed, n_trees):
    m = make_et(hp, seed, n_trees)
    X = train_rows[feature_cols].copy()
    X["__physics__"] = train_rows["soc_phy"].to_numpy(float)
    y = (
        train_rows["target"].to_numpy(float)
        - train_rows["soc_phy"].to_numpy(float)
    )
    m.fit(X, y)
    return m


def predict_residual(model, rows, feature_cols):
    X = rows[feature_cols].copy()
    X["__physics__"] = rows["soc_phy"].to_numpy(float)
    return (
        rows["soc_phy"].to_numpy(float)
        + model.predict(X)
    )


def build_early_context(source_df, feature_cols, window_s):
    # feature_cols intentionally ignored: context comes only from raw telemetry.
    key = float(window_s)
    assert key in RAW_CONTEXT_BANK, key

    ctx = RAW_CONTEXT_BANK[key]
    wanted = set(
        source_df["compound_segment"].astype(str)
    )

    out = ctx[
        ctx["compound_segment"].astype(str).isin(wanted)
    ].copy()

    coverage = (
        out["compound_segment"].nunique()
        / max(len(wanted), 1)
    )
    assert coverage >= 0.95, (window_s, coverage)
    return out.reset_index(drop=True)


def context_feature_columns(context_df):
    return [
        c for c in context_df.columns
        if c != "compound_segment"
    ]


def fit_ridge_slow_target(g):
    p = g["physics_delta"].to_numpy(float)
    r = (
        g["target"].to_numpy(float)
        - g["soc_phy"].to_numpy(float)
    )

    keep = np.isfinite(p) & np.isfinite(r)
    p = p[keep]
    r = r[keep]

    if not len(p):
        return None

    A = np.column_stack([
        p,
        np.ones(len(p)),
    ])
    reg = np.diag([
        RIDGE_A,
        RIDGE_B,
    ])

    coef = np.linalg.solve(
        A.T @ A + reg,
        A.T @ r,
    )

    a = float(np.clip(
        coef[0],
        SLOW_A_CLIP[0],
        SLOW_A_CLIP[1],
    ))
    b = float(np.clip(
        coef[1],
        SLOW_B_CLIP[0],
        SLOW_B_CLIP[1],
    ))

    spread = float(
        np.ptp(p)
        if len(p) > 1
        else abs(p[0])
    )
    n = int(len(p))

    reliability = (
        n / (n + 2.0)
    ) * (
        spread / (spread + 1.0)
    )

    return {
        "slow_a_target": a,
        "slow_b_target": b,
        "slow_target_weight": float(
            max(reliability, 1e-3)
        ),
        "slow_target_n_labels": n,
        "slow_target_spread_pp": spread,
    }


def build_slow_target_table(labeled_df, context_df):
    rows = []

    for seg, g in labeled_df.groupby(
        "compound_segment",
        sort=False,
    ):
        target = fit_ridge_slow_target(g)

        if target is None:
            continue

        target["compound_segment"] = seg
        rows.append(target)

    targets = pd.DataFrame(rows)

    if not len(targets):
        return pd.DataFrame()

    return targets.merge(
        context_df,
        on="compound_segment",
        how="inner",
        validate="one_to_one",
    )


def fit_slow_pair(target_table, ctx_cols, seed):
    ma = make_et(
        SLOW_HP,
        seed,
        N_TREES_SLOW,
    )
    mb = make_et(
        SLOW_HP,
        seed + 1,
        N_TREES_SLOW,
    )

    w = target_table[
        "slow_target_weight"
    ].to_numpy(float)

    ma.fit(
        target_table[ctx_cols],
        target_table["slow_a_target"],
        sample_weight=w,
    )
    mb.fit(
        target_table[ctx_cols],
        target_table["slow_b_target"],
        sample_weight=w,
    )
    return ma, mb


def predict_slow_pair(ma, mb, context_df, ctx_cols):
    out = context_df[[
        "compound_segment"
    ]].copy()

    if (
        ma is None
        or mb is None
        or not len(context_df)
    ):
        out["a_pred"] = 0.0
        out["b_pred"] = 0.0
        out["a_std"] = np.inf
        out["b_std"] = np.inf
        return out

    X = context_df[ctx_cols]

    out["a_pred"] = np.clip(
        ma.predict(X),
        SLOW_A_CLIP[0],
        SLOW_A_CLIP[1],
    )
    out["b_pred"] = np.clip(
        mb.predict(X),
        SLOW_B_CLIP[0],
        SLOW_B_CLIP[1],
    )
    out["a_std"] = tree_prediction_std(ma, X)
    out["b_std"] = tree_prediction_std(mb, X)
    return out


def oof_slow_predictions(labeled_train, train_context, seed):
    target_table = build_slow_target_table(
        labeled_train,
        train_context,
    )

    base = train_context[[
        "compound_segment"
    ]].copy()
    base["a_pred"] = 0.0
    base["b_pred"] = 0.0
    base["a_std"] = np.inf
    base["b_std"] = np.inf

    if (
        len(target_table) < 6
        or target_table["compound_segment"].nunique() < 3
    ):
        return base

    ctx_cols = context_feature_columns(
        train_context
    )

    n_groups = target_table[
        "compound_segment"
    ].nunique()
    n_splits = min(
        OOF_FOLDS,
        n_groups,
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )
    pred_parts = []

    for fold, (fit_idx, hold_idx) in enumerate(
        splitter.split(
            target_table,
            groups=target_table[
                "compound_segment"
            ],
        )
    ):
        tr = target_table.iloc[
            fit_idx
        ].copy()
        ho = target_table.iloc[
            hold_idx
        ].copy()

        ma, mb = fit_slow_pair(
            tr,
            ctx_cols,
            seed + fold * 31,
        )

        pred_parts.append(
            predict_slow_pair(
                ma,
                mb,
                ho,
                ctx_cols,
            )
        )

    pred = pd.concat(
        pred_parts,
        ignore_index=True,
    )

    base = (
        base[["compound_segment"]]
        .merge(
            pred,
            on="compound_segment",
            how="left",
            validate="one_to_one",
        )
    )

    base[["a_pred", "b_pred"]] = (
        base[["a_pred", "b_pred"]]
        .fillna(0.0)
    )
    base[["a_std", "b_std"]] = (
        base[["a_std", "b_std"]]
        .fillna(np.inf)
    )
    return base


def full_slow_models(labeled_train, train_context, seed):
    target_table = build_slow_target_table(
        labeled_train,
        train_context,
    )

    if len(target_table) < 6:
        return None, None, None

    ctx_cols = context_feature_columns(
        train_context
    )
    ma, mb = fit_slow_pair(
        target_table,
        ctx_cols,
        seed,
    )
    return ma, mb, ctx_cols


def add_slow_state(rows_df, slow_pred_table, strength, tau_pp):
    d = rows_df.merge(
        slow_pred_table,
        on="compound_segment",
        how="left",
        validate="many_to_one",
    )

    d[["a_pred", "b_pred"]] = (
        d[["a_pred", "b_pred"]]
        .fillna(0.0)
    )
    d[["a_std", "b_std"]] = (
        d[["a_std", "b_std"]]
        .fillna(np.inf)
    )

    p = d["physics_delta"].to_numpy(float)
    a = d["a_pred"].to_numpy(float)
    b = d["b_pred"].to_numpy(float)
    sa = d["a_std"].to_numpy(float)
    sb = d["b_std"].to_numpy(float)

    slow_raw = a * p + b

    uncertainty = np.sqrt(
        (p * sa) ** 2
        + sb ** 2
    )

    confidence = np.where(
        np.isfinite(uncertainty),
        1.0 / (
            1.0
            + uncertainty / float(tau_pp)
        ),
        0.0,
    )

    correction = (
        float(strength)
        * confidence
        * slow_raw
    )

    d["slow_raw_pp"] = slow_raw
    d["slow_uncertainty_pp"] = uncertainty
    d["slow_confidence"] = confidence
    d["slow_correction_pp"] = correction
    d["sf2_base"] = (
        d["soc_phy"].to_numpy(float)
        + correction
    )
    return d


def causal_eligible_mask(df, early_window_s):
    elapsed = df["elapsed_s"].to_numpy(float)
    assert np.isfinite(elapsed).all()
    return elapsed + 1e-9 >= float(early_window_s)


def enforce_strict_warmup_state(df, early_window_s):
    d = df.copy()
    eligible = causal_eligible_mask(
        d,
        early_window_s,
    )
    d["causal_eligible"] = eligible

    early = ~eligible
    if early.any():
        d.loc[
            early,
            "slow_raw_pp",
        ] = 0.0
        d.loc[
            early,
            "slow_uncertainty_pp",
        ] = np.inf
        d.loc[
            early,
            "slow_confidence",
        ] = 0.0
        d.loc[
            early,
            "slow_correction_pp",
        ] = 0.0
        d.loc[
            early,
            "sf2_base",
        ] = d.loc[
            early,
            "soc_phy",
        ].to_numpy(float)

    return d

# %% Code cell 11 (notebook cell 21)
def fit_predict_sf2(
    labeled_train,
    train_context_source,
    pred_df,
    pred_context_source,
    feature_cols,
    residual_hp,
    seed,
    early_window_s,
    slow_strength,
    tau_pp,
    n_trees,
):
    strength = float(slow_strength)

    if strength == 0.0:
        residual = fit_residual(
            labeled_train,
            feature_cols,
            residual_hp,
            seed,
            n_trees,
        )
        pred = predict_residual(
            residual,
            pred_df,
            feature_cols,
        )
        return pred, {
            "mean_confidence": 0.0,
            "mean_abs_slow_correction_pp": 0.0,
            "fallback_residual": True,
        }

    train_context = build_early_context(
        train_context_source,
        feature_cols,
        early_window_s,
    )
    pred_context = build_early_context(
        pred_context_source,
        feature_cols,
        early_window_s,
    )

    slow_oof = oof_slow_predictions(
        labeled_train,
        train_context,
        seed=seed + 100,
    )

    tr = add_slow_state(
        labeled_train,
        slow_oof,
        strength,
        tau_pp,
    )
    tr = enforce_strict_warmup_state(
        tr,
        early_window_s,
    )

    ma, mb, ctx_cols = full_slow_models(
        labeled_train,
        train_context,
        seed=seed + 500,
    )

    if (
        ma is None
        or mb is None
        or ctx_cols is None
    ):
        slow_pred = pred_context[[
            "compound_segment"
        ]].copy()
        slow_pred["a_pred"] = 0.0
        slow_pred["b_pred"] = 0.0
        slow_pred["a_std"] = np.inf
        slow_pred["b_std"] = np.inf
    else:
        slow_pred = predict_slow_pair(
            ma,
            mb,
            pred_context,
            ctx_cols,
        )

    pp = add_slow_state(
        pred_df,
        slow_pred,
        strength,
        tau_pp,
    )
    pp = enforce_strict_warmup_state(
        pp,
        early_window_s,
    )

    fast_features = list(feature_cols) + [
        "sf2_base",
        "slow_confidence",
        "slow_correction_pp",
    ]

    fast = make_et(
        residual_hp,
        seed + 900,
        n_trees,
    )

    fast.fit(
        tr[fast_features],
        tr["target"].to_numpy(float)
        - tr["sf2_base"].to_numpy(float),
    )

    pred = (
        pp["sf2_base"].to_numpy(float)
        + fast.predict(pp[fast_features])
    )

    eligible = causal_eligible_mask(
        pp,
        early_window_s,
    )

    return pred, {
        "mean_confidence": float(
            np.mean(
                pp.loc[
                    eligible,
                    "slow_confidence",
                ]
            )
            if eligible.any()
            else 0.0
        ),
        "mean_abs_slow_correction_pp": float(
            np.mean(
                np.abs(
                    pp.loc[
                        eligible,
                        "slow_correction_pp",
                    ]
                )
            )
            if eligible.any()
            else 0.0
        ),
        "fallback_residual": False,
    }


def router_features(df, feature_cols):
    X = df[feature_cols].copy()

    pr = df["pred_r"].to_numpy(float)
    ps = df["pred_sf"].to_numpy(float)
    phy = df["soc_phy"].to_numpy(float)

    X["__soc_phy__"] = phy
    X["__physics_delta__"] = (
        df["physics_delta"].to_numpy(float)
    )
    X["__elapsed__"] = (
        df["elapsed_s"].to_numpy(float)
    )
    X["__soc0__"] = (
        df["soc0"].to_numpy(float)
    )
    X["__pred_r__"] = pr
    X["__pred_sf__"] = ps
    X["__sf_minus_r__"] = ps - pr
    X["__expert_gap__"] = np.abs(ps - pr)
    X["__r_to_phy__"] = np.abs(pr - phy)
    X["__sf_to_phy__"] = np.abs(ps - phy)

    assert "target" not in X.columns
    return X


def outer_oof_expert_frame(
    labeled_train,
    full_train_context,
    feature_cols,
    residual_hp,
    sf_protocol,
    seed,
    n_trees,
):
    n_groups = labeled_train[
        "compound_segment"
    ].nunique()

    assert n_groups >= 3, n_groups

    n_splits = min(
        OOF_FOLDS,
        n_groups,
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )
    parts = []

    for fold, (fit_idx, hold_idx) in enumerate(
        splitter.split(
            labeled_train,
            groups=labeled_train[
                "compound_segment"
            ],
        )
    ):
        tr = labeled_train.iloc[
            fit_idx
        ].copy()
        ho = labeled_train.iloc[
            hold_idx
        ].copy()

        fold_seed = int(
            seed + fold * 1009
        )

        mr = fit_residual(
            tr,
            feature_cols,
            residual_hp,
            fold_seed,
            n_trees,
        )
        pr = predict_residual(
            mr,
            ho,
            feature_cols,
        )

        psf, _ = fit_predict_sf2(
            tr,
            full_train_context,
            ho,
            full_train_context,
            feature_cols,
            residual_hp,
            seed=fold_seed,
            n_trees=n_trees,
            **sf_protocol,
        )

        z = ho.copy()
        z["pred_r"] = pr
        z["pred_sf"] = psf

        eligible = causal_eligible_mask(
            z,
            sf_protocol["early_window_s"],
        )

        z = z.loc[
            eligible
        ].copy()

        if not len(z):
            continue

        er = np.abs(
            z["target"].to_numpy(float)
            - z["pred_r"].to_numpy(float)
        )
        es = np.abs(
            z["target"].to_numpy(float)
            - z["pred_sf"].to_numpy(float)
        )

        z["advantage_target"] = (
            (er - es)
            / (er + es + EPS_ADV)
        )

        parts.append(z)

    assert parts, (
        "No post-warm-up OOF rows available."
    )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def fit_router(
    labeled_train,
    full_train_context,
    feature_cols,
    residual_hp,
    sf_protocol,
    seed,
    n_trees_expert,
):
    oof = outer_oof_expert_frame(
        labeled_train,
        full_train_context,
        feature_cols,
        residual_hp,
        sf_protocol,
        seed,
        n_trees_expert,
    )

    assert len(oof) >= 10, len(oof)

    router = make_et(
        ROUTER_HP,
        seed + 7000,
        N_TREES_ROUTER,
    )
    router.fit(
        router_features(
            oof,
            feature_cols,
        ),
        oof["advantage_target"],
    )
    return router, oof


def full_expert_frame(
    labeled_train,
    full_train_context,
    pred_df,
    pred_context_source,
    feature_cols,
    residual_hp,
    sf_protocol,
    seed,
    n_trees,
):
    mr = fit_residual(
        labeled_train,
        feature_cols,
        residual_hp,
        seed,
        n_trees,
    )

    pr = predict_residual(
        mr,
        pred_df,
        feature_cols,
    )

    psf, sf_audit = fit_predict_sf2(
        labeled_train,
        full_train_context,
        pred_df,
        pred_context_source,
        feature_cols,
        residual_hp,
        seed=seed,
        n_trees=n_trees,
        **sf_protocol,
    )

    z = pred_df.copy()
    z["pred_r"] = pr
    z["pred_sf_unmasked"] = psf

    eligible = causal_eligible_mask(
        z,
        sf_protocol["early_window_s"],
    )

    z["causal_eligible"] = eligible
    z["pred_sf"] = np.where(
        eligible,
        psf,
        pr,
    )

    assert np.allclose(
        z.loc[
            ~eligible,
            "pred_sf",
        ].to_numpy(float),
        z.loc[
            ~eligible,
            "pred_r",
        ].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    )

    return z, sf_audit


def route_prediction_final(
    router,
    expert_frame,
    feature_cols,
    mode,
    kappa,
    margin,
    sf_weight,
):
    assert mode == "Routed"

    pr = expert_frame[
        "pred_r"
    ].to_numpy(float)
    ps = expert_frame[
        "pred_sf"
    ].to_numpy(float)
    eligible = expert_frame[
        "causal_eligible"
    ].to_numpy(bool)

    pred = pr.copy()
    switch_all = np.zeros(
        len(expert_frame),
        dtype=bool,
    )

    if float(sf_weight) == 0.0:
        return pred, {
            "switch_rate": 0.0,
            "postwarmup_switch_rate": 0.0,
            "fail_closed": True,
        }

    if eligible.any():
        Z = router_features(
            expert_frame.loc[eligible],
            feature_cols,
        )
        adv_mean = router.predict(Z)
        adv_std = tree_prediction_std(
            router,
            Z,
        )
        lcb = (
            adv_mean
            - float(kappa) * adv_std
        )
        switch = (
            lcb > float(margin)
        )
        w = np.where(
            switch,
            float(sf_weight),
            0.0,
        )

        idx = np.flatnonzero(
            eligible
        )
        switch_all[idx] = switch

        pred[idx] = (
            pr[idx]
            + w * (ps[idx] - pr[idx])
        )

    pred[~eligible] = pr[~eligible]

    return pred, {
        "switch_rate": float(
            np.mean(switch_all)
        ),
        "postwarmup_switch_rate": float(
            np.mean(
                switch_all[eligible]
            )
            if eligible.any()
            else 0.0
        ),
        "fail_closed": False,
    }

# %% Code cell 12 (notebook cell 22)
def segment_delta_stats(df, pred_candidate, pred_residual):
    work = df[[
        "compound_segment",
        "target",
    ]].copy()

    work["candidate"] = np.asarray(
        pred_candidate,
        float,
    )
    work["residual"] = np.asarray(
        pred_residual,
        float,
    )

    deltas = []

    for _, g in work.groupby(
        "compound_segment",
        sort=False,
    ):
        if len(g) < 3:
            continue

        y = g["target"].to_numpy(float)
        mae_r = np.mean(
            np.abs(
                g["residual"].to_numpy(float)
                - y
            )
        )
        mae_c = np.mean(
            np.abs(
                g["candidate"].to_numpy(float)
                - y
            )
        )
        deltas.append(
            mae_c - mae_r
        )

    d = np.asarray(
        deltas,
        float,
    )

    if not len(d):
        return {
            "n_segments_tail": 0,
            "segment_p95_delta_pp": 0.0,
            "segment_worst_delta_pp": 0.0,
        }

    return {
        "n_segments_tail": int(len(d)),
        "segment_p95_delta_pp": float(
            np.percentile(d, 95)
        ),
        "segment_worst_delta_pp": float(
            np.max(d)
        ),
    }


def attach_chronological_validation_blocks(
    df,
    n_blocks=N_STABILITY_BLOCKS,
):
    z = df.copy().reset_index(drop=True)
    z["__row_order__"] = np.arange(
        len(z),
        dtype=int,
    )

    seg = (
        z.groupby(
            "compound_segment",
            as_index=False,
        )
        .agg(
            first_time=(
                "time_value",
                "min",
            ),
            first_row=(
                "__row_order__",
                "min",
            ),
        )
        .sort_values(
            ["first_time", "first_row"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    nb = int(
        min(
            max(1, n_blocks),
            len(seg),
        )
    )

    block = np.floor(
        np.arange(
            len(seg),
            dtype=float,
        )
        * nb
        / max(1, len(seg))
    ).astype(int)

    block = np.minimum(
        block,
        nb - 1,
    )

    seg["validation_block"] = block

    mapping = dict(
        zip(
            seg["compound_segment"],
            seg["validation_block"],
        )
    )

    z["validation_block"] = (
        z["compound_segment"]
        .map(mapping)
        .astype(int)
    )

    return z.drop(
        columns=["__row_order__"]
    ), nb


def validation_block_metrics(
    val_rows,
    pred,
    pred_r,
):
    z, nb = attach_chronological_validation_blocks(
        val_rows
    )

    z["pred_candidate"] = np.asarray(
        pred,
        float,
    )
    z["pred_residual"] = np.asarray(
        pred_r,
        float,
    )

    improvements = []
    block_rows = []

    for b in sorted(
        z["validation_block"].unique()
    ):
        q = z[
            z["validation_block"] == b
        ]

        r_mae = regression_metrics(
            q["target"],
            q["pred_residual"],
        )["MAE"]

        c_mae = regression_metrics(
            q["target"],
            q["pred_candidate"],
        )["MAE"]

        imp = (
            100.0 * (r_mae - c_mae) / max(r_mae, 1e-12)
        )

        improvements.append(
            float(imp)
        )

        block_rows.append({
            "validation_block": int(b),
            "n_rows": int(len(q)),
            "n_segments": int(
                q["compound_segment"].nunique()
            ),
            "Residual_MAE": float(r_mae),
            "Candidate_MAE": float(c_mae),
            "improvement_pct": float(imp),
        })

    arr = np.asarray(
        improvements,
        float,
    )

    return {
        "n_stability_blocks": int(nb),
        "worst_block_improvement_pct": float(arr.min()),
        "mean_block_improvement_pct": float(arr.mean()),
        "all_blocks_non_degrading": bool(
            np.all(arr >= -STABILITY_EPS)
        ),
        "block_rows": block_rows,
    }

# %% [markdown] 10. EVE validation-only selection inside the frozen 10K family

# %% Code cell 13 (notebook cell 24)
selected_10k = {}
residual_grid_rows = []
router_grid_rows = []
block_rows_all = []

for fraction in LABEL_FRACTIONS:
    print("\n", "=" * 80)
    print("10K VALIDATION FRACTION:", fraction)

    labeled_train_tune = exact_sparse_subset(
        train_df,
        fraction,
        TUNING_SEED,
    )

    # 1) Residual candidate selection.
    residual_rows = []

    for hp_id, hp in enumerate(
        RESIDUAL_CANDIDATES[
            float(fraction)
        ]
    ):
        model = fit_residual(
            labeled_train_tune,
            FEATURE_COLUMNS,
            hp,
            TUNING_SEED,
            N_TREES_FAST_TUNE,
        )
        pred = predict_residual(
            model,
            val_df,
            FEATURE_COLUMNS,
        )
        sc = regression_metrics(
            val_df["target"],
            pred,
        )

        row = {
            "fraction": float(fraction),
            "hp_id": int(hp_id),
            **hp,
            "val_MAE": sc["MAE"],
            "val_RMSE": sc["RMSE"],
        }
        residual_rows.append(row)
        residual_grid_rows.append(row)

    residual_grid = pd.DataFrame(
        residual_rows
    )

    assert len(residual_grid) == len(
        RESIDUAL_CANDIDATES[float(fraction)]
    )
    assert np.isfinite(
        residual_grid[["val_MAE", "val_RMSE"]].to_numpy(float)
    ).all()

    best_res = (
        residual_grid.sort_values(
            ["val_MAE", "val_RMSE", "hp_id"],
            kind="stable",
        )
        .iloc[0]
    )

    residual_hp = {
        "min_samples_leaf": int(
            best_res["min_samples_leaf"]
        ),
        "max_features": float(
            best_res["max_features"]
        ),
    }

    print(
        "Selected residual HP:",
        residual_hp,
    )

    # 2/3/4) Joint inherited SF protocol + frozen router-grid selection.
    all_candidates = []

    for sf_id, sf_protocol in enumerate(
        SF_CANDIDATES[
            float(fraction)
        ]
    ):
        router, oof = fit_router(
            labeled_train_tune,
            train_df,
            FEATURE_COLUMNS,
            residual_hp,
            sf_protocol,
            seed=TUNING_SEED,
            n_trees_expert=N_TREES_FAST_TUNE,
        )

        val_expert, _ = full_expert_frame(
            labeled_train_tune,
            train_df,
            val_df,
            val_df,
            FEATURE_COLUMNS,
            residual_hp,
            sf_protocol,
            seed=TUNING_SEED,
            n_trees=N_TREES_FAST_TUNE,
        )

        pred_r = val_expert[
            "pred_r"
        ].to_numpy(float)

        residual_val_mae = regression_metrics(
            val_df["target"],
            pred_r,
        )["MAE"]

        for cfg_id, cfg in enumerate(
            ROUTER_GRID_FINAL
        ):
            pred, route_audit = (
                route_prediction_final(
                    router,
                    val_expert,
                    FEATURE_COLUMNS,
                    **cfg,
                )
            )

            sc = regression_metrics(
                val_df["target"],
                pred,
            )
            tail = segment_delta_stats(
                val_df,
                pred,
                pred_r,
            )
            stability = validation_block_metrics(
                val_df,
                pred,
                pred_r,
            )

            row = {
                "fraction": float(fraction),
                "sf_id": int(sf_id),
                "cfg_id": int(cfg_id),
                "is_fail_closed_null": bool(
                    float(cfg["sf_weight"]) == 0.0
                ),
                **sf_protocol,
                **cfg,
                "Residual_val_MAE": float(
                    residual_val_mae
                ),
                "val_MAE": float(
                    sc["MAE"]
                ),
                "improvement_vs_Residual_pct": float(
                    100.0
                    * (
                        residual_val_mae
                        - sc["MAE"]
                    )
                    / residual_val_mae
                ),
                **route_audit,
                **tail,
                **{
                    k: v
                    for k, v in stability.items()
                    if k != "block_rows"
                },
                "router_oof_postwarmup_rows": int(
                    len(oof)
                ),
            }

            all_candidates.append(row)
            router_grid_rows.append(row)

            for br in stability[
                "block_rows"
            ]:
                block_rows_all.append({
                    "fraction": float(fraction),
                    "sf_id": int(sf_id),
                    "cfg_id": int(cfg_id),
                    **sf_protocol,
                    **cfg,
                    **br,
                })

        del router, oof, val_expert
        gc.collect()

    grid = pd.DataFrame(
        all_candidates
    )

    assert len(grid) == (
        len(SF_CANDIDATES[float(fraction)])
        * len(ROUTER_GRID_FINAL)
    )
    assert np.isfinite(
        grid[
            [
                "Residual_val_MAE",
                "val_MAE",
                "improvement_vs_Residual_pct",
                "worst_block_improvement_pct",
            ]
        ].to_numpy(float)
    ).all()

    eligible = grid[
        grid["all_blocks_non_degrading"]
    ].copy()

    assert len(eligible) >= 1, (
        "No validation-stable 10K candidate. "
        "The routed-null candidate should make this impossible."
    )

    best = (
        eligible.sort_values(
            [
                "worst_block_improvement_pct",
                "improvement_vs_Residual_pct",
                "segment_p95_delta_pp",
                "segment_worst_delta_pp",
                "switch_rate",
                "val_MAE",
                "sf_id",
                "cfg_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    selected_10k[
        float(fraction)
    ] = {
        "residual_hp": residual_hp,
        "sf_protocol": {
            "early_window_s": float(
                best["early_window_s"]
            ),
            "slow_strength": float(
                best["slow_strength"]
            ),
            "tau_pp": float(
                best["tau_pp"]
            ),
        },
        "router_cfg": {
            "mode": "Routed",
            "kappa": float(
                best["kappa"]
            ),
            "margin": float(
                best["margin"]
            ),
            "sf_weight": float(
                best["sf_weight"]
            ),
        },
        "selected_validation_MAE": float(
            best["val_MAE"]
        ),
        "Residual_validation_MAE": float(
            best["Residual_val_MAE"]
        ),
        "worst_block_improvement_pct": float(
            best["worst_block_improvement_pct"]
        ),
        "selected_fail_closed": bool(
            best["is_fail_closed_null"]
        ),
    }

    print(
        "Selected 10K EVE protocol:",
        selected_10k[float(fraction)],
    )

pd.DataFrame(
    residual_grid_rows
).to_csv(
    K10_OUT / "12B_v4_10K_residual_validation_grid.csv",
    index=False,
)

pd.DataFrame(
    router_grid_rows
).to_csv(
    K10_OUT / "12B_v4_10K_sf_router_validation_grid.csv",
    index=False,
)

pd.DataFrame(
    block_rows_all
).to_csv(
    K10_OUT / "12B_v4_10K_validation_block_audit.csv",
    index=False,
)

selected_rows = []

for fraction, cfg in selected_10k.items():
    selected_rows.append({
        "fraction": fraction,
        **{
            f"residual_{k}": v
            for k, v in cfg[
                "residual_hp"
            ].items()
        },
        **cfg["sf_protocol"],
        **cfg["router_cfg"],
        "selected_validation_MAE": cfg[
            "selected_validation_MAE"
        ],
        "Residual_validation_MAE": cfg[
            "Residual_validation_MAE"
        ],
        "worst_block_improvement_pct": cfg[
            "worst_block_improvement_pct"
        ],
        "selected_fail_closed": cfg[
            "selected_fail_closed"
        ],
    })

selected_10k_table = pd.DataFrame(
    selected_rows
)

selected_10k_table.to_csv(
    K10_OUT / "12B_v4_10K_selected_protocols.csv",
    index=False,
)

display(selected_10k_table)

# %% [markdown] 11. Fit and freeze the final five-seed 10K bundles

# %% Code cell 14 (notebook cell 26)
def fit_final_10k_bundle(
    labeled_train,
    full_train,
    feature_cols,
    residual_hp,
    sf_protocol,
    router_cfg,
    seed,
):
    # OOF router exactly as in 10K.
    router, oof = fit_router(
        labeled_train,
        full_train,
        feature_cols,
        residual_hp,
        sf_protocol,
        seed=seed,
        n_trees_expert=N_TREES_FAST_TUNE,
    )

    # Final residual backbone.
    residual_model = fit_residual(
        labeled_train,
        feature_cols,
        residual_hp,
        seed,
        N_TREES_FAST_FINAL,
    )

    train_context = build_early_context(
        full_train,
        feature_cols,
        sf_protocol["early_window_s"],
    )

    slow_oof = oof_slow_predictions(
        labeled_train,
        train_context,
        seed=seed + 100,
    )

    tr_state = add_slow_state(
        labeled_train,
        slow_oof,
        sf_protocol["slow_strength"],
        sf_protocol["tau_pp"],
    )
    tr_state = enforce_strict_warmup_state(
        tr_state,
        sf_protocol["early_window_s"],
    )

    slow_a_model, slow_b_model, ctx_cols = (
        full_slow_models(
            labeled_train,
            train_context,
            seed=seed + 500,
        )
    )

    fast_features = list(feature_cols) + [
        "sf2_base",
        "slow_confidence",
        "slow_correction_pp",
    ]

    fast_model = make_et(
        residual_hp,
        seed + 900,
        N_TREES_FAST_FINAL,
    )

    fast_model.fit(
        tr_state[fast_features],
        tr_state["target"].to_numpy(float)
        - tr_state["sf2_base"].to_numpy(float),
    )

    return {
        "residual_model": residual_model,
        "router": router,
        "slow_a_model": slow_a_model,
        "slow_b_model": slow_b_model,
        "slow_context_columns": ctx_cols,
        "fast_model": fast_model,
        "fast_features": fast_features,
        "feature_columns": list(feature_cols),
        "residual_hp": residual_hp,
        "sf_protocol": sf_protocol,
        "router_cfg": router_cfg,
        "seed": int(seed),
        "beta_pp_per_ah": float(
            beta_pp_per_ah
        ),
        "n_router_oof_rows": int(len(oof)),
    }


def predict_10k_bundle(
    bundle,
    pred_rows,
):
    feature_cols = bundle[
        "feature_columns"
    ]
    sf_protocol = bundle[
        "sf_protocol"
    ]

    pred_r = predict_residual(
        bundle["residual_model"],
        pred_rows,
        feature_cols,
    )

    pred_context = build_early_context(
        pred_rows,
        feature_cols,
        sf_protocol["early_window_s"],
    )

    if (
        bundle["slow_a_model"] is None
        or bundle["slow_b_model"] is None
        or bundle["slow_context_columns"] is None
    ):
        slow_pred = pred_context[[
            "compound_segment"
        ]].copy()
        slow_pred["a_pred"] = 0.0
        slow_pred["b_pred"] = 0.0
        slow_pred["a_std"] = np.inf
        slow_pred["b_std"] = np.inf
    else:
        slow_pred = predict_slow_pair(
            bundle["slow_a_model"],
            bundle["slow_b_model"],
            pred_context,
            bundle["slow_context_columns"],
        )

    pp = add_slow_state(
        pred_rows,
        slow_pred,
        sf_protocol["slow_strength"],
        sf_protocol["tau_pp"],
    )
    pp = enforce_strict_warmup_state(
        pp,
        sf_protocol["early_window_s"],
    )

    pred_sf_unmasked = (
        pp["sf2_base"].to_numpy(float)
        + bundle["fast_model"].predict(
            pp[bundle["fast_features"]]
        )
    )

    eligible = causal_eligible_mask(
        pp,
        sf_protocol["early_window_s"],
    )

    pred_sf = np.where(
        eligible,
        pred_sf_unmasked,
        pred_r,
    )

    expert = pred_rows.copy()
    expert["pred_r"] = pred_r
    expert["pred_sf"] = pred_sf
    expert["causal_eligible"] = eligible

    pred_final, route_audit = (
        route_prediction_final(
            bundle["router"],
            expert,
            feature_cols,
            **bundle["router_cfg"],
        )
    )

    return {
        "Residual": pred_r,
        "StrictSF_ablation": pred_sf,
        "SC_RS_SFPEF": pred_final,
        "route_audit": route_audit,
    }


def sha256_file_local(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


k10_validation_rows = []
k10_bundle_manifest = []

for fraction in LABEL_FRACTIONS:
    cfg = selected_10k[
        float(fraction)
    ]

    for seed in FINAL_SEEDS:
        labeled_train = exact_sparse_subset(
            train_df,
            fraction,
            seed,
        )

        bundle = fit_final_10k_bundle(
            labeled_train,
            train_df,
            FEATURE_COLUMNS,
            cfg["residual_hp"],
            cfg["sf_protocol"],
            cfg["router_cfg"],
            seed,
        )

        bundle_path = (
            K10_BUNDLES
            / f"EVE_10K_fraction{fraction}_seed{seed}.joblib"
        )
        joblib.dump(
            bundle,
            bundle_path,
            compress=3,
        )

        # Validation scoring is allowed and is recorded only for audit.
        pv = predict_10k_bundle(
            bundle,
            val_df,
        )

        for model_name in [
            "Residual",
            "StrictSF_ablation",
            "SC_RS_SFPEF",
        ]:
            sc = regression_metrics(
                val_df["target"],
                pv[model_name],
            )
            k10_validation_rows.append({
                "fraction": float(fraction),
                "seed": int(seed),
                "model": model_name,
                "n_train_labels": int(
                    len(labeled_train)
                ),
                "n_validation": int(
                    len(val_df)
                ),
                "evaluation_role": "validation_only_pre_confirmatory",
                **sc,
            })

        k10_bundle_manifest.append({
            "fraction": float(fraction),
            "seed": int(seed),
            "bundle_path": str(bundle_path),
            "bundle_sha256": sha256_file_local(bundle_path),
            "n_train_labels": int(
                len(labeled_train)
            ),
            "router_oof_rows": int(
                bundle["n_router_oof_rows"]
            ),
        })

        del bundle, pv
        gc.collect()

k10_validation = pd.DataFrame(
    k10_validation_rows
)

k10_validation.to_csv(
    K10_OUT / "12B_v4_10K_validation_runs.csv",
    index=False,
)

pd.DataFrame(
    k10_bundle_manifest
).to_csv(
    K10_OUT / "12B_v4_10K_bundle_manifest.csv",
    index=False,
)

display(
    k10_validation.groupby(
        ["fraction", "model"],
        as_index=False,
    ).agg(
        validation_MAE_mean=("MAE", "mean"),
        validation_MAE_std=("MAE", "std"),
    )
)

# %% [markdown] 12. Frozen causal TCN / LSTM / GRU baselines

# %% Code cell 15 (notebook cell 28)
DEEP_FEATURES = list(FEATURE_COLUMNS)

# Hard leakage guard.
forbidden_deep_tokens = [
    "soc_phy",
    "physics",
    "target",
    "residual",
    "real state of charge",
    "true_soc",
    "future",
]

bad = [
    c for c in DEEP_FEATURES
    if any(
        tok in c.lower()
        for tok in forbidden_deep_tokens
    )
]

assert not bad, bad

with open(
    DEEP_OUT / "12B_v4_deep_feature_contract.json",
    "w",
) as f:
    json.dump(
        {
            "features": DEEP_FEATURES,
            "soc0_allowed": True,
            "soc_phy_allowed": False,
            "target_allowed": False,
            "real_soc_allowed": False,
            "sequence_length": SEQ_LEN,
        },
        f,
        indent=2,
    )


class TrainOnlyPreprocessor:
    def fit(self, X):
        X = np.asarray(
            X,
            dtype=np.float64,
        )
        med = np.nanmedian(
            X,
            axis=0,
        )
        med = np.where(
            np.isfinite(med),
            med,
            0.0,
        )

        Xi = np.where(
            np.isfinite(X),
            X,
            med[None, :],
        )

        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)

        std = np.where(
            np.isfinite(std)
            & (std > 1e-8),
            std,
            1.0,
        )

        self.median_ = med.astype(
            np.float32
        )
        self.mean_ = mean.astype(
            np.float32
        )
        self.std_ = std.astype(
            np.float32
        )
        return self

    def transform(self, X):
        X = np.asarray(
            X,
            dtype=np.float64,
        )
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
        y = np.asarray(
            y,
            dtype=np.float64,
        )
        self.mean_ = float(
            np.mean(y)
        )
        self.std_ = float(
            np.std(y)
        )

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


def prepare_split_frame(df, split):
    z = df[
        df["split"] == split
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
            index=frame[
                "row_uid"
            ].astype(str),
        ).to_dict()

        self.seg_start = (
            frame.groupby(
                "compound_segment",
                sort=False,
            )["seq_pos"]
            .min()
            .astype(int)
            .to_dict()
        )

        positions = []

        for uid in endpoint_row_uids:
            uid = str(uid)
            if uid in self.uid_to_pos:
                positions.append(
                    int(
                        self.uid_to_pos[uid]
                    )
                )

        self.positions = np.asarray(
            positions,
            dtype=int,
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        pos = int(
            self.positions[i]
        )
        row = self.frame.iloc[pos]
        seg = row[
            "compound_segment"
        ]
        seg0 = int(
            self.seg_start[seg]
        )

        start = max(
            seg0,
            pos - self.seq_len + 1,
        )

        x = self.X[
            start : pos + 1
        ]

        n = len(x)
        pad = self.seq_len - n

        out = np.zeros(
            (
                self.seq_len,
                self.X.shape[1] + 1,
            ),
            dtype=np.float32,
        )

        if n:
            out[
                pad:,
                :-1,
            ] = x
            out[
                pad:,
                -1,
            ] = 1.0

        y = float(
            row["target"]
        )
        y_scaled = float(
            self.target_scaler.transform(
                [y]
            )[0]
        )

        return (
            torch.from_numpy(out),
            torch.tensor(
                y_scaled,
                dtype=torch.float32,
            ),
            pos,
        )


def all_uids(frame):
    return frame[
        "row_uid"
    ].astype(str).tolist()

# %% Code cell 16 (notebook cell 29)
class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        kind="LSTM",
        hidden=64,
    ):
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
        return self.head(
            h
        ).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel,
        dilation,
    ):
        super().__init__()

        self.left_pad = (
            (kernel - 1)
            * dilation
        )

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        x = nn.functional.pad(
            x,
            (self.left_pad, 0),
        )
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel,
        dilation,
        dropout,
    ):
        super().__init__()

        self.c1 = CausalConv1d(
            in_ch,
            out_ch,
            kernel,
            dilation,
        )
        self.c2 = CausalConv1d(
            out_ch,
            out_ch,
            kernel,
            dilation,
        )

        self.act = nn.ReLU()
        self.drop = nn.Dropout(
            dropout
        )

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=1,
            )
        )

    def forward(self, x):
        residual = self.skip(x)
        y = self.drop(
            self.act(
                self.c1(x)
            )
        )
        y = self.drop(
            self.act(
                self.c2(y)
            )
        )
        return self.act(
            y + residual
        )


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

        self.tcn = nn.Sequential(
            *blocks
        )

        self.head = nn.Sequential(
            nn.LayerNorm(
                TCN_CHANNELS
            ),
            nn.Linear(
                TCN_CHANNELS,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        y = self.tcn(
            x.transpose(1, 2)
        )
        h = y[:, :, -1]
        return self.head(
            h
        ).squeeze(-1)


def build_deep_model(
    model_name,
    input_dim,
):
    if model_name == "TCN":
        return TCNRegressor(
            input_dim
        )

    if model_name in {
        "LSTM",
        "GRU",
    }:
        return RNNRegressor(
            input_dim,
            kind=model_name,
            hidden=HIDDEN,
        )

    raise ValueError(
        model_name
    )


def count_params(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

# %% Code cell 17 (notebook cell 30)
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cudnn",
    ):
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


def deep_contract_fingerprint(model_name, fraction, seed, features):
    contract = {
        "model": str(model_name),
        "fraction": float(fraction),
        "seed": int(seed),
        "features": list(features),
        "seq_len": int(SEQ_LEN),
        "hidden": int(HIDDEN),
        "tcn_channels": int(TCN_CHANNELS),
        "tcn_dilations": list(TCN_DILATIONS),
        "tcn_kernel": int(TCN_KERNEL),
        "dropout": float(DROPOUT),
        "lr": float(LR),
        "weight_decay": float(WEIGHT_DECAY),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "canonical_fingerprint_sha256": str(CANONICAL_FINGERPRINT),
    }
    payload = json.dumps(contract, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_grad_scaler():
    if not AMP_ENABLED:
        try:
            return torch.amp.GradScaler(
                "cuda",
                enabled=False,
            )
        except Exception:
            try:
                return torch.cuda.amp.GradScaler(
                    enabled=False
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
            enabled=True
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
            enabled=True
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


@torch.no_grad()
def deep_predict_loader(
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

    ps = np.concatenate(
        pred_scaled
    )
    positions = np.concatenate(
        positions
    ).astype(int)

    return (
        positions,
        target_scaler.inverse(ps),
    )


def train_deep_one(
    model_name,
    fraction,
    seed,
    train_frame,
    val_frame,
    X_train,
    X_val,
    train_endpoint_uids,
    preprocessor,
    target_scaler,
):
    set_all_seeds(seed)

    train_ds = CausalSequenceDataset(
        train_frame,
        X_train,
        train_endpoint_uids,
        target_scaler,
        SEQ_LEN,
    )

    val_ds = CausalSequenceDataset(
        val_frame,
        X_val,
        all_uids(val_frame),
        target_scaler,
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

    input_dim = (
        X_train.shape[1] + 1
    )

    model = build_deep_model(
        model_name,
        input_dim,
    ).to(DEVICE)

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
    contract_fp = deep_contract_fingerprint(
        model_name, fraction, seed, DEEP_FEATURES
    )

    best_path = (
        DEEP_CKPT
        / f"{tag}_best.pt"
    )
    resume_path = (
        DEEP_CKPT
        / f"{tag}_resume.pt"
    )
    history_path = (
        DEEP_OUT
        / f"history_{tag}.csv"
    )

    best_val = np.inf
    best_epoch = 0
    stale = 0
    history = []
    start_epoch = 1

    if resume_path.exists():
        state = safe_torch_load(
            resume_path,
            DEVICE,
        )

        assert state.get("contract_fingerprint") == contract_fp, (
            "Resume checkpoint contract mismatch",
            tag,
        )

        model.load_state_dict(
            state[
                "latest_model_state"
            ]
        )
        optimizer.load_state_dict(
            state[
                "optimizer_state"
            ]
        )

        if (
            scaler is not None
            and state.get(
                "scaler_state"
            )
        ):
            scaler.load_state_dict(
                state[
                    "scaler_state"
                ]
            )

        start_epoch = int(
            state["epoch"]
        ) + 1
        best_val = float(
            state["best_val"]
        )
        best_epoch = int(
            state["best_epoch"]
        )
        stale = int(
            state["stale"]
        )
        history = list(
            state["history"]
        )
        restore_rng_state(
            state.get("rng_state")
        )

        print(
            "RESUME",
            tag,
            "at epoch",
            start_epoch,
        )

    t0 = time.perf_counter()

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
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

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_context():
                pred = model(xb)
                loss = criterion(
                    pred,
                    yb,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{tag}: non-finite loss"
                )

            if (
                scaler is None
                or not AMP_ENABLED
            ):
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )
                optimizer.step()
            else:
                scaler.scale(
                    loss
                ).backward()
                scaler.unscale_(
                    optimizer
                )
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )
                scaler.step(
                    optimizer
                )
                scaler.update()

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        pos, val_pred = deep_predict_loader(
            model,
            val_loader,
            target_scaler,
        )
        yv = val_frame.iloc[
            pos
        ]["target"].to_numpy(float)

        val_mae = regression_metrics(
            yv,
            val_pred,
        )["MAE"]

        history.append({
            "epoch": int(epoch),
            "train_mse_scaled": float(
                np.mean(losses)
            ),
            "val_MAE": float(
                val_mae
            ),
        })

        if (
            val_mae
            < best_val - MIN_DELTA
        ):
            best_val = float(
                val_mae
            )
            best_epoch = int(
                epoch
            )
            stale = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "input_dim": int(
                        input_dim
                    ),
                    "best_epoch": best_epoch,
                    "best_val_MAE": best_val,
                    "n_params": count_params(
                        model
                    ),
                    "fraction": float(
                        fraction
                    ),
                    "seed": int(seed),
                    "features": DEEP_FEATURES,
                    "preprocessor": {
                        "median": preprocessor.median_,
                        "mean": preprocessor.mean_,
                        "std": preprocessor.std_,
                    },
                    "target_scaler": {
                        "mean": target_scaler.mean_,
                        "std": target_scaler.std_,
                    },
                    "sequence_length": SEQ_LEN,
                    "contract_fingerprint": contract_fp,
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
            pd.DataFrame(
                history
            ).to_csv(
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
                    "best_val": float(
                        best_val
                    ),
                    "best_epoch": int(
                        best_epoch
                    ),
                    "stale": int(stale),
                    "history": history,
                    "rng_state": capture_rng_state(),
                    "contract_fingerprint": contract_fp,
                },
                resume_path,
            )

        if stale >= PATIENCE:
            break

    final_epoch = int(
        history[-1]["epoch"]
    )

    state = safe_torch_load(
        best_path,
        DEVICE,
    )
    model.load_state_dict(
        state["model_state"]
    )

    pos, val_pred = deep_predict_loader(
        model,
        val_loader,
        target_scaler,
    )
    yv = val_frame.iloc[
        pos
    ]["target"].to_numpy(float)
    val_metrics = regression_metrics(
        yv,
        val_pred,
    )

    if resume_path.exists():
        resume_path.unlink()

    return {
        "best_path": str(best_path),
        "best_epoch": int(
            state["best_epoch"]
        ),
        "final_epoch": final_epoch,
        "hit_epoch_cap": bool(
            final_epoch >= MAX_EPOCHS
        ),
        "stopped_by_patience": bool(
            final_epoch < MAX_EPOCHS
        ),
        "n_params": int(
            state["n_params"]
        ),
        "n_train_labels": int(
            len(train_ds)
        ),
        "n_validation": int(
            len(val_ds)
        ),
        "train_seconds": float(
            time.perf_counter() - t0
        ),
        **{
            f"val_{k}": v
            for k, v in val_metrics.items()
        },
    }

# %% [markdown] 13. Train and freeze all temporal baselines

# %% Code cell 18 (notebook cell 32)
train_frame = prepare_split_frame(
    canon,
    "train",
)
val_frame = prepare_split_frame(
    canon,
    "val",
)

preprocessor = TrainOnlyPreprocessor().fit(
    train_frame[
        DEEP_FEATURES
    ].to_numpy(float)
)

X_train = preprocessor.transform(
    train_frame[
        DEEP_FEATURES
    ].to_numpy(float)
)
X_val = preprocessor.transform(
    val_frame[
        DEEP_FEATURES
    ].to_numpy(float)
)

DEEP_RUNS_PATH = (
    DEEP_OUT
    / "12B_v4_deep_runs_incremental.csv"
)

completed = set()

if DEEP_RUNS_PATH.exists():
    old = pd.read_csv(
        DEEP_RUNS_PATH
    )
    for r in old.itertuples(index=False):
        p = Path(str(r.checkpoint))
        if not p.exists():
            continue

        try:
            state = safe_torch_load(p, "cpu")
            expected_fp = deep_contract_fingerprint(
                str(r.model),
                float(r.fraction),
                int(r.seed),
                DEEP_FEATURES,
            )
            if state.get("contract_fingerprint") == expected_fp:
                completed.add(
                    (
                        float(r.fraction),
                        str(r.model),
                        int(r.seed),
                    )
                )
        except Exception:
            pass

for fraction in LABEL_FRACTIONS:
    for seed in FINAL_SEEDS:
        labeled_train = exact_sparse_subset(
            train_frame,
            fraction,
            seed,
        )

        target_scaler = TargetScaler().fit(
            labeled_train[
                "target"
            ].to_numpy(float)
        )

        endpoint_uids = (
            labeled_train[
                "row_uid"
            ].astype(str).tolist()
        )

        for model_name in DEEP_MODELS:
            key = (
                float(fraction),
                model_name,
                int(seed),
            )

            if key in completed:
                print(
                    "SKIP completed:",
                    key,
                )
                continue

            print(
                "TRAIN",
                key,
                "labels=",
                len(endpoint_uids),
            )

            result = train_deep_one(
                model_name,
                fraction,
                seed,
                train_frame,
                val_frame,
                X_train,
                X_val,
                endpoint_uids,
                preprocessor,
                target_scaler,
            )

            row = {
                "fraction": float(
                    fraction
                ),
                "model": model_name,
                "seed": int(seed),
                "checkpoint": result[
                    "best_path"
                ],
                "checkpoint_sha256": sha256_file_local(
                    result["best_path"]
                ),
                "evaluation_role": "validation_only_pre_confirmatory",
                **{
                    k: v
                    for k, v in result.items()
                    if k != "best_path"
                },
            }

            cur = (
                pd.read_csv(
                    DEEP_RUNS_PATH
                )
                if DEEP_RUNS_PATH.exists()
                else pd.DataFrame()
            )

            cur = pd.concat(
                [
                    cur,
                    pd.DataFrame(
                        [row]
                    ),
                ],
                ignore_index=True,
            )

            cur = cur.drop_duplicates(
                [
                    "fraction",
                    "model",
                    "seed",
                ],
                keep="last",
            )

            cur.to_csv(
                DEEP_RUNS_PATH,
                index=False,
            )

            completed.add(key)

            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

deep_runs = pd.read_csv(
    DEEP_RUNS_PATH
)

expected_runs = (
    len(LABEL_FRACTIONS)
    * len(FINAL_SEEDS)
    * len(DEEP_MODELS)
)

assert len(deep_runs) == expected_runs, (
    len(deep_runs),
    expected_runs,
)

deep_runs.to_csv(
    DEEP_OUT
    / "12B_v4_deep_runs.csv",
    index=False,
)

deep_summary = (
    deep_runs.groupby(
        ["fraction", "model"],
        as_index=False,
    )
    .agg(
        validation_MAE_mean=(
            "val_MAE",
            "mean",
        ),
        validation_MAE_std=(
            "val_MAE",
            "std",
        ),
        best_epoch_mean=(
            "best_epoch",
            "mean",
        ),
        final_epoch_mean=(
            "final_epoch",
            "mean",
        ),
        seeds_hitting_epoch_200=(
            "hit_epoch_cap",
            "sum",
        ),
        n_params=(
            "n_params",
            "first",
        ),
    )
)

deep_summary.to_csv(
    DEEP_OUT
    / "12B_v4_deep_validation_summary.csv",
    index=False,
)

display(deep_summary)

# %% [markdown] 14. Freeze artifacts for one-shot 12C

# %% Code cell 19 (notebook cell 34)
# Integrity checks.
assert set(canon["split"].unique()) == {
    "train",
    "val",
}
assert (canon["timestamp"] < VAL_END_EXCLUSIVE).all()
assert len(selected_10k) == 2
assert len(k10_bundle_manifest) == 10
assert len(deep_runs) == 30

for x in k10_bundle_manifest:
    assert Path(
        x["bundle_path"]
    ).exists()

for r in deep_runs.itertuples(index=False):
    p = Path(str(r.checkpoint))
    assert p.exists()
    assert sha256_file_local(p) == str(r.checkpoint_sha256)

# 12C canonicalization contract.
canonicalization_contract = {
    "feature_paths": {k: v["path"] for k, v in REQUIRED_CHANNELS.items()},
    "feature_columns": FEATURE_COLUMNS,
    "feature_file_ids": {k: int(v["file_id"]) for k, v in REQUIRED_CHANNELS.items()},
    "feature_medians_train_only": feature_medians,
    "prior": "SIGNED_AH",
    "current_sign_handling": (
        "raw EVE signed current as encoded; beta fitted on train only"
    ),
    "coord_definition": "Ah_cum - Ah_at_SOC0_anchor",
    "beta_pp_per_ah": float(
        beta_pp_per_ah
    ),
    "burst_break_s": BURST_BREAK_S,
    "alignment_tolerance_s": ALIGN_TOLERANCE_S,
    "soc0_rule": ("first dashboard SoC observation in each current-defined acquisition segment"),
    "distance_anchor_policy": "exact first canonical row; never first non-null",
    "raw_context_windows_s": unique_windows,
    "raw_context_signals": RAW_CONTEXT_SIGNALS,
    "confirmatory_target_sha256_expected": EXPECTED_TEST_SHA256,
    "canonical_fingerprint_sha256": CANONICAL_FINGERPRINT,
}

with open(
    OUT / "12B_v4_canonicalization_contract_for_12C.json",
    "w",
) as f:
    json.dump(
        canonicalization_contract,
        f,
        indent=2,
    )

manifest = {
    "experiment": "12B_V4_EVE_FROZEN_TRAINVAL_MODEL_PREPARATION",
    "dataset": "EVE Pilot Renault ZOE Q90",
    "target": "dashboard/user-facing BMS-reported State of Charge",
    "confirmatory_test_target_opened": False,
    "confirmatory_test_scored": False,
    "confirmatory_test_expected_rows": int(
        seal12a["n_timestamped_rows"]
    ),
    "confirmatory_test_expected_sha256": EXPECTED_TEST_SHA256,
    "prior": "SIGNED_AH",
    "current_polarity": (
        "not hard-coded; raw signed current + train-only beta"
    ),
    "prior_beta_pp_per_ah_train_only": float(
        beta_pp_per_ah
    ),
    "feature_columns": FEATURE_COLUMNS,
    "feature_file_ids": {k: int(v["file_id"]) for k, v in REQUIRED_CHANNELS.items()},
    "real_state_of_charge_predictor": False,
    "10K": {
        "architecture": (
            "Residual -> strict-causal SF -> OOF residual-safe router -> Residual fallback"
        ),
        "candidate_source_residual": str(
            PROTOCOL_10B
        ),
        "candidate_source_sf": str(
            PROTOCOL_10C
        ),
        "router_grid_frozen": ROUTER_GRID_FINAL,
        "selected_protocols_file": str(
            K10_OUT
            / "12B_v4_10K_selected_protocols.csv"
        ),
        "bundle_manifest_file": str(
            K10_OUT
            / "12B_v4_10K_bundle_manifest.csv"
        ),
        "five_seed_bundles_frozen": True,
    },
    "deep_baselines": {
        "models": DEEP_MODELS,
        "sequence_length": SEQ_LEN,
        "hidden": HIDDEN,
        "TCN_channels": TCN_CHANNELS,
        "TCN_kernel": TCN_KERNEL,
        "TCN_dilations": TCN_DILATIONS,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seeds": FINAL_SEEDS,
        "checkpoint_manifest": str(
            DEEP_OUT
            / "12B_v4_deep_runs.csv"
        ),
        "architecture_tuning_on_EVE": False,
    },
    "supervision_fractions": LABEL_FRACTIONS,
    "validation_only_protocol_selection": True,
    "test_tuning_allowed": False,
    "status": "FROZEN_READY_FOR_ONE_SHOT_12C",
    "implementation_version": "12B-v4",
    "canonical_fingerprint_sha256": CANONICAL_FINGERPRINT,
    "current_sign_audit_file": str(
        OUT / "12B_v4_current_sign_audit_train_only.json"
    ),
    "channel_provenance_file": str(
        OUT / "12B_v4_channel_source_provenance.csv"
    ),
    "network_policy": (
        "offline-first: v3 cache -> v2 cache -> 12A local channel -> Dataverse retry"
    ),
    "12C_rule": (
        "Open the sealed confirmatory target once, reproduce the frozen canonicalization, "
        "load frozen 10K bundles and deep checkpoints, score all methods, and make no changes afterward."
    ),
}

with open(
    OUT / "12B_v4_FREEZE_MANIFEST.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )

print(
    json.dumps(
        manifest,
        indent=2,
    )
)

# %% [markdown] STOP — do not score the confirmatory test here
