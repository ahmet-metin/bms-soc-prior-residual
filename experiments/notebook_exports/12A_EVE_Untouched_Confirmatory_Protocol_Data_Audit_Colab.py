# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/12A_EVE_Untouched_Confirmatory_Protocol_Data_Audit_Colab.ipynb
# Notebook SHA256: f36eba2bc9220c7b4e1039ba420b46c103247e1677ea4f8fe7769e47709bb08b
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 12A — EVE Untouched Confirmatory Protocol + Data Audit

# %% Code cell 1 (notebook cell 2)
# Core packages are already available in Colab.
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import os, io, re, json, time, math, hashlib, warnings
from urllib.parse import unquote
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")

OUT = ROOT / "outputs" / "eve_12A_untouched_audit"
RAW = ROOT / "data" / "EVE_Pilot"
DOCS = RAW / "documentation"
SELECTED = RAW / "selected_channels"

for p in [OUT, RAW, DOCS, SELECTED]:
    p.mkdir(parents=True, exist_ok=True)

DATAVERSE = "https://entrepot.recherche.data.gouv.fr"
PERSISTENT_ID = "doi:10.57745/5O6QIH"

# Published collection window. Split is fixed from metadata BEFORE model evaluation.
PUBLISHED_START = pd.Timestamp("2020-10-26 00:00:00")
PUBLISHED_END = pd.Timestamp("2023-10-18 23:59:59")

# Fixed approximately 60/20/20 chronological calendar split.
# These boundaries are fixed now and must not be moved after seeing model performance.
TRAIN_END_EXCLUSIVE = pd.Timestamp("2022-08-10 00:00:00")
VAL_END_EXCLUSIVE = pd.Timestamp("2023-03-15 00:00:00")

PRIMARY_TARGET_STEM = "state of charge"
FORBIDDEN_TARGET_TOKENS = [
    "state of charge",
    "soc",
]

# Predictor-only gate for signed-Ah eligibility.
# Gate is intentionally conservative because current integration is sensitive to
# intermittent telemetry.
CURRENT_GATE = {
    "min_trainval_current_observations": 1000,
    "max_median_within_burst_gap_s": 10.0,
    "max_p95_within_burst_gap_s": 30.0,
    "min_distance_rows_with_current_within_30s": 0.80,
    "burst_break_s": 300.0,
    "alignment_tolerance_s": 30.0,
}

print("Output:", OUT)
print("Raw data:", RAW)
print("TRAIN:", PUBLISHED_START, "to <", TRAIN_END_EXCLUSIVE)
print("VAL:  ", TRAIN_END_EXCLUSIVE, "to <", VAL_END_EXCLUSIVE)
print("TEST: ", VAL_END_EXCLUSIVE, "to", PUBLISHED_END)

# %% [markdown] 1. Query the official Dataverse metadata

# %% Code cell 2 (notebook cell 4)
def get_json(url, params=None, timeout=120):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

meta_url = f"{DATAVERSE}/api/datasets/:persistentId/"
meta = get_json(
    meta_url,
    params={"persistentId": PERSISTENT_ID},
)

assert meta.get("status") == "OK", meta
data = meta["data"]
latest = data["latestVersion"]

dataset_version = (
    f'{latest.get("versionNumber")}.'
    f'{latest.get("versionMinorNumber")}'
)
dataset_id = data.get("id")
dataset_pid = data.get("persistentUrl") or PERSISTENT_ID

files = latest.get("files", [])
assert len(files) > 0, "No files returned by Dataverse API."

catalog_rows = []

for item in files:
    df = item.get("dataFile", {})
    directory = item.get("directoryLabel", "") or ""
    label = df.get("filename", "")
    description = item.get("description", "") or ""
    categories = item.get("categories", []) or []

    catalog_rows.append({
        "file_id": df.get("id"),
        "filename": label,
        "directory": directory,
        "path": f"{directory}/{label}" if directory else label,
        "content_type": df.get("contentType"),
        "filesize_bytes": df.get("filesize"),
        "description": description,
        "categories": "|".join(categories),
        "md5": (df.get("checksum") or {}).get("value"),
        "checksum_type": (df.get("checksum") or {}).get("type"),
        "restricted": bool(df.get("restricted", False)),
    })

catalog = pd.DataFrame(catalog_rows)
catalog["path_lower"] = catalog["path"].astype(str).str.lower()

catalog.to_csv(OUT / "12A_official_file_catalog.csv", index=False)

print("Dataset version:", dataset_version)
print("Repository files:", len(catalog))
print("Total catalog size (MB):", round(catalog["filesize_bytes"].fillna(0).sum()/1e6, 1))
display(catalog[[
    "file_id", "path", "content_type", "filesize_bytes", "restricted"
]].sort_values("path").head(40))

# %% [markdown] 2. Lock target, forbidden-label channels and candidate predictor channels

# %% Code cell 3 (notebook cell 6)
def normalized_stem(path):
    name = Path(str(path)).name
    stem = re.sub(r"\.(tab|csv|tsv|txt)$", "", name, flags=re.I)
    return re.sub(r"\s+", " ", stem.strip().lower())

catalog["normalized_stem"] = catalog["path"].map(normalized_stem)

target_candidates = catalog[
    catalog["normalized_stem"].eq(PRIMARY_TARGET_STEM)
].copy()

assert len(target_candidates) == 1, (
    "Expected exactly one primary target file named 'State of Charge'.",
    target_candidates[["file_id", "path"]].to_dict("records"),
)

TARGET_ROW = target_candidates.iloc[0].to_dict()

forbidden = catalog[
    catalog["normalized_stem"].str.contains(
        r"(^|[^a-z])soc([^a-z]|$)|state of charge",
        regex=True,
        na=False,
    )
].copy()

# Rank likely traction-battery-current files using names only.
def current_name_score(path):
    s = str(path).lower()
    score = 0
    if "current" in s:
        score += 2
    if "battery" in s:
        score += 3
    if "hv" in s or "high voltage" in s:
        score += 4
    if "traction" in s:
        score += 4

    # Penalize channels that are likely charging-side / auxiliary currents.
    for bad in [
        "charging", "charger", "mains", "pilot",
        "12v", "14v", "cell", "ac current",
    ]:
        if bad in s:
            score -= 4
    return score

current_candidates = catalog[
    catalog["path_lower"].str.contains("current", na=False)
].copy()

if len(current_candidates):
    current_candidates["name_score"] = (
        current_candidates["path"].map(current_name_score)
    )
    current_candidates = current_candidates.sort_values(
        ["name_score", "filesize_bytes"],
        ascending=[False, False],
    )

distance_candidates = catalog[
    catalog["path_lower"].str.contains(
        r"total vehicle distance|odometer|distance",
        regex=True,
        na=False,
    )
].copy()

speed_candidates = catalog[
    catalog["normalized_stem"].eq("speed")
].copy()

selection_audit = {
    "primary_target": TARGET_ROW["path"],
    "primary_target_file_id": int(TARGET_ROW["file_id"]),
    "forbidden_target_like_paths": forbidden["path"].tolist(),
    "current_candidates_ranked": current_candidates["path"].tolist(),
    "distance_candidates": distance_candidates["path"].tolist(),
    "speed_candidates": speed_candidates["path"].tolist(),
}

with open(OUT / "12A_channel_selection_metadata_only.json", "w") as f:
    json.dump(selection_audit, f, indent=2)

print("PRIMARY TARGET:", TARGET_ROW["path"])
print("\nFORBIDDEN TARGET-LIKE FILES:")
display(forbidden[["file_id", "path", "filesize_bytes"]])

print("\nCURRENT CANDIDATES (metadata-only ranking):")
display(current_candidates[[
    "file_id", "path", "name_score", "filesize_bytes"
]].head(20))

print("\nDISTANCE CANDIDATES:")
display(distance_candidates[["file_id", "path", "filesize_bytes"]].head(20))

# %% [markdown] 3. Download repository documentation

# %% Code cell 4 (notebook cell 8)
def download_datafile(file_id, out_dir, fallback_name, force=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{DATAVERSE}/api/access/datafile/{int(file_id)}"
    with requests.get(
        url,
        params={"format": "original"},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()

        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd, flags=re.I)
        if m:
            name = unquote(m.group(1)).strip().strip('"')
        else:
            name = fallback_name

        dest = out_dir / name

        if dest.exists() and dest.stat().st_size > 0 and not force:
            return dest

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return dest


doc_mask = (
    catalog["path_lower"].str.contains("readme", na=False)
    | catalog["path_lower"].str.contains("data-paper", na=False)
    | catalog["path_lower"].str.endswith(".pdf", na=False)
)

docs = catalog[doc_mask].copy()

downloaded_docs = []
for r in docs.itertuples(index=False):
    try:
        p = download_datafile(
            r.file_id,
            DOCS,
            Path(r.path).name,
        )
        downloaded_docs.append({
            "file_id": r.file_id,
            "repository_path": r.path,
            "local_path": str(p),
            "size_bytes": p.stat().st_size,
        })
    except Exception as e:
        downloaded_docs.append({
            "file_id": r.file_id,
            "repository_path": r.path,
            "local_path": None,
            "error": repr(e),
        })

docs_df = pd.DataFrame(downloaded_docs)
docs_df.to_csv(OUT / "12A_documentation_download_audit.csv", index=False)
display(docs_df)

# %% [markdown] 4. Freeze the chronological outer split

# %% Code cell 5 (notebook cell 10)
split_contract = {
    "policy": "fixed_calendar_chronological_approximately_60_20_20",
    "source_of_boundaries": "published repository collection dates; fixed before model evaluation",
    "published_start": PUBLISHED_START.isoformat(),
    "published_end": PUBLISHED_END.isoformat(),
    "train": {
        "start_inclusive": PUBLISHED_START.isoformat(),
        "end_exclusive": TRAIN_END_EXCLUSIVE.isoformat(),
    },
    "validation": {
        "start_inclusive": TRAIN_END_EXCLUSIVE.isoformat(),
        "end_exclusive": VAL_END_EXCLUSIVE.isoformat(),
    },
    "confirmatory_test": {
        "start_inclusive": VAL_END_EXCLUSIVE.isoformat(),
        "end_inclusive": PUBLISHED_END.isoformat(),
        "status": "SEALED_UNTIL_FINAL_SCORING",
    },
}

with open(OUT / "12A_frozen_outer_split.json", "w") as f:
    json.dump(split_contract, f, indent=2)

print(json.dumps(split_contract, indent=2))

# %% [markdown] 5. Download only the minimum channels needed for the audit

# %% Code cell 6 (notebook cell 12)
assert len(distance_candidates) > 0, "No distance-like repository file found."

# Prefer the exact 'Total vehicle distance' channel when available.
exact_distance = distance_candidates[
    distance_candidates["normalized_stem"].eq("total vehicle distance")
]
DIST_ROW = (
    exact_distance.iloc[0].to_dict()
    if len(exact_distance)
    else distance_candidates.iloc[0].to_dict()
)

CURRENT_ROW = (
    current_candidates.iloc[0].to_dict()
    if len(current_candidates)
    else None
)

SPEED_ROW = (
    speed_candidates.iloc[0].to_dict()
    if len(speed_candidates)
    else None
)

selected_rows = [
    ("target", TARGET_ROW),
    ("distance", DIST_ROW),
]
if CURRENT_ROW is not None:
    selected_rows.append(("current_candidate", CURRENT_ROW))
if SPEED_ROW is not None:
    selected_rows.append(("speed", SPEED_ROW))

selected_downloads = []

for role, r in selected_rows:
    p = download_datafile(
        r["file_id"],
        SELECTED,
        Path(r["path"]).name,
    )
    selected_downloads.append({
        "role": role,
        "file_id": int(r["file_id"]),
        "repository_path": r["path"],
        "local_path": str(p),
        "size_bytes": p.stat().st_size,
    })

selected_downloads = pd.DataFrame(selected_downloads)
selected_downloads.to_csv(
    OUT / "12A_selected_channel_downloads.csv",
    index=False,
)
display(selected_downloads)

# %% [markdown] 6. Generic schema audit

# %% Code cell 7 (notebook cell 14)
def sniff_sep(path):
    with open(path, "rb") as f:
        sample = f.read(65536)
    text = sample.decode("utf-8", errors="ignore")
    counts = {
        ",": text.count(","),
        "\t": text.count("\t"),
        ";": text.count(";"),
    }
    return max(counts, key=counts.get)


def read_channel(path):
    path = Path(path)
    sep = sniff_sep(path)

    try:
        df = pd.read_csv(
            path,
            sep=sep,
            low_memory=False,
        )
    except Exception:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
        )

    # Drop fully empty columns.
    df = df.dropna(axis=1, how="all")
    return df


def detect_time_and_value_columns(df):
    assert len(df.columns) >= 2, df.columns.tolist()

    # Timestamp candidates by column name first.
    named_time = [
        c for c in df.columns
        if re.search(r"time|date|timestamp", str(c), flags=re.I)
    ]

    time_scores = []
    for c in df.columns:
        s = df[c]
        parsed = pd.to_datetime(
            s.head(min(5000, len(s))),
            errors="coerce",
            utc=True,
        )
        score = float(parsed.notna().mean())
        bonus = 0.2 if c in named_time else 0.0
        time_scores.append((score + bonus, score, c))

    time_scores.sort(reverse=True, key=lambda x: x[0])
    _, time_parse_rate, time_col = time_scores[0]

    # Numeric value column among non-time fields.
    value_scores = []
    for c in df.columns:
        if c == time_col:
            continue
        x = pd.to_numeric(
            df[c].head(min(5000, len(df))),
            errors="coerce",
        )
        score = float(x.notna().mean())
        value_scores.append((score, c))

    value_scores.sort(reverse=True)
    value_rate, value_col = value_scores[0]

    if time_parse_rate < 0.70:
        raise ValueError(
            f"Could not confidently detect timestamp column. "
            f"Best={time_col}, parse_rate={time_parse_rate:.3f}"
        )
    if value_rate < 0.70:
        raise ValueError(
            f"Could not confidently detect numeric value column. "
            f"Best={value_col}, numeric_rate={value_rate:.3f}"
        )

    return time_col, value_col


channel_frames = {}
schema_rows = []

for r in selected_downloads.itertuples(index=False):
    df = read_channel(r.local_path)
    tcol, vcol = detect_time_and_value_columns(df)

    # Standardized two-column representation.
    z = pd.DataFrame({
        "timestamp": pd.to_datetime(
            df[tcol],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None),
        "value": pd.to_numeric(
            df[vcol],
            errors="coerce",
        ),
    }).dropna(subset=["timestamp"])

    z = z.sort_values("timestamp").reset_index(drop=True)
    channel_frames[r.role] = z

    schema_rows.append({
        "role": r.role,
        "repository_path": r.repository_path,
        "raw_rows": int(len(df)),
        "detected_time_column": str(tcol),
        "detected_value_column": str(vcol),
        "timestamp_nonnull_fraction": float(z["timestamp"].notna().mean()),
        "numeric_nonnull_fraction": float(z["value"].notna().mean()),
        "first_timestamp": (
            z["timestamp"].min().isoformat()
            if len(z) else None
        ),
        "last_timestamp": (
            z["timestamp"].max().isoformat()
            if len(z) else None
        ),
    })

schema_audit = pd.DataFrame(schema_rows)
schema_audit.to_csv(OUT / "12A_selected_channel_schema_audit.csv", index=False)

display(schema_audit)

# %% [markdown] 7. Materialize train/validation target and cryptographically seal confirmatory-test target

# %% Code cell 8 (notebook cell 16)
def split_by_frozen_calendar(z):
    t = z["timestamp"]
    train = z[
        (t >= PUBLISHED_START)
        & (t < TRAIN_END_EXCLUSIVE)
    ].copy()
    val = z[
        (t >= TRAIN_END_EXCLUSIVE)
        & (t < VAL_END_EXCLUSIVE)
    ].copy()
    test = z[
        (t >= VAL_END_EXCLUSIVE)
        & (t <= PUBLISHED_END)
    ].copy()
    return train, val, test


target = channel_frames["target"].copy()
target = target.rename(columns={"value": "target_soc"})

target_train, target_val, target_test = split_by_frozen_calendar(target)

assert len(target_train) > 0
assert len(target_val) > 0
assert len(target_test) > 0

train_path = OUT / "EVE_target_train.parquet"
val_path = OUT / "EVE_target_validation.parquet"
sealed_test_path = OUT / "EVE_target_CONFIRMATORY_TEST_SEALED.parquet"

target_train.to_parquet(train_path, index=False)
target_val.to_parquet(val_path, index=False)
target_test.to_parquet(sealed_test_path, index=False)


def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


seal = {
    "file": str(sealed_test_path),
    "sha256": sha256_file(sealed_test_path),
    "n_timestamped_rows": int(len(target_test)),
    "start_inclusive": VAL_END_EXCLUSIVE.isoformat(),
    "end_inclusive": PUBLISHED_END.isoformat(),
    "rule": (
        "Do not read, summarize, plot, tune on, or score this file "
        "until 10K and all temporal baselines are frozen for EVE."
    ),
}

with open(OUT / "12A_CONFIRMATORY_TEST_SEAL.json", "w") as f:
    json.dump(seal, f, indent=2)

# Deliberately report row counts only.
target_partition_counts = pd.DataFrame([
    {"partition": "train", "n_timestamped_rows": len(target_train)},
    {"partition": "validation", "n_timestamped_rows": len(target_val)},
    {"partition": "confirmatory_test_SEALED", "n_timestamped_rows": len(target_test)},
])

target_partition_counts.to_csv(
    OUT / "12A_target_partition_counts_NO_TEST_STATS.csv",
    index=False,
)

display(target_partition_counts)
print("Confirmatory test SHA256:", seal["sha256"])
print("No confirmatory-test SoC value statistics were computed.")

# %% [markdown] 8. Predictor-only train+validation audit for structured-prior eligibility

# %% Code cell 9 (notebook cell 18)
def trainval_only(z):
    return z[
        (z["timestamp"] >= PUBLISHED_START)
        & (z["timestamp"] < VAL_END_EXCLUSIVE)
    ].copy()


distance_tv = trainval_only(channel_frames["distance"])
distance_tv = distance_tv.dropna(subset=["value"]).copy()

assert len(distance_tv) > 0, "No train+validation distance observations."

audit = {
    "distance_trainval_observations": int(len(distance_tv)),
    "current_candidate_repository_path": (
        CURRENT_ROW["path"]
        if CURRENT_ROW is not None else None
    ),
}

signed_ah_eligible = False

if "current_candidate" in channel_frames:
    current_tv = trainval_only(
        channel_frames["current_candidate"]
    )
    current_tv = current_tv.dropna(subset=["value"]).copy()
    current_tv = current_tv.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    ).sort_values("timestamp")

    dt = current_tv["timestamp"].diff().dt.total_seconds()

    # Evaluate sampling gaps only within acquisition bursts.
    within = dt[
        (dt > 0)
        & (dt <= CURRENT_GATE["burst_break_s"])
    ]

    median_gap = float(within.median()) if len(within) else np.nan
    p95_gap = float(within.quantile(0.95)) if len(within) else np.nan

    # Predictor-only time alignment to distance.
    left = distance_tv[["timestamp"]].sort_values("timestamp")
    right = current_tv[["timestamp"]].copy()
    right["current_seen"] = 1
    right = right.sort_values("timestamp")

    aligned = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(
            seconds=CURRENT_GATE["alignment_tolerance_s"]
        ),
    )

    overlap = float(aligned["current_seen"].notna().mean())

    audit.update({
        "current_trainval_observations": int(len(current_tv)),
        "current_median_within_burst_gap_s": median_gap,
        "current_p95_within_burst_gap_s": p95_gap,
        "fraction_distance_rows_with_current_within_30s": overlap,
    })

    signed_ah_eligible = bool(
        len(current_tv)
        >= CURRENT_GATE["min_trainval_current_observations"]
        and np.isfinite(median_gap)
        and median_gap
        <= CURRENT_GATE["max_median_within_burst_gap_s"]
        and np.isfinite(p95_gap)
        and p95_gap
        <= CURRENT_GATE["max_p95_within_burst_gap_s"]
        and overlap
        >= CURRENT_GATE["min_distance_rows_with_current_within_30s"]
    )

audit["signed_ah_gate"] = CURRENT_GATE
audit["signed_ah_eligible"] = signed_ah_eligible

if signed_ah_eligible:
    selected_prior = "SIGNED_AH"
else:
    selected_prior = "DISTANCE"

audit["predeclared_selected_prior"] = selected_prior
audit["selection_basis"] = (
    "train+validation predictor availability only; no SoC target error and no confirmatory-test metric"
)

with open(OUT / "12A_prior_gate_trainval_predictor_only.json", "w") as f:
    json.dump(audit, f, indent=2)

# Human-readable one-row table.
audit_row = {
    k: v for k, v in audit.items()
    if not isinstance(v, dict)
}
pd.DataFrame([audit_row]).to_csv(
    OUT / "12A_prior_gate_trainval_predictor_only.csv",
    index=False,
)

print("PREDECLARED EVE PRIOR:", selected_prior)
print(json.dumps(audit, indent=2))

# %% [markdown] 9. Build a leakage blacklist for 12B/12C

# %% Code cell 10 (notebook cell 20)
blacklist = {
    "primary_target_repository_path": TARGET_ROW["path"],
    "target_like_forbidden_repository_paths": forbidden["path"].tolist(),
    "forbidden_name_tokens": FORBIDDEN_TARGET_TOKENS,
    "policy": (
        "No target/SoC trajectory other than SOC0 initialization may be used "
        "as a predictor. Confirmatory test target remains sealed."
    ),
}

with open(OUT / "12A_EVE_leakage_blacklist.json", "w") as f:
    json.dump(blacklist, f, indent=2)

print(json.dumps(blacklist, indent=2))

# %% [markdown] 10. Final 12A manifest

# %% Code cell 11 (notebook cell 22)
manifest = {
    "experiment": "12A_EVE_UNTOUCHED_CONFIRMATORY_PROTOCOL_DATA_AUDIT",
    "dataset": {
        "title": "The EVE Pilot: Usage Data from an Electric Car in France",
        "persistent_id": PERSISTENT_ID,
        "dataset_version_from_api": dataset_version,
        "vehicle": "Renault ZOE Q90",
        "repository": DATAVERSE,
        "published_collection_start": PUBLISHED_START.isoformat(),
        "published_collection_end": PUBLISHED_END.isoformat(),
    },
    "confirmatory_status": "UNTOUCHED_NO_MODEL_SCORE_COMPUTED",
    "primary_target": {
        "repository_path": TARGET_ROW["path"],
        "semantic_role": "production/user-facing BMS-reported SoC",
        "real_state_of_charge_used_as_predictor": False,
    },
    "split_contract": split_contract,
    "selected_prior_from_trainval_predictor_gate": selected_prior,
    "prior_gate": CURRENT_GATE,
    "models_frozen_before_confirmatory_test": [
        "10K_SC_RS_SFPEF",
        "TCN",
        "LSTM",
        "GRU",
    ],
    "temporal_baseline_architecture_frozen": True,
    "test_tuning_allowed": False,
    "confirmatory_test_target_file": str(sealed_test_path),
    "confirmatory_test_target_sha256": seal["sha256"],
    "confirmatory_test_target_value_statistics_computed_in_12A": False,
    "model_training_performed_in_12A": False,
    "test_prediction_performed_in_12A": False,
    "next_step": (
        "12B build EVE canonical train/validation data and train frozen methods "
        "without opening confirmatory-test labels; 12C performs one-shot final scoring."
    ),
}

with open(OUT / "12A_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))

# %% [markdown] Stop here
