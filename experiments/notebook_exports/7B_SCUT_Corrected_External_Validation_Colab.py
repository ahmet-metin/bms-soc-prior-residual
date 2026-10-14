# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/7B_SCUT_Corrected_External_Validation_Colab.ipynb
# Notebook SHA256: d12c2e526e03601075257087e3fb732a335a7069c4dffe870eb835fc49441b5d
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 7B — Corrected SCUT External Validation

# %% Code cell 1 (notebook cell 2)
# [NOTEBOOK MAGIC, MANUALLY SET UP] !pip -q install pyarrow scikit-learn joblib requests openpyxl xlrd python-calamine

import re
import json
import math
import time
import hashlib
import random
import zipfile
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

SEEDS = [42, 43, 44, 45, 46]
TUNING_SUBSET_SEED = 101
MODEL_SEED_FOR_TUNING = 777

FRACTIONS = [1.0, 0.10]

N_TREES = 150

ET_GRID = [
    {"min_samples_leaf": 1, "max_features": 0.5},
    {"min_samples_leaf": 2, "max_features": 0.5},
    {"min_samples_leaf": 4, "max_features": 0.5},
    {"min_samples_leaf": 8, "max_features": 0.5},
    {"min_samples_leaf": 2, "max_features": 0.8},
    {"min_samples_leaf": 4, "max_features": 0.8},
]

HISTORY_SECONDS = 120.0
SOC_EPS = 1e-4

MIN_BETA_ANCHORS = 20
MIN_VAL_ANCHORS = 20
MIN_TEST_ANCHORS = 20

CACHE_VERSION = "7B_scut_vehicle7_drivecode_fix_v1"

RUN_WITHIN_VEHICLE_DIAGNOSTIC = True
FORCE_REBUILD_FEATURES = False
FORCE_REDOWNLOAD = False

random.seed(42)
np.random.seed(42)

print("Cache version:", CACHE_VERSION)

# %% [markdown] 1) Drive ve yollar

# %% Code cell 2 (notebook cell 4)
from google.colab import drive
drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")

SCUT_ROOT = (
    ROOT
    / "dataset"
    / "external_validation"
    / "scut"
)

CACHE_ROOT = (
    ROOT
    / "dataset"
    / "external_validation"
    / "feature_cache_7B"
)

OUT7B = (
    ROOT
    / "outputs"
    / "external_validation_7B_scut_corrected"
)

for p in [SCUT_ROOT, CACHE_ROOT, OUT7B]:
    p.mkdir(parents=True, exist_ok=True)

SCUT_VEHICLES = list(range(1, 8))

# Audited from the actual public SCUT workbooks.
SCUT_DRIVING_SIGNAL_BY_VEHICLE = {
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 0,
}

assert set(SCUT_DRIVING_SIGNAL_BY_VEHICLE) == set(SCUT_VEHICLES)

print("Output:", OUT7B)
print("Vehicles:", SCUT_VEHICLES)
print("Driving signal map:", SCUT_DRIVING_SIGNAL_BY_VEHICLE)

# %% [markdown] 2) SCUT download fallback

# %% Code cell 3 (notebook cell 6)
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "EV-SCUT-ExternalValidation-7B/1.0"}
)

SCUT_REPO_API = (
    "https://api.github.com/repos/"
    "Translab-SCUT/Electric-vehicle-operation-data/contents"
)


def request_json(url, params=None, tries=5, timeout=120):
    last = None

    for k in range(tries):
        try:
            r = SESSION.get(
                url,
                params=params,
                timeout=timeout,
            )

            if r.status_code == 200:
                return r.json()

            last = RuntimeError(
                f"HTTP {r.status_code}: {r.text[:500]}"
            )

        except Exception as e:
            last = e

        time.sleep(2 ** k)

    raise RuntimeError(
        f"Request failed: {url}"
    ) from last


def download_stream(
    url,
    dest,
    force=False,
    tries=4,
):
    dest = Path(dest)
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        dest.exists()
        and dest.stat().st_size > 0
        and not force
    ):
        return dest

    tmp = dest.with_suffix(
        dest.suffix + ".partial"
    )

    last = None

    for k in range(tries):
        try:
            with SESSION.get(
                url,
                stream=True,
                timeout=(30, 300),
                allow_redirects=True,
            ) as r:
                r.raise_for_status()

                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(
                        1024 * 1024
                    ):
                        if chunk:
                            f.write(chunk)

            if (
                not tmp.exists()
                or tmp.stat().st_size == 0
            ):
                raise RuntimeError(
                    "empty download"
                )

            tmp.replace(dest)
            return dest

        except Exception as e:
            last = e

            if tmp.exists():
                tmp.unlink()

            time.sleep(2 ** k)

    raise RuntimeError(
        f"Download failed: {url}"
    ) from last


def ensure_scut_vehicle(vid):
    root = (
        SCUT_ROOT
        / "extracted"
        / f"vehicle_{vid}"
    )

    existing = list(
        root.rglob("*.xlsx")
    ) + list(
        root.rglob("*.xls")
    ) + list(
        root.rglob("*.csv")
    )

    if existing and not FORCE_REDOWNLOAD:
        return root

    repo = request_json(
        SCUT_REPO_API
    )

    name = f"vehicle#{vid}.zip"

    hits = [
        x
        for x in repo
        if (
            x.get("type") == "file"
            and x.get("name") == name
        )
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"SCUT archive not found: {name}"
        )

    row = hits[0]

    archive = (
        SCUT_ROOT
        / "archives"
        / name
    )

    candidates = [
        row.get("download_url"),
        (
            "https://media.githubusercontent.com/media/"
            "Translab-SCUT/Electric-vehicle-operation-data/"
            f"main/{name.replace('#', '%23')}"
        ),
        (
            "https://github.com/Translab-SCUT/"
            "Electric-vehicle-operation-data/raw/refs/heads/main/"
            f"{name.replace('#', '%23')}"
        ),
    ]

    ok = False

    for url in candidates:
        if not url:
            continue

        try:
            if archive.exists():
                archive.unlink()

            download_stream(
                url,
                archive,
                force=True,
            )

            if zipfile.is_zipfile(
                archive
            ):
                ok = True
                break

        except Exception as e:
            print(
                "download candidate failed:",
                url,
                repr(e),
            )

    if not ok:
        raise RuntimeError(
            f"Could not obtain valid SCUT ZIP for vehicle {vid}"
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        archive,
        "r",
    ) as zf:
        zf.extractall(root)

    return root


for vid in SCUT_VEHICLES:
    p = ensure_scut_vehicle(vid)
    print(
        "SCUT",
        vid,
        "->",
        p,
    )

# %% [markdown] 3) Excel / schema helpers

# %% Code cell 4 (notebook cell 8)
def _excel_engine():
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except Exception:
        return "openpyxl"


EXCEL_ENGINE = _excel_engine()

print("Excel engine:", EXCEL_ENGINE)


def read_table_auto(
    path,
    nrows=None,
    sheet_name=0,
):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last = None

        for enc in [
            "utf-8",
            "utf-8-sig",
            "gbk",
            "latin1",
        ]:
            try:
                return pd.read_csv(
                    path,
                    encoding=enc,
                    low_memory=False,
                    nrows=nrows,
                )
            except Exception as e:
                last = e

        raise last

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(
            path,
            sheet_name=sheet_name,
            nrows=nrows,
            engine=EXCEL_ENGINE,
        )

    raise ValueError(
        f"Unsupported file: {path}"
    )


def excel_sheet_names(path):
    path = Path(path)

    if path.suffix.lower() not in [
        ".xlsx",
        ".xls",
    ]:
        return [None]

    with pd.ExcelFile(
        path,
        engine=EXCEL_ENGINE,
    ) as xf:
        return list(xf.sheet_names)


def scut_tables(vid):
    root = (
        SCUT_ROOT
        / "extracted"
        / f"vehicle_{vid}"
    )

    files = []

    for ext in [
        "*.xlsx",
        "*.XLSX",
        "*.xls",
        "*.XLS",
        "*.csv",
        "*.CSV",
    ]:
        files.extend(
            root.rglob(ext)
        )

    return sorted(
        set(files)
    )


def scut_sources(vid):
    out = []

    for p in scut_tables(vid):
        if p.suffix.lower() in [
            ".xlsx",
            ".xls",
        ]:
            for sheet in excel_sheet_names(p):
                out.append(
                    {
                        "path": p,
                        "sheet_name": sheet,
                        "source_name": (
                            f"{p.name}::{sheet}"
                        ),
                    }
                )

        else:
            out.append(
                {
                    "path": p,
                    "sheet_name": None,
                    "source_name": p.name,
                }
            )

    return out


def _norm_header(x):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x)
        .replace("\ufeff", "")
        .strip()
        .lower(),
    )


SCUT_ROLE_ALIASES = {
    "timestamp": [
        "time",
        "timestamp",
        "datetime",
    ],
    "speed_kmh": [
        "vhc_speed",
        "vehicle_speed",
        "speed",
    ],
    "charging_signal": [
        "charging_signal",
        "chargingstatus",
        "chargestatus",
    ],
    "voltage_v": [
        "hv_voltage",
        "packvoltage",
        "batteryvoltage",
    ],
    "current_a": [
        "hv_current",
        "packcurrent",
        "batterycurrent",
    ],
    "soc_pct": [
        "bcell_soc",
        "soc",
        "batterysoc",
    ],
    "temp_max": [
        "bcell_maxTemp",
        "maxtemp",
    ],
    "temp_min": [
        "bcell_minTemp",
        "mintemp",
    ],
    "temp_single": [
        "bcell_Temp",
        "bcelltemp",
        "batterytemp",
    ],
}


def resolve_scut_schema(columns):
    by_norm = {
        _norm_header(c): c
        for c in columns
    }

    out = {}

    for role, aliases in (
        SCUT_ROLE_ALIASES.items()
    ):
        hit = None

        for a in aliases:
            key = _norm_header(a)

            if key in by_norm:
                hit = by_norm[key]
                break

        out[role] = hit

    required = [
        "timestamp",
        "speed_kmh",
        "charging_signal",
        "voltage_v",
        "current_a",
        "soc_pct",
    ]

    missing = [
        x
        for x in required
        if out[x] is None
    ]

    return out, missing

# %% [markdown] 4) SCUT packed time parser

# %% Code cell 5 (notebook cell 10)
def parse_scut_time_axis(values):
    s = pd.Series(values)

    num = pd.to_numeric(
        s,
        errors="coerce",
    )

    finite = num[
        np.isfinite(num)
    ]

    if not len(finite):
        return pd.to_datetime(
            s,
            errors="coerce",
        )

    def parse_one(v):
        try:
            iv = int(
                round(float(v))
            )

            txt = str(
                abs(iv)
            ).zfill(10)

            return pd.to_datetime(
                "2020" + txt,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )

        except Exception:
            return pd.NaT

    parsed = num.map(
        parse_one
    )

    ok = float(
        parsed.notna().mean()
    )

    if ok < 0.90:
        raise RuntimeError(
            "SCUT packed time parse rate "
            f"too low: {ok:.3f}"
        )

    # Handle possible year rollover.
    arr = parsed.copy()

    prev = None
    add_year = 0

    for idx, val in arr.items():
        if pd.isna(val):
            continue

        candidate = (
            val
            + pd.DateOffset(
                years=add_year
            )
        )

        if (
            prev is not None
            and candidate
            < prev - pd.Timedelta(
                days=180
            )
        ):
            add_year += 1
            candidate = (
                val
                + pd.DateOffset(
                    years=add_year
                )
            )

        arr.loc[idx] = candidate
        prev = candidate

    return pd.to_datetime(
        arr
    )


def normalize_soc_percent(values):
    s = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).astype(float)

    finite = s[
        np.isfinite(s)
    ]

    if len(finite):
        if (
            np.nanpercentile(
                finite,
                99,
            )
            <= 1.5
        ):
            s *= 100.0

    return s

# %% [markdown] Part A — Corrected driving-state audit

# %% Code cell 6 (notebook cell 12)
driving_state_audit_rows = []

for vid in SCUT_VEHICLES:
    configured = (
        SCUT_DRIVING_SIGNAL_BY_VEHICLE[
            vid
        ]
    )

    found_configured = False

    for src in scut_sources(vid):
        raw = read_table_auto(
            src["path"],
            sheet_name=src[
                "sheet_name"
            ],
        )

        mp, missing = (
            resolve_scut_schema(
                raw.columns
            )
        )

        if missing:
            raise RuntimeError(
                f"{src['source_name']}: "
                f"missing roles {missing}"
            )

        signal = pd.to_numeric(
            raw[
                mp[
                    "charging_signal"
                ]
            ],
            errors="coerce",
        )

        speed = pd.to_numeric(
            raw[
                mp[
                    "speed_kmh"
                ]
            ],
            errors="coerce",
        )

        tmp = pd.DataFrame(
            {
                "signal": signal,
                "speed": speed,
            }
        ).dropna(
            subset=["signal"]
        )

        for state, g in (
            tmp.groupby("signal")
        ):
            spd = g[
                "speed"
            ].dropna()

            moving_fraction = (
                float(
                    (spd > 1.0).mean()
                )
                if len(spd)
                else np.nan
            )

            row = {
                "vehicle_id": vid,
                "source_file": src[
                    "source_name"
                ],
                "charging_signal_state": state,
                "rows": len(g),
                "speed_mean": (
                    float(spd.mean())
                    if len(spd)
                    else np.nan
                ),
                "speed_p95": (
                    float(
                        np.percentile(
                            spd,
                            95,
                        )
                    )
                    if len(spd)
                    else np.nan
                ),
                "speed_max": (
                    float(spd.max())
                    if len(spd)
                    else np.nan
                ),
                "moving_fraction_gt1": (
                    moving_fraction
                ),
                "configured_driving": (
                    float(state)
                    == float(configured)
                ),
            }

            driving_state_audit_rows.append(
                row
            )

            if row[
                "configured_driving"
            ]:
                found_configured = True

                if (
                    len(spd) < 20
                    or np.nanmax(
                        spd
                    ) <= 5.0
                    or moving_fraction <= 0.01
                ):
                    raise RuntimeError(
                        f"SCUT vehicle {vid}: "
                        f"configured driving state "
                        f"{configured} does not show "
                        "plausible movement"
                    )

    if not found_configured:
        raise RuntimeError(
            f"SCUT vehicle {vid}: configured "
            f"driving signal {configured} "
            "not found"
        )


driving_state_audit = pd.DataFrame(
    driving_state_audit_rows
)

driving_state_audit.to_csv(
    OUT7B
    / "scut_driving_state_audit.csv",
    index=False,
)

display(
    driving_state_audit[
        [
            "vehicle_id",
            "charging_signal_state",
            "rows",
            "speed_mean",
            "speed_p95",
            "speed_max",
            "moving_fraction_gt1",
            "configured_driving",
        ]
    ]
)

# %% [markdown] 5) Corrected SCUT canonicalization

# %% Code cell 7 (notebook cell 14)
def canonicalize_scut(
    path,
    vid,
    sheet_name=0,
    source_name=None,
):
    raw = read_table_auto(
        path,
        sheet_name=sheet_name,
    )

    mp, missing = (
        resolve_scut_schema(
            raw.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{Path(path).name}"
            f"[{sheet_name}]: "
            f"unresolved roles {missing}"
        )

    d = pd.DataFrame(
        index=raw.index
    )

    d["timestamp"] = (
        parse_scut_time_axis(
            raw[
                mp["timestamp"]
            ]
        )
    )

    d["charging_signal"] = (
        pd.to_numeric(
            raw[
                mp[
                    "charging_signal"
                ]
            ],
            errors="coerce",
        )
    )

    d["speed_kmh"] = (
        pd.to_numeric(
            raw[
                mp["speed_kmh"]
            ],
            errors="coerce",
        )
    )

    d["voltage_v"] = (
        pd.to_numeric(
            raw[
                mp["voltage_v"]
            ],
            errors="coerce",
        )
    )

    d["current_a"] = (
        pd.to_numeric(
            raw[
                mp["current_a"]
            ],
            errors="coerce",
        )
    )

    d["soc_pct"] = (
        normalize_soc_percent(
            raw[
                mp["soc_pct"]
            ]
        ).to_numpy()
    )

    # CRITICAL 7B FIX:
    driving_code = (
        SCUT_DRIVING_SIGNAL_BY_VEHICLE[
            vid
        ]
    )

    d = d[
        d["charging_signal"]
        == driving_code
    ].copy()

    if (
        mp.get("temp_max")
        is not None
        and mp.get("temp_min")
        is not None
    ):
        mx = pd.to_numeric(
            raw.loc[
                d.index,
                mp["temp_max"],
            ],
            errors="coerce",
        )

        mn = pd.to_numeric(
            raw.loc[
                d.index,
                mp["temp_min"],
            ],
            errors="coerce",
        )

        d["temp_c_optional"] = (
            mx + mn
        ) / 2.0

    elif (
        mp.get("temp_single")
        is not None
    ):
        d["temp_c_optional"] = (
            pd.to_numeric(
                raw.loc[
                    d.index,
                    mp[
                        "temp_single"
                    ],
                ],
                errors="coerce",
            )
        )

    else:
        d["temp_c_optional"] = np.nan

    d["dataset"] = "SCUT"
    d["vehicle_id"] = (
        f"SCUT_{vid}"
    )
    d["source_file"] = (
        source_name
        or Path(path).name
    )

    return d[
        [
            "dataset",
            "vehicle_id",
            "timestamp",
            "current_a",
            "voltage_v",
            "speed_kmh",
            "soc_pct",
            "temp_c_optional",
            "source_file",
        ]
    ]

# %% [markdown] 6) Session segmentation + charge integration

# %% Code cell 8 (notebook cell 16)
def split_sessions(d):
    d = d.copy()

    for c in [
        "current_a",
        "voltage_v",
        "speed_kmh",
        "soc_pct",
    ]:
        d[c] = (
            pd.to_numeric(
                d[c],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    d = d.dropna(
        subset=[
            "timestamp",
            "current_a",
            "voltage_v",
            "speed_kmh",
            "soc_pct",
        ]
    )

    d = d[
        d["soc_pct"].between(
            -1,
            101,
        )
        & d["voltage_v"].between(
            20,
            1000,
        )
        & (
            d["current_a"].abs()
            <= 2000
        )
        & d["speed_kmh"].between(
            0,
            300,
        )
    ].copy()

    d = (
        d.sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
    )

    if len(d) < 3:
        return d.assign(
            session_local=np.nan
        )

    dt = (
        d["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    positive = dt[
        (dt > 0)
        & np.isfinite(dt)
    ]

    median_dt = (
        float(
            positive.median()
        )
        if len(positive)
        else 1.0
    )

    gap_threshold = max(
        30.0,
        5.0 * median_dt,
    )

    soc_jump = (
        d["soc_pct"]
        .diff()
        .abs()
        > 5.0
    )

    new_session = (
        dt.isna()
        | (dt <= 0)
        | (
            dt
            > gap_threshold
        )
        | soc_jump
    )

    d["session_local"] = (
        new_session.cumsum()
    )

    return d


def attach_session_ids(
    d,
    source_name,
):
    if len(d) == 0:
        return d.assign(
            session_id=pd.Series(
                dtype=str
            )
        )

    out = d.copy()

    out["session_id"] = (
        out["vehicle_id"].astype(str)
        + "_"
        + str(source_name)
        + "_"
        + out[
            "session_local"
        ].astype(str)
    )

    return out.drop(
        columns="session_local"
    )


def add_physics_coordinates(df):
    parts = []

    for sid, g in df.groupby(
        "session_id",
        sort=False,
    ):
        g = (
            g.sort_values(
                "timestamp"
            )
            .copy()
        )

        if len(g) < 3:
            continue

        dt = (
            g["timestamp"]
            .diff()
            .dt.total_seconds()
            .to_numpy(float)
        )

        dt[0] = 0.0

        dt[
            ~(
                np.isfinite(dt)
                & (dt >= 0)
            )
        ] = 0.0

        current = g[
            "current_a"
        ].to_numpy(float)

        current_prev = np.r_[
            current[0],
            current[:-1],
        ]

        current_mid = (
            0.5
            * (
                current
                + current_prev
            )
        )

        g["q_signed_ah"] = (
            np.cumsum(
                current_mid
                * dt
                / 3600.0
            )
        )

        g["q_abs_ah"] = (
            np.cumsum(
                0.5
                * (
                    np.abs(
                        current
                    )
                    + np.abs(
                        current_prev
                    )
                )
                * dt
                / 3600.0
            )
        )

        t0 = g[
            "timestamp"
        ].iloc[0]

        g["elapsed_s"] = (
            g["timestamp"]
            - t0
        ).dt.total_seconds()

        g["soc0"] = float(
            g["soc_pct"].iloc[0]
        )

        g["soc_anchor"] = (
            g["soc_pct"]
            .diff()
            .abs()
            > SOC_EPS
        ).astype(np.int8)

        g.iloc[
            0,
            g.columns.get_loc(
                "soc_anchor"
            ),
        ] = 0

        parts.append(g)

    if not parts:
        return pd.DataFrame()

    return pd.concat(
        parts,
        ignore_index=True,
    )

# %% [markdown] 7) Causal 120-second history features

# %% Code cell 9 (notebook cell 18)
PRIMARY_SIGNALS = [
    "current_a",
    "voltage_v",
    "speed_kmh",
    "q_abs_ah",
    "elapsed_s",
    "soc0",
]

STATS = [
    "last",
    "mean",
    "std",
    "min",
    "max",
    "delta",
]


def history_features(
    g,
    window=HISTORY_SECONDS,
):
    g = (
        g.sort_values(
            "timestamp"
        )
        .copy()
    )

    rel = pd.TimedeltaIndex(
        pd.to_timedelta(
            g["elapsed_s"]
            .to_numpy(float),
            unit="s",
        )
    )

    seconds = (
        rel.total_seconds()
        .to_numpy(float)
    )

    out = pd.DataFrame(
        index=g.index
    )

    for signal in PRIMARY_SIGNALS:
        s = pd.Series(
            g[signal]
            .to_numpy(float),
            index=rel,
        )

        rolling = s.rolling(
            f"{int(window)}s",
            min_periods=1,
            closed="both",
        )

        values = s.to_numpy(
            float
        )

        first = []
        left = 0

        for right in range(
            len(values)
        ):
            while (
                seconds[right]
                - seconds[left]
                > window
            ):
                left += 1

            first.append(
                values[left]
            )

        first = np.asarray(
            first,
            float,
        )

        out[
            f"{signal}__last"
        ] = values

        out[
            f"{signal}__mean"
        ] = (
            rolling.mean()
            .to_numpy(float)
        )

        out[
            f"{signal}__std"
        ] = (
            rolling.std(
                ddof=0
            )
            .fillna(0.0)
            .to_numpy(float)
        )

        out[
            f"{signal}__min"
        ] = (
            rolling.min()
            .to_numpy(float)
        )

        out[
            f"{signal}__max"
        ] = (
            rolling.max()
            .to_numpy(float)
        )

        out[
            f"{signal}__delta"
        ] = (
            values - first
        )

    return out


def make_anchor_features(df):
    rows = []

    for sid, g in df.groupby(
        "session_id",
        sort=False,
    ):
        f = history_features(g)

        base = g[
            [
                "dataset",
                "vehicle_id",
                "session_id",
                "timestamp",
                "soc_pct",
                "soc_anchor",
                "q_signed_ah",
                "q_abs_ah",
                "elapsed_s",
                "soc0",
            ]
        ].copy()

        part = base.join(f)

        part = part[
            part["soc_anchor"] > 0
        ].copy()

        if len(part):
            rows.append(part)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )

# %% [markdown] Part B — Build corrected SCUT 1–7 feature caches

# %% Code cell 10 (notebook cell 20)
def feature_cache_path(vid):
    return (
        CACHE_ROOT
        / (
            f"SCUT_{vid}_"
            f"{CACHE_VERSION}.parquet"
        )
    )


quality_rows = []
error_rows = []
feature_parts = []


for vid in SCUT_VEHICLES:
    cache = feature_cache_path(
        vid
    )

    if (
        cache.exists()
        and not FORCE_REBUILD_FEATURES
    ):
        f = pd.read_parquet(
            cache
        )

        print(
            "cache",
            vid,
            "anchors=",
            len(f),
        )

        feature_parts.append(f)
        continue

    local_parts = []

    for src in scut_sources(
        vid
    ):
        source_name = src[
            "source_name"
        ]

        try:
            d = canonicalize_scut(
                src["path"],
                vid,
                sheet_name=src[
                    "sheet_name"
                ],
                source_name=source_name,
            )

            d = split_sessions(d)

            d = attach_session_ids(
                d,
                source_name,
            )

            quality_rows.append(
                {
                    "vehicle_id": (
                        f"SCUT_{vid}"
                    ),
                    "source_file": (
                        source_name
                    ),
                    "rows_after_drive_filter": (
                        len(d)
                    ),
                    "soc_min": (
                        float(
                            d[
                                "soc_pct"
                            ].min()
                        )
                        if len(d)
                        else np.nan
                    ),
                    "soc_max": (
                        float(
                            d[
                                "soc_pct"
                            ].max()
                        )
                        if len(d)
                        else np.nan
                    ),
                    "speed_p95": (
                        float(
                            np.percentile(
                                d[
                                    "speed_kmh"
                                ],
                                95,
                            )
                        )
                        if len(d)
                        else np.nan
                    ),
                }
            )

            if len(d) < 3:
                raise RuntimeError(
                    "fewer than 3 driving rows"
                )

            ph = (
                add_physics_coordinates(
                    d
                )
            )

            ff = (
                make_anchor_features(
                    ph
                )
            )

            if not len(ff):
                raise RuntimeError(
                    "no genuine SoC anchor features"
                )

            local_parts.append(
                ff
            )

            del d, ph, ff
            gc.collect()

        except Exception as e:
            error_rows.append(
                {
                    "vehicle_id": (
                        f"SCUT_{vid}"
                    ),
                    "source_file": (
                        source_name
                    ),
                    "error": repr(e),
                }
            )

    if not local_parts:
        raise RuntimeError(
            f"SCUT_{vid}: no usable "
            "corrected anchor features"
        )

    f = pd.concat(
        local_parts,
        ignore_index=True,
    )

    f.to_parquet(
        cache,
        index=False,
    )

    print(
        "built",
        vid,
        "anchors=",
        len(f),
    )

    feature_parts.append(f)


features = pd.concat(
    feature_parts,
    ignore_index=True,
)

pd.DataFrame(
    quality_rows
).to_csv(
    OUT7B
    / "scut_corrected_raw_quality.csv",
    index=False,
)

pd.DataFrame(
    error_rows
).to_csv(
    OUT7B
    / "scut_corrected_preprocessing_errors.csv",
    index=False,
)

print(
    "Total corrected SCUT anchors:",
    f"{len(features):,}",
)

display(
    features.groupby(
        "vehicle_id"
    ).agg(
        anchors=(
            "soc_pct",
            "size",
        ),
        sessions=(
            "session_id",
            "nunique",
        ),
    )
)

# %% [markdown] Part C — SoC quantization audit

# %% Code cell 11 (notebook cell 22)
quant_rows = []

for vehicle_id, g in (
    features.groupby(
        "vehicle_id"
    )
):
    values = np.sort(
        pd.Series(
            g["soc_pct"]
        )
        .dropna()
        .unique()
        .astype(float)
    )

    diffs = np.diff(
        values
    )

    positive_diffs = diffs[
        diffs > 1e-9
    ]

    resolution = (
        float(
            np.min(
                positive_diffs
            )
        )
        if len(
            positive_diffs
        )
        else np.nan
    )

    soc = g[
        "soc_pct"
    ].to_numpy(float)

    integer_fraction = float(
        np.mean(
            np.isclose(
                soc,
                np.round(soc),
                atol=1e-8,
            )
        )
    )

    quant_rows.append(
        {
            "vehicle_id": vehicle_id,
            "n_anchor_values": (
                len(soc)
            ),
            "n_unique_soc": (
                len(values)
            ),
            "observed_soc_resolution_pp": (
                resolution
            ),
            "integer_soc_fraction": (
                integer_fraction
            ),
        }
    )


quantization_audit = pd.DataFrame(
    quant_rows
)

quantization_audit.to_csv(
    OUT7B
    / "scut_soc_quantization_audit.csv",
    index=False,
)

display(
    quantization_audit
)

# %% [markdown] Part D — Chronological session-level 60/20/20 split

# %% Code cell 12 (notebook cell 24)
session_summary = (
    features.groupby(
        [
            "vehicle_id",
            "session_id",
        ],
        as_index=False,
    )
    .agg(
        start_time=(
            "timestamp",
            "min",
        ),
        anchors=(
            "soc_pct",
            "size",
        ),
    )
)


def assign_split(g):
    g = (
        g.sort_values(
            [
                "start_time",
                "session_id",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    counts = g[
        "anchors"
    ].to_numpy(float)

    total = counts.sum()

    if (
        len(g) < 5
        or total < 20
    ):
        g["split"] = (
            "insufficient"
        )

        return g

    cumulative = np.cumsum(
        counts
    )

    c1 = np.arange(
        1,
        len(g) - 2,
    )

    cut1 = int(
        c1[
            np.argmin(
                np.abs(
                    cumulative[
                        c1 - 1
                    ]
                    - 0.60 * total
                )
            )
        ]
    )

    c2 = np.arange(
        cut1 + 1,
        len(g),
    )

    cut2 = int(
        c2[
            np.argmin(
                np.abs(
                    cumulative[
                        c2 - 1
                    ]
                    - 0.80 * total
                )
            )
        ]
    )

    g["split"] = "test"

    g.loc[
        : cut1 - 1,
        "split",
    ] = "train"

    g.loc[
        cut1 : cut2 - 1,
        "split",
    ] = "val"

    return g


split_assignment = (
    session_summary.groupby(
        "vehicle_id",
        group_keys=False,
    )
    .apply(
        assign_split
    )
    .reset_index(
        drop=True
    )
)

features = features.merge(
    split_assignment[
        [
            "vehicle_id",
            "session_id",
            "split",
        ]
    ],
    on=[
        "vehicle_id",
        "session_id",
    ],
    how="left",
    validate="many_to_one",
)

split_assignment.to_csv(
    OUT7B
    / "scut_corrected_session_split.csv",
    index=False,
)

support = (
    features[
        features["split"].isin(
            [
                "train",
                "val",
                "test",
            ]
        )
    ]
    .groupby(
        [
            "vehicle_id",
            "split",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)

informative = (
    features[
        (features["split"] == "train")
        & (
            features[
                "q_signed_ah"
            ].abs()
            >= 1e-4
        )
    ]
    .groupby(
        "vehicle_id"
    )
    .size()
    .rename(
        "train_beta_informative"
    )
    .reset_index()
)

support = support.merge(
    informative,
    on="vehicle_id",
    how="left",
)

support[
    "train_beta_informative"
] = (
    support[
        "train_beta_informative"
    ]
    .fillna(0)
    .astype(int)
)

for c in [
    "train",
    "val",
    "test",
]:
    if c not in support:
        support[c] = 0

support["eligible"] = (
    (
        support["train"]
        >= MIN_BETA_ANCHORS
    )
    & (
        support[
            "train_beta_informative"
        ]
        >= MIN_BETA_ANCHORS
    )
    & (
        support["val"]
        >= MIN_VAL_ANCHORS
    )
    & (
        support["test"]
        >= MIN_TEST_ANCHORS
    )
)

display(support)

support.to_csv(
    OUT7B
    / "scut_corrected_support_audit.csv",
    index=False,
)

if not bool(
    support[
        support[
            "vehicle_id"
        ]
        == "SCUT_7"
    ]["eligible"].iloc[0]
):
    raise RuntimeError(
        "SCUT_7 is still not eligible "
        "after the driving-state correction."
    )

eligible_vehicles = set(
    support.loc[
        support["eligible"],
        "vehicle_id",
    ]
)

features = features[
    features[
        "vehicle_id"
    ].isin(
        eligible_vehicles
    )
].copy()

print(
    "Eligible vehicles:",
    sorted(
        eligible_vehicles
    ),
)

# %% [markdown] Part E — Vehicle-specific train-only physics

# %% Code cell 13 (notebook cell 26)
def fit_vehicle_beta(
    train_df,
):
    x = train_df[
        "q_signed_ah"
    ].to_numpy(float)

    y = (
        train_df[
            "soc_pct"
        ].to_numpy(float)
        - train_df[
            "soc0"
        ].to_numpy(float)
    )

    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & (
            np.abs(x)
            >= 1e-4
        )
    )

    x = x[keep]
    y = y[keep]

    if (
        len(x)
        < MIN_BETA_ANCHORS
    ):
        raise RuntimeError(
            f"Too few beta anchors: {len(x)}"
        )

    beta = float(
        np.dot(x, y)
        / np.dot(x, x)
    )

    qeff = (
        100.0 / abs(beta)
        if beta != 0
        else np.inf
    )

    return (
        beta,
        qeff,
        len(x),
    )


beta_rows = []
BETA_BY_VEHICLE = {}

for vehicle_id in sorted(
    eligible_vehicles
):
    train = features[
        (features["vehicle_id"] == vehicle_id)
        & (features["split"] == "train")
    ].copy()

    beta, qeff, n = (
        fit_vehicle_beta(
            train
        )
    )

    BETA_BY_VEHICLE[
        vehicle_id
    ] = beta

    beta_rows.append(
        {
            "vehicle_id": vehicle_id,
            "beta": beta,
            "Qeff_Ah_field": qeff,
            "n_beta_anchors": n,
        }
    )


beta_audit = pd.DataFrame(
    beta_rows
)

display(beta_audit)

beta_audit.to_csv(
    OUT7B
    / "scut_corrected_vehicle_beta_audit.csv",
    index=False,
)


def apply_vehicle_beta(df):
    d = df.copy()

    betas = d[
        "vehicle_id"
    ].map(
        BETA_BY_VEHICLE
    )

    if betas.isna().any():
        raise RuntimeError(
            "Missing vehicle beta"
        )

    d["soc_phy_source"] = (
        d["soc0"].to_numpy(float)
        + betas.to_numpy(float)
        * d[
            "q_signed_ah"
        ].to_numpy(float)
    )

    d[
        "residual_target"
    ] = (
        d["soc_pct"]
        - d["soc_phy_source"]
    )

    return d

# %% [markdown] 8) Model helpers

# %% Code cell 14 (notebook cell 28)
HISTORY_FEATURES = [
    f"{signal}__{stat}"
    for signal in PRIMARY_SIGNALS
    for stat in STATS
]

DIRECT_FEATURES = (
    HISTORY_FEATURES
    + ["q_signed_ah"]
)

PHYSICS_FEATURES = (
    DIRECT_FEATURES
    + ["soc_phy_source"]
)

MODEL_SPECS = {
    "direct": {
        "features": DIRECT_FEATURES,
        "target": "soc_pct",
        "mode": "direct",
    },
    "direct_plus_physics": {
        "features": PHYSICS_FEATURES,
        "target": "soc_pct",
        "mode": "direct",
    },
    "residual": {
        "features": PHYSICS_FEATURES,
        "target": "residual_target",
        "mode": "residual",
    },
}


def finite_rows(
    df,
    columns,
):
    mask = np.ones(
        len(df),
        dtype=bool,
    )

    for c in columns:
        mask &= np.isfinite(
            pd.to_numeric(
                df[c],
                errors="coerce",
            ).to_numpy(float)
        )

    return df[
        mask
    ].copy()


def stable_subset(
    df,
    fraction,
    seed,
):
    if fraction >= 1.0:
        return df.copy()

    key = (
        df["vehicle_id"].astype(str)
        + "|"
        + df["session_id"].astype(str)
        + "|"
        + df["timestamp"].astype(str)
        + "|"
        + str(seed)
    )

    u = key.map(
        lambda s: int(
            hashlib.md5(
                s.encode(
                    "utf-8"
                )
            ).hexdigest()[:12],
            16,
        )
        / float(16**12)
    )

    selected = df[
        u < fraction
    ].copy()

    min_n = min(
        len(df),
        max(
            20,
            int(
                round(
                    fraction
                    * len(df)
                )
            ),
        ),
    )

    if len(selected) < min_n:
        selected = (
            df.assign(_u=u)
            .sort_values("_u")
            .head(min_n)
            .drop(
                columns="_u"
            )
        )

    return selected


def regression_metrics(
    y,
    pred,
):
    y = np.asarray(
        y,
        float,
    )

    pred = np.asarray(
        pred,
        float,
    )

    err = pred - y
    ae = np.abs(err)

    return {
        "MAE": float(
            ae.mean()
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    err**2
                )
            )
        ),
        "P95": float(
            np.percentile(
                ae,
                95,
            )
        ),
        "Bias": float(
            err.mean()
        ),
    }


def predict_soc(
    model,
    df,
    spec,
):
    raw = model.predict(
        df[
            spec["features"]
        ]
    )

    if (
        spec["mode"]
        == "direct"
    ):
        return raw

    return (
        df[
            "soc_phy_source"
        ].to_numpy(float)
        + raw
    )

# %% [markdown] Part F — Corrected pooled SCUT main experiment

# %% Code cell 15 (notebook cell 30)
train_raw = features[
    features["split"]
    == "train"
].copy()

val_raw = features[
    features["split"]
    == "val"
].copy()

test_raw = features[
    features["split"]
    == "test"
].copy()

train_full = finite_rows(
    apply_vehicle_beta(
        train_raw
    ),
    PHYSICS_FEATURES
    + [
        "soc_pct",
        "residual_target",
    ],
)

val_full = finite_rows(
    apply_vehicle_beta(
        val_raw
    ),
    PHYSICS_FEATURES
    + [
        "soc_pct",
        "residual_target",
    ],
)

test_full = finite_rows(
    apply_vehicle_beta(
        test_raw
    ),
    PHYSICS_FEATURES
    + [
        "soc_pct",
        "residual_target",
    ],
)

physics_metrics = (
    regression_metrics(
        test_full[
            "soc_pct"
        ],
        test_full[
            "soc_phy_source"
        ],
    )
)

print(
    "Physics test:",
    physics_metrics,
)

# %% Code cell 16 (notebook cell 31)
hp_rows = []
selected_rows = []

for fraction in FRACTIONS:
    tune_train = stable_subset(
        train_full,
        fraction,
        TUNING_SUBSET_SEED,
    )

    for model_name, spec in (
        MODEL_SPECS.items()
    ):
        local = []

        for hp in ET_GRID:
            model = ExtraTreesRegressor(
                n_estimators=N_TREES,
                min_samples_leaf=hp[
                    "min_samples_leaf"
                ],
                max_features=hp[
                    "max_features"
                ],
                random_state=(
                    MODEL_SEED_FOR_TUNING
                ),
                n_jobs=-1,
            )

            model.fit(
                tune_train[
                    spec["features"]
                ],
                tune_train[
                    spec["target"]
                ],
            )

            pred = predict_soc(
                model,
                val_full,
                spec,
            )

            m = regression_metrics(
                val_full[
                    "soc_pct"
                ],
                pred,
            )

            row = {
                "fraction": fraction,
                "model": model_name,
                **hp,
                "n_train": len(
                    tune_train
                ),
                "n_val": len(
                    val_full
                ),
                "val_MAE": m[
                    "MAE"
                ],
            }

            hp_rows.append(row)
            local.append(row)

        best = (
            pd.DataFrame(local)
            .sort_values(
                [
                    "val_MAE",
                    "min_samples_leaf",
                    "max_features",
                ]
            )
            .iloc[0]
        )

        selected_rows.append(
            {
                "fraction": fraction,
                "model": model_name,
                "selected_min_samples_leaf": int(
                    best[
                        "min_samples_leaf"
                    ]
                ),
                "selected_max_features": float(
                    best[
                        "max_features"
                    ]
                ),
                "selection_val_MAE": float(
                    best[
                        "val_MAE"
                    ]
                ),
            }
        )


hp_audit = pd.DataFrame(
    hp_rows
)

selected_hp = pd.DataFrame(
    selected_rows
)

hp_audit.to_csv(
    OUT7B
    / "scut_corrected_hp_audit.csv",
    index=False,
)

selected_hp.to_csv(
    OUT7B
    / "scut_corrected_selected_hp.csv",
    index=False,
)

display(
    selected_hp
)

# %% Code cell 17 (notebook cell 32)
run_rows = []
prediction_frames = []

for fraction in FRACTIONS:
    for seed in SEEDS:
        train = stable_subset(
            train_full,
            fraction,
            seed,
        )

        for (
            model_name,
            spec
        ) in MODEL_SPECS.items():
            hp = selected_hp[
                (
                    selected_hp[
                        "fraction"
                    ]
                    == fraction
                )
                & (
                    selected_hp[
                        "model"
                    ]
                    == model_name
                )
            ].iloc[0]

            model = ExtraTreesRegressor(
                n_estimators=N_TREES,
                min_samples_leaf=int(
                    hp[
                        "selected_min_samples_leaf"
                    ]
                ),
                max_features=float(
                    hp[
                        "selected_max_features"
                    ]
                ),
                random_state=seed,
                n_jobs=-1,
            )

            model.fit(
                train[
                    spec["features"]
                ],
                train[
                    spec["target"]
                ],
            )

            pred = predict_soc(
                model,
                test_full,
                spec,
            )

            m = regression_metrics(
                test_full[
                    "soc_pct"
                ],
                pred,
            )

            run_rows.append(
                {
                    "fraction": fraction,
                    "seed": seed,
                    "model": model_name,
                    "n_train": len(
                        train
                    ),
                    "n_test": len(
                        test_full
                    ),
                    **m,
                }
            )

            p = test_full[
                [
                    "vehicle_id",
                    "session_id",
                    "timestamp",
                    "soc_pct",
                    "soc_phy_source",
                ]
            ].copy()

            p["fraction"] = (
                fraction
            )
            p["seed"] = seed
            p["model"] = (
                model_name
            )
            p["pred_soc"] = pred

            prediction_frames.append(
                p
            )


main_runs = pd.DataFrame(
    run_rows
)

main_summary = (
    main_runs.groupby(
        [
            "fraction",
            "model",
        ],
        as_index=False,
    )
    .agg(
        MAE_mean=(
            "MAE",
            "mean",
        ),
        MAE_std=(
            "MAE",
            "std",
        ),
        RMSE_mean=(
            "RMSE",
            "mean",
        ),
        P95_mean=(
            "P95",
            "mean",
        ),
        n_train_mean=(
            "n_train",
            "mean",
        ),
        n_test=(
            "n_test",
            "first",
        ),
    )
)

display(
    main_summary
)

main_runs.to_csv(
    OUT7B
    / "scut_corrected_main_runs.csv",
    index=False,
)

main_summary.to_csv(
    OUT7B
    / "scut_corrected_main_summary.csv",
    index=False,
)

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

predictions.to_parquet(
    OUT7B
    / "scut_corrected_predictions.parquet",
    index=False,
)

# %% [markdown] 9) Pairwise residual comparison + label degradation

# %% Code cell 18 (notebook cell 34)
pairwise_rows = []

for fraction in FRACTIONS:
    pivot = (
        main_runs[
            main_runs[
                "fraction"
            ]
            == fraction
        ]
        .pivot(
            index="seed",
            columns="model",
            values="MAE",
        )
    )

    for competitor in [
        "direct",
        "direct_plus_physics",
    ]:
        delta = (
            pivot[competitor]
            - pivot["residual"]
        )

        pairwise_rows.append(
            {
                "fraction": fraction,
                "competitor": competitor,
                "residual_MAE": float(
                    pivot[
                        "residual"
                    ].mean()
                ),
                "competitor_MAE": float(
                    pivot[
                        competitor
                    ].mean()
                ),
                "competitor_minus_residual": float(
                    delta.mean()
                ),
                "residual_better_seeds": int(
                    (delta > 0).sum()
                ),
                "n_seeds": len(delta),
            }
        )


pairwise = pd.DataFrame(
    pairwise_rows
)

display(pairwise)

pairwise.to_csv(
    OUT7B
    / "scut_corrected_pairwise.csv",
    index=False,
)


degradation_rows = []

for model_name in MODEL_SPECS:
    full = main_summary[
        (
            main_summary[
                "fraction"
            ]
            == 1.0
        )
        & (
            main_summary[
                "model"
            ]
            == model_name
        )
    ][
        "MAE_mean"
    ].iloc[0]

    sparse = main_summary[
        (
            main_summary[
                "fraction"
            ]
            == 0.10
        )
        & (
            main_summary[
                "model"
            ]
            == model_name
        )
    ][
        "MAE_mean"
    ].iloc[0]

    degradation_rows.append(
        {
            "model": model_name,
            "full_MAE": full,
            "ten_percent_MAE": sparse,
            "relative_degradation_pct": (
                100.0
                * (
                    sparse
                    - full
                )
                / full
            ),
        }
    )


label_efficiency = pd.DataFrame(
    degradation_rows
)

display(label_efficiency)

label_efficiency.to_csv(
    OUT7B
    / "scut_corrected_label_efficiency.csv",
    index=False,
)

# %% [markdown] Part G — Vehicle-level pooled-model results

# %% Code cell 19 (notebook cell 36)
vehicle_rows = []

for (
    fraction,
    seed,
    model_name,
    vehicle_id,
), g in predictions.groupby(
    [
        "fraction",
        "seed",
        "model",
        "vehicle_id",
    ]
):
    m = regression_metrics(
        g["soc_pct"],
        g["pred_soc"],
    )

    vehicle_rows.append(
        {
            "vehicle_id": vehicle_id,
            "fraction": fraction,
            "seed": seed,
            "model": model_name,
            "n": len(g),
            **m,
        }
    )


vehicle_runs = pd.DataFrame(
    vehicle_rows
)

vehicle_summary = (
    vehicle_runs.groupby(
        [
            "vehicle_id",
            "fraction",
            "model",
        ],
        as_index=False,
    )
    .agg(
        MAE_mean=(
            "MAE",
            "mean",
        ),
        MAE_std=(
            "MAE",
            "std",
        ),
        n=(
            "n",
            "first",
        ),
    )
)

display(
    vehicle_summary
)

vehicle_summary.to_csv(
    OUT7B
    / "scut_corrected_vehicle_level_summary.csv",
    index=False,
)

# %% [markdown] Part H — Within-vehicle diagnostic

# %% Code cell 20 (notebook cell 38)
within_rows = []

if RUN_WITHIN_VEHICLE_DIAGNOSTIC:
    for vehicle_id in sorted(
        eligible_vehicles
    ):
        print(
            "within-vehicle:",
            vehicle_id,
        )

        train_v = train_full[
            train_full[
                "vehicle_id"
            ]
            == vehicle_id
        ].copy()

        val_v = val_full[
            val_full[
                "vehicle_id"
            ]
            == vehicle_id
        ].copy()

        test_v = test_full[
            test_full[
                "vehicle_id"
            ]
            == vehicle_id
        ].copy()

        selected = {}

        for (
            model_name,
            spec
        ) in MODEL_SPECS.items():
            rows = []

            for hp in ET_GRID:
                m = ExtraTreesRegressor(
                    n_estimators=N_TREES,
                    min_samples_leaf=hp[
                        "min_samples_leaf"
                    ],
                    max_features=hp[
                        "max_features"
                    ],
                    random_state=(
                        MODEL_SEED_FOR_TUNING
                    ),
                    n_jobs=-1,
                )

                m.fit(
                    train_v[
                        spec[
                            "features"
                        ]
                    ],
                    train_v[
                        spec[
                            "target"
                        ]
                    ],
                )

                pred = predict_soc(
                    m,
                    val_v,
                    spec,
                )

                mm = regression_metrics(
                    val_v[
                        "soc_pct"
                    ],
                    pred,
                )

                rows.append(
                    {
                        **hp,
                        "val_MAE": (
                            mm["MAE"]
                        ),
                    }
                )

            selected[
                model_name
            ] = (
                pd.DataFrame(
                    rows
                )
                .sort_values(
                    [
                        "val_MAE",
                        "min_samples_leaf",
                        "max_features",
                    ]
                )
                .iloc[0]
                .to_dict()
            )

        for (
            model_name,
            spec
        ) in MODEL_SPECS.items():
            hp = selected[
                model_name
            ]

            seed_mae = []

            for seed in SEEDS:
                m = ExtraTreesRegressor(
                    n_estimators=N_TREES,
                    min_samples_leaf=int(
                        hp[
                            "min_samples_leaf"
                        ]
                    ),
                    max_features=float(
                        hp[
                            "max_features"
                        ]
                    ),
                    random_state=seed,
                    n_jobs=-1,
                )

                m.fit(
                    train_v[
                        spec[
                            "features"
                        ]
                    ],
                    train_v[
                        spec[
                            "target"
                        ]
                    ],
                )

                pred = predict_soc(
                    m,
                    test_v,
                    spec,
                )

                mm = regression_metrics(
                    test_v[
                        "soc_pct"
                    ],
                    pred,
                )

                seed_mae.append(
                    mm["MAE"]
                )

            within_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "model": model_name,
                    "MAE_mean": float(
                        np.mean(
                            seed_mae
                        )
                    ),
                    "MAE_std": float(
                        np.std(
                            seed_mae,
                            ddof=1,
                        )
                    ),
                    "n_train": len(
                        train_v
                    ),
                    "n_val": len(
                        val_v
                    ),
                    "n_test": len(
                        test_v
                    ),
                }
            )


within_vehicle = pd.DataFrame(
    within_rows
)

if len(within_vehicle):
    display(
        within_vehicle
    )

    within_vehicle.to_csv(
        OUT7B
        / "scut_within_vehicle_diagnostic.csv",
        index=False,
    )

# %% [markdown] Part I — Diagnostic summary

# %% Code cell 21 (notebook cell 40)
diagnostic_rows = []

if len(within_vehicle):
    pivot = within_vehicle.pivot(
        index="vehicle_id",
        columns="model",
        values="MAE_mean",
    )

    for vehicle_id, r in pivot.iterrows():
        diagnostic_rows.append(
            {
                "vehicle_id": vehicle_id,
                "within_residual_MAE": (
                    r["residual"]
                ),
                "within_direct_plus_physics_MAE": (
                    r[
                        "direct_plus_physics"
                    ]
                ),
                "within_residual_better": bool(
                    r["residual"]
                    < r[
                        "direct_plus_physics"
                    ]
                ),
            }
        )


diagnostic_summary = pd.DataFrame(
    diagnostic_rows
)

if len(diagnostic_summary):
    display(
        diagnostic_summary
    )

    diagnostic_summary.to_csv(
        OUT7B
        / "scut_within_vehicle_residual_decision.csv",
        index=False,
    )

# %% [markdown] 10) Figures

# %% Code cell 22 (notebook cell 42)
models = [
    "direct",
    "direct_plus_physics",
    "residual",
]

labels = [
    "Direct ET",
    "Direct + Physics",
    "Residual ET",
]

x = np.arange(
    len(models)
)

width = 0.34

plt.figure(
    figsize=(8.5, 5.2)
)

for j, fraction in enumerate(
    FRACTIONS
):
    vals = [
        main_summary[
            (
                main_summary[
                    "fraction"
                ]
                == fraction
            )
            & (
                main_summary[
                    "model"
                ]
                == model_name
            )
        ]["MAE_mean"].iloc[0]
        for model_name in models
    ]

    offset = (
        -width / 2
        if j == 0
        else width / 2
    )

    plt.bar(
        x + offset,
        vals,
        width,
        label=(
            "100% labels"
            if fraction == 1.0
            else "10% labels"
        ),
    )

plt.axhline(
    physics_metrics["MAE"],
    linestyle="--",
    linewidth=1,
    label="Physics only",
)

plt.xticks(
    x,
    labels,
    rotation=10,
)

plt.ylabel(
    "SoC MAE [percentage points]"
)

plt.title(
    "Corrected SCUT External Validation — Vehicles 1–7"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUT7B
    / "scut_corrected_label_ablation.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()

# %% [markdown] Part J — Final integrity checks

# %% Code cell 23 (notebook cell 44)
expected = {
    "SCUT_1",
    "SCUT_2",
    "SCUT_3",
    "SCUT_4",
    "SCUT_5",
    "SCUT_6",
    "SCUT_7",
}

observed = set(
    features[
        "vehicle_id"
    ].unique()
)

missing = (
    expected - observed
)

if missing:
    raise RuntimeError(
        f"Missing corrected SCUT vehicles: {sorted(missing)}"
    )

for fraction in FRACTIONS:
    sub = main_runs[
        main_runs[
            "fraction"
        ]
        == fraction
    ]

    got_models = set(
        sub["model"]
    )

    if got_models != set(
        MODEL_SPECS
    ):
        raise RuntimeError(
            f"fraction {fraction}: "
            f"incomplete models {got_models}"
        )

    for model_name in MODEL_SPECS:
        seeds = set(
            sub[
                sub["model"]
                == model_name
            ]["seed"]
        )

        if seeds != set(SEEDS):
            raise RuntimeError(
                f"fraction {fraction}, "
                f"{model_name}: "
                f"incomplete seeds {seeds}"
            )

metric_cols = [
    "MAE",
    "RMSE",
    "P95",
    "Bias",
]

if not np.isfinite(
    main_runs[
        metric_cols
    ].to_numpy(float)
).all():
    raise RuntimeError(
        "Non-finite main metric"
    )

print("Integrity checks passed.")
print("Included SCUT vehicles:", sorted(observed))
print("Physics MAE:", physics_metrics["MAE"])
display(main_summary)
display(pairwise)
display(label_efficiency)

# %% [markdown] 11) Manifest

# %% Code cell 24 (notebook cell 46)
manifest = {
    "experiment": (
        "SCUT_external_validation_7B_corrected"
    ),
    "source": (
        "Translab-SCUT/Electric-vehicle-operation-data"
    ),
    "vehicles": SCUT_VEHICLES,
    "battery_materials": (
        "NCM vehicles 1-6; LFP vehicle 7"
    ),
    "critical_fix": {
        "vehicle_1_to_6_driving_signal": 3,
        "vehicle_7_driving_signal": 0,
        "mapping_is_audited_against_speed": True,
    },
    "cache_version": CACHE_VERSION,
    "history_seconds": HISTORY_SECONDS,
    "physics_beta": (
        "vehicle-specific, train-only"
    ),
    "split": (
        "chronological session-level approx 60/20/20 "
        "anchor mass within vehicle"
    ),
    "fractions": FRACTIONS,
    "seeds": SEEDS,
    "models": list(
        MODEL_SPECS.keys()
    ),
    "within_vehicle_diagnostic": (
        RUN_WITHIN_VEHICLE_DIAGNOSTIC
    ),
    "claim_boundary": (
        "7B corrects the SCUT Vehicle-7 driving-state bug. "
        "Within-vehicle diagnostics are exploratory and are not "
        "used to tune the primary pooled external-validation result."
    ),
}

with open(
    OUT7B
    / "experiment7B_manifest.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )

print("7B outputs:")
for p in sorted(
    OUT7B.glob("*")
):
    print(" -", p.name)

# %% [markdown] En önemli çıktılar
