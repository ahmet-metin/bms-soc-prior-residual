# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/robustness/R6_AnchorAgeRobustness_Colab.ipynb
# Notebook SHA256: 0cf28444b4ac94771faebdfe71cdefc1e916b83c9837821e0ffa1c1f7fbd5150
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R6 — Anchor-Age Robustness Audit

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path("/content/drive/MyDrive/NissanLeaf")
REV2_OUT = ROOT / "Revizyon2" / "outputs"

R2C_OUT = REV2_OUT / "R2C_corrected_sparse_prior_curve"
INPUT = R2C_OUT / "R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet"

OUT = REV2_OUT / "R6_anchor_age_robustness"
FIG_DIR = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R6_ANCHOR_AGE_ROBUSTNESS"

DOMAINS = ["VED", "KU", "SCUT", "TUM", "EVE"]
FRACTIONS = [0.10, 1.00]

DIRECT = "Direct+Prior"
PR = "PriorResidual+Prior"
MODELS = [DIRECT, PR]

# Fixed, manuscript-friendly bins.
BIN_EDGES_MIN = [0, 5, 10, 20, 40, np.inf]
BIN_LABELS = ["0–5", "5–10", "10–20", "20–40", "≥40"]

BOOT_DRAWS = 10000
MIN_SEGMENTS_FOR_INTERPRETATION = 10
RNG_SEED = 60261

print("Input :", INPUT)
print("Output:", OUT)
assert INPUT.exists(), (f"Missing corrected R2C predictions: {INPUT}")

# %% [markdown] 1. Load and fail-closed validation

# %% Code cell 2 (notebook cell 4)
pred = pd.read_parquet(INPUT).copy()

required = {
    "domain",
    "fraction",
    "model",
    "vehicle_id",
    "segment_id",
    "compound_segment",
    "time_value",
    "elapsed_s",
    "target",
    "prediction",
}
missing = sorted(required - set(pred.columns))
assert not missing, f"Missing required prediction columns: {missing}"

pred["domain"] = pred["domain"].astype(str)
pred["vehicle_id"] = pred["vehicle_id"].astype(str)
pred["segment_id"] = pred["segment_id"].astype(str)
pred["compound_segment"] = pred["compound_segment"].astype(str)
pred["fraction"] = pred["fraction"].astype(float)
pred["elapsed_s"] = pred["elapsed_s"].astype(float)
pred["target"] = pred["target"].astype(float)
pred["prediction"] = pred["prediction"].astype(float)

assert np.isfinite(pred["elapsed_s"]).all()
assert (pred["elapsed_s"] >= -1e-9).all(), "elapsed_s contains negative values."
assert np.isfinite(pred["target"]).all()
assert np.isfinite(pred["prediction"]).all()

pred = pred[
    pred["domain"].isin(DOMAINS)
    & pred["model"].isin(MODELS)
    & pred["fraction"].apply(lambda x: any(np.isclose(x, f) for f in FRACTIONS))
].copy()

pred["fraction"] = pred["fraction"].apply(
    lambda x: min(FRACTIONS, key=lambda f: abs(float(x) - f))
)

expected_conditions = len(DOMAINS) * len(FRACTIONS) * len(MODELS)
condition_count = (
    pred[["domain", "fraction", "model"]]
    .drop_duplicates()
    .shape[0]
)
assert condition_count == expected_conditions, (
    condition_count,
    expected_conditions,
    pred[["domain", "fraction", "model"]].drop_duplicates().sort_values(
        ["domain", "fraction", "model"]
    )
)

print("Rows:", len(pred))
print("Conditions:", condition_count)
display(
    pred.groupby(["domain", "fraction", "model"])
        .agg(
            rows=("prediction", "size"),
            segments=("compound_segment", "nunique"),
            max_elapsed_min=("elapsed_s", lambda x: float(np.max(x) / 60.0)),
        )
        .reset_index()
)

# %% [markdown] 2. Verify exact Direct–PriorResidual row pairing

# %% Code cell 3 (notebook cell 6)
PAIR_KEYS = [
    "domain",
    "fraction",
    "vehicle_id",
    "segment_id",
    "compound_segment",
    "time_value",
    "elapsed_s",
    "target",
]

dup = pred.duplicated(PAIR_KEYS + ["model"])
assert not dup.any(), f"Duplicate model predictions found: {int(dup.sum())}"

row_counts = (
    pred.groupby(PAIR_KEYS, dropna=False)["model"]
    .nunique()
    .reset_index(name="n_models")
)
assert (row_counts["n_models"] == 2).all(), (
    "Direct/PR physical-row mismatch",
    row_counts["n_models"].value_counts().to_dict(),
)

wide = (
    pred.pivot(index=PAIR_KEYS, columns="model", values="prediction")
    .reset_index()
)

assert DIRECT in wide.columns and PR in wide.columns
assert wide[[DIRECT, PR]].notna().all().all()

wide["Direct_abs_error"] = np.abs(
    wide[DIRECT].to_numpy(float) - wide["target"].to_numpy(float)
)
wide["PR_abs_error"] = np.abs(
    wide[PR].to_numpy(float) - wide["target"].to_numpy(float)
)

wide["anchor_age_min"] = wide["elapsed_s"] / 60.0
wide["anchor_age_bin"] = pd.cut(
    wide["anchor_age_min"],
    bins=BIN_EDGES_MIN,
    labels=BIN_LABELS,
    right=False,
    include_lowest=True,
)

assert wide["anchor_age_bin"].notna().all()

print("Paired physical rows:", len(wide))
display(
    wide.groupby(["domain", "fraction", "anchor_age_bin"], observed=True)
        .agg(
            rows=("target", "size"),
            segments=("compound_segment", "nunique"),
            vehicles=("vehicle_id", "nunique"),
        )
        .reset_index()
        .sort_values(["domain", "fraction", "anchor_age_bin"])
)

# %% [markdown] 3. Segment-balanced MAE by anchor age

# %% Code cell 4 (notebook cell 8)
segment_rows = (
    wide.groupby(
        [
            "domain",
            "fraction",
            "vehicle_id",
            "compound_segment",
            "anchor_age_bin",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        Direct_MAE=("Direct_abs_error", "mean"),
        PR_MAE=("PR_abs_error", "mean"),
        n_rows=("target", "size"),
        mean_anchor_age_min=("anchor_age_min", "mean"),
        min_anchor_age_min=("anchor_age_min", "min"),
        max_anchor_age_min=("anchor_age_min", "max"),
    )
)

segment_rows["Delta_PR_minus_Direct"] = (
    segment_rows["PR_MAE"] - segment_rows["Direct_MAE"]
)

summary = (
    segment_rows.groupby(
        ["domain", "fraction", "anchor_age_bin"],
        observed=True,
        as_index=False,
    )
    .agg(
        n_segments=("compound_segment", "nunique"),
        n_vehicles=("vehicle_id", "nunique"),
        n_rows=("n_rows", "sum"),
        Direct_segment_balanced_MAE=("Direct_MAE", "mean"),
        PR_segment_balanced_MAE=("PR_MAE", "mean"),
        Delta_PR_minus_Direct=("Delta_PR_minus_Direct", "mean"),
        Median_segment_delta=("Delta_PR_minus_Direct", "median"),
        Mean_anchor_age_min=("mean_anchor_age_min", "mean"),
    )
)

summary["PR_relative_improvement_pct"] = (
    100.0
    * (
        summary["Direct_segment_balanced_MAE"]
        - summary["PR_segment_balanced_MAE"]
    )
    / summary["Direct_segment_balanced_MAE"].replace(0, np.nan)
)

summary["interpretable_bin"] = (
    summary["n_segments"] >= MIN_SEGMENTS_FOR_INTERPRETATION
)

display(summary)

# %% [markdown] 4. Descriptive paired segment-bootstrap confidence intervals

# %% Code cell 5 (notebook cell 10)
def paired_segment_bootstrap_ci(x, draws=BOOT_DRAWS, seed=RNG_SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    point = float(np.mean(x))
    if len(x) == 1:
        return point, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    boot = x[idx].mean(axis=1)

    return (
        point,
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )

boot_rows = []

for (domain, fraction, age_bin), g in segment_rows.groupby(
    ["domain", "fraction", "anchor_age_bin"],
    observed=True,
):
    seed = (
        RNG_SEED
        + DOMAINS.index(str(domain)) * 1000
        + int(round(float(fraction) * 100)) * 10
        + BIN_LABELS.index(str(age_bin))
    )
    point, lo, hi = paired_segment_bootstrap_ci(
        g["Delta_PR_minus_Direct"].to_numpy(float),
        seed=seed,
    )

    boot_rows.append(
        {
            "domain": domain,
            "fraction": float(fraction),
            "anchor_age_bin": str(age_bin),
            "n_segments": int(g["compound_segment"].nunique()),
            "n_vehicles": int(g["vehicle_id"].nunique()),
            "Delta_PR_minus_Direct": point,
            "CI95_low": lo,
            "CI95_high": hi,
            "PR_better_point_estimate": bool(point < 0),
            "CI_entirely_below_zero": bool(
                np.isfinite(lo) and np.isfinite(hi) and hi < 0
            ),
            "interpretable_bin": bool(
                g["compound_segment"].nunique()
                >= MIN_SEGMENTS_FOR_INTERPRETATION
            ),
        }
    )

boot = pd.DataFrame(boot_rows)

final_table = summary.merge(
    boot[
        [
            "domain",
            "fraction",
            "anchor_age_bin",
            "CI95_low",
            "CI95_high",
            "CI_entirely_below_zero",
        ]
    ],
    on=["domain", "fraction", "anchor_age_bin"],
    how="left",
)

display(final_table)

# %% [markdown] 5. Domain-level robustness summary

# %% Code cell 6 (notebook cell 12)
domain_rows = []

for (domain, fraction), g in final_table.groupby(["domain", "fraction"]):
    gi = g[g["interpretable_bin"]].copy()

    if len(gi) == 0:
        domain_rows.append(
            {
                "domain": domain,
                "fraction": float(fraction),
                "n_interpretable_bins": 0,
                "PR_better_bins": 0,
                "PR_better_fraction": np.nan,
                "farthest_interpretable_bin": None,
                "farthest_bin_delta": np.nan,
                "farthest_bin_PR_better": False,
            }
        )
        continue

    gi["_bin_order"] = gi["anchor_age_bin"].astype(str).map(
        {b: i for i, b in enumerate(BIN_LABELS)}
    )
    gi = gi.sort_values("_bin_order")
    far = gi.iloc[-1]

    domain_rows.append(
        {
            "domain": domain,
            "fraction": float(fraction),
            "n_interpretable_bins": int(len(gi)),
            "PR_better_bins": int((gi["Delta_PR_minus_Direct"] < 0).sum()),
            "PR_better_fraction": float(
                (gi["Delta_PR_minus_Direct"] < 0).mean()
            ),
            "farthest_interpretable_bin": str(far["anchor_age_bin"]),
            "farthest_bin_delta": float(far["Delta_PR_minus_Direct"]),
            "farthest_bin_PR_better": bool(
                far["Delta_PR_minus_Direct"] < 0
            ),
        }
    )

domain_summary = pd.DataFrame(domain_rows)
display(domain_summary)

# %% [markdown] 6. Publication-ready figures

# %% Code cell 7 (notebook cell 14)
def make_figure(fraction, filename):
    fig, axes = plt.subplots(
        1, 5,
        figsize=(17.5, 3.8),
        sharey=False,
        constrained_layout=True,
    )

    for ax, domain in zip(axes, DOMAINS):
        g = final_table[
            (final_table["domain"] == domain)
            & np.isclose(final_table["fraction"], fraction)
        ].copy()

        g["_bin_order"] = g["anchor_age_bin"].astype(str).map(
            {b: i for i, b in enumerate(BIN_LABELS)}
        )
        g = g.sort_values("_bin_order")

        x = np.arange(len(g))

        ax.plot(
            x,
            g["Direct_segment_balanced_MAE"],
            marker="o",
            linewidth=1.8,
            label="Direct+Prior",
        )
        ax.plot(
            x,
            g["PR_segment_balanced_MAE"],
            marker="o",
            linewidth=1.8,
            label="PriorResidual+Prior",
        )

        ax.set_title(domain)
        ax.set_xticks(x)
        ax.set_xticklabels(
            g["anchor_age_bin"].astype(str),
            rotation=40,
            ha="right",
        )
        ax.set_xlabel("Time since anchor (min)")
        ax.grid(alpha=0.20)

    axes[0].set_ylabel("Segment-balanced MAE (SoC pp)")
    axes[-1].legend(
        loc="best",
        frameon=False,
        fontsize=8,
    )

    sup = "10% retained training targets" if np.isclose(fraction, 0.10) else "100% training targets"
    fig.suptitle(
        f"Anchor-age robustness — {sup}",
        fontsize=12,
    )

    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path

fig10 = make_figure(
    0.10,
    f"{EXPERIMENT}_10pct_5panel.png",
)
fig100 = make_figure(
    1.00,
    f"{EXPERIMENT}_full_5panel.png",
)

print(fig10)
print(fig100)

# %% [markdown] 7. Export audit files and manuscript notes

# %% Code cell 8 (notebook cell 16)
segment_rows.to_csv(
    OUT / f"{EXPERIMENT}_SEGMENT_LEVEL.csv",
    index=False,
)

final_table.to_csv(
    OUT / f"{EXPERIMENT}_ANCHOR_AGE_BIN_SUMMARY.csv",
    index=False,
)

domain_summary.to_csv(
    OUT / f"{EXPERIMENT}_DOMAIN_SUMMARY.csv",
    index=False,
)

# Compact manuscript-facing table.
manuscript_table = final_table[
    [
        "domain",
        "fraction",
        "anchor_age_bin",
        "n_segments",
        "n_vehicles",
        "Direct_segment_balanced_MAE",
        "PR_segment_balanced_MAE",
        "Delta_PR_minus_Direct",
        "CI95_low",
        "CI95_high",
        "PR_relative_improvement_pct",
        "interpretable_bin",
    ]
].copy()

manuscript_table.to_csv(
    OUT / f"{EXPERIMENT}_MANUSCRIPT_TABLE.csv",
    index=False,
)

input_sha256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()

manifest = {
    "experiment": EXPERIMENT,
    "status": "R6_ANCHOR_AGE_ROBUSTNESS_COMPLETE",
    "retraining_performed": False,
    "source_experiment": "R2C_CORRECTED_SPARSE_PRIOR_CURVE",
    "source_file": str(INPUT),
    "source_sha256": input_sha256,
    "domains": DOMAINS,
    "fractions": FRACTIONS,
    "models": MODELS,
    "anchor_age_source": "elapsed_s from corrected R2C TEST prediction table",
    "anchor_age_bins_minutes": BIN_LABELS,
    "segment_balanced_estimand": True,
    "paired_physical_rows_verified": True,
    "bootstrap": "paired compound-segment bootstrap; descriptive CI only",
    "bootstrap_draws": BOOT_DRAWS,
    "new_multiplicity_controlled_hypothesis_family_created": False,
    "minimum_segments_for_interpretation": MIN_SEGMENTS_FOR_INTERPRETATION,
}

with open(
    OUT / f"{EXPERIMENT}_MANIFEST.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)

notes = []
notes.append("# R6 Anchor-Age Robustness — Manuscript Notes")
notes.append("")
notes.append("## Scope")
notes.append("")
notes.append(
    "This is a post-hoc robustness audit using corrected R2C five-seed averaged TEST predictions. "
    "No model was retrained and no new primary hypothesis family was created."
)
notes.append("")
notes.append("## Fixed bins")
notes.append("")
notes.append(", ".join(BIN_LABELS) + " min.")
notes.append("")
notes.append("## Domain × supervision summary")
notes.append("")
notes.append(domain_summary.to_markdown(index=False))
notes.append("")
notes.append("## Reporting boundary")
notes.append("")
notes.append(
    "- Negative PR-minus-Direct differences favor PriorResidual.\n"
    "- Bootstrap intervals are descriptive paired segment-bootstrap intervals.\n"
    "- Bins with fewer than "
    f"{MIN_SEGMENTS_FOR_INTERPRETATION} segments are reported but not used for manuscript interpretation.\n"
    "- The audit evaluates robustness to time since the genuine segment-start BMS anchor; "
    "it does not convert the BMS reconstruction task into independent electrochemical SoC estimation."
)

notes_path = OUT / f"{EXPERIMENT}_MANUSCRIPT_NOTES.md"
notes_path.write_text("\n".join(notes), encoding="utf-8")

print(json.dumps(manifest, indent=2))
print("\nSaved:")
for p in sorted(OUT.iterdir()):
    print(" -", p.name)

# %% [markdown] Interpretation rule for the manuscript
