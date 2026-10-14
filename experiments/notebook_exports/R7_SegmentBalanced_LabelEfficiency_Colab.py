# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/primary/R7_SegmentBalanced_LabelEfficiency_Colab.ipynb
# Notebook SHA256: 28ff14bbaab7599b4de1f65b645cea78e109f254a523313332e2a1fcf5d93adf
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R7 — Segment-Balanced Label-Efficiency Export + Figure 2

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

R2C = REV2_OUT / "R2C_corrected_sparse_prior_curve"

PRED = R2C / "R2C_CORRECTED_SPARSE_PRIOR_CURVE_SEED_AVERAGED_PREDICTIONS.parquet"
BUDGET = R2C / "R2C_CORRECTED_SPARSE_PRIOR_CURVE_EFFECTIVE_BUDGET_SUMMARY.csv"
PRIMARY = R2C / "R2C_CORRECTED_SPARSE_PRIOR_CURVE_PRIMARY_EFFECTIVE_AULEC_BOOTSTRAP.csv"

OUT = REV2_OUT / "R7_segment_balanced_label_efficiency"
FIG_DIR = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R7_SEGMENT_BALANCED_LABEL_EFFICIENCY"

DOMAINS = ["VED","KU","SCUT","TUM","EVE"]
FRACTIONS = [0.01,0.02,0.05,0.10,0.20,0.50,1.00]
DIRECT = "Direct+Prior"
PR = "PriorResidual+Prior"
MODELS = [DIRECT, PR]

for p in [PRED, BUDGET, PRIMARY]:
    assert p.exists(), f"Missing required input: {p}"

print("Predictions:", PRED)
print("Budget     :", BUDGET)
print("Primary    :", PRIMARY)
print("Output     :", OUT)

# %% [markdown] 1. Load corrected R2C outputs and validate contracts

# %% Code cell 2 (notebook cell 4)
pred = pd.read_parquet(PRED).copy()
budget = pd.read_csv(BUDGET).copy()
primary = pd.read_csv(PRIMARY).copy()

required_pred = {
    "domain","fraction","model","target","prediction",
    "compound_segment"
}
missing = sorted(required_pred - set(pred.columns))
assert not missing, f"Missing prediction columns: {missing}"

pred["domain"] = pred["domain"].astype(str)
pred["fraction"] = pred["fraction"].astype(float)
pred["model"] = pred["model"].astype(str)
pred["target"] = pred["target"].astype(float)
pred["prediction"] = pred["prediction"].astype(float)
pred["compound_segment"] = pred["compound_segment"].astype(str)

assert np.isfinite(pred["target"]).all()
assert np.isfinite(pred["prediction"]).all()

pred = pred[
    pred["domain"].isin(DOMAINS)
    & pred["model"].isin(MODELS)
    & pred["fraction"].apply(lambda x: any(np.isclose(x, f) for f in FRACTIONS))
].copy()

pred["fraction"] = pred["fraction"].apply(
    lambda x: min(FRACTIONS, key=lambda f: abs(float(x)-f))
)

expected = len(DOMAINS)*len(FRACTIONS)*len(MODELS)
n_conditions = pred[["domain","fraction","model"]].drop_duplicates().shape[0]
assert n_conditions == expected, (n_conditions, expected)

# Optional physical-row pairing audit when these keys are present.
pair_keys = [c for c in [
    "domain","fraction","vehicle_id","segment_id",
    "compound_segment","time_value","elapsed_s","target"
] if c in pred.columns]

if len(pair_keys) >= 5:
    dup = pred.duplicated(pair_keys + ["model"])
    assert not dup.any(), f"Duplicate model rows: {int(dup.sum())}"

    nmods = pred.groupby(pair_keys, dropna=False)["model"].nunique()
    assert (nmods == 2).all(), "Direct and PriorResidual do not share identical physical test rows."

print("R2C corrected prediction contract: PASS")
print("Conditions:", n_conditions)

# %% [markdown] 2. Compute the primary per-condition segment-balanced MAE

# %% Code cell 3 (notebook cell 6)
pred["abs_error"] = np.abs(pred["prediction"] - pred["target"])

segment_mae = (
    pred.groupby(
        ["domain","fraction","model","compound_segment"],
        as_index=False
    )
    .agg(
        segment_MAE=("abs_error","mean"),
        n_rows=("abs_error","size"),
    )
)

long_rows = []

for (domain, fraction, model), g in segment_mae.groupby(
    ["domain","fraction","model"]
):
    long_rows.append({
        "domain": domain,
        "fraction": float(fraction),
        "model": model,
        "n_segments": int(g["compound_segment"].nunique()),
        "n_test_rows": int(g["n_rows"].sum()),
        "segment_balanced_MAE": float(g["segment_MAE"].mean()),
        "segment_MAE_std": float(g["segment_MAE"].std(ddof=1)),
        "segment_MAE_median": float(g["segment_MAE"].median()),
    })

long = pd.DataFrame(long_rows)

assert len(long) == len(DOMAINS)*len(FRACTIONS)*len(MODELS)

display(
    long.sort_values(["domain","fraction","model"])
)

# %% [markdown] 3. Merge effective supervision and construct the 35-row manuscript table

# %% Code cell 4 (notebook cell 8)
budget_small = budget[
    [
        "domain",
        "fraction",
        "actual_nominal_fraction_mean",
        "effective_fit_fraction_mean",
        "anchor_overhead_pp_mean",
        "n_selected_labels_mean",
        "n_unique_fit_observations_mean",
    ]
].copy()

for c in ["fraction","actual_nominal_fraction_mean","effective_fit_fraction_mean"]:
    budget_small[c] = budget_small[c].astype(float)

wide = (
    long.pivot(
        index=["domain","fraction","n_segments","n_test_rows"],
        columns="model",
        values="segment_balanced_MAE",
    )
    .reset_index()
)

assert DIRECT in wide.columns and PR in wide.columns

wide = wide.rename(columns={
    DIRECT: "Direct_segment_balanced_MAE",
    PR: "PR_segment_balanced_MAE",
})

wide = wide.merge(
    budget_small,
    on=["domain","fraction"],
    how="left",
    validate="one_to_one",
)

assert wide["effective_fit_fraction_mean"].notna().all()

wide["Delta_PR_minus_Direct"] = (
    wide["PR_segment_balanced_MAE"] -
    wide["Direct_segment_balanced_MAE"]
)

wide["PR_relative_improvement_pct"] = (
    100.0 *
    (
        wide["Direct_segment_balanced_MAE"] -
        wide["PR_segment_balanced_MAE"]
    ) /
    wide["Direct_segment_balanced_MAE"].replace(0, np.nan)
)

wide["PR_lower_segment_balanced_MAE"] = (
    wide["Delta_PR_minus_Direct"] < 0
)

wide["nominal_pct"] = 100.0 * wide["actual_nominal_fraction_mean"]
wide["effective_pct"] = 100.0 * wide["effective_fit_fraction_mean"]

wide = wide.sort_values(
    ["domain","fraction"],
    key=lambda s: s.map({d:i for i,d in enumerate(DOMAINS)}) if s.name=="domain" else s
).reset_index(drop=True)

assert len(wide) == len(DOMAINS)*len(FRACTIONS)

print(
    "PR lower segment-balanced MAE:",
    int(wide["PR_lower_segment_balanced_MAE"].sum()),
    "/",
    len(wide),
)

display(wide)

# %% [markdown] 4. AULEC reconstruction audit

# %% Code cell 5 (notebook cell 10)
def normalized_log_auc(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    assert np.all(x > 0)
    lx = np.log10(x)
    width = float(lx[-1] - lx[0])
    assert width > 0

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(y, lx) / width)

audit_rows = []

for domain in DOMAINS:
    g = wide[wide["domain"]==domain].sort_values("fraction")

    x_eff = g["effective_fit_fraction_mean"].to_numpy(float)
    x_nom = g["actual_nominal_fraction_mean"].to_numpy(float)

    d = g["Direct_segment_balanced_MAE"].to_numpy(float)
    p = g["PR_segment_balanced_MAE"].to_numpy(float)

    direct_eff = normalized_log_auc(x_eff, d)
    pr_eff = normalized_log_auc(x_eff, p)

    direct_nom = normalized_log_auc(x_nom, d)
    pr_nom = normalized_log_auc(x_nom, p)

    frozen = primary[primary["domain"].astype(str)==domain]
    assert len(frozen)==1
    frozen = frozen.iloc[0]

    audit_rows.append({
        "domain": domain,
        "recomputed_Direct_effective_AULEC": direct_eff,
        "frozen_Direct_effective_AULEC": float(
            frozen["Direct_effective_AULEC_segment_balanced"]
        ),
        "abs_diff_Direct_effective": abs(
            direct_eff - float(frozen["Direct_effective_AULEC_segment_balanced"])
        ),
        "recomputed_PR_effective_AULEC": pr_eff,
        "frozen_PR_effective_AULEC": float(
            frozen["PR_effective_AULEC_segment_balanced"]
        ),
        "abs_diff_PR_effective": abs(
            pr_eff - float(frozen["PR_effective_AULEC_segment_balanced"])
        ),
        "recomputed_Direct_nominal_AULEC": direct_nom,
        "frozen_Direct_nominal_AULEC": float(
            frozen["Direct_nominal_AULEC_segment_balanced"]
        ),
        "abs_diff_Direct_nominal": abs(
            direct_nom - float(frozen["Direct_nominal_AULEC_segment_balanced"])
        ),
        "recomputed_PR_nominal_AULEC": pr_nom,
        "frozen_PR_nominal_AULEC": float(
            frozen["PR_nominal_AULEC_segment_balanced"]
        ),
        "abs_diff_PR_nominal": abs(
            pr_nom - float(frozen["PR_nominal_AULEC_segment_balanced"])
        ),
    })

audit = pd.DataFrame(audit_rows)

tol = 1e-10

check_cols = [
    "abs_diff_Direct_effective",
    "abs_diff_PR_effective",
    "abs_diff_Direct_nominal",
    "abs_diff_PR_nominal",
]

assert (audit[check_cols].to_numpy(float) <= tol).all(), (
    "AULEC reconstruction mismatch. Do not use exported table until audited.",
    audit,
)

print("AULEC reconstruction audit: PASS")
display(audit)

# %% [markdown] 5. Publication-ready 5-panel Figure 2

# %% Code cell 6 (notebook cell 12)
fig, axes = plt.subplots(
    1, 5,
    figsize=(18, 4.2),
    sharey=False,
    constrained_layout=True,
)

for ax, domain in zip(axes, DOMAINS):
    g = wide[wide["domain"]==domain].sort_values("fraction")
    x = 100.0 * g["effective_fit_fraction_mean"].to_numpy(float)

    ax.plot(
        x,
        g["Direct_segment_balanced_MAE"].to_numpy(float),
        marker="o",
        linewidth=1.8,
        label="Direct+Prior",
    )
    ax.plot(
        x,
        g["PR_segment_balanced_MAE"].to_numpy(float),
        marker="o",
        linewidth=1.8,
        label="PriorResidual+Prior",
    )

    ax.set_xscale("log")
    ax.set_title(domain)
    ax.set_xlabel("Effective fit observations (%)")
    ax.grid(alpha=0.20)

axes[0].set_ylabel("Segment-balanced MAE (SoC pp)")
axes[-1].legend(frameon=False, fontsize=8, loc="best")

fig.suptitle(
    "Training-label efficiency under effective BMS observation accounting",
    fontsize=12,
)

png = FIG_DIR / f"{EXPERIMENT}_Figure2_5panel.png"
pdf = FIG_DIR / f"{EXPERIMENT}_Figure2_5panel.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.show()

print("Saved:", png)
print("Saved:", pdf)

# %% [markdown] 6. Export manuscript/supplementary files

# %% Code cell 7 (notebook cell 14)
long.to_csv(
    OUT / f"{EXPERIMENT}_LONG_70ROW_TABLE.csv",
    index=False,
)

wide.to_csv(
    OUT / f"{EXPERIMENT}_MANUSCRIPT_35_CONDITION_TABLE.csv",
    index=False,
)

audit.to_csv(
    OUT / f"{EXPERIMENT}_AULEC_RECONSTRUCTION_AUDIT.csv",
    index=False,
)

# Compact supplementary-ready table.
supp = wide[
    [
        "domain",
        "fraction",
        "nominal_pct",
        "effective_pct",
        "n_segments",
        "Direct_segment_balanced_MAE",
        "PR_segment_balanced_MAE",
        "Delta_PR_minus_Direct",
        "PR_relative_improvement_pct",
        "PR_lower_segment_balanced_MAE",
    ]
].copy()

supp.to_csv(
    OUT / f"{EXPERIMENT}_SUPPLEMENTARY_TABLE.csv",
    index=False,
)

# Manuscript notes.
notes = []
notes.append("# R7 Segment-Balanced Label-Efficiency — Manuscript Notes")
notes.append("")
notes.append("## Scope")
notes.append("")
notes.append(
    "No model was retrained. Corrected R2C five-seed averaged TEST predictions "
    "were re-aggregated using the manuscript's primary equal-segment MAE estimand."
)
notes.append("")
notes.append("## Integrity")
notes.append("")
notes.append(
    "- 35 domain × supervision conditions exported.\n"
    f"- PriorResidual has lower segment-balanced MAE in "
    f"{int(wide['PR_lower_segment_balanced_MAE'].sum())}/35 conditions.\n"
    "- Reconstructed effective- and nominal-axis AULEC values exactly reproduce "
    "the frozen R2C primary AULEC table within 1e-10 tolerance."
)
notes.append("")
notes.append("## Reporting use")
notes.append("")
notes.append(
    "Use the 35-condition segment-balanced table as the numerical audit trail "
    "for the primary AULEC result. Use the 5-panel figure in the main manuscript "
    "if space permits; otherwise place it in Supplementary Material."
)

notes_path = OUT / f"{EXPERIMENT}_MANUSCRIPT_NOTES.md"
notes_path.write_text("\n".join(notes), encoding="utf-8")

manifest = {
    "experiment": EXPERIMENT,
    "status": "R7_SEGMENT_BALANCED_LABEL_EFFICIENCY_COMPLETE",
    "retraining_performed": False,
    "source_experiment": "R2C_CORRECTED_SPARSE_PRIOR_CURVE",
    "source_prediction_file": str(PRED),
    "source_prediction_sha256": hashlib.sha256(PRED.read_bytes()).hexdigest(),
    "n_domains": len(DOMAINS),
    "n_fractions": len(FRACTIONS),
    "n_conditions": int(len(wide)),
    "segment_balanced_estimand": True,
    "effective_supervision_axis_primary": True,
    "aulec_reconstruction_tolerance": tol,
    "aulec_reconstruction_passed": True,
    "PR_lower_segment_balanced_MAE_conditions": int(
        wide["PR_lower_segment_balanced_MAE"].sum()
    ),
    "figure_png": str(png),
    "figure_pdf": str(pdf),
}

with open(
    OUT / f"{EXPERIMENT}_MANIFEST.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
print("\nSaved files:")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(" -", p.relative_to(OUT))

# %% [markdown] Manuscript wording suggestion after successful audit
