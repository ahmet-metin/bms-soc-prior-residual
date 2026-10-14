# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/budget_optimization/R2B_EffectiveSupervisionBudget_Colab.ipynb
# Notebook SHA256: 31dc25ce35f20cf90cbc1d6205a7469d2fc156f8ac5e2cfb35049640d4509ec3
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] R2B — Effective BMS Supervision and Observation Budget

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
OUTPUTS = ROOT / "outputs"
REV1_OUT = ROOT / "Revizyon1" / "outputs"
REV2_OUT = ROOT / "Revizyon2" / "outputs"

OUT = REV2_OUT / "R2B_effective_supervision_budget"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "R2B_EFFECTIVE_SUPERVISION_BUDGET"
DEV_DOMAINS = ["VED", "KU", "SCUT", "TUM"]
ALL_DOMAINS = DEV_DOMAINS + ["EVE"]
FRACTIONS = [1.0, 0.10]
SEEDS = [42, 43, 44, 45, 46]

print(EXPERIMENT)
print("Output:", OUT)

# %% Code cell 2 (notebook cell 3)

def find_unique_file(root, filename):
    matches = list(Path(root).rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename!r} under {root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]

def stable_hash_u01(s):
    return (
        int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:12], 16)
        / float(16 ** 12)
    )

def exact_sparse_subset(df, fraction, seed):
    # Same per-vehicle deterministic selector used in R1A/R2A.
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

def add_identity_and_segment(df):
    z = df.copy().reset_index(drop=True)

    z["vehicle_id"] = z["vehicle_id"].astype(str)
    z["segment_id"] = z["segment_id"].astype(str)

    z["compound_segment"] = (
        z["vehicle_id"] + "||" + z["segment_id"]
    )

    # Local canonical-row identity. All label/anchor sets are constructed
    # from the same in-memory canonical frame, so this is exact.
    z["_row_id"] = np.arange(len(z), dtype=np.int64)
    return z

def anchor_rows(train_df):
    # Genuine current-segment start observation.
    sort_cols = ["compound_segment"]

    if "elapsed_s" in train_df.columns:
        sort_cols.append("elapsed_s")

    sort_cols.append("time_value")

    z = (
        train_df
        .sort_values(sort_cols, kind="mergesort")
        .drop_duplicates("compound_segment", keep="first")
        .copy()
    )

    assert z["compound_segment"].is_unique
    return z

# %% Code cell 3 (notebook cell 4)

# Source artifacts used only as integrity references.

CANON = OUTPUTS / "model_screening_10" / "canonical"

R1A_SELECTOR = find_unique_file(
    REV1_OUT,
    "R1A_INFO_MATCHED_GRU_sparse_selector_audit.csv",
)

R2A_SELECTOR = find_unique_file(
    REV2_OUT,
    "R2A_EVE_INFO_MATCHED_GRU_sparse_selector_audit.csv",
)

R2A_MANIFEST = find_unique_file(
    REV2_OUT,
    "R2A_EVE_INFO_MATCHED_GRU_MANIFEST.json",
)

R1C_BURDEN = find_unique_file(
    REV1_OUT,
    "R1C_PERIODIC_REANCHORING_observation_burden.csv",
)

r1a_selector = pd.read_csv(R1A_SELECTOR)
r2a_selector = pd.read_csv(R2A_SELECTOR)
r1c_burden = pd.read_csv(R1C_BURDEN)

with open(R2A_MANIFEST) as f:
    r2a_manifest = json.load(f)

assert r2a_manifest["status"] == "POST_CONFIRMATORY_EVE_INFO_MATCHED_FAIRNESS_AUDIT_COMPLETE"
assert r2a_manifest["target"] == "dashboard/user-facing BMS-reported State of Charge"
assert r2a_manifest["exact_information_match_between_new_models"] is True
assert r2a_manifest["n_runs"] == r2a_manifest["expected_runs"] == 20

print("R1A selector:", R1A_SELECTOR)
print("R2A selector:", R2A_SELECTOR)
print("R1C burden:", R1C_BURDEN)
print("R2A integrity: PASS")

# %% Code cell 4 (notebook cell 5)

# Freeze the accounting definitions before inspecting the numerical budget.

preflight = {
    "experiment": EXPERIMENT,
    "status": "ACCOUNTING_DEFINITIONS_FROZEN",
    "target": "BMS-reported SoC",
    "nominal_label_fraction_definition": (
        "fraction of TRAIN rows carrying target labels for model fitting"
    ),
    "training_anchor_definition": (
        "one genuine BMS-reported SoC observation at the start of each TRAIN compound segment"
    ),
    "unique_fit_observation_definition": (
        "set union of selected TRAIN target-label rows and TRAIN anchor rows"
    ),
    "anchor_label_overlap_is_double_counted": False,
    "validation_supervision_definition": (
        "all validation target rows used for model selection/early stopping; reported separately"
    ),
    "test_targets_count_as_supervision": False,
    "deployment_initial_anchor_definition": (
        "one BMS-reported SoC observation per deployment/test compound segment"
    ),
    "deployment_refresh_definition": (
        "additional trusted BMS SoC observations from the frozen R1C 5/10/20 min policies"
    ),
    "deployment_refresh_counted_as_training_label": False,
    "fractions": FRACTIONS,
    "seeds": SEEDS,
    "development_domains": DEV_DOMAINS,
    "EVE_included_in_training_budget": True,
    "R1C_refresh_budget_domains": DEV_DOMAINS,
    "no_model_training": True,
    "no_posthoc_accounting_rule_changes": True,
}

with open(OUT / f"{EXPERIMENT}_PREFLIGHT.json", "w") as f:
    json.dump(preflight, f, indent=2)

print(json.dumps(preflight, indent=2))

# %% Code cell 5 (notebook cell 6)

# Load development canonical domains.

domain_frames = {}
domain_meta = {}

for domain in DEV_DOMAINS:
    p = CANON / f"{domain}_canonical.parquet"
    m = CANON / f"{domain}_metadata.json"

    assert p.exists(), p
    assert m.exists(), m

    d = add_identity_and_segment(pd.read_parquet(p))

    with open(m) as f:
        meta = json.load(f)

    assert "split" in d.columns
    assert {"train", "val", "test"}.issubset(set(d["split"].unique()))
    assert np.isfinite(d["target"].to_numpy(float)).all()

    domain_frames[domain] = d
    domain_meta[domain] = meta

    print(
        domain,
        d["split"].value_counts().to_dict(),
        "segments=",
        d["compound_segment"].nunique(),
    )

# %% Code cell 6 (notebook cell 7)

# Load EVE frozen train/validation canonical data.

EVE_TRAINVAL = OUTPUTS / "eve_12B_v4_frozen_trainval" / "12B_v4_EVE_canonical_trainval.parquet"
EVE_META = OUTPUTS / "eve_12B_v4_frozen_trainval" / "12B_v4_EVE_canonical_metadata.json"

assert EVE_TRAINVAL.exists(), EVE_TRAINVAL
assert EVE_META.exists(), EVE_META

eve = add_identity_and_segment(pd.read_parquet(EVE_TRAINVAL))

with open(EVE_META) as f:
    eve_meta = json.load(f)

eve = eve[eve["split"].isin(["train", "val"])].copy()

assert int((eve["split"] == "train").sum()) == 180878
assert int((eve["split"] == "val").sum()) == 45023
assert np.isfinite(eve["target"].to_numpy(float)).all()

domain_frames["EVE"] = eve
domain_meta["EVE"] = eve_meta

print(
    "EVE",
    eve["split"].value_counts().to_dict(),
    "segments=",
    eve["compound_segment"].nunique(),
)

# %% Code cell 7 (notebook cell 8)

# Exact per-seed training observation accounting.

budget_rows = []

for domain in ALL_DOMAINS:
    d = domain_frames[domain]

    train = d[d["split"] == "train"].copy()
    val = d[d["split"] == "val"].copy()

    anchors = anchor_rows(train)
    anchor_ids = set(anchors["_row_id"].astype(int).tolist())

    n_train = int(len(train))
    n_val = int(len(val))
    n_train_segments = int(train["compound_segment"].nunique())
    n_val_segments = int(val["compound_segment"].nunique())

    assert len(anchor_ids) == n_train_segments

    for fraction in FRACTIONS:
        for seed in SEEDS:
            selected = exact_sparse_subset(train, fraction, seed)
            label_ids = set(selected["_row_id"].astype(int).tolist())

            overlap_ids = label_ids.intersection(anchor_ids)
            union_ids = label_ids.union(anchor_ids)

            n_labels = int(len(label_ids))
            n_anchors = int(len(anchor_ids))
            n_overlap = int(len(overlap_ids))
            n_unique_fit = int(len(union_ids))
            n_extra_anchor = int(n_unique_fit - n_labels)

            assert n_unique_fit == n_labels + n_anchors - n_overlap
            assert n_unique_fit <= n_train
            assert n_extra_anchor >= 0

            # Validation target rows are fully observed for model selection.
            # Since every validation row has target, any validation anchor is
            # already contained in the validation-target set and adds no unique
            # observation beyond n_val.
            n_pretest_unique = int(n_unique_fit + n_val)

            budget_rows.append({
                "domain": domain,
                "fraction": float(fraction),
                "seed": int(seed),

                "n_train_rows_potential": n_train,
                "n_validation_rows": n_val,
                "n_train_plus_validation_rows": int(n_train + n_val),

                "n_train_compound_segments": n_train_segments,
                "n_validation_compound_segments": n_val_segments,

                "n_selected_training_labels": n_labels,
                "nominal_training_label_rate": float(n_labels / n_train),

                "n_initial_training_anchors": n_anchors,
                "n_anchor_rows_already_selected_as_labels": n_overlap,
                "n_additional_unique_anchor_observations": n_extra_anchor,

                "n_unique_fit_BMS_observations": n_unique_fit,
                "effective_fit_BMS_observation_rate": float(n_unique_fit / n_train),

                "nominal_training_label_reduction_pct": float(
                    100.0 * (1.0 - n_labels / n_train)
                ),
                "effective_fit_observation_reduction_pct": float(
                    100.0 * (1.0 - n_unique_fit / n_train)
                ),
                "anchor_overhead_percentage_points": float(
                    100.0 * (n_unique_fit - n_labels) / n_train
                ),

                "n_validation_target_observations_for_model_selection": n_val,
                "n_total_unique_pretest_development_BMS_observations": n_pretest_unique,
                "pretest_development_observation_rate": float(
                    n_pretest_unique / (n_train + n_val)
                ),

                "test_target_observations_counted_as_supervision": False,
            })

budget_seed = pd.DataFrame(budget_rows)

budget_seed.to_csv(
    OUT / f"{EXPERIMENT}_TRAINING_BUDGET_BY_SEED.csv",
    index=False,
)

display(budget_seed.head(20))

# %% Code cell 8 (notebook cell 9)

# Cross-check nominal selected-label counts against the frozen R1A/R2A audits.

for row in budget_seed.itertuples(index=False):
    if row.domain in DEV_DOMAINS:
        q = r1a_selector[
            (r1a_selector["domain"] == row.domain)
            & np.isclose(r1a_selector["fraction"].astype(float), row.fraction)
            & (r1a_selector["seed"].astype(int) == row.seed)
        ]

        assert len(q) == 1, (row.domain, row.fraction, row.seed)

        # R1A audit may use n_selected or equivalent selected-count column.
        candidate_cols = [
            c for c in q.columns
            if c.lower() in {
                "n_selected",
                "selected",
                "n_train_labels",
                "selected_count",
            }
        ]

        if not candidate_cols:
            raise RuntimeError(
                f"Cannot find selected-count column in R1A selector: {q.columns.tolist()}"
            )

        expected = int(q.iloc[0][candidate_cols[0]])
        assert expected == row.n_selected_training_labels, (
            row.domain, row.fraction, row.seed,
            expected, row.n_selected_training_labels
        )

    else:
        q = r2a_selector[
            np.isclose(r2a_selector["fraction"].astype(float), row.fraction)
            & (r2a_selector["seed"].astype(int) == row.seed)
        ]

        assert len(q) == 1
        expected = int(q.iloc[0]["n_selected"])
        assert expected == row.n_selected_training_labels

print("Frozen sparse-selector count cross-check: PASS")

# %% Code cell 9 (notebook cell 10)

# Manuscript-ready training-budget summary across the five seeds.

agg = {
    "n_train_rows_potential": "first",
    "n_validation_rows": "first",
    "n_train_compound_segments": "first",

    "n_selected_training_labels": ["mean", "min", "max"],
    "nominal_training_label_rate": "mean",

    "n_initial_training_anchors": "first",
    "n_anchor_rows_already_selected_as_labels": ["mean", "min", "max"],
    "n_additional_unique_anchor_observations": ["mean", "min", "max"],

    "n_unique_fit_BMS_observations": ["mean", "min", "max"],
    "effective_fit_BMS_observation_rate": ["mean", "min", "max"],

    "nominal_training_label_reduction_pct": "mean",
    "effective_fit_observation_reduction_pct": ["mean", "min", "max"],
    "anchor_overhead_percentage_points": ["mean", "min", "max"],

    "n_validation_target_observations_for_model_selection": "first",
    "n_total_unique_pretest_development_BMS_observations": ["mean", "min", "max"],
    "pretest_development_observation_rate": ["mean", "min", "max"],
}

summary = (
    budget_seed
    .groupby(["domain", "fraction"], as_index=False)
    .agg(agg)
)

summary.columns = [
    "__".join([str(x) for x in c if str(x)])
    if isinstance(c, tuple)
    else str(c)
    for c in summary.columns
]

# Clean common group-column names.
summary = summary.rename(columns={
    "domain": "domain",
    "fraction": "fraction",
    "domain__": "domain",
    "fraction__": "fraction",
})

summary.to_csv(
    OUT / f"{EXPERIMENT}_TRAINING_BUDGET_SUMMARY.csv",
    index=False,
)

display(summary)

# %% Code cell 10 (notebook cell 11)

# Compact 10%-label table for the main manuscript.

sparse = summary[
    np.isclose(summary["fraction"].astype(float), 0.10)
].copy()

def col_like(prefix, suffix):
    matches = [
        c for c in sparse.columns
        if c.startswith(prefix) and c.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError((prefix, suffix, matches))
    return matches[0]

main_table = pd.DataFrame({
    "domain": sparse["domain"],
    "potential_train_rows": sparse[col_like("n_train_rows_potential", "first")].astype(int),
    "training_labels_mean": sparse[col_like("n_selected_training_labels", "mean")],
    "initial_train_anchors": sparse[col_like("n_initial_training_anchors", "first")].astype(int),
    "anchor_label_overlap_mean": sparse[col_like("n_anchor_rows_already_selected_as_labels", "mean")],
    "additional_unique_anchor_obs_mean": sparse[col_like("n_additional_unique_anchor_observations", "mean")],
    "nominal_label_rate_pct": 100.0 * sparse[col_like("nominal_training_label_rate", "mean")],
    "effective_fit_BMS_observation_rate_pct": 100.0 * sparse[col_like("effective_fit_BMS_observation_rate", "mean")],
    "anchor_overhead_pp_mean": sparse[col_like("anchor_overhead_percentage_points", "mean")],
    "effective_fit_observation_reduction_pct": sparse[col_like("effective_fit_observation_reduction_pct", "mean")],
    "validation_targets_for_model_selection": sparse[col_like("n_validation_target_observations_for_model_selection", "first")].astype(int),
    "pretest_development_observation_rate_pct": 100.0 * sparse[col_like("pretest_development_observation_rate", "mean")],
})

main_table.to_csv(
    OUT / f"{EXPERIMENT}_MANUSCRIPT_TRAINING_TABLE.csv",
    index=False,
)

display(main_table)

# %% Code cell 11 (notebook cell 12)

# Deployment BMS observation budget from frozen R1C geometry.
#
# R1C already reports additional refresh observations/hour.
# We now add the unavoidable one initial anchor per deployment segment.

required_burden_cols = {
    "domain",
    "refresh_policy",
    "n_segments",
    "n_refresh_observations",
    "total_acquisition_hours",
    "refresh_observations_per_hour",
}
missing = required_burden_cols - set(r1c_burden.columns)
assert not missing, missing

deployment = r1c_burden.copy()

deployment["initial_anchor_observations"] = deployment["n_segments"].astype(int)
deployment["initial_anchors_per_hour"] = (
    deployment["initial_anchor_observations"]
    / deployment["total_acquisition_hours"]
)

deployment["additional_refresh_observations"] = deployment["n_refresh_observations"].astype(int)
deployment["additional_refresh_observations_per_hour"] = deployment["refresh_observations_per_hour"]

deployment["total_BMS_observations"] = (
    deployment["initial_anchor_observations"]
    + deployment["additional_refresh_observations"]
)

deployment["total_BMS_observations_per_hour"] = (
    deployment["total_BMS_observations"]
    / deployment["total_acquisition_hours"]
)

deployment["training_supervision"] = False
deployment["deployment_observation_cost"] = True

deployment.to_csv(
    OUT / f"{EXPERIMENT}_DEPLOYMENT_BUDGET_BY_DOMAIN.csv",
    index=False,
)

display(deployment)

# %% Code cell 12 (notebook cell 13)

# Acquisition-hour-weighted aggregate deployment budget across development domains.

deployment_summary_rows = []

for policy, g in deployment.groupby("refresh_policy", sort=False):
    total_hours = float(g["total_acquisition_hours"].sum())
    total_anchors = int(g["initial_anchor_observations"].sum())
    total_refresh = int(g["additional_refresh_observations"].sum())
    total_obs = total_anchors + total_refresh

    deployment_summary_rows.append({
        "refresh_policy": policy,
        "development_domains": int(g["domain"].nunique()),
        "total_acquisition_hours": total_hours,
        "initial_anchor_observations": total_anchors,
        "additional_refresh_observations": total_refresh,
        "total_BMS_observations": total_obs,
        "initial_anchors_per_hour_weighted": float(total_anchors / total_hours),
        "additional_refreshes_per_hour_weighted": float(total_refresh / total_hours),
        "total_BMS_observations_per_hour_weighted": float(total_obs / total_hours),
    })

deployment_summary = pd.DataFrame(deployment_summary_rows)

policy_order = {
    "no_refresh": 0,
    "20min": 1,
    "10min": 2,
    "5min": 3,
}

deployment_summary["_order"] = (
    deployment_summary["refresh_policy"]
    .map(policy_order)
    .fillna(99)
)

deployment_summary = (
    deployment_summary
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

deployment_summary.to_csv(
    OUT / f"{EXPERIMENT}_DEPLOYMENT_BUDGET_SUMMARY.csv",
    index=False,
)

display(deployment_summary)

# %% Code cell 13 (notebook cell 14)

# EVE deployment initial-anchor-only burden (no R1C re-anchoring experiment was run on EVE).
# This is descriptive and is not mixed with the R1C refresh trade-off.

EVE_TEST = OUTPUTS / "eve_12C_one_shot_confirmatory" / "12C_EVE_confirmatory_canonical.parquet"
assert EVE_TEST.exists(), EVE_TEST

eve_test = add_identity_and_segment(pd.read_parquet(EVE_TEST))

if "split" in eve_test.columns and (eve_test["split"] == "test").any():
    eve_test = eve_test[eve_test["split"] == "test"].copy()

durations = (
    eve_test
    .groupby("compound_segment")["elapsed_s"]
    .agg(lambda x: float(np.max(x) - np.min(x)))
)

eve_hours = float(durations.sum() / 3600.0)
eve_segments = int(eve_test["compound_segment"].nunique())

eve_deployment = pd.DataFrame([{
    "domain": "EVE",
    "policy": "initial_anchor_only_descriptive",
    "n_segments": eve_segments,
    "total_acquisition_hours": eve_hours,
    "initial_anchor_observations": eve_segments,
    "initial_anchors_per_hour": (
        float(eve_segments / eve_hours)
        if eve_hours > 0
        else np.nan
    ),
    "R1C_refresh_experiment_available": False,
}])

eve_deployment.to_csv(
    OUT / f"{EXPERIMENT}_EVE_DEPLOYMENT_ANCHOR_ONLY.csv",
    index=False,
)

display(eve_deployment)

# %% Code cell 14 (notebook cell 15)

# Integrity checks that directly guard against the "hidden labels in anchors" objection.

# Full-label case: anchors cannot increase unique fit observations.
full_rows = budget_seed[np.isclose(budget_seed["fraction"], 1.0)]

assert (
    full_rows["n_unique_fit_BMS_observations"]
    == full_rows["n_train_rows_potential"]
).all()

assert (
    full_rows["n_additional_unique_anchor_observations"] == 0
).all()

# Sparse case: effective rate must be >= nominal rate, but never >100%.
sparse_rows = budget_seed[np.isclose(budget_seed["fraction"], 0.10)]

assert (
    sparse_rows["effective_fit_BMS_observation_rate"]
    >= sparse_rows["nominal_training_label_rate"] - 1e-12
).all()

assert (
    sparse_rows["effective_fit_BMS_observation_rate"] <= 1.0
).all()

# Exact set-union identity.
assert (
    budget_seed["n_unique_fit_BMS_observations"]
    == (
        budget_seed["n_selected_training_labels"]
        + budget_seed["n_initial_training_anchors"]
        - budget_seed["n_anchor_rows_already_selected_as_labels"]
    )
).all()

# Refresh observations are not training supervision.
assert not deployment["training_supervision"].any()

print("Effective-supervision accounting integrity checks: PASS")

# %% Code cell 15 (notebook cell 16)

# Figures.

# 1) Nominal vs effective training observation rate under 10% labels.
plot_sparse = main_table.sort_values("domain").copy()
x = np.arange(len(plot_sparse))
width = 0.36

plt.figure(figsize=(9, 5))
plt.bar(
    x - width/2,
    plot_sparse["nominal_label_rate_pct"],
    width=width,
    label="Nominal training-label rate",
)
plt.bar(
    x + width/2,
    plot_sparse["effective_fit_BMS_observation_rate_pct"],
    width=width,
    label="Effective unique fit-observation rate",
)
plt.xticks(x, plot_sparse["domain"])
plt.ylabel("BMS target observations (% of potential train rows)")
plt.xlabel("Domain")
plt.title("R2B — Nominal 10% labels vs effective fit observation budget")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIG / "R2B_nominal_vs_effective_training_rate.png",
    dpi=220,
)
plt.show()

# 2) Deployment BMS observation burden.
ds = deployment_summary.copy()
x = np.arange(len(ds))

plt.figure(figsize=(8, 5))
plt.bar(
    x,
    ds["total_BMS_observations_per_hour_weighted"],
)
plt.xticks(x, ds["refresh_policy"])
plt.ylabel("BMS SoC observations per acquisition hour")
plt.xlabel("Deployment refresh policy")
plt.title("R2B — Effective deployment BMS observation burden")
plt.tight_layout()
plt.savefig(
    FIG / "R2B_deployment_observations_per_hour.png",
    dpi=220,
)
plt.show()

# %% Code cell 16 (notebook cell 17)

# Machine-readable final manifest.

manifest = {
    **preflight,
    "status": "R2B_EFFECTIVE_SUPERVISION_BUDGET_COMPLETE",
    "domains_training_budget": ALL_DOMAINS,
    "development_domains_deployment_budget": DEV_DOMAINS,
    "seed_level_training_budget_rows": int(len(budget_seed)),
    "expected_seed_level_training_budget_rows": int(
        len(ALL_DOMAINS) * len(FRACTIONS) * len(SEEDS)
    ),
    "manuscript_sparse_training_rows": int(len(main_table)),
    "deployment_domain_policy_rows": int(len(deployment)),
    "deployment_policy_summary_rows": int(len(deployment_summary)),
    "training_anchor_overlap_removed_before_counting": True,
    "validation_targets_reported_separately": True,
    "deployment_refreshes_reported_separately": True,
    "test_targets_excluded_from_supervision_budget": True,
    "R1C_refreshes_are_training_labels": False,
    "no_model_training": True,
    "no_retuning": True,
}

assert (
    manifest["seed_level_training_budget_rows"]
    == manifest["expected_seed_level_training_budget_rows"]
)

with open(OUT / f"{EXPERIMENT}_MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))

# %% Code cell 17 (notebook cell 18)

# Auto-generate manuscript-ready notes using the computed values.

lines = []
lines.append("# R2B Manuscript Notes")
lines.append("")
lines.append("## Training supervision")
lines.append("")
lines.append(
    "The nominal sparse setting uses 10% of training target rows. "
    "Because the structured prior additionally requires one genuine segment-start "
    "BMS-reported SoC anchor per training segment, we also report the number of "
    "unique BMS observations after taking the set union of sparse labels and anchors. "
    "Anchor rows already selected as training labels are not counted twice."
)
lines.append("")
lines.append("| Domain | Nominal label rate | Effective unique fit rate | Anchor overhead | Effective fit-label reduction |")
lines.append("|---|---:|---:|---:|---:|")

for r in main_table.sort_values("domain").itertuples(index=False):
    lines.append(
        f"| {r.domain} | "
        f"{r.nominal_label_rate_pct:.2f}% | "
        f"{r.effective_fit_BMS_observation_rate_pct:.2f}% | "
        f"{r.anchor_overhead_pp_mean:.2f} pp | "
        f"{r.effective_fit_observation_reduction_pct:.2f}% |"
    )

lines.append("")
lines.append(
    "**Interpretation boundary:** the 10% statement refers to TRAIN target labels. "
    "Validation targets used for early stopping/model selection are reported separately "
    "and are not hidden inside that percentage."
)
lines.append("")
lines.append("## Deployment observation budget")
lines.append("")
lines.append(
    "Deployment accounting is separate from training supervision. "
    "Each segment requires one initial BMS SoC anchor; R1C refresh policies add trusted "
    "BMS observations at the frozen 5/10/20 min schedules."
)
lines.append("")
lines.append("| Policy | Initial anchors/h | Additional refreshes/h | Total BMS observations/h |")
lines.append("|---|---:|---:|---:|")

for r in deployment_summary.itertuples(index=False):
    lines.append(
        f"| {r.refresh_policy} | "
        f"{r.initial_anchors_per_hour_weighted:.3f} | "
        f"{r.additional_refreshes_per_hour_weighted:.3f} | "
        f"{r.total_BMS_observations_per_hour_weighted:.3f} |"
    )

notes_path = OUT / f"{EXPERIMENT}_MANUSCRIPT_NOTES.md"
notes_path.write_text("\n".join(lines), encoding="utf-8")

print(notes_path.read_text(encoding="utf-8"))

# %% [markdown] Main outputs
