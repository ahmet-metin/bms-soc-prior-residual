# Dataset input contract (not included in this repository)

The R2C runner expects a user-provided data root containing **preprocessed, frozen canonical tables**. The original 2026 research workspace is named `NissanLeaf`; the root name is *not* part of the scientific contract. Required files are:

```text
DATA_ROOT/
  outputs/
    model_screening_10/canonical/
      VED_canonical.parquet       VED_metadata.json
      KU_canonical.parquet        KU_metadata.json
      SCUT_canonical.parquet      SCUT_metadata.json
      TUM_canonical.parquet       TUM_metadata.json
    deep_baselines_11B_100epoch_vs_10K/11B_manifest.json
    eve_12B_v4_frozen_trainval/
      12B_v4_EVE_canonical_trainval.parquet
      12B_v4_EVE_canonical_metadata.json
    eve_12C_one_shot_confirmatory/
      12C_EVE_confirmatory_canonical.parquet
  Revizyon2/outputs/
    P0_sparse_prior_calibration_audit_v2/P0_SELECTOR_MANIFEST_SHA256.csv
    R2C_corrected_sparse_prior_curve/
      R2C_CORRECTED_SPARSE_PRIOR_CURVE_FROZEN_DOMAIN_TRAINING_BUDGET.csv
      R2C_CORRECTED_SPARSE_PRIOR_CURVE_ALL_CURVE_RUNS.csv  # OPTIONAL historical compare
```

The original experiments used separate preprocessing notebooks in `legacy_sources/notebooks/preprocessing`. Those notebooks are **not a standalone public downloader**. External parties must obtain the public source datasets from their licensors and reconstruct or receive legally distributable frozen canonical tables. This is an open reproducibility dependency, **not** a claimed solved part of the release.

## Data handling

BMS-reported SoC is the supervised target, **not electrochemically measured ground-truth SoC**. The initial BMS SoC segment anchor is a required trusted input and must be accounted for as a measurement. EVE primary post-anchor scoring excludes measured segment starts and does not re-establish a fresh untouched test once the original test has been opened. Do not publish vehicle identifiers, individual trajectories or raw logs without legal/privacy review.
