# Causal prior–residual learning for BMS-reported electric-vehicle SoC

Research code and experiment sources accompanying an **Applied Energy** manuscript on label-efficient reconstruction of **production BMS-reported state of charge (SoC)**. The target is the value reported by the vehicle/BMS — **not** independently established electrochemical true SoC.

The model uses a *trusted BMS SoC measurement at the start of each physical segment*. This measurement is an input, not a free observation. Periodic re-anchoring, when used, requires **additional trusted BMS observations**, whose costs are reported explicitly.

## Get started

Requirements: Python 3.10+; a CUDA GPU only for fresh GRU training. The `all` extra installs PyTorch, which the offline training-contract tests import (a GPU is **not** required to run those tests). From the repository root:

```bash
python -m pip install -e '.[all]'
python -m unittest discover -s tests -v
python scripts/verify_sources.py
python scripts/export_notebooks_to_py.py --check
python scripts/check_archived_aulec.py
python scripts/figure2_from_archived.py
```

These commands test the code and reproduce an **archived aggregate** label-efficiency plot *without* downloading telemetry or retraining models. For GPU training, supply your own legally obtained canonical data; see [data layout](docs/DATA_LAYOUT.md) and [runbook](docs/RUNBOOK.md).

```bash
# Optional: validate locally supplied canonical inputs
python scripts/check_inputs.py --root /path/to/DATA_ROOT
python -m soc_repro.prepare --root /path/to/DATA_ROOT

# Optional: run/resume R2C 5 × 7 × 5 × 2 training grid, requires CUDA
python -m soc_repro --root /path/to/DATA_ROOT --run-root /path/to/RUN_RESULTS --max-hours 7
python -m soc_repro --run-root /path/to/RUN_RESULTS --status --verify-hashes

# Optional: rederive R7/R6/Rev3B tables from a complete frozen prediction file
python -m soc_repro.report \
  --predictions /path/to/R2C_SEED_AVERAGED_PREDICTIONS.parquet \
  --budget /path/to/R2C_EFFECTIVE_BUDGET_SUMMARY.csv \
  --output-dir /path/to/REPORTS
```

## Python source files

**Yes — the main R2C algorithms are available as `.py` modules in `soc_repro/`.** For code review and reuse, this v07 release also supplies `.py` mirrors of all 28 original plus 2 corrected experiment notebooks under `experiments/notebook_exports/`. These mirrors are Colab-dependent source exports, **not independently verified standalone scripts**. See [Python entry points](docs/PYTHON_ENTRYPOINTS.md) and regenerate/verify them with `python scripts/export_notebooks_to_py.py --check`.

## Repository structure

| Directory | Purpose |
|---|---|
| `experiments/notebook_exports/` | Readable `.py` mirrors of 30 research notebooks; Colab/Drive-dependent, not standalone-certified |
| `soc_repro/` | Main reusable R2C trainer, resume logic, inputs/audit, label-efficiency, anchor-age, and re-anchoring analyses |
| `legacy_sources/notebooks/` | 28 frozen experiment source notebooks across preprocessing, baselines, inference, robustness and fixed-time studies; original source hashes retained |
| `notebooks/updated_colab/` | Output-free corrected Rev3B and Rev3E notebooks, separately from original archived sources |
| `references/` | Small archived **aggregate** results for checks and demonstration plots; not raw telemetry |
| `tests/` | Offline code, archive and synthetic-contract tests |
| `scripts/` | Source-integrity checks, input validation and archived plot calculation |
| `docs/` | Data schema, runbook, experiment/source mapping, verification status and publication caveats |

## What has actually been verified?

- **28 original notebook sources:** archived and independently hash-checked in the repository. Their presence does not mean each was rerun from scratch.
- **Offline software tests:** 22 tests passed when assembling this package.
- **Archived summary calculations:** AULEC arithmetic verified for VED, KU, SCUT, TUM and EVE. This uses stored aggregate CSVs, not new model training.
- **Stage 06 robustness output audit (14 October 2026):** R6, Rev3B, Rev3C, Rev3D and Rev3E saved output files checked; Rev3B still requires historical R2C source-version review.
- **Stage 07 fixed-time output audit (14 October 2026):** VED/SCUT/TUM output comparisons passed 107 checks, with 0 reported check errors. This is a validation of *stored* predictions, **not** independent raw-to-model reconstruction.
- **Separate cleanroom R2C run:** a prior checkpoint reported 40/350 completed fits. **Do not claim all 350 independently reproduced from the public code.** The historical/creator-side complete R2C manifest is distinct from this independent rerun.

See [verification status](docs/VERIFICATION_STATUS.md) for the precise boundaries and [experiment map](docs/EXPERIMENT_MAP.md) for manuscript component coverage.

## Data, reproducibility, and limitations

Source EV telemetry, per-vehicle predictions, private Drive files, and model checkpoint weights are **not included**. Legal access to the original datasets and generation of the canonical input tables are required for a complete reproduction. The repository does not automatically fetch private datasets.

The stored R2C five-seed-averaged predictions changed **binary SHA-256** between a historical 2 October and a current 13 October Drive revision; archived SHA and new SHA are recorded in [source provenance](docs/SOURCE_PROVENANCE.md). Since old revision bytes were not accessible, **scientific-table equivalence has not been proven**. No hash is silently reassigned to imply equivalence.

The original EVE test set had already been exposed during research. Analyses using it are not newly independent confirmatory tests. Segment-level bootstrap intervals and small vehicle counts must not be described as independent cross-vehicle generalization or safety certification.

**Source code availability does not imply open-source reuse permission.** No open-source license is asserted until all relevant rights holders authorize one. See [licensing status](LICENSE_STATUS.md). Cite the accompanying manuscript when available; see `CITATION.cff`.

## Further documentation

[Runbook](docs/RUNBOOK.md) · [Data layout](docs/DATA_LAYOUT.md) · [Experiment map](docs/EXPERIMENT_MAP.md) · [Verification and limitations](docs/VERIFICATION_STATUS.md) · [Release gates](docs/RELEASE_GATES.md) · [GitHub upload guide](docs/GITHUB_UPLOAD.md)
