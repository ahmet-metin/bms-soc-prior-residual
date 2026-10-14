# Causal Prior–Residual Learning for BMS-Reported State-of-Charge Estimation

Research code accompanying the manuscript **“Causal Prior–Residual Learning for Label-Efficient Reconstruction of BMS State of Charge.”**

This repository investigates data-efficient estimation of the **state of charge (SoC) reported by a vehicle's production battery management system (BMS)**. The prediction target is BMS-reported SoC, **not an independently measured electrochemical ground-truth SoC**.

## Overview

The approach combines a structured, domain-informed SoC prior with a learned residual correction:

$$
\widehat{\mathrm{SoC}}(t) = \mathrm{SoC}_{\mathrm{prior}}(t) + \widehat{r}(t).
$$

The structured prior uses available operating information, including charge throughput and/or distance-based signals as appropriate to each dataset. The residual model learns the remaining discrepancy. We evaluate this *PriorResidual+Prior* formulation against an information-matched *Direct+Prior* model, including experiments with limited training-target observations.

A trusted **BMS SoC measurement at the beginning of each segment** provides the initial anchor. Where periodic re-anchoring is studied, additional BMS observations are counted explicitly rather than treated as free information.

The research spans five EV telemetry domains (**VED, KU, SCUT, TUM, EVE**) and includes label-efficiency, segment-aware evaluation, robustness, trajectory, and fixed-time sensitivity analyses.

## Repository structure

| Path | Contents |
|---|---|
| [`soc_repro/`](soc_repro/) | Python implementation of the primary training and evaluation pipeline |
| [`scripts/`](scripts/) | Dataset checks, source checks, and figure-generation utilities |
| [`experiments/notebook_exports/`](experiments/notebook_exports/) | Readable `.py` exports of the experiment notebooks |
| [`legacy_sources/notebooks/`](legacy_sources/notebooks/) | Original research notebooks (`.ipynb`) |
| [`notebooks/updated_colab/`](notebooks/updated_colab/) | Updated Colab notebooks for re-anchoring analyses |
| [`references/`](references/) | Small, archived aggregate tables used for reproducible summary checks |
| [`tests/`](tests/) | Automated tests |
| [`docs/`](docs/) | Dataset specification, reproduction guide, and experiment-to-code mapping |

**Note:** The `.py` notebook exports preserve the original notebook code for inspection. Some require Google Colab or experiment-specific data and are not standalone command-line programs. Use `soc_repro/` for the supported Python entry points.

## Installation

Python **3.10+** is required. A **CUDA-capable GPU** is required for fresh neural-network training; the checks and archived-figure generation can run without one.

```bash
git clone https://github.com/ahmet-metin/bms-soc-prior-residual.git
cd bms-soc-prior-residual
python -m pip install -e '.[all]'
```

## Reproducing results

### Code checks and an archived aggregate figure

These commands use files included in this repository; they do **not** retrain models or require the original EV telemetry.

```bash
python -m unittest discover -s tests -v
python scripts/verify_sources.py
python scripts/check_archived_aulec.py
python scripts/figure2_from_archived.py
```

The last command writes an aggregate label-efficiency figure under `outputs/`.

### Training with your own prepared datasets

The main training pipeline requires **frozen, preprocessed canonical tables**, which are not bundled with the source code. After preparing the inputs described in [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md):

```bash
python scripts/check_inputs.py --root /path/to/DATA_ROOT
python -m soc_repro --root /path/to/DATA_ROOT --run-root /path/to/RESULTS --max-hours 7
python -m soc_repro --run-root /path/to/RESULTS --status --verify-hashes
```

The runner supports resumable execution. To regenerate supported summary analyses from a complete seed-averaged prediction table and its observation-budget summary:

```bash
python -m soc_repro.report \
  --predictions /path/to/SEED_AVERAGED_PREDICTIONS.parquet \
  --budget /path/to/EFFECTIVE_BUDGET_SUMMARY.csv \
  --output-dir /path/to/REPORTS
```

See the [reproduction guide](docs/RUNBOOK.md) for input requirements, scope, and experiment-specific workflows.

## Data availability and reproducibility

- **Raw EV telemetry, private data files, trained checkpoints, and individual vehicle prediction records are not distributed** in this repository. Obtain the underlying datasets from their respective providers subject to applicable terms, and prepare the documented canonical inputs.
- The included aggregate tables support checking selected published calculations and generating a reference visualization. They are **not a substitute for training the models on the original data**.
- Several supplementary studies are preserved as original notebooks with readable Python exports; their execution may require additional dependencies or Colab-specific setup.
- The repository documents source provenance and the scope of verification in [`docs/`](docs/). Availability of code and archived results should not be interpreted as proof of an independent end-to-end retraining of every experiment.

## Citation

If you use this code or build on the method, please cite the accompanying manuscript. Bibliographic details will be updated upon publication; see [`CITATION.cff`](CITATION.cff) for the citation metadata.

## License

No open-source license has been assigned yet. See [`LICENSE_STATUS.md`](LICENSE_STATUS.md) for current reuse and permission information.
