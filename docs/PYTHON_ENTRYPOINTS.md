# Python (`.py`) code and notebook-source mirrors

This repository now includes two deliberately separate types of `.py` code:

## 1. Primary portable Python package — `soc_repro/`

These modules are the **supported Python entry points**, where documented data dependencies are provided. They implement R2C model training, supervision-budget logic, source/input validation, inference reports, robustness re-anchoring summaries, and progress tracking. They were already part of v06; v07 does not rewrite their scientific algorithms.

From the repository root (Python 3.10+):

```bash
python -m pip install -e '.[all]'
python -m unittest discover -s tests -v
python scripts/verify_sources.py
python scripts/export_notebooks_to_py.py --check
python scripts/check_archived_aulec.py
python -m soc_repro.experiments
```

For user-provided canonical inputs and optional CUDA training:

```bash
python scripts/check_inputs.py --root /ABSOLUTE/DATA_ROOT
python -m soc_repro.prepare --root /ABSOLUTE/DATA_ROOT
python -m soc_repro --root /ABSOLUTE/DATA_ROOT --run-root /ABSOLUTE/RUN_RESULTS --max-hours 7
python -m soc_repro --run-root /ABSOLUTE/RUN_RESULTS --status --verify-hashes
```

To regenerate post-hoc tables/plots from a **complete** frozen prediction file and supervision-budget table:

```bash
python -m soc_repro.report --predictions /ABSOLUTE/R2C_SEED_AVERAGED_PREDICTIONS.parquet --budget /ABSOLUTE/R2C_EFFECTIVE_BUDGET_SUMMARY.csv --output-dir /ABSOLUTE/REPORTS
```

These are not a claim that a full 350-model independent retraining has finished or that raw-to-canonical data preparation is completely automated; see [Runbook](RUNBOOK.md).

## 2. Notebook code mirrors — `experiments/notebook_exports/`

All **28 frozen original** experiment notebooks and **2 corrected Rev3B/Rev3E** notebooks are represented as readable `.py` source files, preserving notebook code-cell order and annotating cell boundaries. They are organized by experiment theme and can be searched, diffed, cited, and inspected directly on GitHub.

**Important limitation:** These exports are **NOT guaranteed to work directly with `python filename.py`**. They retain notebook dependencies such as `google.colab`, mounted private Drive paths, cell-order state and domain-specific prepared data. Notebook shell setup commands like `!pip install ...` are marked as comments and are never silently executed. Run the original notebook in Colab or refactor the relevant code for a new environment; do not describe a source export as newly verified standalone training.

Scientific source code is preserved; no experimental method was silently modified during conversion. The original `.ipynb` is authoritative. The manifest [`PY_EXPORT_MANIFEST.json`](PY_EXPORT_MANIFEST.json) records the source and export SHA-256 hashes. To verify the one-to-one conversion, run:

```bash
python scripts/export_notebooks_to_py.py --check
```

To regenerate the `.py` mirrors deterministically after editing an **updated** notebook, run:

```bash
python scripts/export_notebooks_to_py.py
```

Do not change files in `legacy_sources/notebooks/` in place; their historical source hashes are tracked in the original source manifest and used in verification tests.

## Optional dependencies for historical Colab notebooks

The archived notebooks sometimes require packages beyond the main R2C runtime: `scikit-learn`, `joblib`, `requests`, `openpyxl`, `xlrd`, `python-calamine`, and a Colab runtime with PyTorch. Their exact installation was notebook-specific; see commented `!pip` lines in exports or use the source `.ipynb` in Colab. These are **not automatically installed** by running a Python export.
