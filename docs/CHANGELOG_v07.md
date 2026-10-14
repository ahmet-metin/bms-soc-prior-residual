# Changes in GitHub-ready v07

- Kept the existing `soc_repro/` reusable Python modules and 22 offline software tests unchanged.
- Added 30 readable `.py` source mirrors (28 frozen notebook sources + 2 corrected Rev3B/Rev3E Colab sources) under `experiments/notebook_exports/` grouped by experiment theme.
- Added deterministic `scripts/export_notebooks_to_py.py` and `docs/PY_EXPORT_MANIFEST.json` to verify code-cell extraction against the notebook sources and SHA-256 hashes.
- Added `docs/PYTHON_ENTRYPOINTS.md` to distinguish executable R2C package code from Colab/Drive-dependent legacy source exports.
- Did not assert independent retraining, BMS-to-true-SoC correspondence, retrospective-source equivalence, or permissions to redistribute raw datasets.
- No change to archived notebooks, model/training algorithms, or frozen result tables.
