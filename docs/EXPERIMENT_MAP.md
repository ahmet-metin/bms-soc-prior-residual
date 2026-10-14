# REV8 manuscript component → source code map

Use `python -m soc_repro.experiments` for the machine-verified inventory. The complete registry at `docs/MANUSCRIPT_CODE_REGISTRY.json` maps all 17 explicitly labeled LaTex tables/figures to an actual file and records current validation status.

- **R2C main training**: `soc_repro/r2c_engine.py`, `soc_repro/cli.py`. Five domains, seven sparse-label fractions, five seeds and two input/parameter-matched formulations. Training consumes canonical tables and training-only calibration rules. Confirm full 350 receipts before declaring rerun complete.
- **R7 main inference, AULEC and Figure 2**: `soc_repro/postprocess.py` and `soc_repro/report.py`. Original notebook in `legacy_sources/notebooks/primary/R7*`. EVE scoring excludes the start-anchor, while all fitted label/anchor costs remain counted.
- **R6 anchor age / Figure 3**: `soc_repro/robustness.py`, `soc_repro/report.py`, original `R6*` notebook. Only bins with at least ten physical segments support interpretation.
- **Rev3B / Figure 4**: `soc_repro/reanchoring.py`, `soc_repro/report.py`, original `Rev3B*` notebook. Interval-based trusted refresh is a paid-for BMS observation and refresh rows are excluded from the matched-unobserved comparison.
- **R2D, R3A, R4, R5A, R2B, Rev3C/D/E and fixed-clock Rev3I/J/K**: full original notebook source archived, not yet demonstrated as an entirely self-contained, independently verified standalone command.
- **Figure 1**: explanatory artwork, not a model training output.

The 28 original notebook sources have not been modified. The two later corrected runnable Colab variants for Rev3B/Rev3E are in `notebooks/updated_colab/` and are **not** replacements for the hash-verified originals. Stage 06–07 source-output audit results are summarized in `docs/VERIFICATION_STATUS.md`; source preservation does not establish independent retraining.
