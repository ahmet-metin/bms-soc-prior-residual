# v05 — repository code implementation update

- Added standalone `soc_repro.prepare` validation (not a raw-data downloader).
- Preserved existing matched 350-condition R2C trainer and safe-resume runner; original stored results remain untouched.
- Implemented `soc_repro.postprocess` for R7 segment-balanced MAE, EVE revised post-anchor primary scoring and log-effective-observation AULEC.
- Implemented `soc_repro.robustness` for R6 age bins, paired segment percentile CI and farthest-supported-bin table.
- Implemented `soc_repro.reanchoring` for Rev3B fixed periodic refresh offsets, matched-unobserved evaluation and counted BMS observation cost.
- Implemented `soc_repro.report` for derived Tables and Figures 2–4 from frozen five-seed-averaged predictions (no model training).
- Created `MANUSCRIPT_CODE_REGISTRY.json` mapping all 17 labelled REV8 tables/figures to code and verification state.
- Updated CLI/packaging/docs and added 12 new synthetic offline tests. 22/22 test suite passes.
- Kept all 28 original experiment/reproduction notebooks unchanged and hash-verifiable.
- Explicitly retained barriers: separate R2D/R3A/R4 model trainer ports, all fresh reruns, Rev5 null-based inference, data/legal rights and final license.
