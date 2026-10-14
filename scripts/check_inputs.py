"""Display frozen canonical-file requirements before training (no data read, no GPU)."""
import argparse
from pathlib import Path

REQ = {
    'VED': [
        'outputs/model_screening_10/canonical/VED_canonical.parquet',
        'outputs/model_screening_10/canonical/VED_metadata.json',
    ],
    'KU': [
        'outputs/model_screening_10/canonical/KU_canonical.parquet',
        'outputs/model_screening_10/canonical/KU_metadata.json',
    ],
    'SCUT': [
        'outputs/model_screening_10/canonical/SCUT_canonical.parquet',
        'outputs/model_screening_10/canonical/SCUT_metadata.json',
    ],
    'TUM': [
        'outputs/model_screening_10/canonical/TUM_canonical.parquet',
        'outputs/model_screening_10/canonical/TUM_metadata.json',
    ],
    'EVE': [
        'outputs/eve_12B_v4_frozen_trainval/12B_v4_EVE_canonical_trainval.parquet',
        'outputs/eve_12B_v4_frozen_trainval/12B_v4_EVE_canonical_metadata.json',
        'outputs/eve_12C_one_shot_confirmatory/12C_EVE_confirmatory_canonical.parquet',
    ],
}
COMMON = [
    'outputs/deep_baselines_11B_100epoch_vs_10K/11B_manifest.json',
    'Revizyon2/outputs/P0_sparse_prior_calibration_audit_v2/P0_SELECTOR_MANIFEST_SHA256.csv',
    'Revizyon2/outputs/R2C_corrected_sparse_prior_curve/R2C_CORRECTED_SPARSE_PRIOR_CURVE_FROZEN_DOMAIN_TRAINING_BUDGET.csv',
]
OPTIONAL_ARCHIVE = 'Revizyon2/outputs/R2C_corrected_sparse_prior_curve/R2C_CORRECTED_SPARSE_PRIOR_CURVE_ALL_CURVE_RUNS.csv'

def required_files(domains):
    return COMMON + [p for d in domains for p in REQ[d]]

def check(root, domains):
    present = []
    missing = []
    for relative in required_files(domains):
        (present if (root/relative).is_file() else missing).append(relative)
    return present, missing

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Root of canonical data and manifests')
    parser.add_argument('--domain', choices=['ALL'] + list(REQ), default='ALL')
    args = parser.parse_args(argv)
    domains=list(REQ) if args.domain == 'ALL' else [args.domain]
    root=args.root.expanduser().resolve()
    present,missing=check(root, domains)
    print(f'Data root: {root}')
    print(f'Domains: {", ".join(domains)}')
    print(f'Files present: {len(present)}/{len(present)+len(missing)}')
    for p in missing: print(f'MISSING: {p}')
    if not (root/OPTIONAL_ARCHIVE).is_file():
        print('OPTIONAL: archived R2C comparison CSV absent; new training can still produce fresh receipts.')
    if missing:
        print('BLOCKED: obtain/produce the correct frozen inputs before training. Do not substitute arbitrary data.')
        return 2
    print('PASS: required input files exist. NOT proof of correct hashes, schema or data-generation reproducibility.')
    return 0

if __name__=='__main__': raise SystemExit(main())
