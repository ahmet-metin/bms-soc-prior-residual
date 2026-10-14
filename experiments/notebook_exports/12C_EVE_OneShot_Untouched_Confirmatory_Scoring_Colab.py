# -*- coding: utf-8 -*-
# Extracted research code; source notebook remains authoritative.
# Source: legacy_sources/notebooks/preprocessing/12C_EVE_OneShot_Untouched_Confirmatory_Scoring_Colab.ipynb
# Notebook SHA256: fdba02538ef304c68ba8b2739abfae1ec8ce41c50310f19081ceb4a2c98926b8
# WARNING: Colab/Drive-dependent; NOT certified as a standalone script.
# Google Drive mounts, canonical inputs, external data permissions and GPUs
# are still needed wherever the source experiment requires them.
# Leading !pip / % notebook magics are commented, not silently executed.
# Use original .ipynb in Colab to reproduce the original runtime setup.

# %% [markdown] 12C — EVE One-Shot Untouched Confirmatory Scoring

# %% Code cell 1 (notebook cell 2)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os, gc, json, math, time, random, hashlib, warnings, re, shutil

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

ROOT = Path('/content/drive/MyDrive/NissanLeaf')
A12 = ROOT / 'outputs' / 'eve_12A_untouched_audit'
B12 = ROOT / 'outputs' / 'eve_12B_v4_frozen_trainval'
OUT = ROOT / 'outputs' / 'eve_12C_one_shot_confirmatory'
RAW12C = ROOT / 'data' / 'EVE_Pilot' / '12C_channels'
CACHE12C = OUT / 'channel_cache_confirmatory_predictor_only'
PRED_OUT = OUT / 'predictions'
FIG_OUT = OUT / 'figures'

for p in [OUT, RAW12C, CACHE12C, PRED_OUT, FIG_OUT]:
    p.mkdir(parents=True, exist_ok=True)

SEALED_TEST = A12 / 'EVE_target_CONFIRMATORY_TEST_SEALED.parquet'
SEAL_12A = A12 / '12A_CONFIRMATORY_TEST_SEAL.json'
FREEZE_12B = B12 / '12B_v4_FREEZE_MANIFEST.json'
CONTRACT_12B = B12 / '12B_v4_canonicalization_contract_for_12C.json'
FEATURE_CONTRACT_12B = B12 / '12B_v4_raw_feature_contract.csv'
CANON_TRAINVAL_12B = B12 / '12B_v4_EVE_canonical_trainval.parquet'
K10_MANIFEST = B12 / '10K' / '12B_v4_10K_bundle_manifest.csv'
K10_SELECTED = B12 / '10K' / '12B_v4_10K_selected_protocols.csv'
DEEP_RUNS = B12 / 'deep' / '12B_v4_deep_runs.csv'

FINAL_MANIFEST = OUT / '12C_FINAL_MANIFEST.json'
UNSEAL_MARKER = OUT / '12C_UNSEAL_MARKER.json'
PREUNSEAL_AUDIT = OUT / '12C_PREUNSEAL_AUDIT.json'

for p in [SEALED_TEST, SEAL_12A, FREEZE_12B, CONTRACT_12B,
          FEATURE_CONTRACT_12B, CANON_TRAINVAL_12B,
          K10_MANIFEST, K10_SELECTED, DEEP_RUNS]:
    assert p.exists(), f'Missing frozen input: {p}'

# Hard one-shot guard.
if FINAL_MANIFEST.exists():
    raise RuntimeError(
        '12C has already completed. Do NOT rerun the confirmatory scoring notebook.'
    )
if UNSEAL_MARKER.exists():
    raise RuntimeError(
        'A prior 12C run already opened the sealed target but did not finish. '
        'Do not silently rerun. Review and document the protocol deviation first.'
    )

DATAVERSE = 'https://entrepot.recherche.data.gouv.fr'
FINAL_SEEDS = [42, 43, 44, 45, 46]
LABEL_FRACTIONS = [1.0, 0.10]
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 12612

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INFER_BATCH = 2048 if DEVICE.type == 'cuda' else 512

print('Device:', DEVICE)
print('12C output:', OUT)
print('One-shot guard: PASS')

# %% [markdown] 1. Frozen-contract and seal preflight

# %% Code cell 2 (notebook cell 4)
def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


with open(SEAL_12A) as f:
    seal12a = json.load(f)
with open(FREEZE_12B) as f:
    freeze12b = json.load(f)
with open(CONTRACT_12B) as f:
    contract12b = json.load(f)

assert freeze12b['status'] == 'FROZEN_READY_FOR_ONE_SHOT_12C'
assert freeze12b['confirmatory_test_target_opened'] is False
assert freeze12b['confirmatory_test_scored'] is False
assert freeze12b['test_tuning_allowed'] is False
assert freeze12b['validation_only_protocol_selection'] is True
assert freeze12b['real_state_of_charge_predictor'] is False
assert freeze12b['implementation_version'] == '12B-v4'

EXPECTED_TEST_ROWS = int(freeze12b['confirmatory_test_expected_rows'])
EXPECTED_TEST_SHA256 = str(freeze12b['confirmatory_test_expected_sha256'])
EXPECTED_CANON_FP = str(freeze12b['canonical_fingerprint_sha256'])

assert EXPECTED_TEST_ROWS == int(seal12a['n_timestamped_rows']) == 48838
assert EXPECTED_TEST_SHA256 == seal12a['sha256']
assert EXPECTED_TEST_SHA256 == contract12b['confirmatory_target_sha256_expected']
assert EXPECTED_CANON_FP == contract12b['canonical_fingerprint_sha256']

ACTUAL_TEST_SHA256 = sha256_file(SEALED_TEST)
assert ACTUAL_TEST_SHA256 == EXPECTED_TEST_SHA256, (
    'Sealed confirmatory parquet bytes changed',
    EXPECTED_TEST_SHA256,
    ACTUAL_TEST_SHA256,
)

TEST_START = pd.Timestamp(seal12a['start_inclusive'])
TEST_END = pd.Timestamp(seal12a['end_inclusive'])
BURST_BREAK_S = float(contract12b['burst_break_s'])
ALIGN_TOLERANCE_S = float(contract12b['alignment_tolerance_s'])
RAW_START = TEST_START - pd.Timedelta(seconds=ALIGN_TOLERANCE_S)
BETA_PP_PER_AH = float(contract12b['beta_pp_per_ah'])
FEATURE_COLUMNS = list(contract12b['feature_columns'])
FEATURE_MEDIANS = {k: float(v) for k, v in contract12b['feature_medians_train_only'].items()}
RAW_CONTEXT_SIGNALS = list(contract12b['raw_context_signals'])
CONTEXT_WINDOWS = sorted(float(x) for x in contract12b['raw_context_windows_s'])

# Frozen 10K constants needed only for inference/deserialization-compatible behavior.
SLOW_A_CLIP = (-1.5, 1.5)
SLOW_B_CLIP = (-10.0, 10.0)

# Frozen deep architecture contract.
DEEP_MODELS = list(freeze12b['deep_baselines']['models'])
SEQ_LEN = int(freeze12b['deep_baselines']['sequence_length'])
HIDDEN = int(freeze12b['deep_baselines']['hidden'])
TCN_CHANNELS = int(freeze12b['deep_baselines']['TCN_channels'])
TCN_KERNEL = int(freeze12b['deep_baselines']['TCN_kernel'])
TCN_DILATIONS = list(freeze12b['deep_baselines']['TCN_dilations'])
DROPOUT = float(freeze12b['deep_baselines']['dropout'])
MAX_EPOCHS = int(freeze12b['deep_baselines']['max_epochs'])
PATIENCE = int(freeze12b['deep_baselines']['patience'])
LR = float(freeze12b['deep_baselines']['lr'])
WEIGHT_DECAY = float(freeze12b['deep_baselines']['weight_decay'])

assert DEEP_MODELS == ['TCN', 'LSTM', 'GRU']
assert freeze12b['deep_baselines']['architecture_tuning_on_EVE'] is False
assert list(freeze12b['supervision_fractions']) == [1.0, 0.1]

print('Frozen 12B contract: PASS')
print('Sealed target SHA-256: PASS')
print('Expected sealed rows:', EXPECTED_TEST_ROWS)
print('Test interval:', TEST_START, '->', TEST_END)
print('Canonical fingerprint:', EXPECTED_CANON_FP)

# %% [markdown] 2. Verify all frozen model artifacts before unsealing

# %% Code cell 3 (notebook cell 6)
k10_manifest = pd.read_csv(K10_MANIFEST)
k10_selected = pd.read_csv(K10_SELECTED)
deep_runs = pd.read_csv(DEEP_RUNS)

assert len(k10_manifest) == 10
assert len(deep_runs) == 30
assert set(k10_manifest['seed'].astype(int)) == set(FINAL_SEEDS)
assert set(deep_runs['seed'].astype(int)) == set(FINAL_SEEDS)
assert set(np.round(k10_manifest['fraction'].astype(float), 2)) == {0.1, 1.0}
assert set(np.round(deep_runs['fraction'].astype(float), 2)) == {0.1, 1.0}
assert set(deep_runs['model']) == set(DEEP_MODELS)

artifact_audit = []

for r in k10_manifest.itertuples(index=False):
    p = Path(str(r.bundle_path))
    assert p.exists(), p
    got = sha256_file(p)
    assert got == str(r.bundle_sha256), ('10K bundle hash mismatch', p)
    artifact_audit.append({
        'artifact_type': '10K_bundle',
        'fraction': float(r.fraction),
        'model': 'SC_RS_SFPEF',
        'seed': int(r.seed),
        'path': str(p),
        'sha256': got,
        'status': 'PASS',
    })

for r in deep_runs.itertuples(index=False):
    p = Path(str(r.checkpoint))
    assert p.exists(), p
    got = sha256_file(p)
    assert got == str(r.checkpoint_sha256), ('Deep checkpoint hash mismatch', p)
    artifact_audit.append({
        'artifact_type': 'deep_checkpoint',
        'fraction': float(r.fraction),
        'model': str(r.model),
        'seed': int(r.seed),
        'path': str(p),
        'sha256': got,
        'status': 'PASS',
    })

artifact_audit_df = pd.DataFrame(artifact_audit)
artifact_audit_df.to_csv(OUT / '12C_frozen_artifact_integrity.csv', index=False)
print('Frozen artifacts verified:', len(artifact_audit_df), '/', 40)

# %% [markdown] 3. Prepare confirmatory predictor channels — still pre-unseal

# %% Code cell 4 (notebook cell 8)
feature_contract = pd.read_csv(FEATURE_CONTRACT_12B)
assert len(feature_contract) == 7

SOURCE_DIRS = [
    ROOT / 'data' / 'EVE_Pilot' / '12B_v2_channels',
    ROOT / 'data' / 'EVE_Pilot' / '12B_v3_channels',
    ROOT / 'data' / 'EVE_Pilot' / '12B_v4_channels',
    ROOT / 'data' / 'EVE_Pilot' / 'selected_channels',
]


def normalized_stem(path):
    name = Path(str(path)).name
    stem = re.sub(r'\.(tab|csv|tsv|txt)$', '', name, flags=re.I)
    return re.sub(r'\s+', ' ', stem.strip().lower())


def find_local_source(repository_path, official_md5):
    wanted = normalized_stem(repository_path)
    exact_name = Path(str(repository_path)).name.lower()

    for d in SOURCE_DIRS:
        if not d.exists():
            continue
        candidates = [p for p in d.iterdir() if p.is_file() and normalized_stem(p.name) == wanted]
        if not candidates:
            continue
        # Prefer exact original filename if present.
        candidates = sorted(candidates, key=lambda p: p.name.lower() != exact_name)
        p = candidates[0]
        if p.name.lower() == exact_name:
            got = md5_file(p)
            if got.lower() != str(official_md5).lower():
                continue
            return p, 'local_original_md5_verified'
        return p, 'trusted_local_transformed_copy'
    return None, None


def retry_download_original(file_id, fallback_name, expected_md5, max_attempts=6):
    dest = RAW12C / fallback_name
    if dest.exists() and dest.stat().st_size > 0:
        if md5_file(dest).lower() == str(expected_md5).lower():
            return dest
        dest.unlink()

    url = f'{DATAVERSE}/api/access/datafile/{int(file_id)}'
    waits = [2, 5, 10, 20, 40, 60]
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        tmp = dest.with_suffix(dest.suffix + '.partial')
        tmp.unlink(missing_ok=True)
        try:
            print(f'Dataverse {attempt}/{max_attempts}: {fallback_name}')
            with requests.get(url, params={'format': 'original'}, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            got = md5_file(tmp)
            assert got.lower() == str(expected_md5).lower(), (expected_md5, got)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < max_attempts:
                time.sleep(waits[min(attempt - 1, len(waits)-1)])

    raise ConnectionError(f'Dataverse download failed: {file_id}') from last_exc


def sniff_sep(path):
    with open(path, 'rb') as f:
        sample = f.read(65536)
    text = sample.decode('utf-8', errors='ignore')
    counts = {'\t': text.count('\t'), ',': text.count(','), ';': text.count(';')}
    return max(counts, key=counts.get)


def detect_columns(df):
    named_time = [c for c in df.columns if re.search(r'time|date|timestamp', str(c), flags=re.I)]
    ts = []
    for c in df.columns:
        x = pd.to_datetime(df[c].head(min(5000, len(df))), errors='coerce', utc=True)
        rate = float(x.notna().mean())
        ts.append((rate + (0.2 if c in named_time else 0.0), rate, c))
    ts.sort(reverse=True, key=lambda x: x[0])
    _, tr, tcol = ts[0]
    assert tr >= 0.70, (tcol, tr)

    vs = []
    for c in df.columns:
        if c == tcol:
            continue
        x = pd.to_numeric(df[c].head(min(5000, len(df))), errors='coerce')
        vs.append((float(x.notna().mean()), c))
    vs.sort(reverse=True)
    vr, vcol = vs[0]
    assert vr >= 0.70, (vcol, vr)
    return tcol, vcol


def build_confirmatory_predictor_cache(row):
    feature = str(row['feature'])
    cache_path = CACHE12C / f'{feature}.parquet'
    meta_path = CACHE12C / f'{feature}.json'

    expected = {
        'feature': feature,
        'file_id': int(row['file_id']),
        'repository_path': str(row['repository_path']),
        'official_md5': str(row['official_md5']).lower(),
        'raw_start_inclusive': RAW_START.isoformat(),
        'test_end_inclusive': TEST_END.isoformat(),
    }

    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                old = json.load(f)
            if all(old.get(k) == v for k, v in expected.items()):
                z = pd.read_parquet(cache_path)
                assert len(z) > 0
                return cache_path, old
        except Exception:
            pass
        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    source, source_mode = find_local_source(row['repository_path'], row['official_md5'])
    if source is None:
        source = retry_download_original(
            int(row['file_id']),
            Path(str(row['repository_path'])).name,
            str(row['official_md5']),
        )
        source_mode = 'Dataverse_original_md5_verified'

    sep = sniff_sep(source)
    df = pd.read_csv(source, sep=sep, low_memory=False).dropna(axis=1, how='all')
    tcol, vcol = detect_columns(df)

    z = pd.DataFrame({
        'timestamp': pd.to_datetime(df[tcol], errors='coerce', utc=True).dt.tz_convert(None),
        feature: pd.to_numeric(df[vcol], errors='coerce'),
    })
    z = z[
        z['timestamp'].notna()
        & (z['timestamp'] >= RAW_START)
        & (z['timestamp'] <= TEST_END)
    ].copy()
    z = z.sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True)

    assert len(z) > 0, feature
    assert z['timestamp'].is_monotonic_increasing
    assert float(z[feature].notna().mean()) > 0.95

    z.to_parquet(cache_path, index=False)
    meta = {
        **expected,
        'source_mode': source_mode,
        'source_file': str(source),
        'n_predictor_rows': int(len(z)),
        'finite_fraction': float(z[feature].notna().mean()),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    del df, z
    gc.collect()
    return cache_path, meta


channel_cache = {}
channel_provenance = []
for _, row in feature_contract.iterrows():
    p, meta = build_confirmatory_predictor_cache(row)
    feature = str(row['feature'])
    channel_cache[feature] = p
    channel_provenance.append({
        'feature': feature,
        'file_id': int(row['file_id']),
        'repository_path': str(row['repository_path']),
        'source_mode': meta['source_mode'],
        'n_predictor_rows': int(meta['n_predictor_rows']),
        'finite_fraction': float(meta['finite_fraction']),
        'cache_path': str(p),
    })

channel_provenance_df = pd.DataFrame(channel_provenance)
channel_provenance_df.to_csv(OUT / '12C_predictor_channel_provenance.csv', index=False)
assert set(channel_cache) == set(feature_contract['feature'])
display(channel_provenance_df)

# %% Code cell 5 (notebook cell 9)
# Predictor-only current state, prepared before target unsealing.
test_current = pd.read_parquet(channel_cache['current_a']).copy()
test_current = test_current.dropna(subset=['timestamp', 'current_a']).sort_values('timestamp').reset_index(drop=True)

dt = test_current['timestamp'].diff().dt.total_seconds()
new_seg = dt.isna() | (dt <= 0) | (dt > BURST_BREAK_S)
test_current['raw_segment_id'] = new_seg.cumsum().astype(int) - 1

prev_i = test_current['current_a'].shift(1)
same_seg = test_current['raw_segment_id'].eq(test_current['raw_segment_id'].shift(1))

step_ah = np.where(
    same_seg,
    0.5 * (
        test_current['current_a'].to_numpy(float)
        + prev_i.fillna(test_current['current_a']).to_numpy(float)
    ) * dt.fillna(0).to_numpy(float) / 3600.0,
    0.0,
)
step_qabs = np.where(
    same_seg,
    0.5 * (
        np.abs(test_current['current_a'].to_numpy(float))
        + np.abs(prev_i.fillna(test_current['current_a']).to_numpy(float))
    ) * dt.fillna(0).to_numpy(float) / 3600.0,
    0.0,
)

test_current['ah_cum_segment'] = pd.Series(step_ah).groupby(test_current['raw_segment_id']).cumsum().to_numpy(float)
test_current['qabs_cum_segment'] = pd.Series(step_qabs).groupby(test_current['raw_segment_id']).cumsum().to_numpy(float)

test_current.to_parquet(OUT / '12C_predictor_only_current_state.parquet', index=False)

predictor_only_audit = {
    'test_target_parsed': False,
    'raw_start_inclusive': RAW_START.isoformat(),
    'test_end_inclusive': TEST_END.isoformat(),
    'current_rows': int(len(test_current)),
    'current_segments': int(test_current['raw_segment_id'].nunique()),
    'sealed_target_sha256_verified': True,
}
with open(OUT / '12C_predictor_only_audit.json', 'w') as f:
    json.dump(predictor_only_audit, f, indent=2)

print(predictor_only_audit)

# %% [markdown] 4. Frozen inference code

# %% Code cell 6 (notebook cell 11)
def regression_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    e = pred - y
    ae = np.abs(e)
    return {
        'MAE': float(ae.mean()),
        'RMSE': float(np.sqrt(np.mean(e ** 2))),
        'P95': float(np.percentile(ae, 95)),
        'Bias': float(e.mean()),
    }


def tree_prediction_std(model, X):
    pred = np.vstack([t.predict(X) for t in model.estimators_])
    return np.std(pred, axis=0, ddof=0)


def predict_residual(model, rows, feature_cols):
    X = rows[feature_cols].copy()
    X['__physics__'] = rows['soc_phy'].to_numpy(float)
    return rows['soc_phy'].to_numpy(float) + model.predict(X)


def build_early_context(source_df, feature_cols, window_s):
    key = float(window_s)
    assert key in RAW_CONTEXT_BANK, key
    wanted = set(source_df['compound_segment'].astype(str))
    out = RAW_CONTEXT_BANK[key][
        RAW_CONTEXT_BANK[key]['compound_segment'].astype(str).isin(wanted)
    ].copy()
    coverage = out['compound_segment'].nunique() / max(len(wanted), 1)
    assert coverage >= 0.95, (window_s, coverage)
    return out.reset_index(drop=True)


def predict_slow_pair(ma, mb, context_df, ctx_cols):
    out = context_df[['compound_segment']].copy()
    if ma is None or mb is None or not len(context_df):
        out['a_pred'] = 0.0
        out['b_pred'] = 0.0
        out['a_std'] = np.inf
        out['b_std'] = np.inf
        return out
    X = context_df[ctx_cols]
    out['a_pred'] = np.clip(ma.predict(X), SLOW_A_CLIP[0], SLOW_A_CLIP[1])
    out['b_pred'] = np.clip(mb.predict(X), SLOW_B_CLIP[0], SLOW_B_CLIP[1])
    out['a_std'] = tree_prediction_std(ma, X)
    out['b_std'] = tree_prediction_std(mb, X)
    return out


def add_slow_state(rows_df, slow_pred_table, strength, tau_pp):
    d = rows_df.merge(slow_pred_table, on='compound_segment', how='left', validate='many_to_one')
    d[['a_pred', 'b_pred']] = d[['a_pred', 'b_pred']].fillna(0.0)
    d[['a_std', 'b_std']] = d[['a_std', 'b_std']].fillna(np.inf)

    p = d['physics_delta'].to_numpy(float)
    a = d['a_pred'].to_numpy(float)
    b = d['b_pred'].to_numpy(float)
    sa = d['a_std'].to_numpy(float)
    sb = d['b_std'].to_numpy(float)

    slow_raw = a * p + b
    uncertainty = np.sqrt((p * sa) ** 2 + sb ** 2)
    confidence = np.where(
        np.isfinite(uncertainty),
        1.0 / (1.0 + uncertainty / float(tau_pp)),
        0.0,
    )
    correction = float(strength) * confidence * slow_raw

    d['slow_raw_pp'] = slow_raw
    d['slow_uncertainty_pp'] = uncertainty
    d['slow_confidence'] = confidence
    d['slow_correction_pp'] = correction
    d['sf2_base'] = d['soc_phy'].to_numpy(float) + correction
    return d


def causal_eligible_mask(df, early_window_s):
    elapsed = df['elapsed_s'].to_numpy(float)
    assert np.isfinite(elapsed).all()
    return elapsed + 1e-9 >= float(early_window_s)


def enforce_strict_warmup_state(df, early_window_s):
    d = df.copy()
    eligible = causal_eligible_mask(d, early_window_s)
    d['causal_eligible'] = eligible
    early = ~eligible
    if early.any():
        d.loc[early, 'slow_raw_pp'] = 0.0
        d.loc[early, 'slow_uncertainty_pp'] = np.inf
        d.loc[early, 'slow_confidence'] = 0.0
        d.loc[early, 'slow_correction_pp'] = 0.0
        d.loc[early, 'sf2_base'] = d.loc[early, 'soc_phy'].to_numpy(float)
    return d


def router_features(df, feature_cols):
    X = df[feature_cols].copy()
    pr = df['pred_r'].to_numpy(float)
    ps = df['pred_sf'].to_numpy(float)
    phy = df['soc_phy'].to_numpy(float)
    X['__soc_phy__'] = phy
    X['__physics_delta__'] = df['physics_delta'].to_numpy(float)
    X['__elapsed__'] = df['elapsed_s'].to_numpy(float)
    X['__soc0__'] = df['soc0'].to_numpy(float)
    X['__pred_r__'] = pr
    X['__pred_sf__'] = ps
    X['__sf_minus_r__'] = ps - pr
    X['__expert_gap__'] = np.abs(ps - pr)
    X['__r_to_phy__'] = np.abs(pr - phy)
    X['__sf_to_phy__'] = np.abs(ps - phy)
    assert 'target' not in X.columns
    return X


def route_prediction_final(router, expert_frame, feature_cols, mode, kappa, margin, sf_weight):
    assert mode == 'Routed'
    pr = expert_frame['pred_r'].to_numpy(float)
    ps = expert_frame['pred_sf'].to_numpy(float)
    eligible = expert_frame['causal_eligible'].to_numpy(bool)
    pred = pr.copy()
    switch_all = np.zeros(len(expert_frame), dtype=bool)

    if float(sf_weight) == 0.0:
        return pred, {'switch_rate': 0.0, 'postwarmup_switch_rate': 0.0, 'fail_closed': True}

    if eligible.any():
        Z = router_features(expert_frame.loc[eligible], feature_cols)
        adv_mean = router.predict(Z)
        adv_std = tree_prediction_std(router, Z)
        lcb = adv_mean - float(kappa) * adv_std
        switch = lcb > float(margin)
        w = np.where(switch, float(sf_weight), 0.0)
        idx = np.flatnonzero(eligible)
        switch_all[idx] = switch
        pred[idx] = pr[idx] + w * (ps[idx] - pr[idx])

    pred[~eligible] = pr[~eligible]
    return pred, {
        'switch_rate': float(np.mean(switch_all)),
        'postwarmup_switch_rate': float(np.mean(switch_all[eligible])) if eligible.any() else 0.0,
        'fail_closed': False,
    }


def predict_10k_bundle(bundle, pred_rows):
    feature_cols = bundle['feature_columns']
    sf_protocol = bundle['sf_protocol']
    pred_r = predict_residual(bundle['residual_model'], pred_rows, feature_cols)
    pred_context = build_early_context(pred_rows, feature_cols, sf_protocol['early_window_s'])

    if bundle['slow_a_model'] is None or bundle['slow_b_model'] is None or bundle['slow_context_columns'] is None:
        slow_pred = pred_context[['compound_segment']].copy()
        slow_pred['a_pred'] = 0.0
        slow_pred['b_pred'] = 0.0
        slow_pred['a_std'] = np.inf
        slow_pred['b_std'] = np.inf
    else:
        slow_pred = predict_slow_pair(
            bundle['slow_a_model'], bundle['slow_b_model'],
            pred_context, bundle['slow_context_columns']
        )

    pp = add_slow_state(
        pred_rows, slow_pred,
        sf_protocol['slow_strength'], sf_protocol['tau_pp']
    )
    pp = enforce_strict_warmup_state(pp, sf_protocol['early_window_s'])
    pred_sf_unmasked = pp['sf2_base'].to_numpy(float) + bundle['fast_model'].predict(pp[bundle['fast_features']])
    eligible = causal_eligible_mask(pp, sf_protocol['early_window_s'])
    pred_sf = np.where(eligible, pred_sf_unmasked, pred_r)

    expert = pred_rows.copy()
    expert['pred_r'] = pred_r
    expert['pred_sf'] = pred_sf
    expert['causal_eligible'] = eligible
    pred_final, route_audit = route_prediction_final(
        bundle['router'], expert, feature_cols, **bundle['router_cfg']
    )
    return {
        'Residual': pred_r,
        'StrictSF_ablation': pred_sf,
        'SC_RS_SFPEF': pred_final,
        'route_audit': route_audit,
    }

# %% Code cell 7 (notebook cell 12)
class RNNRegressor(nn.Module):
    def __init__(self, input_dim, kind='LSTM', hidden=64):
        super().__init__()
        if kind == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        elif kind == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(32, 1),
        )

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1, :]).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.c2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        y = self.drop(self.act(self.c1(x)))
        y = self.drop(self.act(self.c2(y)))
        return self.act(y + residual)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        blocks = []
        in_ch = input_dim
        for dilation in TCN_DILATIONS:
            blocks.append(TCNBlock(in_ch, TCN_CHANNELS, TCN_KERNEL, dilation, DROPOUT))
            in_ch = TCN_CHANNELS
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(TCN_CHANNELS), nn.Linear(TCN_CHANNELS, 32), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(32, 1),
        )

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)


def build_deep_model(model_name, input_dim):
    if model_name == 'TCN':
        return TCNRegressor(input_dim)
    if model_name in {'LSTM', 'GRU'}:
        return RNNRegressor(input_dim, kind=model_name, hidden=HIDDEN)
    raise ValueError(model_name)


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def deep_contract_fingerprint(model_name, fraction, seed, features):
    contract = {
        'model': str(model_name),
        'fraction': float(fraction),
        'seed': int(seed),
        'features': list(features),
        'seq_len': int(SEQ_LEN),
        'hidden': int(HIDDEN),
        'tcn_channels': int(TCN_CHANNELS),
        'tcn_dilations': list(TCN_DILATIONS),
        'tcn_kernel': int(TCN_KERNEL),
        'dropout': float(DROPOUT),
        'lr': float(LR),
        'weight_decay': float(WEIGHT_DECAY),
        'max_epochs': int(MAX_EPOCHS),
        'patience': int(PATIENCE),
        'canonical_fingerprint_sha256': str(EXPECTED_CANON_FP),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode('utf-8')).hexdigest()


class CausalInferenceDataset(Dataset):
    def __init__(self, frame, X_scaled, seq_len):
        self.frame = frame.reset_index(drop=True)
        self.X = np.asarray(X_scaled, np.float32)
        self.seq_len = int(seq_len)
        self.positions = np.arange(len(self.frame), dtype=int)
        self.seg_start = (
            self.frame.groupby('compound_segment', sort=False)['seq_pos']
            .min().astype(int).to_dict()
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        pos = int(self.positions[i])
        row = self.frame.iloc[pos]
        seg0 = int(self.seg_start[row['compound_segment']])
        start = max(seg0, pos - self.seq_len + 1)
        x = self.X[start:pos+1]
        n = len(x)
        pad = self.seq_len - n
        out = np.zeros((self.seq_len, self.X.shape[1] + 1), dtype=np.float32)
        if n:
            out[pad:, :-1] = x
            out[pad:, -1] = 1.0
        return torch.from_numpy(out), pos


def prepare_inference_frame(df):
    z = df.sort_values(['compound_segment', 'time_value'], kind='stable').reset_index(drop=True).copy()
    z['seq_pos'] = np.arange(len(z), dtype=int)
    return z


@torch.no_grad()
def predict_deep_checkpoint(checkpoint_path, frame, expected_model, expected_fraction, expected_seed):
    state = safe_torch_load(checkpoint_path, DEVICE)
    assert state['model_name'] == expected_model
    assert np.isclose(float(state['fraction']), float(expected_fraction))
    assert int(state['seed']) == int(expected_seed)
    assert list(state['features']) == FEATURE_COLUMNS
    assert int(state['sequence_length']) == SEQ_LEN
    expected_fp = deep_contract_fingerprint(expected_model, expected_fraction, expected_seed, FEATURE_COLUMNS)
    assert state.get('contract_fingerprint') == expected_fp

    prep = state['preprocessor']
    X = frame[FEATURE_COLUMNS].to_numpy(float)
    med = np.asarray(prep['median'], np.float32)
    mean = np.asarray(prep['mean'], np.float32)
    std = np.asarray(prep['std'], np.float32)
    X = np.where(np.isfinite(X), X, med[None, :])
    X = ((X - mean[None, :]) / std[None, :]).astype(np.float32)

    ds = CausalInferenceDataset(frame, X, int(state['sequence_length']))
    loader = DataLoader(ds, batch_size=INFER_BATCH, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    model = build_deep_model(expected_model, int(state['input_dim'])).to(DEVICE)
    model.load_state_dict(state['model_state'])
    model.eval()

    pred_scaled = []
    pos_all = []
    for xb, pos in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        p = model(xb)
        pred_scaled.append(p.detach().cpu().numpy())
        pos_all.append(np.asarray(pos))

    pred_scaled = np.concatenate(pred_scaled)
    pos_all = np.concatenate(pos_all).astype(int)
    target_scale = state['target_scaler']
    pred = pred_scaled.astype(float) * float(target_scale['std']) + float(target_scale['mean'])

    out = pd.DataFrame({
        'row_uid': frame.iloc[pos_all]['row_uid'].astype(str).to_numpy(),
        'prediction': pred,
    })
    assert len(out) == len(frame)
    assert np.isfinite(out['prediction'].to_numpy(float)).all()

    del model, ds, loader, state
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return out

# %% [markdown] 5. Pre-unseal validation smoke test

# %% Code cell 8 (notebook cell 14)
# Existing 12B contexts provide an exact smoke-test environment.
val_all = pd.read_parquet(CANON_TRAINVAL_12B)
val_all = val_all[val_all['split'] == 'val'].copy()

smoke_segments = (
    val_all.sort_values('timestamp')['compound_segment']
    .drop_duplicates().head(10).tolist()
)
val_smoke = val_all[val_all['compound_segment'].isin(smoke_segments)].copy()
assert len(val_smoke) > 0

RAW_CONTEXT_BANK = {}
for w in CONTEXT_WINDOWS:
    p = B12 / 'context' / f'12B_v4_EVE_raw_context_{int(w)}s.parquet'
    assert p.exists(), p
    RAW_CONTEXT_BANK[float(w)] = pd.read_parquet(p)

smoke_rows = []

# One representative bundle from each supervision regime; all bundle bytes were hash-verified above.
for fraction in LABEL_FRACTIONS:
    r = k10_manifest[np.isclose(k10_manifest['fraction'].astype(float), fraction)].sort_values('seed').iloc[0]
    bundle = joblib.load(Path(r['bundle_path']))
    assert list(bundle['feature_columns']) == FEATURE_COLUMNS
    assert np.isclose(float(bundle['beta_pp_per_ah']), BETA_PP_PER_AH)
    p = predict_10k_bundle(bundle, val_smoke)
    for name in ['Residual', 'StrictSF_ablation', 'SC_RS_SFPEF']:
        assert len(p[name]) == len(val_smoke)
        assert np.isfinite(np.asarray(p[name], float)).all()
    smoke_rows.append({'family': '10K', 'fraction': fraction, 'model': 'SC_RS_SFPEF', 'seed': int(r['seed']), 'status': 'PASS'})
    del bundle, p
    gc.collect()

# One representative checkpoint per architecture and supervision regime.
val_smoke_frame = prepare_inference_frame(val_smoke)
for fraction in LABEL_FRACTIONS:
    for model_name in DEEP_MODELS:
        r = deep_runs[
            np.isclose(deep_runs['fraction'].astype(float), fraction)
            & deep_runs['model'].eq(model_name)
        ].sort_values('seed').iloc[0]
        pp = predict_deep_checkpoint(
            Path(r['checkpoint']), val_smoke_frame,
            model_name, fraction, int(r['seed'])
        )
        assert len(pp) == len(val_smoke_frame)
        smoke_rows.append({'family': 'deep', 'fraction': fraction, 'model': model_name, 'seed': int(r['seed']), 'status': 'PASS'})

smoke_df = pd.DataFrame(smoke_rows)
smoke_df.to_csv(OUT / '12C_preunseal_validation_smoke.csv', index=False)
assert (smoke_df['status'] == 'PASS').all()
print('Pre-unseal validation smoke: PASS', len(smoke_df), 'checks')

# %% Code cell 9 (notebook cell 15)
preunseal = {
    'stage': 'PRE_UNSEAL_COMPLETE',
    'sealed_target_parquet_parsed': False,
    'sealed_target_sha256_expected': EXPECTED_TEST_SHA256,
    'sealed_target_sha256_actual': ACTUAL_TEST_SHA256,
    'sealed_target_sha256_match': True,
    'expected_test_rows': EXPECTED_TEST_ROWS,
    'canonical_fingerprint_sha256': EXPECTED_CANON_FP,
    'frozen_10K_bundles_verified': int((artifact_audit_df['artifact_type'] == '10K_bundle').sum()),
    'frozen_deep_checkpoints_verified': int((artifact_audit_df['artifact_type'] == 'deep_checkpoint').sum()),
    'predictor_channels_ready': int(len(channel_cache)),
    'validation_smoke_checks_passed': int(len(smoke_df)),
    'test_tuning_allowed': False,
    'safe_to_unseal': True,
}
with open(PREUNSEAL_AUDIT, 'w') as f:
    json.dump(preunseal, f, indent=2)
print(json.dumps(preunseal, indent=2))

# %% [markdown] 🔒 → 🔓 UNSEAL BOUNDARY

# %% Code cell 10 (notebook cell 17)
assert PREUNSEAL_AUDIT.exists()
assert not UNSEAL_MARKER.exists()
assert not FINAL_MANIFEST.exists()

unseal_time = pd.Timestamp.utcnow().isoformat()
unseal_marker = {
    'event': 'EVE_CONFIRMATORY_TARGET_UNSEALED_ONCE',
    'utc_time': unseal_time,
    'sha256': ACTUAL_TEST_SHA256,
    'expected_rows': EXPECTED_TEST_ROWS,
    'rule': 'No retuning, retraining, feature changes, or silent rerun after this marker.',
}
with open(UNSEAL_MARKER, 'w') as f:
    json.dump(unseal_marker, f, indent=2)

# THE ONLY PARSE/OPEN OF THE SEALED CONFIRMATORY TARGET IN 12C.
sealed_target = pd.read_parquet(SEALED_TEST).copy()

assert set(sealed_target.columns) >= {'timestamp', 'target_soc'}
sealed_target['timestamp'] = pd.to_datetime(sealed_target['timestamp'])
assert len(sealed_target) == EXPECTED_TEST_ROWS
assert sealed_target['timestamp'].notna().all()
assert np.isfinite(sealed_target['target_soc'].to_numpy(float)).all()
assert (sealed_target['timestamp'] >= TEST_START).all()
assert (sealed_target['timestamp'] <= TEST_END).all()

print('CONFIRMATORY TARGET UNSEALED ONCE')
print('Rows:', len(sealed_target))
print('No performance metric has been computed yet.')

# %% [markdown] 6. Frozen confirmatory canonicalization

# %% Code cell 11 (notebook cell 19)
targets = (
    sealed_target[['timestamp', 'target_soc']]
    .rename(columns={'target_soc': 'target'})
    .sort_values('timestamp', kind='stable')
    .reset_index(drop=True)
)

test_current_merge = test_current[[
    'timestamp', 'raw_segment_id', 'current_a',
    'ah_cum_segment', 'qabs_cum_segment'
]].copy().rename(columns={'timestamp': 'current_source_timestamp'})

test_canon = pd.merge_asof(
    targets,
    test_current_merge.sort_values('current_source_timestamp'),
    left_on='timestamp', right_on='current_source_timestamp',
    direction='backward',
    tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
)

test_canon = test_canon.dropna(subset=['raw_segment_id', 'current_a', 'current_source_timestamp']).copy()
test_canon['raw_segment_id'] = test_canon['raw_segment_id'].astype(int)
test_canon['current_alignment_age_s'] = (
    test_canon['timestamp'] - test_canon['current_source_timestamp']
).dt.total_seconds()
assert (
    (test_canon['current_alignment_age_s'] >= -1e-9)
    & (test_canon['current_alignment_age_s'] <= ALIGN_TOLERANCE_S + 1e-9)
).all()

# Frozen SOC0 rule: exact first aligned target row in each current-defined segment.
anchor_rows = (
    test_canon.sort_values(['raw_segment_id', 'timestamp'], kind='stable')
    .groupby('raw_segment_id', sort=False, group_keys=False)
    .head(1)[[
        'raw_segment_id', 'timestamp', 'current_source_timestamp',
        'target', 'ah_cum_segment', 'qabs_cum_segment'
    ]]
    .rename(columns={
        'timestamp': 'anchor_timestamp',
        'current_source_timestamp': 'anchor_current_timestamp',
        'target': 'soc0',
        'ah_cum_segment': 'anchor_ah_cum',
        'qabs_cum_segment': 'anchor_qabs_cum',
    })
)
assert not anchor_rows['raw_segment_id'].duplicated().any()

test_canon = test_canon.merge(anchor_rows, on='raw_segment_id', how='left', validate='many_to_one')
test_canon['elapsed_s'] = (test_canon['timestamp'] - test_canon['anchor_timestamp']).dt.total_seconds()
test_canon['coord'] = (
    test_canon['ah_cum_segment'].to_numpy(float)
    - test_canon['anchor_ah_cum'].to_numpy(float)
)
test_canon['q_abs_ah'] = (
    test_canon['qabs_cum_segment'].to_numpy(float)
    - test_canon['anchor_qabs_cum'].to_numpy(float)
)
assert (test_canon['elapsed_s'] >= -1e-9).all()
assert (test_canon['q_abs_ah'] >= -1e-8).all()

# Frozen causal backward joins for all remaining predictors.
for feature in [
    'voltage_v', 'battery_temp_c', 'speed_kmh',
    'peb_torque_nm', 'hydraulic_brake_torque_req_nm', 'distance_km'
]:
    z = pd.read_parquet(channel_cache[feature])[['timestamp', feature]].copy()
    source_col = f'{feature}__source_timestamp'
    z = z.rename(columns={'timestamp': source_col}).sort_values(source_col)
    test_canon = pd.merge_asof(
        test_canon.sort_values('timestamp'), z,
        left_on='timestamp', right_on=source_col,
        direction='backward',
        tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
    )
    age = (test_canon['timestamp'] - test_canon[source_col]).dt.total_seconds().dropna().to_numpy(float)
    if len(age):
        assert age.min() >= -1e-9
        assert age.max() <= ALIGN_TOLERANCE_S + 1e-9

# Frozen distance-anchor policy: exact first canonical row; never future first-non-null.
first_distance_rows = (
    test_canon.sort_values(['raw_segment_id', 'timestamp'], kind='stable')
    .groupby('raw_segment_id', sort=False, group_keys=False)
    .head(1)[['raw_segment_id', 'distance_km']]
    .rename(columns={'distance_km': 'anchor_distance_km'})
)
test_canon = test_canon.merge(first_distance_rows, on='raw_segment_id', how='left', validate='many_to_one')
anchor_distance_valid = np.isfinite(test_canon['anchor_distance_km'].to_numpy(float))
distance_valid = np.isfinite(test_canon['distance_km'].to_numpy(float))
test_canon['distance_since_anchor_km'] = np.where(
    anchor_distance_valid & distance_valid,
    test_canon['distance_km'].to_numpy(float) - test_canon['anchor_distance_km'].to_numpy(float),
    0.0,
)
test_canon['distance_anchor_available'] = anchor_distance_valid.astype(np.int8)

test_canon['vehicle_id'] = 'EVE_ZOE_Q90'
test_canon['segment_id'] = test_canon['raw_segment_id'].astype(str)
test_canon['compound_segment'] = test_canon['vehicle_id'] + '||' + test_canon['segment_id']
test_canon['time_value'] = test_canon['timestamp'].astype('int64') / 1e9
test_canon['row_uid'] = (
    test_canon['vehicle_id'] + '||' + test_canon['segment_id'] + '||' + test_canon['time_value'].astype(str)
)
test_canon['split'] = 'test'

# Frozen train-only imputation.
for c in FEATURE_COLUMNS:
    test_canon[c] = pd.to_numeric(test_canon[c], errors='coerce').fillna(FEATURE_MEDIANS[c])
assert np.isfinite(test_canon[FEATURE_COLUMNS].to_numpy(float)).all()

# Frozen train-only structured-prior coefficient.
test_canon['physics_delta'] = BETA_PP_PER_AH * test_canon['coord'].to_numpy(float)
test_canon['soc_phy'] = test_canon['soc0'].to_numpy(float) + test_canon['physics_delta'].to_numpy(float)

assert not test_canon['row_uid'].duplicated().any()
assert len(test_canon) > 0

TEST_CANON_PATH = OUT / '12C_EVE_confirmatory_canonical.parquet'
test_canon.to_parquet(TEST_CANON_PATH, index=False)

canonical_audit = {
    'sealed_rows': int(len(sealed_target)),
    'scored_canonical_rows': int(len(test_canon)),
    'row_coverage_fraction': float(len(test_canon) / len(sealed_target)),
    'segments': int(test_canon['compound_segment'].nunique()),
    'max_current_alignment_age_s': float(test_canon['current_alignment_age_s'].max()),
    'beta_pp_per_ah_frozen': BETA_PP_PER_AH,
    'canonical_fingerprint_source_12B': EXPECTED_CANON_FP,
    'real_state_of_charge_predictor': False,
    'soc0_only_target_trajectory_input': True,
}
with open(OUT / '12C_canonicalization_audit.json', 'w') as f:
    json.dump(canonical_audit, f, indent=2)
print(json.dumps(canonical_audit, indent=2))

# %% [markdown] 7. Frozen strict-causal early-context bank for test inference

# %% Code cell 12 (notebook cell 21)
anchor_map = (
    test_canon[[
        'raw_segment_id', 'compound_segment', 'anchor_timestamp',
        'anchor_current_timestamp', 'soc0', 'anchor_ah_cum',
        'anchor_qabs_cum', 'anchor_distance_km', 'distance_anchor_available'
    ]]
    .drop_duplicates('raw_segment_id')
)

ctx_base = test_current.merge(anchor_map, on='raw_segment_id', how='inner', validate='many_to_one')
ctx_base['context_elapsed_s'] = (
    ctx_base['timestamp'] - ctx_base['anchor_current_timestamp']
).dt.total_seconds()
ctx_base = ctx_base[
    (ctx_base['context_elapsed_s'] >= -1e-9)
    & (ctx_base['context_elapsed_s'] <= max(CONTEXT_WINDOWS) + 1e-9)
].copy()

ctx_base['coord'] = ctx_base['ah_cum_segment'].to_numpy(float) - ctx_base['anchor_ah_cum'].to_numpy(float)
ctx_base['q_abs_ah'] = ctx_base['qabs_cum_segment'].to_numpy(float) - ctx_base['anchor_qabs_cum'].to_numpy(float)
ctx_base['physics_delta'] = BETA_PP_PER_AH * ctx_base['coord']
ctx_base['soc_phy'] = ctx_base['soc0'] + ctx_base['physics_delta']

for feature in [
    'voltage_v', 'battery_temp_c', 'speed_kmh',
    'peb_torque_nm', 'hydraulic_brake_torque_req_nm', 'distance_km'
]:
    z = pd.read_parquet(channel_cache[feature])[['timestamp', feature]].sort_values('timestamp')
    ctx_base = pd.merge_asof(
        ctx_base.sort_values('timestamp'), z,
        on='timestamp', direction='backward',
        tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
    )

anchor_dist = ctx_base['anchor_distance_km'].to_numpy(float)
distance_now = ctx_base['distance_km'].to_numpy(float)
ctx_base['distance_since_anchor_km'] = np.where(
    np.isfinite(anchor_dist) & np.isfinite(distance_now)
    & (ctx_base['distance_anchor_available'].to_numpy(int) == 1),
    distance_now - anchor_dist,
    0.0,
)
ctx_base['elapsed_s'] = ctx_base['context_elapsed_s']
ctx_base['time_value'] = ctx_base['timestamp'].astype('int64') / 1e9


def six_stats(vals):
    vals = pd.to_numeric(pd.Series(vals), errors='coerce').to_numpy(float)
    finite = vals[np.isfinite(vals)]
    if not len(finite):
        return {'last': 0.0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'delta': 0.0}
    return {
        'last': float(finite[-1]),
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'delta': float(finite[-1] - finite[0]),
    }


def summarize_context(window_s):
    rows = []
    for seg, g in ctx_base.groupby('compound_segment', sort=False):
        h = g[g['context_elapsed_s'] <= float(window_s) + 1e-9].sort_values('timestamp')
        if not len(h):
            continue
        tv = h['time_value'].to_numpy(float)
        gaps = np.diff(tv)
        gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
        row = {
            'compound_segment': seg,
            'ctx_rows': float(len(h)),
            'ctx_duration_s': float(h['context_elapsed_s'].max() - h['context_elapsed_s'].min()),
            'ctx_median_gap_s': float(np.median(gaps)) if len(gaps) else 0.0,
            'ctx_p90_gap_s': float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
            'ctx_sampling_hz': float((len(h)-1) / max(tv[-1]-tv[0], 1e-9)) if len(tv) >= 2 else 0.0,
            'ctx_soc0': float(h['soc0'].iloc[0]),
            'ctx_phy_last': float(h['soc_phy'].iloc[-1]),
            'ctx_phy_delta_last': float(h['physics_delta'].iloc[-1]),
        }
        for c in RAW_CONTEXT_SIGNALS:
            st = six_stats(h[c])
            for stat, val in st.items():
                row[f'raw_{stat}__{c}'] = val
            vals = pd.to_numeric(h[c], errors='coerce').to_numpy(float)
            row[f'raw_finite_frac__{c}'] = float(np.mean(np.isfinite(vals)))
        rows.append(row)
    out = pd.DataFrame(rows)
    assert len(out)
    assert not out['compound_segment'].duplicated().any()
    return out


RAW_CONTEXT_BANK = {}
context_audit_rows = []
for w in CONTEXT_WINDOWS:
    ctx = summarize_context(w)
    RAW_CONTEXT_BANK[float(w)] = ctx
    p = OUT / f'12C_EVE_raw_context_{int(w)}s.parquet'
    ctx.to_parquet(p, index=False)
    coverage = ctx['compound_segment'].nunique() / test_canon['compound_segment'].nunique()
    context_audit_rows.append({
        'early_window_s': float(w),
        'canonical_segments': int(test_canon['compound_segment'].nunique()),
        'context_segments': int(ctx['compound_segment'].nunique()),
        'coverage': float(coverage),
        'n_context_features': int(len(ctx.columns)-1),
    })
    assert coverage >= 0.95, (w, coverage)

context_audit = pd.DataFrame(context_audit_rows)
context_audit.to_csv(OUT / '12C_raw_context_coverage.csv', index=False)
display(context_audit)

# %% [markdown] 8. Score frozen 10K bundles — no fitting

# %% Code cell 13 (notebook cell 23)
k10_seed_parts = []
router_audits = []

for r in k10_manifest.sort_values(['fraction', 'seed']).itertuples(index=False):
    fraction = float(r.fraction)
    seed = int(r.seed)
    bundle_path = Path(str(r.bundle_path))
    assert sha256_file(bundle_path) == str(r.bundle_sha256)

    bundle = joblib.load(bundle_path)
    assert list(bundle['feature_columns']) == FEATURE_COLUMNS
    assert int(bundle['seed']) == seed
    assert np.isclose(float(bundle['beta_pp_per_ah']), BETA_PP_PER_AH)

    selected = k10_selected[np.isclose(k10_selected['fraction'].astype(float), fraction)].iloc[0]
    assert int(bundle['residual_hp']['min_samples_leaf']) == int(selected['residual_min_samples_leaf'])
    assert np.isclose(float(bundle['residual_hp']['max_features']), float(selected['residual_max_features']))
    assert np.isclose(float(bundle['sf_protocol']['early_window_s']), float(selected['early_window_s']))
    assert np.isclose(float(bundle['sf_protocol']['slow_strength']), float(selected['slow_strength']))
    assert np.isclose(float(bundle['sf_protocol']['tau_pp']), float(selected['tau_pp']))
    assert np.isclose(float(bundle['router_cfg']['kappa']), float(selected['kappa']))
    assert np.isclose(float(bundle['router_cfg']['margin']), float(selected['margin']))
    assert np.isclose(float(bundle['router_cfg']['sf_weight']), float(selected['sf_weight']))

    p = predict_10k_bundle(bundle, test_canon)
    part = pd.DataFrame({
        'row_uid': test_canon['row_uid'].astype(str).to_numpy(),
        'fraction': fraction,
        'seed': seed,
        'Residual': p['Residual'],
        'StrictSF': p['StrictSF_ablation'],
        'SC_RS_SFPEF': p['SC_RS_SFPEF'],
    })
    k10_seed_parts.append(part)
    router_audits.append({
        'fraction': fraction,
        'seed': seed,
        **p['route_audit'],
    })

    del bundle, p, part
    gc.collect()

k10_seed_pred = pd.concat(k10_seed_parts, ignore_index=True)
k10_seed_pred.to_parquet(PRED_OUT / '12C_10K_seed_predictions.parquet', index=False)
router_audit_df = pd.DataFrame(router_audits)
router_audit_df.to_csv(OUT / '12C_10K_router_test_audit.csv', index=False)
assert len(k10_seed_pred) == len(test_canon) * 10
print('10K confirmatory predictions complete:', len(k10_seed_pred))

# %% [markdown] 9. Score all 30 frozen TCN / LSTM / GRU checkpoints — no retraining

# %% Code cell 14 (notebook cell 25)
test_frame = prepare_inference_frame(test_canon)
deep_seed_parts = []

for r in deep_runs.sort_values(['fraction', 'model', 'seed']).itertuples(index=False):
    fraction = float(r.fraction)
    model_name = str(r.model)
    seed = int(r.seed)
    checkpoint = Path(str(r.checkpoint))
    assert sha256_file(checkpoint) == str(r.checkpoint_sha256)

    pp = predict_deep_checkpoint(checkpoint, test_frame, model_name, fraction, seed)
    pp['fraction'] = fraction
    pp['model'] = model_name
    pp['seed'] = seed
    deep_seed_parts.append(pp)
    print('scored', fraction, model_name, seed)


deep_seed_pred = pd.concat(deep_seed_parts, ignore_index=True)
deep_seed_pred.to_parquet(PRED_OUT / '12C_deep_seed_predictions.parquet', index=False)
assert len(deep_seed_pred) == len(test_frame) * 30
print('Deep confirmatory predictions complete:', len(deep_seed_pred))

# %% [markdown] 10. Seed-averaged primary confirmatory predictions

# %% Code cell 15 (notebook cell 27)
primary = test_canon[[
    'row_uid', 'timestamp', 'compound_segment', 'segment_id',
    'target', 'soc0', 'soc_phy', 'elapsed_s'
]].copy()
primary['Prior'] = primary['soc_phy'].to_numpy(float)

for fraction in LABEL_FRACTIONS:
    tag = 'full' if fraction == 1.0 else 'sparse10'
    q = k10_seed_pred[np.isclose(k10_seed_pred['fraction'].astype(float), fraction)]
    avg = q.groupby('row_uid', as_index=False)[['Residual', 'StrictSF', 'SC_RS_SFPEF']].mean()
    avg = avg.rename(columns={
        'Residual': f'Residual_{tag}',
        'StrictSF': f'StrictSF_{tag}',
        'SC_RS_SFPEF': f'10K_{tag}',
    })
    primary = primary.merge(avg, on='row_uid', how='left', validate='one_to_one')

    for model_name in DEEP_MODELS:
        d = deep_seed_pred[
            np.isclose(deep_seed_pred['fraction'].astype(float), fraction)
            & deep_seed_pred['model'].eq(model_name)
        ]
        avgd = d.groupby('row_uid', as_index=False)['prediction'].mean().rename(
            columns={'prediction': f'{model_name}_{tag}'}
        )
        primary = primary.merge(avgd, on='row_uid', how='left', validate='one_to_one')

pred_cols = [c for c in primary.columns if c == 'Prior' or c.endswith('_full') or c.endswith('_sparse10')]
assert np.isfinite(primary[pred_cols].to_numpy(float)).all()

PRIMARY_PRED_PATH = PRED_OUT / '12C_CONFIRMATORY_seed_averaged_predictions.parquet'
primary.to_parquet(PRIMARY_PRED_PATH, index=False)
print('Primary prediction columns:', pred_cols)

# %% [markdown] 11. Final confirmatory metrics and seed robustness

# %% Code cell 16 (notebook cell 29)
def segment_balanced_mae(df, pred_col):
    seg = (
        df.assign(abs_err=np.abs(df[pred_col].to_numpy(float) - df['target'].to_numpy(float)))
        .groupby('compound_segment', as_index=False)['abs_err'].mean()
    )
    return float(seg['abs_err'].mean())


summary_rows = []

# Shared prior baseline.
sc = regression_metrics(primary['target'], primary['Prior'])
summary_rows.append({
    'fraction': 'shared', 'method': 'Prior',
    'n_rows': len(primary), 'n_segments': primary['compound_segment'].nunique(),
    'segment_balanced_MAE': segment_balanced_mae(primary, 'Prior'),
    **sc,
})

for fraction in LABEL_FRACTIONS:
    tag = 'full' if fraction == 1.0 else 'sparse10'
    label = 'full' if fraction == 1.0 else '10%'
    mapping = {
        'Residual': f'Residual_{tag}',
        'StrictSF': f'StrictSF_{tag}',
        '10K_SC_RS_SFPEF': f'10K_{tag}',
        'TCN': f'TCN_{tag}',
        'LSTM': f'LSTM_{tag}',
        'GRU': f'GRU_{tag}',
    }
    for method, col in mapping.items():
        sc = regression_metrics(primary['target'], primary[col])
        summary_rows.append({
            'fraction': label,
            'method': method,
            'n_rows': len(primary),
            'n_segments': primary['compound_segment'].nunique(),
            'segment_balanced_MAE': segment_balanced_mae(primary, col),
            **sc,
        })

final_table = pd.DataFrame(summary_rows)
final_table.to_csv(OUT / 'FINAL_EVE_CONFIRMATORY_TABLE.csv', index=False)

# Seed-level metrics for transparency.
seed_rows = []
y_map = primary.set_index('row_uid')['target']
seg_map = primary.set_index('row_uid')['compound_segment']

for (fraction, seed), q in k10_seed_pred.groupby(['fraction', 'seed']):
    q = q.copy()
    y = y_map.loc[q['row_uid']].to_numpy(float)
    for method in ['Residual', 'StrictSF', 'SC_RS_SFPEF']:
        sc = regression_metrics(y, q[method])
        seed_rows.append({'fraction': float(fraction), 'model': method, 'seed': int(seed), **sc})

for (fraction, model, seed), q in deep_seed_pred.groupby(['fraction', 'model', 'seed']):
    y = y_map.loc[q['row_uid']].to_numpy(float)
    sc = regression_metrics(y, q['prediction'])
    seed_rows.append({'fraction': float(fraction), 'model': str(model), 'seed': int(seed), **sc})

seed_metrics = pd.DataFrame(seed_rows)
seed_metrics.to_csv(OUT / '12C_seed_level_metrics.csv', index=False)
seed_summary = (
    seed_metrics.groupby(['fraction', 'model'], as_index=False)
    .agg(MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
         RMSE_mean=('RMSE', 'mean'), P95_mean=('P95', 'mean'), Bias_mean=('Bias', 'mean'))
)
seed_summary.to_csv(OUT / '12C_seed_metric_summary.csv', index=False)

display(final_table)
display(seed_summary)

# %% [markdown] 12. Segment-balanced paired bootstrap versus 10K

# %% Code cell 17 (notebook cell 31)
def paired_segment_bootstrap(df, pred_10k_col, pred_comp_col, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    work = df[['compound_segment', 'target', pred_10k_col, pred_comp_col]].copy()
    work['ae_10k'] = np.abs(work[pred_10k_col].to_numpy(float) - work['target'].to_numpy(float))
    work['ae_comp'] = np.abs(work[pred_comp_col].to_numpy(float) - work['target'].to_numpy(float))
    seg = work.groupby('compound_segment', as_index=False)[['ae_10k', 'ae_comp']].mean()
    d = (seg['ae_comp'] - seg['ae_10k']).to_numpy(float)
    n = len(d)
    assert n > 1

    rng = np.random.default_rng(seed)
    # Chunked vectorized bootstrap to keep memory bounded.
    boot = []
    remaining = int(B)
    while remaining > 0:
        b = min(1000, remaining)
        idx = rng.integers(0, n, size=(b, n))
        boot.append(d[idx].mean(axis=1))
        remaining -= b
    boot = np.concatenate(boot)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    p10k = float(np.mean(boot > 0))
    p2 = float(min(1.0, 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    if lo > 0:
        interpretation = '10K_lower_segment_balanced_MAE'
    elif hi < 0:
        interpretation = 'comparator_lower_segment_balanced_MAE'
    else:
        interpretation = 'CI_crosses_zero'

    row_delta = (
        np.mean(np.abs(df[pred_comp_col].to_numpy(float) - df['target'].to_numpy(float)))
        - np.mean(np.abs(df[pred_10k_col].to_numpy(float) - df['target'].to_numpy(float)))
    )
    return {
        'n_segments': int(n),
        'delta_segment_MAE_comp_minus_10K': float(d.mean()),
        'CI95_low': float(lo),
        'CI95_high': float(hi),
        'P_bootstrap_10K_lower': p10k,
        'two_sided_bootstrap_p': p2,
        'row_MAE_delta_comp_minus_10K': float(row_delta),
        'interpretation': interpretation,
    }


boot_rows = []
for fraction in LABEL_FRACTIONS:
    tag = 'full' if fraction == 1.0 else 'sparse10'
    label = 'full' if fraction == 1.0 else '10%'
    kcol = f'10K_{tag}'
    comparators = {
        'Residual': f'Residual_{tag}',
        'StrictSF': f'StrictSF_{tag}',
        'TCN': f'TCN_{tag}',
        'LSTM': f'LSTM_{tag}',
        'GRU': f'GRU_{tag}',
    }
    for i, (name, col) in enumerate(comparators.items()):
        res = paired_segment_bootstrap(primary, kcol, col, seed=BOOTSTRAP_SEED + i + (0 if fraction == 1.0 else 100))
        boot_rows.append({'fraction': label, 'comparator': name, **res})

bootstrap_df = pd.DataFrame(boot_rows)
bootstrap_df.to_csv(OUT / '12C_paired_segment_bootstrap_vs_10K.csv', index=False)
display(bootstrap_df)

# %% [markdown] 13. Full → sparse supervision degradation and seed-paired checks

# %% Code cell 18 (notebook cell 33)
degradation_rows = []
for method in ['Residual', 'StrictSF', '10K', 'TCN', 'LSTM', 'GRU']:
    full_col = f'{method}_full'
    sparse_col = f'{method}_sparse10'
    full_mae = regression_metrics(primary['target'], primary[full_col])['MAE']
    sparse_mae = regression_metrics(primary['target'], primary[sparse_col])['MAE']
    degradation_rows.append({
        'method': method,
        'full_MAE': full_mae,
        'sparse10_MAE': sparse_mae,
        'absolute_MAE_increase': sparse_mae - full_mae,
        'relative_degradation_pct': 100.0 * (sparse_mae - full_mae) / max(full_mae, 1e-12),
    })
degradation_df = pd.DataFrame(degradation_rows)
degradation_df.to_csv(OUT / '12C_full_to_sparse_degradation.csv', index=False)

# Same-seed MAE comparison: 10K main vs each deep baseline.
paired_seed_rows = []
for fraction in LABEL_FRACTIONS:
    for seed in FINAL_SEEDS:
        k = seed_metrics[
            np.isclose(seed_metrics['fraction'].astype(float), fraction)
            & seed_metrics['model'].eq('SC_RS_SFPEF')
            & seed_metrics['seed'].eq(seed)
        ].iloc[0]
        for model in DEEP_MODELS:
            d = seed_metrics[
                np.isclose(seed_metrics['fraction'].astype(float), fraction)
                & seed_metrics['model'].eq(model)
                & seed_metrics['seed'].eq(seed)
            ].iloc[0]
            paired_seed_rows.append({
                'fraction': float(fraction),
                'seed': int(seed),
                'deep_model': model,
                'MAE_10K': float(k['MAE']),
                'MAE_deep': float(d['MAE']),
                'delta_deep_minus_10K': float(d['MAE'] - k['MAE']),
                '10K_lower_MAE': bool(k['MAE'] < d['MAE']),
            })
paired_seed_df = pd.DataFrame(paired_seed_rows)
paired_seed_df.to_csv(OUT / '12C_seed_paired_10K_vs_deep.csv', index=False)

display(degradation_df)
display(paired_seed_df.groupby(['fraction', 'deep_model'])['10K_lower_MAE'].agg(['sum', 'count']).reset_index())

# %% [markdown] 14. Confirmatory figures

# %% Code cell 19 (notebook cell 35)
figure_audit = {'status': 'PASS', 'error': None}
try:
    # Figure 1: primary MAE full vs sparse.
    models = ['10K_SC_RS_SFPEF', 'TCN', 'LSTM', 'GRU']
    full_mae = [float(final_table[(final_table['fraction']=='full') & (final_table['method']==m)]['MAE'].iloc[0]) for m in models]
    sparse_mae = [float(final_table[(final_table['fraction']=='10%') & (final_table['method']==m)]['MAE'].iloc[0]) for m in models]

    x = np.arange(len(models))
    w = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - w/2, full_mae, width=w, label='Full')
    plt.bar(x + w/2, sparse_mae, width=w, label='10% sparse')
    plt.xticks(x, ['10K', 'TCN', 'LSTM', 'GRU'])
    plt.ylabel('Confirmatory MAE (SoC percentage points)')
    plt.title('EVE untouched confirmatory performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_OUT / '12C_confirmatory_MAE_full_vs_sparse.png', dpi=180)
    plt.show()

    # Figure 2: supervision degradation.
    plt.figure(figsize=(8, 5))
    plt.bar(degradation_df['method'], degradation_df['relative_degradation_pct'])
    plt.ylabel('Full → 10% MAE degradation (%)')
    plt.title('Sensitivity to 90% label removal')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(FIG_OUT / '12C_full_to_sparse_degradation.png', dpi=180)
    plt.show()

    # Figure 3: sparse segment-balanced bootstrap vs 10K.
    sp = bootstrap_df[bootstrap_df['fraction'] == '10%'].copy()
    y = np.arange(len(sp))
    center = sp['delta_segment_MAE_comp_minus_10K'].to_numpy(float)
    lo = sp['CI95_low'].to_numpy(float)
    hi = sp['CI95_high'].to_numpy(float)
    plt.figure(figsize=(8, 5))
    plt.errorbar(center, y, xerr=np.vstack([center-lo, hi-center]), fmt='o', capsize=4)
    plt.axvline(0.0, linewidth=1)
    plt.yticks(y, sp['comparator'])
    plt.xlabel('Segment-balanced ΔMAE = comparator − 10K')
    plt.title('Sparse confirmatory paired bootstrap')
    plt.tight_layout()
    plt.savefig(FIG_OUT / '12C_sparse_paired_bootstrap_vs_10K.png', dpi=180)
    plt.show()
except Exception as exc:
    figure_audit = {'status': 'FAILED_OPTIONAL', 'error': f'{type(exc).__name__}: {exc}'}
    print('Optional figure generation failed; scoring artifacts remain valid:', figure_audit['error'])

with open(OUT / '12C_figure_audit.json', 'w') as f:
    json.dump(figure_audit, f, indent=2)

# %% [markdown] 15. Final one-shot manifest

# %% Code cell 20 (notebook cell 37)
# Compact result snapshot for audit.
def get_mae(frac, method):
    return float(final_table[(final_table['fraction'] == frac) & (final_table['method'] == method)]['MAE'].iloc[0])

result_snapshot = {
    'full': {
        '10K': get_mae('full', '10K_SC_RS_SFPEF'),
        'TCN': get_mae('full', 'TCN'),
        'LSTM': get_mae('full', 'LSTM'),
        'GRU': get_mae('full', 'GRU'),
    },
    '10%': {
        '10K': get_mae('10%', '10K_SC_RS_SFPEF'),
        'TCN': get_mae('10%', 'TCN'),
        'LSTM': get_mae('10%', 'LSTM'),
        'GRU': get_mae('10%', 'GRU'),
    },
}

final_manifest = {
    'experiment': '12C_EVE_ONE_SHOT_UNTOUCHED_CONFIRMATORY_SCORING',
    'dataset': freeze12b['dataset'],
    'target': freeze12b['target'],
    'status': 'COMPLETE_ONE_SHOT_CONFIRMATORY_SCORED_NO_RETUNING',
    'confirmatory_target_opened': True,
    'confirmatory_target_parse_count_in_notebook': 1,
    'unseal_utc_time': unseal_time,
    'sealed_target_sha256_expected': EXPECTED_TEST_SHA256,
    'sealed_target_sha256_actual': ACTUAL_TEST_SHA256,
    'sealed_rows': int(len(sealed_target)),
    'scored_rows': int(len(test_canon)),
    'scored_row_coverage_fraction': float(len(test_canon) / len(sealed_target)),
    'scored_segments': int(test_canon['compound_segment'].nunique()),
    'canonical_fingerprint_sha256': EXPECTED_CANON_FP,
    'prior_beta_pp_per_ah_frozen': BETA_PP_PER_AH,
    'frozen_10K_bundles_scored': 10,
    'frozen_deep_checkpoints_scored': 30,
    'model_retraining_in_12C': False,
    'hyperparameter_tuning_in_12C': False,
    'feature_selection_in_12C': False,
    'threshold_selection_in_12C': False,
    'post_unseal_changes_allowed': False,
    'bootstrap_segments_as_resampling_unit': True,
    'bootstrap_replicates': BOOTSTRAP_B,
    'primary_prediction_rule': 'five-seed prediction average',
    'result_snapshot_MAE': result_snapshot,
    'outputs': {
        'final_table': str(OUT / 'FINAL_EVE_CONFIRMATORY_TABLE.csv'),
        'seed_metrics': str(OUT / '12C_seed_level_metrics.csv'),
        'paired_bootstrap': str(OUT / '12C_paired_segment_bootstrap_vs_10K.csv'),
        'degradation': str(OUT / '12C_full_to_sparse_degradation.csv'),
        'primary_predictions': str(PRIMARY_PRED_PATH),
        'router_audit': str(OUT / '12C_10K_router_test_audit.csv'),
        'preunseal_audit': str(PREUNSEAL_AUDIT),
        'unseal_marker': str(UNSEAL_MARKER),
    },
    'interpretation_rule': (
        'This confirmatory set may now be reported but must not be used to alter models, '
        'features, hyperparameters, or thresholds.'
    ),
}

with open(FINAL_MANIFEST, 'w') as f:
    json.dump(final_manifest, f, indent=2)

print(json.dumps(final_manifest, indent=2))
print('\n12C COMPLETE. DO NOT RERUN OR RETUNE ON EVE CONFIRMATORY RESULTS.')

# %% [markdown] END — EVE confirmatory evidence is now consumed
