"""CPU-only Optuna search for spectral-library retrieval parameters.

This tunes the direct spectral-library channel. The validation protocol is a
spectrum-level split; report it as a library-hit proxy, not unseen-structure LB.
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np
import optuna
import pandas as pd

ADDUCTS = {
    '[M+H]+': (1, 1, 1.007276466621), '[M+NH4]+': (1, 1, 18.033823),
    '[M+Na]+': (1, 1, 22.989218), '[M+K]+': (1, 1, 38.963158),
    '[M-H2O+H]+': (1, 1, -17.003288), '[M-2H2O+H]+': (1, 1, -35.013852),
    '[M+2H]2+': (1, 2, 2.014552933242), '[M]+': (1, 1, 0.0),
    '[M-H2O]+': (1, 1, -18.010565), '[M+CH3OH+H]+': (1, 1, 33.03349),
    '[M+CH3CN+H]+': (1, 1, 42.03382), '[M-H]-': (1, 1, -1.007276466621),
    '[M-H2O-H]-': (1, 1, -19.01784), '[M+CH2O2-H]-': (1, 1, 44.9982),
    '[M+C2H4O2-H]-': (1, 1, 59.0139), '[M+Cl]-': (1, 1, 34.9694),
    '[M]-': (1, 1, 0.0), '[M-2H]2-': (1, 2, -2.014552933242),
    '[M+Na-2H]-': (1, 1, 20.9747), '[2M+H]+': (2, 1, 1.007276466621),
    '[2M+Na]+': (2, 1, 22.989218), '[2M+NH4]+': (2, 1, 18.033823),
    '[2M+K]+': (2, 1, 38.963158), '[2M-H]-': (2, 1, -1.007276466621),
    '[2M+CH2O2-H]-': (2, 1, 44.9982), '[2M+C2H4O2-H]-': (2, 1, 59.0139),
    '[3M+H]+': (3, 1, 1.007276466621), '[3M-H]-': (3, 1, -1.007276466621),
}


def neutral_mass(mz, adduct):
    try:
        n, z, delta = ADDUCTS.get(str(adduct), (1, 1, np.nan))
        return (float(mz) * z - delta) / n
    except (TypeError, ValueError):
        return np.nan


def get_peaks(row, max_peaks=256, floor=0.002, power=1.0):
    mz = row.get('ms2_mzs', None)
    inten = row.get('ms2_normalized_intensities', None)
    if mz is None or inten is None:
        # Some releases encode a list of dictionaries instead.
        specs = row.get('spectra', None)
        if isinstance(specs, list) and specs:
            mz, inten = specs[0].get('mz'), specs[0].get('intensity')
    try:
        mz = np.asarray(mz, dtype=np.float32).reshape(-1)
        inten = np.asarray(inten, dtype=np.float32).reshape(-1)
    except Exception:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    n = min(mz.size, inten.size)
    mz, inten = mz[:n], inten[:n]
    ok = np.isfinite(mz) & np.isfinite(inten) & (mz > 0) & (inten > 0) & (mz <= 2000)
    mz, inten = mz[ok], inten[ok]
    if not len(mz): return mz, inten
    inten = inten / max(float(inten.max()), 1e-12)
    keep = inten >= floor
    mz, inten = mz[keep], inten[keep]
    if len(mz) > max_peaks:
        ids = np.argsort(inten)[-max_peaks:]
        mz, inten = mz[ids], inten[ids]
    inten = np.power(inten, power)
    norm = float(np.linalg.norm(inten))
    if norm > 0: inten /= norm
    order = np.argsort(mz)
    return mz[order], inten[order]


def entropy_cosine(a, b, tol):
    amz, ai = a; bmz, bi = b
    if len(amz) == 0 or len(bmz) == 0: return 0.0
    # Two-pointer greedy matching; intensity vectors are L2-normalized.
    i = j = 0; score = 0.0
    while i < len(amz) and j < len(bmz):
        delta = float(amz[i] - bmz[j])
        if abs(delta) <= tol:
            score += float(ai[i] * bi[j]); i += 1; j += 1
        elif delta < 0: i += 1
        else: j += 1
    return max(0.0, min(1.0, score))


def load_rows(path, max_rows=0):
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    needed = {'precursor_mz', 'adduct'}
    missing = needed - set(df.columns)
    if missing: raise ValueError(f'Missing required columns {sorted(missing)}. Available: {list(df.columns)}')
    label_col = next((c for c in ('inchikey14', 'normalized_smiles', 'smiles') if c in df.columns), None)
    if label_col is None: raise ValueError('No structure label column found (expected inchikey14/normalized_smiles/smiles).')
    df = df.copy()
    df['_label'] = df[label_col].fillna('').astype(str)
    df['_neutral_mass'] = [neutral_mass(m, a) for m, a in zip(df.precursor_mz, df.adduct)]
    df['_peaks'] = [get_peaks(r) for r in df.to_dict('records')]
    df = df[(df._label != '') & np.isfinite(df._neutral_mass) & df._peaks.map(lambda p: len(p[0]) >= 3)].reset_index(drop=True)
    return df


def mrr_at_25(ranked_labels, truth):
    try: rank = ranked_labels.index(truth) + 1
    except ValueError: return 0.0
    return 1.0 / rank if rank <= 25 else 0.0


def _query_scores(query, library, params, top_k=25):
    ppm = params['precursor_tolerance_ppm']
    qmass = float(query['_neutral_mass'])
    tol_da = max(qmass * ppm / 1e6, 0.002)
    mass = library['mass']
    ix = np.flatnonzero(np.abs(mass - qmass) <= tol_da)
    if not len(ix):
        ix = np.flatnonzero(np.abs(mass - qmass) <= max(tol_da * 3, 0.02))
    if len(ix) > params['max_mass_candidates']:
        ix = ix[np.argsort(np.abs(mass[ix] - qmass))[:params['max_mass_candidates']]]
    scores = {}
    labels, peaks = library['labels'], library['peaks']
    for j in ix:
        key = str(labels[j])
        if not key or key == 'nan': continue
        score = entropy_cosine(query['_peaks'], peaks[j], params['mz_tolerance_da'])
        scores.setdefault(key, []).append(score)
    if params.get('aggregation', 'max') == 'top3_mean':
        scores = {k: float(np.mean(sorted(v, reverse=True)[:3])) for k, v in scores.items()}
    else:
        scores = {k: float(max(v)) for k, v in scores.items()}
    return [k for k, v in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]


def run_retrieval_nas(train_df, config, timeout_seconds=36000, n_trials=100000):
    seed = int(config['project'].get('seed', 42))
    n = len(train_df)
    if n < 10: raise ValueError(f'Need at least 10 usable training spectra, found {n}.')
    rng = np.random.default_rng(seed)
    # Fixed holdout for fair comparisons. Spectrum-level split evaluates direct-library retrieval.
    perm = rng.permutation(n)
    n_val = min(max(1, int(n * float(config['nas'].get('validation_fraction', .2)))),
                int(config['nas'].get('max_validation_molecules', 1200)))
    val_ids = perm[:n_val]
    lib_ids = perm[n_val:]
    max_lib = int(config['nas'].get('max_library_spectra', 250000))
    if len(lib_ids) > max_lib:
        lib_ids = rng.choice(lib_ids, max_lib, replace=False)
    val = train_df.iloc[val_ids].reset_index(drop=True)
    library = train_df.iloc[lib_ids].reset_index(drop=True)
    # Rebuild query/library peak arrays under each trial's preprocessing settings.
    raw_val = val.drop(columns=['_peaks'], errors='ignore').to_dict('records')
    raw_lib = library.drop(columns=['_peaks'], errors='ignore').to_dict('records')
    db = str(config['nas'].get('storage', 'sqlite:///outputs/nas_cpu_10h.sqlite3'))
    if db.startswith('sqlite:///'):
        db_path = Path(db.replace('sqlite:///', '', 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=int(config['nas'].get('n_startup_trials', 40)), multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1)
    study = optuna.create_study(study_name='casmi26_cpu_retrieval_nas', direction='maximize', sampler=sampler,
                                pruner=pruner, storage=db, load_if_exists=True)

    def objective(trial):
        params = {
            'precursor_tolerance_ppm': trial.suggest_float('precursor_tolerance_ppm', 4.0, 35.0, log=True),
            'mz_tolerance_da': trial.suggest_float('mz_tolerance_da', 0.003, 0.05, log=True),
            'intensity_floor': trial.suggest_float('intensity_floor', 0.0005, 0.01, log=True),
            'intensity_power': trial.suggest_float('intensity_power', 0.5, 1.5),
            'max_peaks': trial.suggest_categorical('max_peaks', [64, 96, 128, 192, 256]),
            'aggregation': trial.suggest_categorical('aggregation', ['max', 'top3_mean']),
            'max_mass_candidates': trial.suggest_categorical('max_mass_candidates', [250, 500, 1000, 2000]),
        }
        # Apply trial preprocessing. Each candidate row is one observed reference spectrum.
        val_trial = val.copy()
        lib_trial = library.copy()
        val_trial['_peaks'] = [get_peaks(r, params['max_peaks'], params['intensity_floor'], params['intensity_power']) for r in raw_val]
        lib_trial['_peaks'] = [get_peaks(r, params['max_peaks'], params['intensity_floor'], params['intensity_power']) for r in raw_lib]
        lib_arrays = {'mass': lib_trial['_neutral_mass'].to_numpy(dtype=np.float64),
                      'labels': lib_trial['_label'].to_numpy(), 'peaks': lib_trial['_peaks'].tolist()}
        scores = []
        for qi, (_, q) in enumerate(val_trial.iterrows()):
            # Exclude exact row by split construction; other spectra of same compound may exist.
            ranked = _query_scores(q, lib_arrays, params)
            scores.append(mrr_at_25(ranked, str(q['_label'])))
            if qi in (max(1, len(val_trial)//3), max(2, 2*len(val_trial)//3)):
                trial.report(float(np.mean(scores)), step=qi)
                if trial.should_prune(): raise optuna.TrialPruned()
        value = float(np.mean(scores)) if scores else 0.0
        trial.set_user_attr('validation_spectra', len(scores))
        trial.set_user_attr('library_spectra', len(lib_trial))
        trial.set_user_attr('protocol', 'spectrum-level direct-library proxy; not unseen-structure validation')
        return value

    start = time.time()
    study.optimize(objective, timeout=int(timeout_seconds), n_trials=int(n_trials), n_jobs=1,
                   gc_after_trial=True, show_progress_bar=True)
    out = Path(config['paths'].get('output_dir', 'outputs')); out.mkdir(parents=True, exist_ok=True)
    completed = sorted((t for t in study.trials if t.value is not None), key=lambda t: t.value, reverse=True)
    result = {
        'profile': config['nas'].get('profile', 'custom'), 'requested_timeout_seconds': int(timeout_seconds),
        'elapsed_seconds': round(time.time() - start, 2), 'completed_trials': len(completed),
        'best_value_mrr25': study.best_value if completed else None,
        'best_params': study.best_params if completed else {},
        'validation_protocol': 'Spectrum-level split for direct spectral-library matching. This can include other spectra from the same compound in the library; it is an optimistic direct-hit proxy and must not be presented as unseen-compound competition accuracy.',
        'library_spectra': len(library), 'validation_spectra': len(val),
    }
    (out / 'nas_best.json').write_text(json.dumps(result, indent=2))
    top = [{'rank': i+1, 'mrr25': t.value, 'params': t.params} for i, t in enumerate(completed[:int(config['nas'].get('top_k', 10))])]
    (out / 'nas_top_configs.json').write_text(json.dumps(top, indent=2))
    study.trials_dataframe().to_csv(out / 'nas_history.csv', index=False)
    print(json.dumps(result, indent=2))
    return study
