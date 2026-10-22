#!/usr/bin/env python3
"""Generate a conservative direct-library + precursor-mass baseline submission.

Uses observed training spectra only; it never fabricates MS/MS peaks from molecular mass.
The mass-only fallback is a candidate-generation safety net, not a learned prediction.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.nas.runner import neutral_mass, get_peaks, entropy_cosine

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / 'data' / 'competition'
OUT = ROOT / 'outputs'


def load_structures(train):
    cache = ROOT / 'cache' / 'candidate_structures.pkl'
    if cache.exists():
        with cache.open('rb') as f: return pickle.load(f)
    pool = {}
    for col in ('normalized_smiles', 'smiles'):
        if col in train:
            for smi in train[col].dropna().astype(str).unique():
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    can = Chem.MolToSmiles(mol, isomericSmiles=True)
                    pool[can] = float(Descriptors.ExactMolWt(mol))
    coconut = ROOT / 'data' / 'coconut'
    for sdf in list(coconut.rglob('*.sdf')) + list(coconut.rglob('*.sdf.gz')):
        try:
            supplier = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True)
            for mol in supplier:
                if mol is None: continue
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                if smi and smi not in pool: pool[smi] = float(Descriptors.ExactMolWt(mol))
        except Exception as exc:
            print(f'[warn] could not read {sdf}: {exc}', file=sys.stderr)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open('wb') as f: pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pool


def main():
    train_path, test_path = COMP / 'train.parquet', COMP / 'test.parquet'
    if not train_path.exists() or not test_path.exists():
        raise SystemExit('Competition data missing. Run `bash setup.sh --core` after configuring Kaggle credentials.')
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    label_col = next((c for c in ('normalized_smiles', 'smiles') if c in train.columns), None)
    if not label_col: raise SystemExit(f'No SMILES column in train. Available: {list(train.columns)}')
    needed = {'precursor_mz', 'adduct', 'ms2_mzs', 'ms2_normalized_intensities'}
    if not needed.issubset(test.columns):
        raise SystemExit(f'Test schema missing {sorted(needed - set(test.columns))}; run scripts/inspect_data.py')

    print(f'Loading {len(train):,} train rows and {len(test):,} test spectra...')
    lib = []
    for row in train.to_dict('records'):
        smi = row.get(label_col)
        if not isinstance(smi, str) or not smi: continue
        nm = neutral_mass(row.get('precursor_mz'), row.get('adduct'))
        if not np.isfinite(nm): continue
        lib.append({'mass': nm, 'smiles': smi, 'peaks': get_peaks(row), 'source': 'competition_train'})
    gnps_count = 0
    gnps_dir = ROOT / 'data' / 'gnps' / 'processed'
    if gnps_dir.exists():
        for part in sorted(gnps_dir.glob('part-*.parquet')):
            try: gdf = pd.read_parquet(part)
            except Exception as exc:
                print(f'[warn] failed to read GNPS chunk {part.name}: {exc}', file=sys.stderr); continue
            for row in gdf.to_dict('records'):
                smi = row.get('normalized_smiles') or row.get('smiles')
                if not isinstance(smi, str) or not smi: continue
                nm = neutral_mass(row.get('precursor_mz'), row.get('adduct'))
                if not np.isfinite(nm): continue
                lib.append({'mass': nm, 'smiles': smi, 'peaks': get_peaks(row), 'source': 'GNPS'})
                gnps_count += 1
    masses = np.asarray([x['mass'] for x in lib], dtype=np.float64)
    order = np.argsort(masses)
    masses = masses[order]; lib = [lib[i] for i in order]
    print(f'Observed spectral library: {len(lib):,} spectra ({gnps_count:,} from processed GNPS)')
    pool = load_structures(train)
    cand_smiles = np.asarray(list(pool.keys()), dtype=object)
    cand_masses = np.asarray(list(pool.values()), dtype=np.float64)
    print(f'Structure candidate pool: {len(cand_smiles):,} structures')

    # The challenge uses one row per molecule in the sample file; combine spectra if molecule_id exists.
    id_col = 'molecule_id' if 'molecule_id' in test.columns else None
    if id_col:
        groups = list(test.groupby(id_col, sort=False))
    else:
        groups = [(i, test.iloc[[i]]) for i in range(len(test))]
    rows = []
    for gi, (mid, sub) in enumerate(groups):
        observed_scores = {}
        target_masses = [neutral_mass(r.get('precursor_mz'), r.get('adduct')) for r in sub.to_dict('records')]
        target_masses = [m for m in target_masses if np.isfinite(m) and m > 0]
        for r in sub.to_dict('records'):
            nm = neutral_mass(r.get('precursor_mz'), r.get('adduct'))
            if not np.isfinite(nm): continue
            mz, inten = get_peaks(r)
            if len(mz) < 3: continue
            tol = max(nm * 15e-6, 0.01)
            lo, hi = np.searchsorted(masses, [nm - tol, nm + tol])
            if hi == lo:
                tol = max(nm * 40e-6, 0.03)
                lo, hi = np.searchsorted(masses, [nm - tol, nm + tol])
            q = (mz, inten)
            for entry in lib[lo:hi]:
                sim = entropy_cosine(q, entry['peaks'], 0.01)
                smi = entry['smiles']
                observed_scores[smi] = max(observed_scores.get(smi, 0.0), sim)
        ranked = [s for s, _ in sorted(observed_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        # Candidate fallback: structures in neutral exact-mass windows, nearest mass first.
        if target_masses:
            target = float(np.median(target_masses))
            delta_ppm = np.abs(cand_masses - target) / max(target, 1.0) * 1e6
            mass_ix = np.flatnonzero(delta_ppm <= 50.0)
            if len(mass_ix) < 25:
                mass_ix = np.argsort(delta_ppm)[:max(100, 25)]
            mass_ix = mass_ix[np.argsort(delta_ppm[mass_ix])]
            for j in mass_ix:
                smi = str(cand_smiles[j])
                if smi not in observed_scores and smi not in ranked:
                    ranked.append(smi)
                if len(ranked) >= 25: break
        # Pad only with valid known structures if the mass window was sparse.
        if len(ranked) < 25:
            for smi in cand_smiles:
                smi = str(smi)
                if smi not in ranked: ranked.append(smi)
                if len(ranked) >= 25: break
        rows.append({'molecule_id': mid, 'smiles': ';'.join(ranked[:25])})
        if gi % 100 == 0 or gi == len(groups) - 1:
            print(f'Processed {gi+1:,}/{len(groups):,} molecules', flush=True)

    subm = pd.DataFrame(rows)
    sample_path = COMP / 'sample_submission.csv'
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        idname = sample.columns[0]
        if idname != 'molecule_id' and 'molecule_id' in subm.columns:
            subm = subm.rename(columns={'molecule_id': idname})
        subm = sample[[idname]].merge(subm, left_on=idname, right_on=idname, how='left')
        subm['smiles'] = subm['smiles'].fillna('')
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'submission.csv'
    subm.to_csv(target, index=False)
    report = {'train_spectra': len(train), 'test_spectra': len(test), 'test_molecules': len(subm),
              'observed_library_spectra': len(lib), 'processed_gnps_spectra': gnps_count, 'candidate_structures': len(cand_smiles),
              'method': 'observed spectral cosine + precursor-mass candidate fallback',
              'warning': 'Fallback candidates are mass-ranked, not confidence-calibrated. Validate MRR@25; this baseline is not expected to match the multi-channel notebook solution.'}
    (OUT / 'baseline_report.json').write_text(json.dumps(report, indent=2))
    print(f'Wrote {target}\n{json.dumps(report, indent=2)}')

if __name__ == '__main__': main()
