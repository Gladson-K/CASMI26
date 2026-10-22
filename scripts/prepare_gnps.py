#!/usr/bin/env python3
"""Convert annotated GNPS MGF spectra into chunked Parquet for CPU retrieval.

By default processes up to 250k records to keep the starter pipeline practical.
Use --max-spectra 0 to process the entire library (large disk/time requirement).
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from matchms.importing import load_from_mgf

ROOT = Path(__file__).resolve().parents[1]

def first(meta, *keys):
    for k in keys:
        v = meta.get(k)
        if v not in (None, '', 'N/A', 'n/a'):
            return v
    return None

def norm_adduct(meta):
    a = first(meta, 'adduct', 'precursor_type', 'ion_type')
    if a:
        return str(a).strip()
    mode = str(first(meta, 'ionmode', 'ion_mode', 'polarity') or '').lower()
    charge = first(meta, 'charge')
    try: z = abs(int(str(charge).replace('+','').replace('-',''))) if charge else 1
    except ValueError: z = 1
    if 'negative' in mode or mode in ('neg', '-'):
        return '[M-2H]2-' if z == 2 else '[M-H]-'
    if 'positive' in mode or mode in ('pos', '+'):
        return '[M+2H]2+' if z == 2 else '[M+H]+'
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/gnps/GNPS-LIBRARY.mgf')
    p.add_argument('--output', default='data/gnps/processed')
    p.add_argument('--max-spectra', type=int, default=250000)
    p.add_argument('--chunk-size', type=int, default=10000)
    args = p.parse_args()
    source = ROOT / args.input; out = ROOT / args.output
    if not source.exists(): raise SystemExit(f'Missing {source}; run `python scripts/bootstrap_assets.py --gnps` first.')
    out.mkdir(parents=True, exist_ok=True)
    count = accepted = 0; chunk = []
    # Clean old chunks so reruns cannot mix different caps.
    for old in out.glob('part-*.parquet'): old.unlink()
    for spectrum in load_from_mgf(str(source)):
        count += 1
        meta = spectrum.metadata or {}
        smi = first(meta, 'smiles', 'SMILES', 'canonical_smiles', 'normalized_smiles')
        precursor = first(meta, 'precursor_mz', 'pepmass', 'parent_mass')
        adduct = norm_adduct(meta)
        if not smi or not precursor or not adduct:
            if args.max_spectra and count >= args.max_spectra: break
            continue
        try:
            precursor = float(precursor[0] if isinstance(precursor, (tuple, list)) else precursor)
            mz = np.asarray(spectrum.peaks.mz, dtype=np.float32)
            inten = np.asarray(spectrum.peaks.intensities, dtype=np.float32)
            if len(mz) < 3 or len(inten) != len(mz) or not np.isfinite(precursor): raise ValueError('invalid peaks')
            inten = inten / max(float(inten.max()), 1e-12)
        except Exception:
            if args.max_spectra and count >= args.max_spectra: break
            continue
        chunk.append({'precursor_mz': precursor, 'adduct': adduct, 'normalized_smiles': str(smi),
                      'inchikey14': str(first(meta, 'inchikey14', 'inchikey', 'InChIKey') or ''),
                      'ms2_mzs': mz.tolist(), 'ms2_normalized_intensities': inten.tolist(),
                      'source': 'GNPS'})
        accepted += 1
        if len(chunk) >= args.chunk_size:
            pd.DataFrame(chunk).to_parquet(out / f'part-{accepted:09d}.parquet', index=False)
            print(f'Parsed {count:,} spectra; retained {accepted:,} annotated spectra', flush=True); chunk = []
        if args.max_spectra and count >= args.max_spectra: break
    if chunk: pd.DataFrame(chunk).to_parquet(out / f'part-{accepted:09d}.parquet', index=False)
    report = {'input': str(source.relative_to(ROOT)), 'records_scanned': count, 'annotated_spectra_retained': accepted,
              'chunk_size': args.chunk_size, 'max_spectra': args.max_spectra, 'output': str(out.relative_to(ROOT)),
              'note': 'Metadata fields vary across GNPS library releases; inspect retained row counts and licenses.'}
    (out / 'prepare_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
