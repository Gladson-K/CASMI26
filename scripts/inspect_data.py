#!/usr/bin/env python3
"""Inspect competition/benchmark files without assuming a fixed schema."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for dirname in ('competition', 'benchmark', 'coconut', 'gnps', 'massbank'):
    base = ROOT / 'data' / dirname
    print(f'\n## {base}')
    files = sorted(list(base.rglob('*.parquet')) + list(base.rglob('*.csv')) + list(base.rglob('*.mgf')) + list(base.rglob('*.sdf')))
    if not files:
        print('No recognized files found.'); continue
    for f in files:
        print(f'\n-- {f.relative_to(ROOT)} ({f.stat().st_size:,} bytes)')
        try:
            if f.suffix == '.parquet':
                df = pd.read_parquet(f)
                print('shape:', df.shape)
                print('columns:', list(df.columns))
                print(df.head(2).to_string(max_cols=12, max_colwidth=90))
            elif f.suffix == '.csv':
                df = pd.read_csv(f, nrows=5)
                print('columns:', list(df.columns)); print(df.to_string(index=False, max_cols=12))
            else:
                print('File present; use a format-specific reader to inspect records.')
        except Exception as e:
            print('Could not inspect:', repr(e))
