#!/usr/bin/env python3
"""Check output/submission.csv against the sample template and parse candidate SMILES."""
from pathlib import Path
import sys
import pandas as pd
from rdkit import Chem
ROOT = Path(__file__).resolve().parents[1]
sub_path = ROOT / 'outputs' / 'submission.csv'
sample_path = ROOT / 'data' / 'competition' / 'sample_submission.csv'
if not sub_path.exists(): raise SystemExit(f'Missing {sub_path}; run scripts/run_baseline.py first.')
sub = pd.read_csv(sub_path)
if sample_path.exists():
    sample = pd.read_csv(sample_path)
    if len(sub) != len(sample): raise SystemExit(f'Row count mismatch: {len(sub)} != {len(sample)}')
    if sub.columns[0] != sample.columns[0]: raise SystemExit(f'ID column mismatch: {sub.columns[0]} != {sample.columns[0]}')
if 'smiles' not in sub: raise SystemExit(f'Missing smiles column. Found: {list(sub.columns)}')
errors=[]; total=0; valid=0
for idx, value in enumerate(sub['smiles'].fillna('').astype(str)):
    cand=[x for x in value.split(';') if x]
    if len(cand) != 25: errors.append(f'row {idx}: expected 25 candidates, got {len(cand)}')
    if len(cand) != len(set(cand)): errors.append(f'row {idx}: duplicate candidates')
    for smi in cand:
        total += 1
        if Chem.MolFromSmiles(smi) is not None: valid += 1
        else: errors.append(f'row {idx}: invalid SMILES {smi[:60]!r}')
print(f'Rows={len(sub):,}; candidates={total:,}; valid_SMILES={valid:,}; valid_rate={valid/max(total,1):.3%}')
if errors:
    print('\n'.join(errors[:50])); print(f'{len(errors)} issue(s) found.'); sys.exit(1)
print('Submission structure checks passed. This does not establish prediction accuracy or leaderboard validity.')
