"""Load competition parquet and COCONUT SDF into memory."""
from pathlib import Path
import pandas as pd
from rdkit import Chem

def load_competition(cfg):
    d = Path(cfg['paths']['competition_dir'])
    train = pd.read_parquet(d / "train.parquet")
    test = pd.read_parquet(d / "test.parquet")
    return train, test

def load_coconut_smiles(cfg):
    """Return list of SMILES from COCONUT SDF files."""
    d = Path(cfg['paths']['coconut_dir'])
    smiles = []
    for sdf in d.glob("*.sdf"):
        suppl = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if smi:
                smiles.append(smi)
    return list(set(smiles))

def load_benchmark(cfg):
    d = Path(cfg['paths']['benchmark_dir'])
    files = list(d.glob("*.parquet"))
    return {f.stem: pd.read_parquet(f) for f in files}