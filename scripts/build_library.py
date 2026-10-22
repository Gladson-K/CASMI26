"""Build spectral library from COCONUT + competition SMILES."""
import yaml, pickle
from pathlib import Path
from rdkit import Chem
from src.data.loader import load_competition, load_coconut_smiles
from src.data.preprocess import clean_spectrum, make_spectrum
from src.retrieval.spectral_library import build_library

def main():
    cfg = yaml.safe_load(open('config.yaml'))
    train, test = load_competition(cfg)
    smiles = load_coconut_smiles(cfg)

    # Also add SMILES from training set
    train_smiles = train['smiles'].dropna().unique().tolist()
    smiles = list(set(smiles) | set(train_smiles))

    print(f"Building library with {len(smiles)} unique SMILES...")
    entries = build_library(smiles, cache_path=Path(cfg['paths']['cache_dir']) / 'library.pkl')
    print(f"Library built: {len(entries)} entries")

if __name__ == '__main__':
    main()
