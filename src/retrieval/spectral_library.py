"""Build searchable spectral library from reference SMILES + predicted spectra."""
import pickle
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.data.preprocess import make_spectrum

def smiles_to_library_entry(smi, pseudo_spectrum=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mass = Descriptors.ExactMolWt(mol)
    entry = {
        'smiles': Chem.MolToSmiles(mol),
        'mass': mass,
        'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
    }
    if pseudo_spectrum is not None:
        entry['spectrum'] = pseudo_spectrum
    return entry

def build_library(smiles_list, cache_path=None):
    """Build library from SMILES list. Optionally use in-silico spectra."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    entries = []
    for smi in smiles_list:
        e = smiles_to_library_entry(smi)
        if e is not None:
            entries.append(e)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, f)
    return entries

def attach_spectra(entries, spectra_by_smiles):
    """Attach observed spectra to library entries when available."""
    for e in entries:
        if e['smiles'] in spectra_by_smiles:
            e['spectra'] = spectra_by_smiles[e['smiles']]
    return entries