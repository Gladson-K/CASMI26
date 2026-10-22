"""Feature engineering for the reranker."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def _mol_props(mol):
    try:
        return [
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.ExactMolWt(mol),
            rdMolDescriptors.CalcNumHeavyAtoms(mol),
        ]
    except Exception:
        return [0.0] * 8

def build_features(spectrum, candidate_smiles, similarity_score, params):
    """Return 1-D feature vector for a (spectrum, candidate) pair."""
    feats = [similarity_score]

    mol = Chem.MolFromSmiles(candidate_smiles)
    if mol is None:
        return None

    # Precursor mass discrepancy
    if params.get('use_precursor_mass_feature', True):
        cand_mass = Descriptors.ExactMolWt(mol)
        prec = spectrum.get("precursor_mz", 0.0)
        feats.append(abs(cand_mass - prec))
        feats.append(abs(cand_mass - prec) / max(prec, 1.0))

    # Spectral features
    if params.get('use_peak_count', True):
        feats.append(len(spectrum.peaks.mz))

    # Molecular descriptors
    feats.extend(_mol_props(mol))

    return np.asarray(feats, dtype=np.float32)