"""Fast cosine similarity retrieval over a spectral library."""
import numpy as np
from numba import njit
from matchms import Spectrum
from matchms.similarity import CosineGreedy, ModifiedCosine

@njit(cache=True)
def _peak_cosine(mz1, int1, mz2, int2, tol):
    """Greedy peak matching cosine similarity."""
    used = np.zeros(len(mz2), dtype=np.bool_)
    score = 0.0
    for i in range(len(mz1)):
        best = -1
        best_diff = tol + 1.0
        for j in range(len(mz2)):
            if used[j]:
                continue
            d = abs(mz1[i] - mz2[j])
            if d <= tol and d < best_diff:
                best_diff = d
                best = j
        if best >= 0:
            used[best] = True
            score += int1[i] * int2[best]
    return score

def retrieve_topk(query_spec, library_specs, library_smiles,
                  metric='cosine', tol=0.01, top_k=100, precursor_tol=0.02):
    """Retrieve top-k candidate SMILES for a query spectrum."""
    q_mz = query_spec.peaks.mz.astype(np.float64)
    q_int = query_spec.peaks.intensities.astype(np.float64)
    q_prec = query_spec.get("precursor_mz", 0.0)

    scores = []
    for spec, smi in zip(library_specs, library_smiles):
        # precursor mass filter
        p_prec = spec.get("precursor_mz", 0.0)
        if q_prec > 0 and p_prec > 0:
            if abs(q_prec - p_prec) / max(q_prec, 1.0) > precursor_tol:
                continue
        l_mz = spec.peaks.mz.astype(np.float64)
        l_int = spec.peaks.intensities.astype(np.float64)
        s = _peak_cosine(q_mz, q_int, l_mz, l_int, tol)
        scores.append((smi, s))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]