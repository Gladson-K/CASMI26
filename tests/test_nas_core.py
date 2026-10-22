import numpy as np
from src.nas.runner import neutral_mass, get_peaks, entropy_cosine, mrr_at_25


def test_neutral_mass_protonated():
    assert abs(neutral_mass(101.007276466621, '[M+H]+') - 100.0) < 1e-6


def test_get_peaks_normalizes_and_sorts():
    mz, inten = get_peaks({'ms2_mzs': [100, 50, 200], 'ms2_normalized_intensities': [1, 2, 0.001]}, max_peaks=10, floor=0.0001)
    assert list(mz) == [50.0, 100.0, 200.0]
    assert abs(float(np.linalg.norm(inten)) - 1.0) < 1e-5


def test_cosine_and_mrr():
    a = (np.array([50., 100.]), np.array([0.8, 0.6]))
    assert entropy_cosine(a, a, 0.01) > 0.99
    assert mrr_at_25(['wrong', 'truth'], 'truth') == 0.5
    assert mrr_at_25(['wrong'], 'truth') == 0.0
