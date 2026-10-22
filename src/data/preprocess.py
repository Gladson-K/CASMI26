"""Spectrum preprocessing: normalize, filter, standardize."""
import numpy as np
from matchms import Spectrum
from matchms.filtering import (
    default_filters,
    normalize_intensities,
    select_by_mz,
    select_by_relative_intensity,
    add_precursor_mz,
)
from matchms.importing import load_from_mgf

def make_spectrum(mz, intensity, precursor_mz, metadata=None):
    s = Spectrum(mz=np.asarray(mz, dtype=np.float64),
                 intensities=np.asarray(intensity, dtype=np.float64),
                 metadata=metadata or {})
    s.set("precursor_mz", float(precursor_mz))
    return s

def clean_spectrum(spec, cfg):
    s = default_filters(spec)
    s = add_precursor_mz(s)
    s = select_by_mz(s, mz_from=cfg['preprocess']['mz_range'][0],
                        mz_to=cfg['preprocess']['mz_range'][1])
    s = select_by_relative_intensity(s, intensity_from=0.001)
    s = normalize_intensities(s)
    # Truncate to max peaks by intensity
    max_p = cfg['preprocess']['max_peaks']
    if s is not None and len(s.peaks.mz) > max_p:
        order = np.argsort(s.peaks.intensities)[::-1][:max_p]
        s = Spectrum(mz=s.peaks.mz[order], intensities=s.peaks.intensities[order],
                     metadata=s.metadata)
    return s

def process_dataframe(df, cfg, spectra_col="spectra"):
    """Apply cleaning to every spectrum in a dataframe column."""
    out = []
    for _, row in df.iterrows():
        cleaned = []
        for spec_dict in row[spectra_col]:
            s = make_spectrum(spec_dict['mz'], spec_dict['intensity'],
                              spec_dict['precursor_mz'])
            s = clean_spectrum(s, cfg)
            if s is not None:
                cleaned.append(s)
        out.append(cleaned)
    df = df.copy()
    df['clean_spectra'] = out
    return df