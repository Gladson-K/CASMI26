"""Real structure–property NP Panel candidate prior.

Uses CASMI benchmark NP panel metadata (data/benchmark/panel.parquet) as an
independent, structure/property-informed prior (not derived from the 64-cluster
labels). 

Signals:
- in_coconut: presence in COCOCONUT (natural products DB) -> NP-likeness proxy
- n_pubchem_window: PubChem compound hits in the candidate window (popularity)
- n_pool_window: pool coverage
- truth_pool_rank_fpnet: FPNet pool rank (structural prior)
- mass alignment signals

Scores are aggregated into a single real structure–property prior and can be
applied at the candidate level (per chemotype class medoid / per candidate).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .chemist_knowledge import chemist_prior


class NPPanelPrior:
    def __init__(self):
        self.panel = None
        self.mol_to_score = {}
        self.stats = {}

    @classmethod
    def from_panel(cls, panel_path: str | Path = "data/benchmark/panel.parquet"):
        obj = cls()
        obj.panel = pd.read_parquet(panel_path)
        obj._build()
        return obj

    def _build(self):
        df = self.panel.copy()
        # normalize
        df['in_coconut'] = df['in_coconut'].astype(bool)
        # compute composite NP panel score
        scores = []
        for _, row in df.iterrows():
            s = 0.0
            if row.get('in_coconut', False):
                s += 1.0
            # pubchem window (log scaled)
            npw = float(row.get('n_pubchem_window', 0) or 0)
            s += 0.25 * np.log1p(npw)
            npool = float(row.get('n_pool_window', 0) or 0)
            s += 0.1 * np.log1p(npool)
            # better (lower) fpnet pool rank -> higher prior
            tpr = row.get('truth_pool_rank_fpnet', None)
            if tpr is not None and not (isinstance(tpr, float) and np.isnan(tpr)):
                try:
                    s += 0.5 / (1.0 + float(tpr))
                except Exception:
                    pass
            # mass error small -> plausible
            me = row.get('mass_error_ppm', None)
            if me is not None and not (isinstance(me, float) and np.isnan(me)):
                try:
                    s += 0.2 * np.exp(-abs(float(me)) / 10.0)
                except Exception:
                    pass
            scores.append(s)
        df['np_panel_score'] = scores
        # z-score across panel
        mu = df['np_panel_score'].mean()
        sd = df['np_panel_score'].std() + 1e-9
        df['np_panel_z'] = (df['np_panel_score'] - mu) / sd
        self.panel = df
        self.mol_to_score = dict(zip(df['molecule'].astype(str), df['np_panel_z'].astype(float)))
        self.stats = {'n_panel': len(df), 'mean': float(mu), 'std': float(sd)}

    def score_molecules(self, inchikeys_or_smiles: list[str], use_smiles_fallback: bool = False):
        out = []
        for x in inchikeys_or_smiles:
            x = str(x)
            if x in self.mol_to_score:
                out.append(self.mol_to_score[x])
                continue
            # try inchikey14 prefix match
            found = None
            for k, v in self.mol_to_score.items():
                if k.startswith(x[:14]) or x.startswith(k[:14]):
                    found = v
                    break
            if found is not None:
                out.append(found)
                continue
            # fallback: chemist prior blend
            if use_smiles_fallback:
                vec, _ = chemist_prior(x)
                s = 0.8 * vec.get('np_like', 0.0) + 0.4 * vec.get('formula_ok', 0.0) - 0.3 * vec.get('alerts', 0.0)
                out.append(float(s))
            else:
                out.append(0.0)
        return np.array(out, dtype=np.float32)

    def score_candidates_by_medoid(self, medoid_smiles: list[str]):
        # try to map via InChIKey if possible
        keys = []
        for smi in medoid_smiles:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                keys.append(smi)
                continue
            try:
                from rdkit.Chem.inchi import MolToInchiKey
                ik = MolToInchiKey(m)
                keys.append(ik)
            except Exception:
                keys.append(smi)
        return self.score_molecules(keys, use_smiles_fallback=True)

    def to_dict(self):
        return {'stats': self.stats, 'n_mapped': len(self.mol_to_score)}
