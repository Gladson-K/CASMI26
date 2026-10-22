"""Chemist-knowledge injection for SMILES scoring (CASMI structure-ranking).

The public CASMI26 benchmark shows the learned fingerprint score alone caps the
PubChem channel at MRR@25 ~0.51, while adding a *documentation/popularity prior*
raises it to ~0.97 (see data/benchmark/README.md). The lesson generalises: a
structure ranker must fuse a learned chemical-linguistic score with explicit
chemist knowledge.

This module encodes that knowledge in two layers, both usable standalone or
fused with the ChemBERTa head via the evaluator's scheme
``s_final = z(model_score) + lam * prior``:

  1. ``chemist_prior``  - an RDKit-derived "is-this-a-real-well-formed-NP" prior:
     natural-product-likeness (NP-likeness proxies), formula/precursor-mass
     consistency, structural-alert counts, and a documentation prior if a
     popularity table is supplied (the exact lever the benchmark proves works).
  2. ``modal_fingerprint_prior`` - a *learned* chemist prior: builds a
     molecule-class fingerprint model by averaging Morgan fingerprints of the
     labelled molecules in each chemotype cluster, then scores candidates by
     cosine similarity to those cluster prototypes. This is "chemical knowledge
     written directly into the model" as a non-parametric memory of the classes.

Fusion helpers implement the evaluator's ``mrr25``/``mrr25_from_scores`` and the
``z + lam*pop`` re-ranking used by np_eval.py / merge_eval.py.
"""
import json
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluation.mrr25 import mrr25_from_scores

RDLogger.DisableLog('rdApp.*')

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Substrings / SMARTS capturing motifs chemists expect in natural products.
_NP_SMARTS = [
    ('[#6;R]([#6;R])([#6;R])[#6;R]', 1.0),        # fused carbocycle
    ('O=C(O)[CX4]', 0.9),                          # carboxylic acid
    ('[OX2H]', 0.6),                               # alcohol / phenol
    ('[CX4;R][OX2H]', 0.8),                        # sugar-like hydroxyl on ring
    ('C1OC1', 0.8),                                # epoxide
    ('[#6]=[#6]', 0.4),                            # unsaturation
    ('c1ccccc1', 0.5),                             # aromatic ring
    ('C(=O)C', 0.4),                               # ketone
    ('[NX3;H2,H1]', 0.4),                          # amine
    ('[OX2]([CX4])[CX4]', 0.6),                    # ether
]
# Reactive / assay-interference alerts (higher = less likely a clean NP answer).
_ALERT_SMARTS = [
    '[CX3]=[CX3][CX3]=[CX3]',                      # polyene (fragile)
    '[N+](=O)[O-]',                                # nitro
    '[CX3](=O)[CX3](=O)',                          # 1,3-dicarbonyl
    '[Cl,Br,I]',                                   # halogen
    '[#7]#[#6]',                                   # nitrile
]


def _fp(mol):
    return _GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def chemist_prior(smiles, popularity=None):
    """Per-molecule chemist prior. Returns (prior_vector_dict, total).

    Keys:
      np_like        natural-product-likeness in [0, 1] (weighted motif matches)
      mass_ok        1 if a plausible neutral mass, else 0
      halogen_pen    1 if a halogen is present (unusual for many NP panels)
      alerts         number of reactive/assay-interference alerts
      formula_ok     1 if RDKit assigns a formula
      pop            documentation prior (log_substances + log_pubmed) if given
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(np_like=0.0, mass_ok=0.0, halogen_pen=1.0, alerts=5.0,
                    formula_ok=0.0, pop=0.0 if popularity is None else float(popularity)), 0.0
    np_scores = [w for smarts, w in _NP_SMARTS if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
    np_like = float(np.clip(sum(np_scores) / (sum(w for _, w in _NP_SMARTS)), 0.0, 1.0))
    mw = Descriptors.ExactMolWt(mol)
    mass_ok = 1.0 if 100.0 <= mw <= 1500.0 else 0.0
    halogen_pen = 1.0 if any(a in smiles for a in ('Cl', 'Br', 'I')) else 0.0
    alerts = float(sum(1 for s in _ALERT_SMARTS if mol.HasSubstructMatch(Chem.MolFromSmarts(s))))
    try:
        formula_ok = 1.0 if rdMolDescriptors.CalcMolFormula(mol) else 0.0
    except Exception:
        formula_ok = 0.0
    pop = float(popularity) if popularity is not None else 0.0
    vec = dict(np_like=np_like, mass_ok=mass_ok, halogen_pen=halogen_pen,
               alerts=alerts, formula_ok=formula_ok, pop=pop)
    total = (0.5 * np_like + 0.2 * mass_ok + 0.1 * formula_ok
             - 0.15 * alerts - 0.1 * halogen_pen)
    if popularity is not None:
        total += 0.5 * pop
    return vec, float(total)


class ModalFingerprintPrior:
    """A non-parametric 'chemist memory': Morgan prototypes of each chemotype class.

    Fit on labelled train molecules: average the Morgan-2048 fingerprint of the
    members of each cluster -> a prototype a chemist would call 'the phenotype of
    that class'. Score a candidate by the best cosine similarity to any prototype
    (and by the softmax over prototypes for a calibrated class distribution).
    Injecting this into a head model is the fingerprint analogue of the
    popularity prior: explicit chemical knowledge fused at score time.
    """

    def __init__(self, radius=2, fp_size=2048):
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        self.fp_size = fp_size
        self.prototypes = None
        self.classes_ = None

    def fit(self, smiles, labels, medoids=False):
        """Streaming fit: accumulate per-class fingerprint sums (memory O(n_classes)).

        With ``medoids=True`` a second pass picks, per class, the training molecule
        whose fingerprint is closest to the class centroid (the 'phenotype' a
        chemist would call the representative of that chemotype).
        """
        classes = np.unique(np.asarray([y for y in labels if y >= 0]))
        cls_idx = {c: i for i, c in enumerate(classes)}
        sums = np.zeros((len(classes), self.fp_size), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for s, y in zip(smiles, labels):
            if y < 0:
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            i = cls_idx[int(y)]
            sums[i] += self.gen.GetFingerprintAsNumPy(m)
            counts[i] += 1
        protos = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
        self.prototypes = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        self.classes_ = classes
        self.medoid_smiles_ = None
        if medoids:
            best = np.full(len(classes), -np.inf)
            best_smi = [None] * len(classes)
            for s, y in zip(smiles, labels):
                if y < 0:
                    continue
                m = Chem.MolFromSmiles(s)
                if m is None:
                    continue
                i = cls_idx[int(y)]
                f = self.gen.GetFingerprintAsNumPy(m).astype(np.float32)
                c = float(self.prototypes[i] @ (f / (np.linalg.norm(f) + 1e-8)))
                if c > best[i]:
                    best[i] = c
                    best_smi[i] = s
            self.medoid_smiles_ = best_smi
        return self

    def scores(self, smiles):
        """Return (n, n_classes) cosine similarity to each class prototype."""
        out = np.zeros((len(smiles), len(self.classes_)), dtype=np.float32)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            f = self.gen.GetFingerprintAsNumPy(m).astype(np.float32)
            out[i] = self.prototypes @ (f / (np.linalg.norm(f) + 1e-8))
        return out

    def save(self, path):
        np.savez(path, prototypes=self.prototypes, classes=self.classes_)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obj = cls()
        obj.prototypes = z['prototypes']
        obj.classes_ = z['classes']
        return obj


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def fuse(model_scores, prior, lam):
    """Evaluator fusion: z(model) + lam * prior (elementwise, same length)."""
    return zscore(model_scores) + lam * np.asarray(prior, dtype=np.float64)


def evaluate_fusion(model_scores, prior, y_true, lam_grid=(0.0, 0.1, 0.25, 0.5, 1.0)):
    """Sweep lam and return MRR@25 (evaluator metric) per lam, plus the best."""
    rows = {}
    for lam in lam_grid:
        s = fuse(model_scores, prior, lam)
        rows[lam] = float(np.mean([mrr25_from_scores(s[i], {int(y_true[i])})
                                   for i in range(len(y_true))]))
    best_lam = max(rows, key=rows.get)
    return rows, best_lam, rows[best_lam]


def _smoke():
    smis = ['CC(=O)O', 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O',
            'CN1CCC2=CC(=C(C=C2C1)O)O', 'ClC(Cl)=C']
    for s in smis:
        vec, tot = chemist_prior(s)
        print(f'{s:35s} prior={tot:+.3f} {vec}')
    y = np.array([0, 1, 2, 0])
    mfp = ModalFingerprintPrior().fit(smis, y)
    sc = mfp.scores(smis)
    print('prototype scores shape', sc.shape)
    rows, lam, best = evaluate_fusion(sc.max(1), np.array([chemist_prior(s)[1] for s in smis]),
                                      y, lam_grid=(0.0, 0.5, 1.0))
    print('fusion sweep', rows, 'best lam', lam, 'mrr25', best)


if __name__ == '__main__':
    _smoke()
