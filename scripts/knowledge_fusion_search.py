"""Finer search over how to inject the chemist knowledge into the full-data model.

Uses the cached scores (outputs/smiles_detector_full/fusion_cache.npz):
  probs : (n_val, 64) ChemBERTa-head posteriors (3-seed ensemble)
  chem  : (n_val, 64) Morgan class-prototype cosines (feature-level chemist memory)
  yva   : true class per val molecule

Fusion families
  prob mix     : p' = (1-eps) p + eps * minmax(chem)
  logit mix    : z(log p) + lam * z(chem)
  rank fusion  : RRF 1/(k+rank_p) + 1/(k+rank_chem)  (rank-only, scale free)
  prob*exp     : p'_j = p_j * exp(lam * chem_j)       (multiplicative / likelihood)

Writes outputs/smiles_detector_full/knowledge_fusion_search.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.mrr25 import mrr25_from_scores

OUTF = 'outputs/smiles_detector_full'


def mrr_many(scores2d, y):
    return float(np.mean([mrr25_from_scores(scores2d[i], {int(y[i])}) for i in range(len(y))]))


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def minmax(a):
    lo = a.min(1, keepdims=True)
    hi = a.max(1, keepdims=True)
    return (a - lo) / (hi - lo + 1e-8)


def main():
    z = np.load(f'{OUTF}/fusion_cache.npz')
    probs, chem, yva = z['probs'], z['chem'], z['yva']
    n = len(yva)
    print('cached', probs.shape, chem.shape, flush=True)
    res = {}
    res['model_only'] = mrr_many(probs, yva)
    res['chem_prior_only'] = mrr_many(chem, yva)
    print('model only', round(res['model_only'], 4),
          '| chem prior only', round(res['chem_prior_only'], 4), flush=True)

    chem_n = minmax(chem)
    for eps in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96,
                0.97, 0.98, 0.99, 1.0):
        res[f'probmix_eps_{eps}'] = mrr_many(probs * (1 - eps) + eps * chem_n, yva)

    zlogp = zscore(np.log(probs + 1e-9))
    zchem = zscore(chem)
    for lam in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 12.0,
                14.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0, 100.0):
        res[f'logitmix_lam_{lam}'] = mrr_many(zlogp + lam * zchem, yva)

    # softmax-of-chem at several temperatures, mixed with the posterior
    for T in (0.02, 0.05, 0.1, 0.2):
        cs = np.exp((chem - chem.max(1, keepdims=True)) / T)
        cs /= cs.sum(1, keepdims=True)
        for eps in (0.5, 0.8, 0.9, 0.95):
            res[f'probmix_softmaxT{T}_eps_{eps}'] = mrr_many(probs * (1 - eps) + eps * cs, yva)

    for lam in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        res[f'mul_lam_{lam}'] = mrr_many(probs * np.exp(lam * (chem - chem.max(1, keepdims=True))), yva)

    rp = np.argsort(-probs, axis=1).argsort(axis=1).astype(np.float64)
    rc = np.argsort(-chem, axis=1).argsort(axis=1).astype(np.float64)
    for k in (5, 10, 30, 60):
        res[f'rrf_k_{k}'] = mrr_many(1.0 / (k + rp) + 1.0 / (k + rc), yva)
    for w in (0.3, 0.5, 0.7):
        res[f'rrf_k60_w_{w}'] = mrr_many(1.0 / (60 + rp) + w / (60 + rc), yva)

    # best-of reporting
    top = sorted(res.items(), key=lambda kv: -kv[1])[:8]
    for k, v in top:
        print(f'  {k}: {v:.4f}', flush=True)
    best_key, best_val = top[0]
    # top-1 / top-5 of the best fusion
    fam, _, param = best_key.rpartition('_')
    if fam == 'logitmix_lam':
        fused = zlogp + float(param) * zchem
    elif fam.startswith('probmix_eps'):
        fused = probs * (1 - float(param)) + float(param) * chem_n
    else:
        fused = probs
    t1 = float(np.mean(fused.argmax(1) == yva))
    t5 = float(np.mean((np.argsort(-fused, axis=1)[:, :5] == yva[:, None]).any(1)))
    print(f'best fusion top1 {t1:.4f} top5 {t5:.4f}', flush=True)
    out = dict(n_val=n, results=res, best=dict(name=best_key, mrr25=best_val,
                                                top1=t1, top5=t5),
               model_only=res['model_only'], chem_prior_only=res['chem_prior_only'],
               gain_vs_model=best_val - res['model_only'],
               model_only_top1=float(np.mean(probs.argmax(1) == yva)),
               model_only_top5=float(np.mean((np.argsort(-probs, axis=1)[:, :5] == yva[:, None]).any(1))),
               note='metric = src/evaluation/mrr25.py mrr25_from_scores; '
                    'chem prior = Morgan-2048 class prototypes fitted on the 220,408-molecule '
                    'train fold of the full-data 64-cluster taxonomy')
    json.dump(out, open(f'{OUTF}/knowledge_fusion_search.json', 'w'), indent=2)
    print('saved', f'{OUTF}/knowledge_fusion_search.json', flush=True)


if __name__ == '__main__':
    main()
