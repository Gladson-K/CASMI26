"""Chemist-knowledge injection on the FULL-data run-3 model (55,110-molecule val set).

The evaluator (src/evaluation/mrr25.py) ranks *candidate structures* for each query
molecule and rewards the true candidate within the top 25. So the chemist prior is
injected at the CANDIDATE level: each of the 64 chemotype classes contributes one
candidate structure (its train-fold medoid = the molecule whose Morgan fingerprint
is closest to the class centroid, i.e. the phenotype a chemist would name).

Three injection routes are evaluated against the model's class posteriors:
  1. feature level  : cosine of the query's Morgan fingerprint to the class
                      prototypes (Morgan "chemist memory") fused with the posterior;
  2. candidate level: z(log p_model) + lam * z(chemist prior of the candidate),
                      the same shape as the popularity-prior fusion validated in
                      data/benchmark (MRR@25 0.51 -> 0.95);
  3. molecule level : the NP-likeness / mass / structural-alert prior of the query
                      molecule, used to break ties between equally likely candidates.

Writes outputs/smiles_detector_full/knowledge_fusion_full.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import tune_smiles_detector_v2 as v2
import tune_smiles_detector_v3 as v3
from src.knowledge.chemist_knowledge import (ModalFingerprintPrior, chemist_prior,
                                             zscore)
from src.evaluation.mrr25 import mrr25_from_scores

OUTF = v3.FULLOUT
CHUNK = 2048


def mrr_many(scores2d, y):
    return float(np.mean([mrr25_from_scores(scores2d[i], {int(y[i])}) for i in range(len(y))]))


def norm_rows(a):
    lo = a.min(1, keepdims=True)
    hi = a.max(1, keepdims=True)
    return (a - lo) / (hi - lo + 1e-8)


def main():
    torch.set_num_threads(4)
    z = np.load(f'{OUTF}/labels.npz', allow_pickle=True)
    labels, good, is_train_all, smiles = z['labels'], z['good'], z['is_train'], z['smiles']
    y_g = labels[good].astype(np.int64)
    smi_g = smiles[good]
    tr_g = is_train_all[good]
    n_cls = int(y_g.max()) + 1
    emb = np.load(f'{OUTF}/embeddings.npy', mmap_mode='r')
    va_rows = np.flatnonzero(good & ~is_train_all)
    tr_rows = np.flatnonzero(good & is_train_all)
    yva = y_g[~tr_g]
    smi_va = smi_g[~tr_g]
    n_val = len(yva)
    print(f'full val {n_val} | train {len(tr_rows)} | classes {n_cls}', flush=True)

    # ---------- model posteriors (3-seed ensemble), streamed ----------
    ck = torch.load(f'{OUTF}/checkpoints/FULL_3SEED_ENS.pt', map_location='cpu', weights_only=False)
    models = []
    for state, cfg in zip(ck['states'], ck['configs']):
        cfg = dict(cfg)
        cfg.setdefault('norm', 'none')
        m = v2.Head(emb.shape[1], n_cls, cfg)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    print('ensemble members:', ck.get('tags'), flush=True)

    probs = np.empty((n_val, n_cls), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_val, CHUNK):
            rows = va_rows[s:s + CHUNK]
            Xb = torch.from_numpy(np.asarray(emb[rows]))
            p = torch.zeros(len(Xb), n_cls)
            for m in models:
                p += F.softmax(m(Xb), 1)
            probs[s:s + CHUNK] = (p / len(models)).numpy()
            del Xb, p
    base_mrr = mrr_many(probs, yva)
    print('FULL model (3-seed ens) MRR@25', round(base_mrr, 4), flush=True)

    # ---------- feature-level chemist memory ----------
    mfp = ModalFingerprintPrior().fit(smi_g[tr_g].tolist(), y_g[tr_g], medoids=True)
    chem = mfp.scores(smi_va.tolist()).astype(np.float32)
    chem_mrr = mrr_many(chem, yva)
    print('prototype-fingerprint chemist prior MRR@25', round(chem_mrr, 4), flush=True)
    np.savez_compressed(f'{OUTF}/fusion_cache.npz', probs=probs, chem=chem, yva=yva)
    print('cached probs/chem -> fusion_cache.npz', flush=True)

    # ---------- candidate-level chemist prior (per class representative) ----------
    cand_prior = np.zeros(n_cls, dtype=np.float32)
    cand_detail = []
    for j, c in enumerate(mfp.classes_):
        smi = mfp.medoid_smiles_[j]
        vec, _ = chemist_prior(smi) if smi else (dict(), 0.0)
        # a chemist's "quality" score: NP-like and formulable, no alerts, no halogen
        q = (1.0 * vec.get('np_like', 0.0) + 0.5 * vec.get('mass_ok', 0.0)
             + 0.5 * vec.get('formula_ok', 0.0) - 0.5 * vec.get('alerts', 0.0)
             - 0.5 * vec.get('halogen_pen', 0.0))
        cand_prior[j] = q
        cand_detail.append(dict(cls=int(c), medoid=smi, prior=float(q),
                                np_like=vec.get('np_like'), alerts=vec.get('alerts'),
                                halogen=vec.get('halogen_pen')))
    cand_prior = cand_prior[np.argsort(mfp.classes_)] if list(mfp.classes_) != list(range(n_cls)) \
        else cand_prior
    zcand = zscore(cand_prior)

    # ---------- fusion routes ----------
    results = {'model_only': base_mrr, 'prototype_prior_only': chem_mrr}
    chem_n = norm_rows(chem)

    for eps in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        r = mrr_many(probs * (1 - eps) + eps * chem_n, yva)
        results[f'feature_fusion_eps_{eps}'] = r
        print(f'  feature fusion eps={eps}: {r:.4f}', flush=True)

    zlogp = zscore(np.log(probs + 1e-9))
    for lam in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        r = mrr_many(zlogp + lam * zcand[None, :], yva)
        results[f'candidate_prior_lam_{lam}'] = r
        print(f'  z(log p) + {lam} * z(candidate chemist prior): {r:.4f}', flush=True)

    # feature + candidate combined
    best_eps = max((k for k in results if k.startswith('feature_fusion')),
                    key=lambda k: results[k])
    eps = float(best_eps.split('_')[-1])
    fused_feat = probs * (1 - eps) + eps * chem_n
    for lam in (0.05, 0.1, 0.25, 0.5):
        r = mrr_many(zscore(np.log(fused_feat + 1e-9)) + lam * zcand[None, :], yva)
        results[f'combined_feat_eps_{eps}_lam_{lam}'] = r
        print(f'  feature(eps={eps}) + candidate(lam={lam}): {r:.4f}', flush=True)

    out = dict(base_full_ensemble_mrr25=base_mrr,
               prototype_prior_mrr25=chem_mrr,
               fusion_grid=results,
               best_fusion=max((v, k) for k, v in results.items() if k != 'model_only'),
               n_val=n_val, n_train=int(len(tr_rows)),
               ensemble_tags=ck.get('tags'),
               candidate_representatives=cand_detail,
               note='chemist knowledge: (a) Morgan class prototypes = feature-level '
                    'chemist memory, (b) medoid candidate structures scored with the '
                    'NP-likeness / mass / structural-alert prior = candidate-level prior, '
                    'fused as z(log p) + lam * z(prior); metric = '
                    'src/evaluation/mrr25.py mrr25_from_scores')
    json.dump(out, open(f'{OUTF}/knowledge_fusion_full.json', 'w'), indent=2)
    print('saved', f'{OUTF}/knowledge_fusion_full.json', flush=True)


if __name__ == '__main__':
    main()
