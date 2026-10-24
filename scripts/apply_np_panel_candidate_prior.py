"""Apply the REAL structure–property NP Panel as a candidate prior on top of
the latest full-data ensemble weights (outputs/smiles_detector_full/checkpoints/FULL_3SEED_ENS.pt).

This uses data/benchmark/panel.parquet (COCOCONUT membership, PubChem window,
FPNet pool rank, mass error) as an independent structure/property signal,
mapped to chemotype class candidates (medoids) and fused as z(log p_model) + lam * z(np_panel_candidate_prior).
Results stored in outputs/smiles_detector_full/np_panel_candidate_prior_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import tune_smiles_detector_v2 as v2
import tune_smiles_detector_v3 as v3
from src.knowledge.np_panel_prior import NPPanelPrior
from src.evaluation.mrr25 import mrr25_from_scores

OUTF = v3.FULLOUT


def mrr_many(scores2d, y):
    return float(np.mean([mrr25_from_scores(scores2d[i], {int(y[i])}) for i in range(len(y))]))


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


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

    # load latest ensemble
    ck = torch.load(f'{OUTF}/checkpoints/FULL_3SEED_ENS.pt', map_location='cpu', weights_only=False)
    models = []
    for state, cfg in zip(ck['states'], ck['configs']):
        cfg = dict(cfg); cfg.setdefault('norm', 'none')
        m = v2.Head(emb.shape[1], n_cls, cfg)
        m.load_state_dict(state); m.eval(); models.append(m)
    print('ensemble:', ck.get('tags'), flush=True)

    probs = np.empty((n_val, n_cls), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            rows = va_rows[s:s+2048]
            Xb = torch.from_numpy(np.asarray(emb[rows]))
            p = torch.zeros(len(Xb), n_cls)
            for m in models:
                p += F.softmax(m(Xb), 1)
            probs[s:s+2048] = (p/len(models)).numpy()
            del Xb,p
    base_mrr = mrr_many(probs, yva)
    print('BASE ensemble MRR@25', round(base_mrr,4), flush=True)

    # build NP panel prior and map to class medoids
    try:
        npp = NPPanelPrior.from_panel('data/benchmark/panel.parquet')
        print('NP panel stats:', npp.to_dict(), flush=True)
    except Exception as e:
        print('ERR loading panel', e, flush=True)
        raise

    # find medoids per class (same as before)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    classes = np.arange(n_cls)
    medoids = [None]*n_cls
    best_sim = np.full(n_cls, -1.0)
    # first pass centroids
    cent = np.zeros((n_cls,2048), dtype=np.float64)
    cnt = np.zeros(n_cls, dtype=np.int64)
    for i in range(len(tr_rows)):
        y = int(y_g[tr_rows[i]])
        smi = smi_g[tr_rows[i]]
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        fp = gen.GetFingerprint(m)
        arr = np.zeros((2048,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        cent[y] += arr; cnt[y] += 1
    cent = cent / np.maximum(cnt,1)[:,None]
    centn = cent / (np.linalg.norm(cent,axis=1,keepdims=True)+1e-8)
    # second pass medoids
    for i in range(len(tr_rows)):
        y = int(y_g[tr_rows[i]])
        smi = smi_g[tr_rows[i]]
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        fp = gen.GetFingerprint(m)
        arr = np.zeros((2048,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arrn = arr / (np.linalg.norm(arr)+1e-8)
        sim = float(centn[y] @ arrn)
        if sim > best_sim[y]:
            best_sim[y]=sim; medoids[y]=smi
    cand_prior = npp.score_candidates_by_medoid(medoids).astype(np.float32)
    print('candidate prior stats:', cand_prior.mean(), cand_prior.std(), flush=True)

    zlogp = zscore(np.log(probs+1e-9)); zc = zscore(cand_prior)
    res = {'base_ensemble_mrr25': base_mrr, 'np_panel_stats': npp.to_dict()}
    for lam in (0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0):
        r = mrr_many(zlogp + lam*zc[None,:], yva)
        res[f'cand_np_panel_lam_{lam}'] = r
        print(f'  z(log p)+{lam}*z(NP_panel_candidate): {r:.4f}', flush=True)
    # combine with fingerprint prototype
    cache = np.load(f'{OUTF}/fusion_cache.npz')
    chem = cache['chem']
    zchem = zscore(chem)
    for lam_c in (0.01,0.05,0.1,0.25,0.5,1.0):
        for lam_n in (0.01,0.05,0.1,0.25,0.5):
            r = mrr_many(zlogp + lam_c*zchem + lam_n*zc[None,:], yva)
            res[f'proto_lam{lam_c}_np_lam{lam_n}'] = r
    # best
    cand_keys = [k for k in res if k not in ('base_ensemble_mrr25','np_panel_stats')]
    bestk = max(cand_keys, key=lambda k: res[k]) if cand_keys else None
    res['best'] = {bestk: res[bestk]} if bestk else {}
    # compute top1/top5
    lam_c, lam_n = 0.0, 0.0
    if bestk and bestk.startswith('proto_lam'):
        # format proto_lam{c}_np_lam{n}
        try:
            lam_c = float(bestk.split('proto_lam')[1].split('_np_lam')[0])
            lam_n = float(bestk.split('_np_lam')[1])
        except Exception:
            pass
    fused = zlogp + lam_c*zchem + lam_n*zc[None,:]
    res['best_top1'] = float((fused.argmax(1)==yva).mean())
    res['best_top5'] = float((np.argsort(-fused,axis=1)[:,:5]==yva[:,None]).any(1).mean())
    Path(f'{OUTF}/np_panel_candidate_prior_results.json').write_text(json.dumps(res,indent=2))
    print('saved np_panel_candidate_prior_results.json')
    print(json.dumps(res,indent=2)[:800])

if __name__=='__main__':
    main()
