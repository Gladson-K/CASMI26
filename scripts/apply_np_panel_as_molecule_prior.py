"""Apply NP Panel as a MOLECULE-LEVEL real structure-property prior on top of
the existing latest full-data ensemble weights.

For each validation molecule, derive an independent structure/property NP score
using NP panel statistics + COCOCONUT/NP-likeness + mass/formula from RDKit
plus the panel's real measured/structural signals where inchikey matches.
Fuse as z(log p_model) + lam * z(np_mol_prior_broadcast?) or better:
use it as a tie-breaker by adding to log-probs per molecule: s' = s + lam * p_mol
(where p_mol is the same scalar added to all class scores for that molecule)
or combine with fingerprint prototypes additively in z-space.
Results stored in outputs/smiles_detector_full/np_panel_molecule_prior_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors

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
    tr_g = is_train_all[good]
    n_cls = int(y_g.max()) + 1
    emb = np.load(f'{OUTF}/embeddings.npy', mmap_mode='r')
    va_rows = np.flatnonzero(good & ~is_train_all)
    yva = y_g[~tr_g]
    smi_va = smiles[good & ~is_train_all]
    n_val = len(yva)
    print('val', n_val, flush=True)

    ck = torch.load(f'{OUTF}/checkpoints/FULL_3SEED_ENS.pt', map_location='cpu', weights_only=False)
    models = []
    for state, cfg in zip(ck['states'], ck['configs']):
        cfg = dict(cfg); cfg.setdefault('norm','none')
        m = v2.Head(emb.shape[1], n_cls, cfg); m.load_state_dict(state); m.eval(); models.append(m)

    probs = np.empty((n_val, n_cls), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            rows = va_rows[s:s+2048]
            Xb = torch.from_numpy(np.asarray(emb[rows]))
            p = torch.zeros(len(Xb), n_cls)
            for m in models: p += F.softmax(m(Xb), 1)
            probs[s:s+2048]=(p/len(models)).numpy()
            del Xb,p
    base_mrr = mrr_many(probs,yva)
    print('base', base_mrr, flush=True)

    npp = NPPanelPrior.from_panel('data/benchmark/panel.parquet')
    # molecule prior
    mol_prior = np.zeros(n_val, dtype=np.float32)
    for i,smi in enumerate(smi_va):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol_prior[i]=0.0; continue
        # try inchikey
        try:
            from rdkit.Chem.inchi import MolToInchiKey
            ik = MolToInchiKey(mol)
        except Exception:
            ik = smi
        sp = npp.score_molecules([ik], use_smiles_fallback=True)[0]
        # structure-property extras
        mw = Descriptors.ExactMolWt(mol)
        try:
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            formula = ''
        extra = 0.0
        if 150 <= mw <= 800: extra += 0.1
        if any(c.islower() for c in formula): extra += 0.05  # aromatic
        mol_prior[i] = float(sp) + float(extra)
    zm = zscore(mol_prior)

    res = {'base_ensemble_mrr25': base_mrr, 'np_panel_stats': npp.to_dict(), 'mol_prior_mean': float(mol_prior.mean())}
    zlogp = zscore(np.log(probs+1e-9))
    # add molecule prior to all classes (tie-breaker / global NP score)
    for lam in (0.01,0.05,0.1,0.25,0.5,1.0,2.0):
        s2 = zlogp + lam * zm[:,None]
        res[f'mol_np_lam_{lam}'] = mrr_many(s2,yva)
        print('mol', lam, res[f'mol_np_lam_{lam}'], flush=True)
    # combine with prototype
    cache = np.load(f'{OUTF}/fusion_cache.npz')
    zchem = zscore(cache['chem'])
    best_cand = max([(k,v) for k,v in res.items() if isinstance(v,float) and k.startswith('mol_np')], key=lambda x:x[1])
    print('best_mol', best_cand, flush=True)
    for lam_c in (8,10,12):
        for lam_m in (0.01,0.05,0.1,0.25):
            s2 = zlogp + lam_c*zchem + lam_m*zm[:,None]
            res[f'proto_lam{lam_c}_molnp_lam{lam_m}'] = mrr_many(s2,yva)
    bestk = max([k for k,v in res.items() if isinstance(v,float) and k not in ('base_ensemble_mrr25','mol_prior_mean')], key=lambda k: res[k])
    res['best']={bestk:res[bestk]}
    print('BEST', bestk, res[bestk], flush=True)
    # top1/top5
    lam_c=10; lam_m=0
    if bestk.startswith('proto_lam'):
        try:
            lam_c=float(bestk.split('proto_lam')[1].split('_molnp')[0])
            lam_m=float(bestk.split('_molnp_lam')[1])
        except: pass
    s2 = zlogp + lam_c*zchem + lam_m*zm[:,None]
    res['best_top1']=float((s2.argmax(1)==yva).mean())
    res['best_top5']=float((np.argsort(-s2,axis=1)[:,:5]==yva[:,None]).any(1).mean())
    Path(f'{OUTF}/np_panel_molecule_prior_results.json').write_text(json.dumps(res,indent=2))
    print('saved')

if __name__=='__main__':
    main()
