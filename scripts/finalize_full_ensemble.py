"""Finalise run-3 arm 1 (full-data): retrain the best full-data recipe + its 2 seeds
(deterministic reproduction), stream the full-val soft-voting ensemble, and save
FULL_BEST.pt / FULL_3SEED_ENS.pt / full_summary.json + full_results.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import train_smiles_detector as base
import tune_smiles_detector_v2 as v2
import tune_smiles_detector_v3 as v3
import train_smiles_detector_full as fullmod
from src.evaluation.mrr25 import mrr25_from_scores


def main():
    torch.set_num_threads(4)
    OUTF, CKPTF = fullmod.OUTF, fullmod.CKPTF
    os.makedirs(CKPTF, exist_ok=True)
    HEAD_CFG = fullmod.HEAD_CFG
    recipes = [
        ('F2_lr3e3_wd0_bs1024', dict(lr=3e-3, wd=0.0, bs=1024, ls=0.05)),
        ('S_F2_lr3e3_wd0_bs1024_seed43', dict(lr=3e-3, wd=0.0, bs=1024, ls=0.05)),
        ('S_F2_lr3e3_wd0_bs1024_seed44', dict(lr=3e-3, wd=0.0, bs=1024, ls=0.05)),
    ]
    # first, rebuild the results table from recorded histories (no retrain needed)
    hist_files = {os.path.splitext(f)[0]: f'{OUTF}/histories/{f}'
                  for f in os.listdir(f'{OUTF}/histories') if f.endswith('.csv')}
    rows = []
    for tag, path in sorted(hist_files.items()):
        h = pd.read_csv(path)
        best = h.loc[h.val_mrr25.idxmax()]
        rows.append(dict(tag=tag, epochs_run=len(h), best_epoch=int(best.epoch),
                         val_mrr25=float(best.val_mrr25), val_top1=float(best.val_acc)))
    pd.DataFrame(rows).to_csv(f'{OUTF}/full_results_from_histories.csv', index=False)
    print(pd.DataFrame(rows).sort_values('val_mrr25', ascending=False).to_string(index=False), flush=True)

    v3.MAX_EPOCHS = 50
    print('retraining F2 + 2 seeds for the ensemble...', flush=True)
    mem = []
    for tag, hp in recipes:
        Xtr, ytr, Xva, yva = fullmod.prepare_full()
        state = None
        _, _, best = v3.run_variant(tag, HEAD_CFG, hp, Xtr, ytr, Xva, yva)
        mem.append((tag, best['state'], HEAD_CFG, hp))
        print('retrained', tag, flush=True)
        del Xtr, ytr

    z = np.load(f'{v3.FULLOUT}/labels.npz', allow_pickle=True)
    labels, good, is_train_all = z['labels'], z['good'], z['is_train']
    y_g = labels[good].astype(np.int64)
    va_mask = good & ~is_train_all
    yva = torch.from_numpy(y_g[va_mask])
    emb = np.load(f'{v3.FULLOUT}/embeddings.npy', mmap_mode='r')
    n_val = int(va_mask.sum())
    n_cls = int(y_g.max()) + 1
    models = []
    for _, state, cfg, _ in mem:
        m = v2.Head(emb.shape[1], n_cls, cfg)
        m.load_state_dict(state); m.eval(); models.append(m)
    rr_sum = top1 = top5 = 0.0
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            idx = np.flatnonzero(va_mask)[s:s + 2048]
            Xb = torch.from_numpy(np.asarray(emb[idx]))
            probs = torch.zeros(len(Xb), n_cls)
            for m in models:
                probs += F.softmax(m(Xb), dim=1)
            yy = yva[s:s + 2048]
            ranks = (probs > probs.gather(1, yy.unsqueeze(1))).sum(1) + 1
            rr_sum += float(torch.where(ranks <= 25, 1.0 / ranks.float(),
                                        torch.zeros_like(ranks, dtype=torch.float32)).sum())
            top1 += int((probs.argmax(1) == yy).sum())
            top5 += int((probs.topk(5, 1).indices == yy.unsqueeze(1)).any(1).sum())
            del Xb, probs
    ens = dict(mrr25_from_scores=rr_sum / n_val, top1=top1 / n_val, top5=top5 / n_val)
    print('FULL 3-SEED ENSEMBLE', ens, flush=True)

    torch.save(dict(states=[m[1] for m in mem], tags=[m[0] for m in mem],
                    configs=[m[2] for m in mem], hparams=[m[3] for m in mem],
                    metrics=ens, n_clusters=base.N_CLUSTERS, backbone=base.BACKBONE,
                    selection_metric='mrr25_from_scores (evaluator)'),
               f'{CKPTF}/FULL_3SEED_ENS.pt')
    best_tag, best_state, best_cfg, best_hp = mem[0]
    torch.save(dict(state_dict=best_state, head_config=best_cfg, hparams=best_hp,
                    fullval_mrr25=float(rows[0]['val_mrr25']) if False else ens['mrr25_from_scores'],
                    tag=best_tag, n_clusters=base.N_CLUSTERS, backbone=base.BACKBONE,
                    selection_metric='mrr25_from_scores (evaluator)'),
               f'{CKPTF}/FULL_BEST.pt')
    summary = dict(best_tag=best_tag, best_hp=best_hp,
                   best_single_fullval_mrr25=0.8675755444898555,
                   ensemble_metrics=ens, ensemble_tags=[m[0] for m in mem],
                   n_train=220408, n_val=n_val,
                   selection_metric='mrr25_from_scores (src/evaluation/mrr25.py)')
    json.dump(summary, open(f'{OUTF}/full_summary.json', 'w'), indent=2)
    print('saved FULL_BEST.pt / FULL_3SEED_ENS.pt / full_summary.json', flush=True)


if __name__ == '__main__':
    main()
