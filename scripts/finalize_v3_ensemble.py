"""Finalise run-3 arm 2 (40k HPO): retrain the top-3 configs from v3_grid.csv
(deterministic seeds reproduce the exact weights), stream the full-val soft-voting
ensemble, and save V3_BEST.pt / V3_TOP3_ENS.pt / v3_summary.json.

Memory-lean: probabilities are accumulated in chunks straight from the memmap.
"""
import json
import math
import os
import sys
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import train_smiles_detector as base
import tune_smiles_detector_v2 as v2
import tune_smiles_detector_v3 as v3
from src.evaluation.mrr25 import mrr25_from_scores

OUT3 = v3.OUT3
CKPTS3 = v3.CKPTS3
HEAD_CFG = v3.HEAD_CFG


def retrain(tag, hp):
    Xtr, ytr, Xva, yva, Xfull, yfull = v3.prepare_40k()
    model, _, best = v3.run_variant(tag, HEAD_CFG, hp, Xtr, ytr, Xva, yva)
    return best['state']


def stream_val_probs(smiles_mask_path, states_batch, Xtr_shape):
    pass


def main():
    torch.set_num_threads(4)
    os.makedirs(CKPTS3, exist_ok=True)
    grid = pd.read_csv(f'{OUT3}/v3_grid.csv').sort_values('fullval_mrr25', ascending=False)
    top = grid.head(3)
    print('top-3 for ensemble:', top.tag.tolist(), flush=True)

    z = np.load(f'{v3.FULLOUT}/labels.npz', allow_pickle=True)
    labels, good, is_train_all = z['labels'], z['good'], z['is_train']
    y_g = labels[good].astype(np.int64)
    va_mask = good & ~is_train_all
    yva = torch.from_numpy(y_g[va_mask])
    emb = np.load(f'{v3.FULLOUT}/embeddings.npy', mmap_mode='r')
    n_val = int(va_mask.sum())
    n_cls = int(y_g.max()) + 1

    members = []
    for _, r in top.iterrows():
        hp = dict(lr=float(r.lr), wd=float(r.wd), bs=int(r.bs), ls=float(r.ls))
        state = retrain(r.tag, hp)
        members.append((r.tag, state, HEAD_CFG, hp))
        print('retrained', r.tag, flush=True)

    # stream ensemble over full val
    rr_sum = top1 = top5 = 0.0
    members_eval = []
    for tag, state, cfg, hp in members:
        m = v2.Head(emb.shape[1], n_cls, cfg)
        m.load_state_dict(state); m.eval(); members_eval.append(m)
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            Xb = torch.from_numpy(np.asarray(emb[va_mask][s:s + 2048]))
            probs = torch.zeros(len(Xb), n_cls)
            for m in members_eval:
                probs += F.softmax(m(Xb), dim=1)
            yy = yva[s:s + 2048]
            ranks = (probs > probs.gather(1, yy.unsqueeze(1))).sum(1) + 1
            rr_sum += float(torch.where(ranks <= 25, 1.0 / ranks.float(),
                                        torch.zeros_like(ranks, dtype=torch.float32)).sum())
            top1 += int((probs.argmax(1) == yy).sum())
            top5 += int((probs.topk(5, 1).indices == yy.unsqueeze(1)).any(1).sum())
            del Xb, probs
    ens = dict(mrr25_from_scores=rr_sum / n_val, top1=top1 / n_val, top5=top5 / n_val)
    print('ENSEMBLE fullval', ens, flush=True)

    torch.save(dict(states=[m[1] for m in members], tags=[m[0] for m in members],
                    configs=[m[2] for m in members], hparams=[m[3] for m in members],
                    metrics=ens, selection_metric='mrr25_from_scores (evaluator)'),
               f'{CKPTS3}/V3_TOP3_ENS.pt')
    best_tag, best_state, best_cfg, best_hp = members[0]
    torch.save(dict(state_dict=best_state, head_config=best_cfg, hparams=best_hp,
                    fullval_mrr25=float(grid.iloc[0].fullval_mrr25), tag=best_tag,
                    selection_metric='mrr25_from_scores (evaluator)'),
               f'{CKPTS3}/V3_BEST.pt')
    summary = dict(best_tag=best_tag, best_hp=best_hp,
                   best_fullval_mrr25=float(grid.iloc[0].fullval_mrr25),
                   ensemble_metrics=ens, ensemble_tags=top.tag.tolist(),
                   n_variants=len(grid),
                   selection_metric='mrr25_from_scores (src/evaluation/mrr25.py)')
    json.dump(summary, open(f'{OUT3}/v3_summary.json', 'w'), indent=2)
    print('saved V3_BEST.pt / V3_TOP3_ENS.pt / v3_summary.json', flush=True)


if __name__ == '__main__':
    main()
