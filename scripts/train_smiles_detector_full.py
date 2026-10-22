"""Run-3 experiment 1: train the best 40k recipe on ALL 220k train molecules
(frozen ChemBERTa embeddings, shared full-data 64-cluster taxonomy), scored by the
evaluator's src/evaluation/mrr25.py.

Scope (user-selected): ONE recipe (the 40k winner J_ls0.1: lr 2e-3, wd 3e-4,
bs 1024, ls 0.1) with a capped epoch budget, plus a short 3-seed ensemble probe
only if the single run is fast enough.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_smiles_detector as base
import tune_smiles_detector_v3 as v3

OUTF = f'{v3.FULLOUT}/runs'
CKPTF = f'{v3.FULLOUT}/checkpoints'
HISTF = f'{OUTF}/histories'
HEAD_CFG = dict(hidden=[2048, 1024], act='tanh', dropout=0.0, norm='none')

# Best 40k recipe (J_ls0.1). Capped epochs for the 220k run.
BEST_HP = dict(lr=2e-3, wd=3e-4, bs=1024, ls=0.1)
CAP_EPOCHS = int(os.environ.get('FULL_CAP_EPOCHS', '12'))
VARIANTS = [dict(tag='FULL_J_ls0.1_cap', **BEST_HP)]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class MemmapIndex:
    """Row-indexable, torch-returning view over an embedding memmap (memory-lean).

    Supports the `X[idx]` and `len(X)` operations that v3.run_variant uses, plus
    `X.shape[1]`, without ever holding the full 220k x 768 array in RAM.
    """

    def __init__(self, memmap, mask):
        self.mm = memmap
        self.rows = np.flatnonzero(mask)
        self.shape = (len(self.rows), memmap.shape[1])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(len(self.rows)))
        elif torch.is_tensor(idx):
            idx = idx.numpy()
        idx = np.asarray(idx, dtype=np.int64)
        return torch.from_numpy(np.asarray(self.mm[self.rows[idx]]))


def prepare_full():
    z = np.load(f'{v3.FULLOUT}/labels.npz', allow_pickle=True)
    labels, good, is_train_all = z['labels'], z['good'], z['is_train']
    emb = np.load(f'{v3.FULLOUT}/embeddings.npy', mmap_mode='r')
    y_g = labels[good].astype(np.int64)
    tr_g = is_train_all[good]
    Xtr = MemmapIndex(emb, good & tr_g)
    Xva = MemmapIndex(emb, good & ~tr_g)
    ytr = torch.from_numpy(y_g[tr_g])
    yva = torch.from_numpy(y_g[~tr_g])
    log('full arm: train/val', len(ytr), '/', len(yva), '| classes', int(y_g.max()) + 1)
    return Xtr, ytr, Xva, yva


def main():
    os.makedirs(OUTF, exist_ok=True)
    os.makedirs(CKPTF, exist_ok=True)
    torch.set_num_threads(4)
    v3.MAX_EPOCHS = CAP_EPOCHS
    log('full arm capped at', CAP_EPOCHS, 'epochs, recipe', BEST_HP)
    Xtr, ytr, Xva, yva = prepare_full()
    v3.HIST = HISTF
    rows, store = [], []
    for v in VARIANTS:
        hp = dict(lr=v['lr'], wd=v['wd'], bs=v['bs'], ls=v['ls'])
        row, _, _ = v3.run_variant(v['tag'], HEAD_CFG, hp, Xtr, ytr, Xva, yva,
                                   Xval_full=Xva, yval_full=yva, state_store=store)
        rows.append(dict(round='F', **row))
        pd.DataFrame(rows).to_csv(f'{OUTF}/full_results.csv', index=False)

    best = store[0]
    torch.save(dict(state_dict=best[2], head_config=best[3], hparams=best[4],
                    fullval_mrr25=best[0], tag=best[1], n_clusters=base.N_CLUSTERS,
                    backbone=base.BACKBONE, cap_epochs=CAP_EPOCHS,
                    selection_metric='mrr25_from_scores (evaluator)'),
               f'{CKPTF}/FULL_BEST.pt')
    summary = dict(best_tag=best[1], best_hp=best[4], best_fullval_mrr25=best[0],
                   cap_epochs=CAP_EPOCHS, n_train=int(len(ytr)),
                   selection_metric='mrr25_from_scores (src/evaluation/mrr25.py)')
    with open(f'{OUTF}/full_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).to_csv(f'{OUTF}/full_results.csv', index=False)
    log('saved FULL_BEST.pt / full_summary.json')
    log('FULL DONE')


if __name__ == '__main__':
    main()
