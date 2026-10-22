"""Unified run-3 comparison table.

Scores every stored head checkpoint with the evaluator's metric
(src/evaluation/mrr25.py) on the validation set that belongs to its own label
space:
  * run-1 / run-2 checkpoints -> 40k taxonomy (fold_identity==0 val split, 7,982 mols)
  * run-3 v3 (40k-trained) and full-data checkpoints -> full-data taxonomy
    (220,408 train / 55,110 val mols, same 64-cluster label space)

Writes outputs/checkpoint_metrics_run3.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_smiles_detector as base
import tune_smiles_detector_v2 as v2
import tune_smiles_detector_v3 as v3
from src.evaluation.mrr25 import mrr25_from_scores

OUT = 'outputs'


def score_probs(probs, y):
    y = np.asarray(y)
    ranks = (probs > probs[np.arange(len(y)), y][:, None]).sum(1) + 1
    rr = np.where(ranks <= 25, 1.0 / ranks, 0.0)
    return dict(mrr25=float(rr.mean()), top1=float((probs.argmax(1) == y).mean()),
                top5=float((np.argsort(-probs, axis=1)[:, :5] == y[:, None]).any(1).mean()))


def load_full_space():
    z = np.load(f'{v3.FULLOUT}/labels.npz', allow_pickle=True)
    labels, good, is_train = z['labels'], z['good'], z['is_train']
    y_g = labels[good].astype(np.int64)
    tr = is_train[good]
    emb = np.load(f'{v3.FULLOUT}/embeddings.npy', mmap_mode='r')
    rows_va = np.flatnonzero(good & ~tr)
    rows_tr = np.flatnonzero(good & tr)
    return emb, y_g, tr, rows_va, rows_tr


def main():
    torch.set_num_threads(4)
    rows = []

    # ---------------- full-data taxonomy (run-3 v3 + full arm) ----------------
    emb, y_g, tr, rows_va, rows_tr = load_full_space()
    yva = y_g[~tr]
    n_cls = int(y_g.max()) + 1
    print('full taxonomy: val', len(yva), 'classes', n_cls, flush=True)

    def eval_states(states, cfgs, tag, ckpt):
        models = []
        for st, cfg in zip(states, cfgs):
            cfg = dict(cfg); cfg.setdefault('norm', 'none')
            m = v2.Head(emb.shape[1], n_cls, cfg)
            m.load_state_dict(st); m.eval(); models.append(m)
        probs = np.zeros((len(yva), n_cls), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(yva), 4096):
                Xb = torch.from_numpy(np.asarray(emb[rows_va[s:s + 4096]]))
                p = torch.zeros(len(Xb), n_cls)
                for m in models:
                    p += F.softmax(m(Xb), 1)
                probs[s:s + 4096] = (p / len(models)).numpy()
                del Xb, p
        met = score_probs(probs, yva)
        rows.append(dict(arm=tag, checkpoint=ckpt, n_val=len(yva), label_space='full-64',
                         **met))
        print(rows[-1], flush=True)

    # v3 (40k-trained, full-data taxonomy)
    ck = torch.load(f'{v3.OUT3}/checkpoints/V3_BEST.pt', map_location='cpu', weights_only=False)
    eval_states([ck['state_dict']], [ck['head_config']], 'run3-v3-40k-best', 'V3_BEST.pt')
    ck = torch.load(f'{v3.OUT3}/checkpoints/V3_TOP3_ENS.pt', map_location='cpu', weights_only=False)
    eval_states(ck['states'], ck['configs'], 'run3-v3-40k-top3-ens', 'V3_TOP3_ENS.pt')

    # full-data arm
    ck = torch.load(f'{v3.FULLOUT}/checkpoints/FULL_BEST.pt', map_location='cpu', weights_only=False)
    eval_states([ck['state_dict']], [ck['head_config']], 'run3-full-220k-best', 'FULL_BEST.pt')
    ck = torch.load(f'{v3.FULLOUT}/checkpoints/FULL_3SEED_ENS.pt', map_location='cpu', weights_only=False)
    eval_states(ck['states'], ck['configs'], 'run3-full-220k-3seed-ens', 'FULL_3SEED_ENS.pt')

    # chemist-knowledge fusion on the full-data ensemble
    cache = np.load(f'{v3.FULLOUT}/fusion_cache.npz')
    probs, chem = cache['probs'], cache['chem']
    from scripts.knowledge_fusion_search import zscore  # noqa: F401  (local helper)
    def z_(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / (x.std() + 1e-9)
    met = score_probs(z_(np.log(probs + 1e-9)) + 10.0 * z_(chem), yva)
    rows.append(dict(arm='run3-full-220k-3seed-ens+chemist-prior', checkpoint='fusion z(log p)+10 z(chem)',
                     n_val=len(yva), label_space='full-64', **met))
    print(rows[-1], flush=True)

    # ---------------- 40k taxonomy (run-1 / run-2) ----------------
    df = base.load_dataset()
    is_train = (df.fold_identity.values != 0)
    labels, good, _ = base.cluster_labels(df.smiles.values, is_train)
    emb40 = base.embed_smiles(df.smiles.values[good].tolist(), f'{base.OUT}/embeddings.npz')
    y40 = labels[good]
    tr40 = is_train[good]
    _, _, Xva40, yva40 = base.tensorize(emb40, y40, tr40)
    n_cls40 = int(y40.max()) + 1
    print('40k taxonomy: val', len(yva40), 'classes', n_cls40, flush=True)

    def eval40(states, cfgs, tag, ckpt):
        models = []
        for st, cfg in zip(states, cfgs):
            cfg = dict(cfg); cfg.setdefault('norm', 'none')
            m = v2.Head(Xva40.shape[1], n_cls40, cfg)
            m.load_state_dict(st); m.eval(); models.append(m)
        with torch.no_grad():
            p = sum(F.softmax(m(Xva40), 1) for m in models) / len(models)
        met = score_probs(p.numpy(), yva40.numpy())
        rows.append(dict(arm=tag, checkpoint=ckpt, n_val=len(yva40), label_space='40k-64', **met))
        print(rows[-1], flush=True)

    for path, tag in [(f'{base.OUT}/checkpoints/V2_BEST_MODEL.pt', 'run2-best'),
                      (f'{base.OUT}/checkpoints/V2_ENSEMBLE.pt', 'run2-ensemble')]:
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if 'states' in ck:
            eval40(ck['states'], ck.get('configs', [ck.get('head_config')] * len(ck['states'])),
                   tag, os.path.basename(path))
        else:
            eval40([ck['state_dict']], [ck['head_config']], tag, os.path.basename(path))

    out = pd.DataFrame(rows).sort_values('mrr25', ascending=False)
    out.to_csv(f'{OUT}/checkpoint_metrics_run3.csv', index=False)
    json.dump(rows, open(f'{OUT}/checkpoint_metrics_run3.json', 'w'), indent=2)
    print()
    print(out.to_string(index=False))
    print('written', f'{OUT}/checkpoint_metrics_run3.csv')


if __name__ == '__main__':
    main()
