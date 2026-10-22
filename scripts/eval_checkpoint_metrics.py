"""Evaluate saved head checkpoints with the full metric set, including the
repo's competition accuracy logic: mrr25 = mean(1/rank) for rank <= 25 else 0.

usage: python eval_checkpoint_metrics.py --ckpts path1.pt path2.pt [--out out.csv]
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_smiles_detector as base
from tune_smiles_detector_v2 import Head, full_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--out', default=f'{base.OUT}/checkpoint_metrics.csv')
    args = ap.parse_args()
    torch.set_num_threads(4)
    base.set_seed()
    df = base.load_dataset()
    is_train = (df.fold_identity.values != 0)
    labels, good, _ = base.cluster_labels(df.smiles.values, is_train)
    emb = base.embed_smiles(df.smiles.values[good].tolist(), f'{base.OUT}/embeddings.npz')
    y = labels[good]
    tr = is_train[good]
    _, _, Xva, yva = base.tensorize(emb, y, tr)
    rows = []
    for path in args.ckpts:
        c = torch.load(path, map_location='cpu', weights_only=False)
        cfg = dict(c['head_config'])
        cfg.setdefault('norm', 'none')
        model = Head(Xva.shape[1], int(y.max()) + 1, cfg)
        model.load_state_dict(c['state_dict'])
        m = full_metrics(model, Xva, yva)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append(dict(checkpoint=os.path.basename(path),
                         saved_tag=c.get('tag', c.get('stage', '')),
                         saved_metric=round(float(c.get('val_acc', c.get('metrics', 0.0))), 4),
                         val_mrr25=round(m['mrr25'], 4), val_top1=round(m['acc'], 4),
                         val_top5=round(m['top5'], 4), val_macro_f1=round(m['macro_f1'], 4),
                         val_balanced_acc=round(m['balanced_acc'], 4), head_params=n_params))
        base.log(os.path.basename(path), rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print()
    print(out.to_string(index=False))
    base.log('written', args.out)


if __name__ == '__main__':
    main()
