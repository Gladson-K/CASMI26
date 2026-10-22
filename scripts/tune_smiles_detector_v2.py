"""Extended experiment (run 2): richer accuracy metrics + hyperparameter tuning.

Reuses run-1 data pipeline (same 40k structures, labels, cached frozen ChemBERTa
embeddings) so results are directly comparable.

Rounds:
  A  new head configs (50 epochs, early stop after 3 straight val-acc drops)
  B  hyperparameter sweep (lr / weight decay / batch size / label smoothing) on the round-A winner
  C  soft-voting ensemble of the top-3 variants

Accuracy per run uses the repo's mrr_at_25 logic - mean(1/rank) for rank <= 25 else 0 (val_mrr25) -
which drives best-model selection and the 3-consecutive-drop early stop; val_top1, val_top5,
val_macro_f1 and val_balanced_acc are also reported.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_smiles_detector as base

OUT = base.OUT
CKPTS = base.CKPTS
V2HIST = f'{OUT}/v2_histories'
MAX_EPOCHS = 50
WARMUP_EPOCHS = 3
DROP_PATIENCE = 3

ROUND_A_HP = dict(lr=1e-3, wd=1e-4, bs=512, ls=0.05)

ROUND_A_CONFIGS = {
    'control_2x1024_tanh': dict(hidden=[1024, 512], act='tanh', dropout=0.0, norm='none'),
    'mlp_3x1024_tanh': dict(hidden=[1024, 1024, 512], act='tanh', dropout=0.0, norm='none'),
    'mlp_2x2048_tanh': dict(hidden=[2048, 1024], act='tanh', dropout=0.0, norm='none'),
    'mlp_2x1024_tanh_bn': dict(hidden=[1024, 512], act='tanh', dropout=0.0, norm='bn'),
    'mlp_2x1024_relu_bn': dict(hidden=[1024, 512], act='relu', dropout=0.0, norm='bn'),
    'mlp_4x512_tanh': dict(hidden=[512, 512, 512, 256], act='tanh', dropout=0.0, norm='none'),
    'mlp_2x1024_tanh_do01': dict(hidden=[1024, 512], act='tanh', dropout=0.1, norm='none'),
}

ROUND_B_HPS = [
    dict(lr=3e-4, wd=0.0, bs=512, ls=0.05),
    dict(lr=3e-4, wd=1e-3, bs=512, ls=0.05),
    dict(lr=3e-4, wd=1e-2, bs=512, ls=0.05),
    dict(lr=1e-3, wd=0.0, bs=512, ls=0.05),
    dict(lr=1e-3, wd=1e-3, bs=512, ls=0.05),
    dict(lr=1e-3, wd=1e-2, bs=512, ls=0.05),
    dict(lr=3e-3, wd=0.0, bs=512, ls=0.05),
    dict(lr=3e-3, wd=1e-3, bs=512, ls=0.05),
    dict(lr=3e-3, wd=1e-2, bs=512, ls=0.05),
    dict(lr=5e-4, wd=1e-3, bs=256, ls=0.05),
    dict(lr=1e-3, wd=1e-3, bs=1024, ls=0.05),
    dict(lr=5e-4, wd=1e-3, bs=512, ls=0.0),
]


class Head(nn.Module):
    def __init__(self, din, n_classes, cfg):
        super().__init__()
        layers, d = [], din
        act = nn.ReLU if cfg['act'] == 'relu' else nn.Tanh
        for h in cfg['hidden']:
            layers.append(nn.Linear(d, h))
            if cfg.get('norm') == 'bn':
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if cfg.get('dropout', 0) > 0:
                layers.append(nn.Dropout(cfg['dropout']))
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def mrr25_from_logits(logits, y):
    """Repo mrr_at_25 logic: mean of 1/rank when rank <= 25, else 0."""
    ranks = (logits > logits.gather(1, y.unsqueeze(1))).sum(1) + 1
    recip = torch.where(ranks <= 25, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float32))
    return float(recip.mean())


@torch.no_grad()
def full_metrics(model, X, y):
    model.eval()
    logits = model(X)
    preds = logits.argmax(1)
    out = dict(
        mrr25=mrr25_from_logits(logits, y),
        acc=float((preds == y).float().mean()),
        top5=float((logits.topk(5, 1).indices == y.unsqueeze(1)).any(1).float().mean()),
        macro_f1=float(f1_score(y.numpy(), preds.numpy(), average='macro', zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y.numpy(), preds.numpy())),
    )
    return out


def cosine_lambda(total_steps):
    warm = WARMUP_EPOCHS * total_steps // MAX_EPOCHS

    def fn(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total_steps - warm)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return fn


def run_variant(tag, cfg, hp, Xtr, ytr, Xva, yva, state_store=None):
    seed = base.SEED + zlib.crc32(tag.encode()) % 10000
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    model = Head(Xtr.shape[1], int(ytr.max()) + 1, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'], betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(ytr) / hp['bs'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(steps_per_epoch * MAX_EPOCHS))
    bs = hp['bs']
    history = []
    best = dict(mrr25=-1.0, state=None, epoch=-1, metrics=None)
    consec = 0
    stop = 'max_epochs'
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], label_smoothing=hp['ls'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        tr = full_metrics(model, Xtr, ytr)
        va = full_metrics(model, Xva, yva)
        lr_now = opt.param_groups[0]['lr']
        history.append(dict(epoch=epoch, lr=lr_now, train_loss=float(loss.item()),
                            train_acc=tr['acc'], train_mrr25=tr['mrr25'],
                            val_mrr25=va['mrr25'], val_acc=va['acc'], val_top5=va['top5'],
                            val_macro_f1=va['macro_f1'], val_balanced_acc=va['balanced_acc']))
        if va['mrr25'] > best['mrr25']:
            best = dict(mrr25=va['mrr25'], state=copy.deepcopy(model.state_dict()),
                        epoch=epoch, metrics=dict(va))
            consec = 0
        elif epoch > 1 and va['mrr25'] < history[-2]['val_mrr25']:
            consec += 1
            if consec >= DROP_PATIENCE:
                stop = 'early_stop_3_consecutive_acc_drops'
                base.log(f'[{tag}] early stop at epoch {epoch} (MRR@25 accuracy dropped {consec}x straight)')
                break
        else:
            consec = 0
    model.load_state_dict(best['state'])
    pd.DataFrame(history).to_csv(f'{V2HIST}/{tag}.csv', index=False)
    m = best['metrics']
    row = dict(tag=tag, arch='+'.join(map(str, cfg['hidden'])), act=cfg['act'],
               norm=cfg.get('norm'), dropout=cfg.get('dropout'), hidden_params=n_params,
               lr=hp['lr'], wd=hp['wd'], bs=hp['bs'], ls=hp['ls'],
               epochs_run=len(history), stop_reason=stop, best_epoch=best['epoch'],
               val_mrr25=m['mrr25'], val_top1=m['acc'], val_top5=m['top5'],
               val_macro_f1=m['macro_f1'], val_balanced_acc=m['balanced_acc'],
               train_mrr25=history[best['epoch'] - 1]['train_mrr25'],
               train_acc=history[best['epoch'] - 1]['train_acc'],
               secs=round(time.time() - t0, 1))
    base.log(f"[{tag}] MRR@25={m['mrr25']:.4f} top1={m['acc']:.4f} top5={m['top5']:.4f} "
             f"f1={m['macro_f1']:.4f} bal={m['balanced_acc']:.4f} @e{best['epoch']} ({stop})")
    if state_store is not None:
        state_store.append((m['mrr25'], tag, copy.deepcopy(best['state']), cfg, hp))
    return row, model, best


def prepare():
    os.makedirs(V2HIST, exist_ok=True)
    base.set_seed()
    torch.set_num_threads(4)
    df = base.load_dataset()
    is_train = (df.fold_identity.values != 0)
    labels, good, _ = base.cluster_labels(df.smiles.values, is_train)
    emb = base.embed_smiles(df.smiles.values[good].tolist(), f'{OUT}/embeddings.npz')
    y = labels[good]
    tr = is_train[good]
    base.log('train/val:', int(tr.sum()), '/', int((~tr).sum()), '| classes:', int(y.max()) + 1)
    return base.tensorize(emb, y, tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', default='all', choices=['a', 'b', 'c', 'all'])
    args = ap.parse_args()
    Xtr, ytr, Xva, yva = prepare()
    all_rows, store = [], []

    if args.round in ('a', 'all'):
        base.log('=== RUN2 ROUND A: new configs (50 ep, cosine + warmup, label smoothing) ===')
        for i, (name, cfg) in enumerate(ROUND_A_CONFIGS.items()):
            row, _, _ = run_variant(f'A_{name}', cfg, dict(ROUND_A_HP),
                                    Xtr, ytr, Xva, yva, state_store=store)
            all_rows.append(dict(round='A', **row))
        pd.DataFrame(all_rows).to_csv(f'{OUT}/v2_round_a.csv', index=False)

    if args.round in ('b', 'all'):
        if not all_rows:
            prev = pd.read_csv(f'{OUT}/v2_round_a.csv')
            winner = prev.sort_values('val_mrr25', ascending=False).iloc[0]
            wcfg = next(c for n, c in ROUND_A_CONFIGS.items() if n in winner.tag)
        else:
            winner = max(all_rows, key=lambda r: r['val_mrr25'])
            wcfg = next(c for n, c in ROUND_A_CONFIGS.items() if n in winner['tag'])
        wname = winner['tag'] if isinstance(winner, dict) else winner.tag
        base.log('=== RUN2 ROUND B: HPO on', wname, '===')
        for i, hp in enumerate(ROUND_B_HPS, 1):
            row, _, _ = run_variant(f'B_v{i}', wcfg, hp, Xtr, ytr, Xva, yva, state_store=store)
            all_rows.append(dict(round='B', **row))
        pd.DataFrame([r for r in all_rows if r.get('round') == 'B']).to_csv(f'{OUT}/v2_round_b.csv', index=False)

    if args.round in ('c', 'all'):
        if not store:
            base.log('no in-memory states; rebuilding top-3 from saved histories is not supported - run --round all')
            return
        all_rows_df = pd.DataFrame(all_rows)
        pd.concat([pd.read_csv(f'{OUT}/v2_round_a.csv'), all_rows_df]).to_csv(f'{OUT}/v2_all_results.csv', index=False)
        top3 = sorted(store, key=lambda s: -s[0])[:3]
        base.log('=== RUN2 ROUND C: ensemble of', [t[1] for t in top3], '===')
        probs = torch.zeros(len(yva), int(ytr.max()) + 1)
        for acc_, tag, state, cfg, hp in top3:
            m = Head(Xtr.shape[1], int(ytr.max()) + 1, cfg)
            m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                probs += F.softmax(m(Xva), dim=1)
        preds = probs.argmax(1)
        ens = dict(
            mrr25=mrr25_from_logits(probs, yva),
            acc=float((preds == yva).float().mean()),
            top5=float((probs.topk(5, 1).indices == yva.unsqueeze(1)).any(1).float().mean()),
            macro_f1=float(f1_score(yva.numpy(), preds.numpy(), average='macro', zero_division=0)),
            balanced_acc=float(balanced_accuracy_score(yva.numpy(), preds.numpy())),
        )
        base.log(f"ENSEMBLE MRR@25={ens['mrr25']:.4f} top1={ens['acc']:.4f} top5={ens['top5']:.4f} "
                 f"f1={ens['macro_f1']:.4f} bal={ens['balanced_acc']:.4f}")
        torch.save(dict(states=[s[2] for s in top3], tags=[s[1] for s in top3],
                        configs=[s[3] for s in top3], hparams=[s[4] for s in top3],
                        metrics=ens, n_clusters=base.N_CLUSTERS, backbone=base.BACKBONE,
                        selection_metric='mrr25 (1/rank, rank<=25)'),
                   f'{CKPTS}/V2_ENSEMBLE.pt')
        best = max(store, key=lambda s: s[0])
        torch.save(dict(state_dict=best[2], head_config=best[3], hparams=best[4],
                        metrics=best[0], tag=best[1], n_clusters=base.N_CLUSTERS,
                        backbone=base.BACKBONE, selection_metric='mrr25 (1/rank, rank<=25)'),
                   f'{CKPTS}/V2_BEST_MODEL.pt')
        summary = dict(best_single_tag=best[1], best_single_mrr25=best[0],
                       ensemble_metrics=ens, ensemble_tags=[s[1] for s in top3],
                       protocol=dict(max_epochs=MAX_EPOCHS,
                                     early_stop='3_consecutive_val_mrr25_drops',
                                     schedule='cosine+warmup3',
                                     selection_metric='mrr25 = mean(1/rank) for rank<=25 else 0'))
        with open(f'{OUT}/v2_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        all_df = pd.concat([pd.read_csv(f'{OUT}/v2_round_a.csv'), pd.read_csv(f'{OUT}/v2_round_b.csv')])
        all_df.sort_values('val_mrr25', ascending=False).to_csv(f'{OUT}/v2_all_results.csv', index=False)
        base.log('saved V2_BEST_MODEL.pt / V2_ENSEMBLE.pt / v2_summary.json')
    base.log('DONE')


if __name__ == '__main__':
    main()
