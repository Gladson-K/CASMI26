"""Run-3 experiment 2: deeper HPO around the run-2 winner (B_v11: 2x2048 tanh, lr 1e-3,
wd 1e-3, bs 1024, ls 0.05) on the 40k subset, scored by the evaluator's own
src/evaluation/mrr25.py::mrr25_from_scores.

All run-3 arms share the full-data 64-cluster taxonomy (labels.npz from prep_full_data.py),
so the 40k-trained and full-data-trained models are directly comparable on the same
55k fold-0 validation set (fullval_* columns).

Rounds:
  H  lr x wd grid at bs 1024, ls 0.05          (18 runs)
  I  batch-size sweep at the best lr/wd        (512, 2048)
  J  label-smoothing sweep at the best hp      (0.0, 0.02, 0.1)
  S  3-seed ensemble of the best recipe        (+ top-3 soft-voting ensemble of all runs)
  P  unfreeze probe: unfreeze last 2 ChemBERTa layers on a 3k subset (feasibility check)
"""
import argparse
import gc
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_smiles_detector as base
import tune_smiles_detector_v2 as v2
from src.evaluation.mrr25 import mrr25_from_scores

OUT3 = 'outputs/smiles_detector_v3'
FULLOUT = 'outputs/smiles_detector_full'
HIST = f'{OUT3}/histories'
CKPTS3 = f'{OUT3}/checkpoints'
MAX_EPOCHS = 50
WARMUP_EPOCHS = 3
DROP_PATIENCE = 3
HEAD_CFG = dict(hidden=[2048, 1024], act='tanh', dropout=0.0, norm='none')

H_GRID = [dict(lr=lr, wd=wd, bs=1024, ls=0.05)
          for lr in (5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3)
          for wd in (3e-4, 1e-3, 3e-3)]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def mrr25_eval(scores2d, y):
    """Evaluator-exact mrr25_from_scores, vectorised over rows (no 55k python loop).

    Mirrors src/evaluation/mrr25.py: rank of the true class when scores are sorted
    descending; 1/rank if rank<=25 else 0; mean over molecules.
    """
    s = np.asarray(scores2d, dtype=np.float64)
    y = np.asarray(y)
    order = np.argsort(-s, axis=1, kind='stable')
    pos = np.argmax(order == y[:, None], axis=1) + 1
    return float(np.where(pos <= 25, 1.0 / pos, 0.0).mean())


def eval_fullval(model, Xval_full, yval_full):
    """Streaming, memory-lean evaluation on the 55k full validation set.

    Never materialises the full (n_val x n_classes) logits tensor: accumulates
    MRR@25 (evaluator logic), top-1 and top-5 in one pass.
    """
    model.eval()
    y_np = yval_full.numpy()
    n = len(y_np)
    rr_sum = 0.0
    top1_hits = 0
    top5_hits = 0
    with torch.no_grad():
        for s in range(0, n, 4096):
            lg = model(Xval_full[s:s + 4096])
            yy = yval_full[s:s + 4096]
            ranks = (lg > lg.gather(1, yy.unsqueeze(1))).sum(1) + 1
            rr_sum += float(torch.where(ranks <= 25, 1.0 / ranks.float(),
                                        torch.zeros_like(ranks, dtype=torch.float32)).sum())
            top1_hits += int((lg.argmax(1) == yy).sum())
            top5_hits += int((lg.topk(min(5, lg.shape[1]), 1).indices == yy.unsqueeze(1)).any(1).sum())
            del lg
    mrr_torch = rr_sum / n
    return dict(mrr25_from_scores=mrr_torch, mrr25_torch=mrr_torch,
                top1=top1_hits / n, top5=top5_hits / n)


@torch.no_grad()
def cheap_metrics(model, X, y):
    """Batched train/val metrics for per-epoch checks (no full-set logits held)."""
    model.eval()
    n = len(y)
    correct = rr_sum = 0.0
    for s in range(0, n, 4096):
        lg = model(X[s:s + 4096])
        yy = y[s:s + 4096]
        ranks = (lg > lg.gather(1, yy.unsqueeze(1))).sum(1) + 1
        rr_sum += float(torch.where(ranks <= 25, 1.0 / ranks.float(),
                                    torch.zeros_like(ranks, dtype=torch.float32)).sum())
        correct += float((lg.argmax(1) == yy).float().sum())
        del lg
    return dict(acc=correct / n, mrr25=rr_sum / n)


def cosine_lambda(total_steps):
    warm = WARMUP_EPOCHS * total_steps // MAX_EPOCHS

    def fn(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total_steps - warm)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return fn


def run_variant(tag, cfg, hp, Xtr, ytr, Xva, yva, Xval_full=None, yval_full=None, state_store=None):
    seed = base.SEED + zlib.crc32(tag.encode()) % 10000
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    model = v2.Head(Xtr.shape[1], int(ytr.max()) + 1, cfg)
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
        tr = cheap_metrics(model, Xtr, ytr)
        va = cheap_metrics(model, Xva, yva)
        history.append(dict(epoch=epoch, lr=opt.param_groups[0]['lr'], train_loss=float(loss.item()),
                            train_acc=tr['acc'], train_mrr25=tr['mrr25'],
                            val_mrr25=va['mrr25'], val_acc=va['acc']))
        if va['mrr25'] > best['mrr25']:
            best = dict(mrr25=va['mrr25'], state=copy.deepcopy(model.state_dict()),
                        epoch=epoch, metrics=dict(va))
            consec = 0
        elif epoch > 1 and va['mrr25'] < history[-2]['val_mrr25']:
            consec += 1
            if consec >= DROP_PATIENCE:
                stop = 'early_stop_3_consecutive_mrr25_drops'
                log(f'[{tag}] early stop at epoch {epoch} (MRR@25 dropped {consec}x straight)')
                break
        else:
            consec = 0
    model.load_state_dict(best['state'])
    os.makedirs(HIST, exist_ok=True)
    pd.DataFrame(history).to_csv(f'{HIST}/{tag}.csv', index=False)
    gc.collect()
    m = best['metrics']
    row = dict(tag=tag, arch='+'.join(map(str, cfg['hidden'])), act=cfg['act'],
               lr=hp['lr'], wd=hp['wd'], bs=hp['bs'], ls=hp['ls'],
               epochs_run=len(history), stop_reason=stop, best_epoch=best['epoch'],
               val40k_mrr25=m['mrr25'], val40k_top1=m['acc'],
               secs=round(time.time() - t0, 1))
    if Xval_full is not None:
        fv = eval_fullval(model, Xval_full, yval_full)
        row.update(fullval_mrr25=fv['mrr25_from_scores'], fullval_mrr25_torch=fv['mrr25_torch'],
                   fullval_top1=fv['top1'], fullval_top5=fv['top5'])
        log(f"[{tag}] 40k MRR@25={m['mrr25']:.4f} | FULLVAL MRR@25={fv['mrr25_from_scores']:.4f} "
            f"(torch {fv['mrr25_torch']:.4f}) top1={fv['top1']:.4f} @e{best['epoch']} ({stop})")
    else:
        log(f"[{tag}] 40k MRR@25={m['mrr25']:.4f} @e{best['epoch']} ({stop})")
    if state_store is not None:
        state_store.append([row.get('fullval_mrr25', m['mrr25']), tag,
                            copy.deepcopy(best['state']), dict(cfg), dict(hp)])
        state_store.sort(key=lambda s: -s[0])
        del state_store[5:]
    return row, model, best


def prepare_40k():
    z = np.load(f'{FULLOUT}/labels.npz', allow_pickle=True)
    labels, good, is_train_all = z['labels'], z['good'], z['is_train']
    emb = np.load(f'{FULLOUT}/embeddings.npy', mmap_mode='r')
    y_g = labels[good].astype(np.int64)
    tr_g = is_train_all[good]
    rng = np.random.default_rng(base.SEED)
    idx40 = rng.choice(len(good), 40000, replace=False)
    m40 = np.zeros(len(good), dtype=bool)
    m40[idx40] = True
    tr40 = m40 & tr_g
    va40 = m40 & ~tr_g
    def T(mask):
        return torch.from_numpy(np.asarray(emb[mask])), torch.from_numpy(y_g[mask])
    Xtr, ytr = T(tr40)
    Xva, yva = T(va40)
    Xfull, yfull = T(~tr_g)
    log('40k arm: train/val40k/fullval', len(ytr), '/', len(yva), '/', len(yfull),
        '| classes', int(y_g.max()) + 1)
    return Xtr, ytr, Xva, yva, Xfull, yfull


def ensemble_eval(members, X, y, label):
    probs = torch.zeros(len(y), int(y.max()) + 1)
    for _, _, state, cfg, _ in members:
        m = v2.Head(X.shape[1], int(y.max()) + 1, cfg)
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            probs += F.softmax(m(X), dim=1)
    preds = probs.argmax(1)
    y_np = y.numpy()
    mrr_scores = mrr25_eval(probs.numpy(), y_np)
    ens = dict(mrr25_from_scores=mrr_scores,
               mrr25_torch=v2.mrr25_from_logits(probs, y),
               top1=float((preds == y).float().mean()),
               top5=float((probs.topk(5, 1).indices == y.unsqueeze(1)).any(1).float().mean()),
               macro_f1=float(f1_score(y_np, preds.numpy(), average='macro', zero_division=0)),
               balanced_acc=float(balanced_accuracy_score(y_np, preds.numpy())))
    log(f"ENSEMBLE[{label}] MRR@25(scores)={ens['mrr25_from_scores']:.4f} "
        f"top1={ens['top1']:.4f} top5={ens['top5']:.4f} members={[t[1] for t in members]}")
    return ens


def unfreeze_probe(Xva, yva, n_probe=3000, epochs=3):
    from transformers import AutoTokenizer
    import train_smiles_detector as base_train
    df = base_train.load_dataset()
    z = np.load(f'{FULLOUT}/labels.npz', allow_pickle=True)
    labels, good = z['labels'], z['good']
    is_train_all = z['is_train']
    y_g = labels[good].astype(np.int64)
    rng = np.random.default_rng(base.SEED + 7)
    idx40 = rng.choice(len(good), 40000, replace=False)
    tr_pool = idx40[is_train_all[good][idx40]]
    pick = rng.choice(tr_pool, n_probe, replace=False)
    smiles = df.smiles.values[good][pick]
    y = torch.from_numpy(y_g[pick])
    tok = AutoTokenizer.from_pretrained(base_train.BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(base_train.BACKBONE)
    for p in bb.parameters():
        p.requires_grad_(False)
    layers = bb.roberta.encoder.layer
    for layer in layers[-2:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    head = v2.Head(bb.config.hidden_size, int(y.max()) + 1, HEAD_CFG)
    opt = torch.optim.AdamW([{'params': head.parameters(), 'lr': 1e-3},
                             {'params': layers[-2].parameters(), 'lr': 2e-5},
                             {'params': layers[-1].parameters(), 'lr': 2e-5}], weight_decay=1e-3)
    bs = 256
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        bb.train()
        head.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            batch = tok(list(smiles[idx]), padding=True, truncation=True, max_length=128,
                        return_tensors='pt')
            out = bb(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            emb = (out * mask).sum(1).div(mask.sum(1).clamp(min=1))
            loss = F.cross_entropy(head(emb), y[idx], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(layers[-2:].parameters()), 1.0)
            opt.step()
        log(f'  [P_unfreeze] epoch {epoch} loss={float(loss.item()):.4f} {time.time() - t0:.0f}s')
    bb.eval()
    head.eval()
    logits = []
    with torch.no_grad():
        for s in range(0, len(Xva), 2048):
            logits.append(head(Xva[s:s + 2048]))
    logits = torch.cat(logits)
    mrr = v2.mrr25_from_logits(logits, yva)
    top1 = float((logits.argmax(1) == yva).float().mean())
    log(f'[P_unfreeze] val40k MRR@25={mrr:.4f} top1={top1:.4f} '
        f'(baseline frozen B_v11 ~0.7778/0.6525) {time.time() - t0:.0f}s')
    return dict(probe_mrr25=mrr, probe_top1=top1, n_probe=n_probe, epochs=epochs,
                unfrozen='last 2 ChemBERTa layers, lr 2e-5, head lr 1e-3')


def _load_done():
    if os.path.exists(f'{OUT3}/v3_grid.csv'):
        df = pd.read_csv(f'{OUT3}/v3_grid.csv')
        return df, set(df.tag.tolist())
    return pd.DataFrame(), set()


def _save(rows):
    pd.DataFrame(rows).to_csv(f'{OUT3}/v3_grid.csv', index=False)


def _best_row(df):
    return df.sort_values('fullval_mrr25', ascending=False).iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', default='all', choices=['h', 'i', 'j', 's', 'all'])
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT3, exist_ok=True)
    os.makedirs(CKPTS3, exist_ok=True)
    torch.set_num_threads(4)
    Xtr, ytr, Xva, yva, Xfull, yfull = prepare_40k()
    prev_df, done = _load_done()
    rows = prev_df.to_dict('records') if len(prev_df) else []
    store = []
    log('resuming with', len(done), 'completed variants:', sorted(done))

    if args.round in ('h', 'all'):
        log('=== V3 ROUND H: lr x wd grid on 2x2048 tanh (bs1024 ls0.05) ===')
        for hp in H_GRID:
            tag = f"H_lr{hp['lr']:g}_wd{hp['wd']:g}"
            if tag in done:
                continue
            row, _, _ = run_variant(tag, HEAD_CFG, hp, Xtr, ytr, Xva, yva, Xfull, yfull, store)
            rows.append(dict(round='H', **row)); _save(rows); gc.collect()

    if args.round in ('i', 'all'):
        best_hp = _best_row(pd.DataFrame(rows))
        log('=== V3 ROUND I: batch sweep at lr', best_hp.lr, 'wd', best_hp.wd, '===')
        for bs in (512, 2048):
            tag = f'I_bs{bs}'
            if tag in done:
                continue
            hp = dict(lr=float(best_hp.lr), wd=float(best_hp.wd), bs=bs, ls=0.05)
            row, _, _ = run_variant(tag, HEAD_CFG, hp, Xtr, ytr, Xva, yva, Xfull, yfull, store)
            rows.append(dict(round='I', **row)); _save(rows); gc.collect()

    if args.round in ('j', 'all'):
        best_hp = _best_row(pd.DataFrame(rows))
        log('=== V3 ROUND J: label-smoothing sweep at lr', best_hp.lr, 'wd', best_hp.wd, 'bs', best_hp.bs, '===')
        for ls in (0.0, 0.02, 0.1):
            tag = f'J_ls{ls:g}'
            if tag in done:
                continue
            hp = dict(lr=float(best_hp.lr), wd=float(best_hp.wd), bs=int(best_hp.bs), ls=ls)
            row, _, _ = run_variant(tag, HEAD_CFG, hp, Xtr, ytr, Xva, yva, Xfull, yfull, store)
            rows.append(dict(round='J', **row)); _save(rows); gc.collect()

    if args.round in ('s', 'all'):
        best = _best_row(pd.DataFrame(rows))
        hp = dict(lr=float(best.lr), wd=float(best.wd), bs=int(best.bs), ls=float(best.ls))
        log('=== V3 ROUND S: 3-seed ensemble of', best.tag, hp, '===')
        have_best = any(s[1] == best.tag for s in store)
        if not have_best:
            row, _, _ = run_variant(f'S_{best.tag}_base', HEAD_CFG, hp,
                                    Xtr, ytr, Xva, yva, Xfull, yfull, store)
            rows.append(dict(round='S', **row)); _save(rows); gc.collect()
        for seed_tag in ('seed43', 'seed44'):
            tag = f'S_{best.tag}_{seed_tag}'
            if tag in done:
                continue
            row, _, _ = run_variant(tag, HEAD_CFG, hp, Xtr, ytr, Xva, yva, Xfull, yfull, store)
            rows.append(dict(round='S', **row)); _save(rows); gc.collect()
        if not store:
            log('no in-memory states to ensemble; rerun --round s in the same session as H/I/J')
        else:
            seed_members = [s for s in store if s[1] == best.tag or s[1].startswith(f'S_{best.tag}')]
            seed_ens = ensemble_eval(seed_members, Xfull, yfull, '3seed_fullval') if seed_members else {}
            top3 = store[:3]
            top3_ens = ensemble_eval(top3, Xfull, yfull, 'top3_fullval')
            torch.save(dict(states=[s[2] for s in top3], tags=[s[1] for s in top3],
                            configs=[s[3] for s in top3], hparams=[s[4] for s in top3],
                            metrics=top3_ens, selection_metric='mrr25_from_scores (evaluator)'),
                       f'{CKPTS3}/V3_TOP3_ENS.pt')
            best_m = store[0]
            torch.save(dict(state_dict=best_m[2], head_config=best_m[3], hparams=best_m[4],
                            fullval_mrr25=best_m[0], tag=best_m[1],
                            selection_metric='mrr25_from_scores (evaluator)'),
                       f'{CKPTS3}/V3_BEST.pt')
            summary = dict(best_tag=best_m[1], best_hp=best_m[4], best_fullval_mrr25=best_m[0],
                           seed_ensemble=seed_ens, top3_ensemble=top3_ens,
                           selection_metric='mrr25_from_scores (src/evaluation/mrr25.py)')
            with open(f'{OUT3}/v3_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            _save(rows)
            log('saved V3_BEST.pt / V3_TOP3_ENS.pt / v3_summary.json')

    if args.probe:
        log('=== V3 ROUND P: unfreeze last-2-layers probe ===')
        res = unfreeze_probe(Xva, yva)
        with open(f'{OUT3}/v3_probe.json', 'w') as f:
            json.dump(res, f, indent=2)

    log('V3 DONE')


if __name__ == '__main__':
    main()
