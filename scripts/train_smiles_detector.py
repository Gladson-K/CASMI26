"""SMILES 2D molecule-class detection with a frozen pretrained ChemBERTa backbone.

Stages:
  1  config search   - train several head architectures for 3 epochs each, compare val accuracy
  2  full train      - best config, up to 50 epochs, early stop after 3 straight val-acc drops,
                       best weights saved as a separate checkpoint
  3  hyperparameter  - vary lr / weight decay / batch size for the winning config, keep the best
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel, AutoTokenizer

RDLogger.DisableLog('rdApp.*')

OUT = 'outputs/smiles_detector'
CKPTS = f'{OUT}/checkpoints'
SEED = 42
N_STRUCTS = 40000
N_CLUSTERS = 64
BACKBONE = 'DeepChem/ChemBERTa-100M-MLM'
MAX_LEN = 128
EMB_BATCH = 64
FOLDS = 'data/benchmark/casmi26-identity-disjoint-validation-folds/structures.parquet'

HEAD_CONFIGS = {
    'linear':        dict(arch='linear', hidden=[], act='relu', dropout=0.0),
    'mlp_1x256':     dict(arch='mlp', hidden=[256], act='relu', dropout=0.0),
    'mlp_2x512':     dict(arch='mlp', hidden=[512, 256], act='relu', dropout=0.0),
    'mlp_2x512_drop': dict(arch='mlp', hidden=[512, 256], act='relu', dropout=0.3),
    'mlp_2x1024_tanh': dict(arch='mlp', hidden=[1024, 512], act='tanh', dropout=0.0),
}


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class Head(nn.Module):
    def __init__(self, din, n_classes, cfg):
        super().__init__()
        layers = []
        d = din
        act = nn.ReLU if cfg['act'] == 'relu' else nn.Tanh
        for h in cfg['hidden']:
            layers += [nn.Linear(d, h), act()]
            if cfg['dropout'] > 0:
                layers.append(nn.Dropout(cfg['dropout']))
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_dataset():
    df = pd.read_parquet(FOLDS, columns=['inchikey14', 'smiles', 'fold_identity', 'formula', 'n_spectra'])
    df = df.dropna(subset=['smiles']).drop_duplicates('inchikey14').reset_index(drop=True)
    df = df[df.smiles.str.len().between(3, 200)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    if len(df) > N_STRUCTS:
        df = df.iloc[rng.choice(len(df), N_STRUCTS, replace=False)].reset_index(drop=True)
    log('dataset', len(df), 'structures | fold0 (val):', int((df.fold_identity == 0).sum()))
    return df


def cluster_labels(smiles, is_train):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False); fps.append(np.zeros(2048, dtype=np.uint8))
        else:
            ok.append(True)
            fps.append(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
    X = np.asarray(fps, dtype=np.float32)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=2048, n_init=5)
    km.fit(X[is_train])
    labels = km.predict(X).astype(np.int64)
    labels = np.where(np.asarray(ok), labels, -1)
    good = labels >= 0
    log('clusters', N_CLUSTERS, '| usable', int(good.sum()), '| dropped', int((~good).sum()))
    return labels, good, X


def embed_smiles(smiles, cache_path):
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if len(z['smiles']) == len(smiles) and (z['smiles'] == np.asarray(smiles, dtype=object)).all():
            log('embeddings loaded from cache', cache_path, z['emb'].shape)
            return z['emb']
    log('tokenising + embedding', len(smiles), 'SMILES with frozen', BACKBONE)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(BACKBONE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    embs = np.zeros((len(smiles), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(smiles), EMB_BATCH):
            batch = tok(list(smiles[s:s + EMB_BATCH]), padding='max_length', truncation=True,
                        max_length=MAX_LEN, return_tensors='pt')
            out = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            embs[s:s + EMB_BATCH] = (out * mask).sum(1).div(mask.sum(1).clamp(min=1)).numpy()
            if (s // EMB_BATCH) % 100 == 0:
                log(f'  {s + EMB_BATCH}/{len(smiles)} {(time.time() - t0):.0f}s')
    np.savez_compressed(cache_path, smiles=np.asarray(smiles, dtype=object), emb=embs)
    log('embeddings done', embs.shape, f'{time.time() - t0:.0f}s')
    return embs


def acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def run_head(cfg, hparams, Xtr, ytr, Xva, yva, max_epochs, tag):
    model = Head(Xtr.shape[1], int(ytr.max()) + 1, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=hparams['lr'],
                            weight_decay=hparams['weight_decay'])
    bs = hparams['batch']
    n = len(ytr)
    history, best = [], dict(acc=-1.0, state=None, epoch=-1)
    consec_drops = 0
    stop_reason = 'max_epochs'
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = acc(model(Xva), yva)
        history.append(dict(epoch=epoch, val_acc=va, loss=float(loss.item())))
        if va > best['acc']:
            best = dict(acc=va, state=copy.deepcopy(model.state_dict()), epoch=epoch)
            consec_drops = 0
        elif epoch > 1 and va < history[-2]['val_acc']:
            consec_drops += 1
            if consec_drops >= 3:
                stop_reason = 'early_stop_3_consecutive_acc_drops'
                log(f'[{tag}] early stop at epoch {epoch} (val acc dropped {consec_drops}x in a row)')
                break
        else:
            consec_drops = 0
    model.load_state_dict(best['state'])
    return model, history, best, stop_reason


def tensorize(emb, labels, tr_mask):
    X = torch.from_numpy(emb)
    y = torch.from_numpy(labels.astype(np.int64))
    return X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask]


def save_ckpt(path, model, cfg, hparams, meta):
    torch.save(dict(state_dict=model.state_dict(), head_config=cfg, hparams=hparams, **meta), path)
    log('saved', path)


def stage1(Xtr, ytr, Xva, yva):
    log('=== STAGE 1: 3-epoch algorithm search over', len(HEAD_CONFIGS), 'head configs ===')
    rows = []
    for name, cfg in HEAD_CONFIGS.items():
        set_seed()
        t0 = time.time()
        _, hist, best, _ = run_head(cfg, dict(lr=1e-3, weight_decay=1e-4, batch_size=512, batch=512),
                                    Xtr, ytr, Xva, yva, max_epochs=3, tag=name)
        row = dict(config=name, val_acc_epoch3=hist[-1]['val_acc'], best_val_acc=best['acc'],
                   best_epoch=best['epoch'], secs=round(time.time() - t0, 1),
                   accs=' '.join(f"{h['val_acc']:.4f}" for h in hist))
        rows.append(row)
        log(f"  {name:18s} accs=[{row['accs']}] best={best['acc']:.4f}")
    df = pd.DataFrame(rows).sort_values('best_val_acc', ascending=False)
    df.to_csv(f'{OUT}/stage1_config_search.csv', index=False)
    winner = df.iloc[0].config
    log('STAGE 1 RESULT (ranked):')
    print(df.to_string(index=False))
    log('winner:', winner)
    return winner, df


def stage2(winner, Xtr, ytr, Xva, yva):
    log('=== STAGE 2: best config', winner, 'up to 50 epochs, stop after 3 straight acc drops ===')
    set_seed()
    cfg = HEAD_CONFIGS[winner]
    hp = dict(lr=1e-3, weight_decay=1e-4, batch=512)
    t0 = time.time()
    model, hist, best, stop_reason = run_head(cfg, hp, Xtr, ytr, Xva, yva, max_epochs=50, tag=winner)
    log(f'stop_reason={stop_reason} | best val acc {best["acc"]:.4f} @ epoch {best["epoch"]} | {time.time() - t0:.0f}s')
    pd.DataFrame(hist).to_csv(f'{OUT}/stage2_history_{winner}.csv', index=False)
    save_ckpt(f'{CKPTS}/stage2_{winner}_best.pt', model, cfg, hp,
              dict(val_acc=best['acc'], best_epoch=best['epoch'], stop_reason=stop_reason,
                   backbone=BACKBONE, n_clusters=N_CLUSTERS, stage='2_full_train'))
    final_model = Head(Xtr.shape[1], int(ytr.max()) + 1, cfg)
    final_model.load_state_dict(best['state'])
    save_ckpt(f'{CKPTS}/stage2_{winner}_early_stopped_SEPARATE.pt', final_model, cfg, hp,
              dict(val_acc=best['acc'], best_epoch=best['epoch'], stop_reason=stop_reason,
                   backbone=BACKBONE, n_clusters=N_CLUSTERS, stage='2_early_stopped_separate'))
    tail = ' '.join(f"e{h['epoch']}={h['val_acc']:.4f}" for h in hist[-10:])
    log('last epochs:', tail)
    return best['acc'], hist


def stage3(winner, Xtr, ytr, Xva, yva):
    log('=== STAGE 3: hyperparameter search on config', winner, '===')
    cfg = HEAD_CONFIGS[winner]
    variants = [
        dict(lr=3e-4, weight_decay=1e-4, batch=512),
        dict(lr=1e-3, weight_decay=0.0, batch=512),
        dict(lr=1e-3, weight_decay=1e-2, batch=512),
        dict(lr=3e-3, weight_decay=1e-4, batch=512),
        dict(lr=1e-3, weight_decay=1e-4, batch=256),
        dict(lr=3e-4, weight_decay=1e-2, batch=256),
        dict(lr=3e-3, weight_decay=0.0, batch=256),
        dict(lr=5e-4, weight_decay=1e-3, batch=1024),
    ]
    rows, best = [], dict(acc=-1.0)
    for i, hp in enumerate(variants, 1):
        set_seed(100 + i)
        t0 = time.time()
        model, hist, bst, stop_reason = run_head(cfg, hp, Xtr, ytr, Xva, yva, max_epochs=50,
                                                tag=f'hpo{i}')
        row = dict(variant=i, **{k: hp[k] for k in ('lr', 'weight_decay', 'batch')},
                   best_val_acc=bst['acc'], best_epoch=bst['epoch'], epochs_run=len(hist),
                   stop_reason=stop_reason, secs=round(time.time() - t0, 1))
        rows.append(row)
        log(f"  v{i} lr={hp['lr']:.0e} wd={hp['weight_decay']:.0e} bs={hp['batch']:4d} "
            f"-> best {bst['acc']:.4f} @ e{bst['epoch']} ({stop_reason})")
        if bst['acc'] > best['acc']:
            best = dict(acc=bst['acc'], epoch=bst['epoch'], hp=hp, state=copy.deepcopy(bst['state']),
                        stop_reason=stop_reason, epochs_run=len(hist))
    df = pd.DataFrame(rows).sort_values('best_val_acc', ascending=False)
    df.to_csv(f'{OUT}/stage3_hpo_results.csv', index=False)
    log('STAGE 3 RESULT (ranked):')
    print(df.to_string(index=False))
    final = Head(Xtr.shape[1], int(ytr.max()) + 1, cfg)
    final.load_state_dict(best['state'])
    save_ckpt(f'{CKPTS}/BEST_MODEL_final.pt', final, cfg, best['hp'],
              dict(val_acc=best['acc'], best_epoch=best['epoch'], stop_reason=best['stop_reason'],
                   backbone=BACKBONE, n_clusters=N_CLUSTERS, stage='3_hpo_best',
                   note='frozen ChemBERTa-100M-MLM backbone; this head is the trained model'))
    with open(f'{OUT}/best_config.json', 'w') as f:
        json.dump(dict(head_config=winner, arch=cfg, hparams=best['hp'], val_acc=best['acc'],
                       best_epoch=best['epoch'], stop_reason=best['stop_reason'],
                       backbone=BACKBONE, n_clusters=N_CLUSTERS, seed=SEED), f, indent=2)
    log(f"BEST MODEL: val_acc={best['acc']:.4f} hp={best['hp']} -> {CKPTS}/BEST_MODEL_final.pt")
    return df, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['1', '2', '3', 'all'])
    ap.add_argument('--winner', default='')
    args = ap.parse_args()
    os.makedirs(CKPTS, exist_ok=True)
    set_seed()
    torch.set_num_threads(4)
    df = load_dataset()
    is_train = (df.fold_identity.values != 0)
    labels, good, _ = cluster_labels(df.smiles.values, is_train)
    m = good
    emb = embed_smiles(df.smiles.values[m].tolist(), f'{OUT}/embeddings.npz')
    y = labels[m]
    tr = is_train[m]
    log('train/val:', int(tr.sum()), '/', int((~tr).sum()), '| classes:', int(y.max()) + 1)
    Xtr, ytr, Xva, yva = tensorize(emb, y, tr)
    winner = args.winner
    if args.stage in ('1', 'all'):
        winner, _ = stage1(Xtr, ytr, Xva, yva)
    if args.stage in ('2', 'all'):
        if not winner:
            raise SystemExit('--winner needed for stage 2 (config name from stage 1)')
        stage2(winner, Xtr, ytr, Xva, yva)
    if args.stage in ('3', 'all'):
        if not winner:
            raise SystemExit('--winner needed for stage 3')
        stage3(winner, Xtr, ytr, Xva, yva)
    log('DONE')


if __name__ == '__main__':
    main()
