"""Run-3 experiment 1 prep: full-data labels + frozen ChemBERTa embeddings for ALL 275k structures.

- labels: MiniBatchKMeans (64 clusters) over Morgan-2048 fps, fit on the FULL train fold
  via partial_fit in chunks (RAM-safe), shared taxonomy for both run-3 arms
- embeddings: frozen DeepChem/ChemBERTa-100M-MLM, mean-pooled, dynamic per-batch padding
  (pads are attention-masked, so mathematically identical to max_length padding)

Outputs (outputs/smiles_detector_full/):
  embeddings.npy   float32 (n_good, 768) .npy memmap, resumable via embed_progress.json
  labels.npz       labels, good, is_train, smiles, km_centers
  prep_meta.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel, AutoTokenizer

RDLogger.DisableLog('rdApp.*')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import train_smiles_detector as base

OUT = 'outputs/smiles_detector_full'
SEED = base.SEED
N_CLUSTERS = base.N_CLUSTERS
BACKBONE = base.BACKBONE
MAX_LEN = 128
EMB_BATCH = 128
FP_CHUNK = 4096
KM_CHUNK = 16384


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_full():
    df = pd.read_parquet(base.FOLDS, columns=['inchikey14', 'smiles', 'fold_identity', 'formula', 'n_spectra'])
    df = df.dropna(subset=['smiles']).drop_duplicates('inchikey14').reset_index(drop=True)
    df = df[df.smiles.str.len().between(3, 200)].reset_index(drop=True)
    log('dataset', len(df), 'structures | fold0 (val):', int((df.fold_identity == 0).sum()))
    return df


def cluster_full(smiles, is_train):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    n = len(smiles)
    fps = np.zeros((n, 2048), dtype=np.uint8)
    ok = np.ones(n, dtype=bool)
    t0 = time.time()
    for s in range(0, n, FP_CHUNK):
        for j, smi in enumerate(smiles[s:s + FP_CHUNK]):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                ok[s + j] = False
            else:
                fps[s + j] = gen.GetFingerprintAsNumPy(m)
        if (s // FP_CHUNK) % 20 == 0:
            log(f'  fps {s + FP_CHUNK}/{n} {time.time() - t0:.0f}s')
    np.random.seed(SEED)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=2048, random_state=SEED, n_init=5)
    trX = fps[is_train]
    for s in range(0, len(trX), KM_CHUNK):
        km.partial_fit(trX[s:s + KM_CHUNK].astype(np.float32))
    for _ in range(2):
        for s in range(0, len(trX), KM_CHUNK):
            km.partial_fit(trX[s:s + KM_CHUNK].astype(np.float32))
    labels = np.full(n, -1, dtype=np.int16)
    for s in range(0, n, KM_CHUNK):
        labels[s:s + KM_CHUNK] = km.predict(fps[s:s + KM_CHUNK].astype(np.float32))
    labels = np.where(ok, labels, -1).astype(np.int16)
    log('clusters', N_CLUSTERS, '| usable', int((labels >= 0).sum()),
        '| dropped', int((labels < 0).sum()), f'{time.time() - t0:.0f}s')
    return labels, ok, km


def embed_full(smiles, emb_path, progress_path):
    n = len(smiles)
    done = 0
    if os.path.exists(progress_path) and os.path.exists(emb_path):
        done = json.load(open(progress_path))['done']
        embs = np.lib.format.open_memmap(emb_path, mode='r+', dtype=np.float32, shape=(n, 768))
        if done >= n:
            log('embeddings already complete', emb_path, embs.shape)
            return embs
        log(f'resuming embeddings from {done}/{n}')
    else:
        embs = np.lib.format.open_memmap(emb_path, mode='w+', dtype=np.float32, shape=(n, 768))
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(BACKBONE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    t0 = time.time()
    with torch.no_grad():
        for s in range(done, n, EMB_BATCH):
            batch = tok(list(smiles[s:s + EMB_BATCH]), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors='pt')
            out = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            embs[s:s + EMB_BATCH] = (out * mask).sum(1).div(mask.sum(1).clamp(min=1)).numpy()
            if (s // EMB_BATCH) % 100 == 0:
                el = time.time() - t0
                rate = max(1e-6, (s + EMB_BATCH - done) / el)
                eta_min = (n - s - EMB_BATCH) / rate / 60
                log(f'  {s + EMB_BATCH}/{n} {el:.0f}s eta {eta_min:.0f}min')
                embs.flush()
                json.dump({'done': s + EMB_BATCH}, open(progress_path, 'w'))
    json.dump({'done': n}, open(progress_path, 'w'))
    embs.flush()
    log('embeddings done', (n, 768), f'{time.time() - t0:.0f}s')
    return embs


def main():
    os.makedirs(OUT, exist_ok=True)
    base.set_seed()
    torch.set_num_threads(int(os.environ.get('EMB_THREADS', '6')))
    df = load_full()
    is_train = (df.fold_identity.values != 0)
    labels_path = f'{OUT}/labels.npz'
    if os.path.exists(labels_path):
        z = np.load(labels_path, allow_pickle=True)
        labels, good = z['labels'], z['good']
        log('labels loaded from cache', labels_path)
    else:
        labels, _, km = cluster_full(df.smiles.values, is_train)
        good = labels >= 0
        np.savez_compressed(labels_path, labels=labels, good=good, is_train=is_train,
                            smiles=df.smiles.values.astype('U200'),
                            km_centers=km.cluster_centers_.astype(np.float32))
        log('saved', labels_path)
    smi_g = df.smiles.values[good]
    emb = embed_full(smi_g.tolist(), f'{OUT}/embeddings.npy', f'{OUT}/embed_progress.json')
    tr = is_train[good]
    y = labels[good].astype(np.int64)
    log('train/val:', int(tr.sum()), '/', int((~tr).sum()), '| classes:', int(y.max()) + 1)
    meta = dict(n=int(good.sum()), n_train=int(tr.sum()), n_val=int((~tr).sum()),
                n_clusters=N_CLUSTERS, backbone=BACKBONE, seed=SEED,
                embedding='frozen ChemBERTa-100M-MLM mean-pool, dynamic padding',
                labels='MiniBatchKMeans 64 on Morgan-2048, fit on full train fold',
                metric='src/evaluation/mrr25.py (mrr25_from_scores)')
    with open(f'{OUT}/prep_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    log('PREP DONE')


if __name__ == '__main__':
    main()
