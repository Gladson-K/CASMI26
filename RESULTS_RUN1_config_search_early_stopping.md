# Run 1 — Config Search, Early Stopping and Hyperparameter Tuning (SMILES 2D Molecule Detection)

**Date:** 2026-10-19
**Script:** `scripts/train_smiles_detector.py`
**Artifacts:** `outputs/smiles_detector/`

## 1. Task and setup

* **Task** — SMILES 2D molecule-class detection: classify a molecule from its SMILES string into one of
  **64 chemotype clusters** (MiniBatchKMeans over Morgan-2048 fingerprints, fit on the training folds only).
* **Data** — 40,000 unique structures from `data/benchmark/casmi26-identity-disjoint-validation-folds/structures.parquet`.
  Validation is **identity-disjoint** (`fold_identity == 0`): 32,018 train / 7,982 val molecules, no identity overlap.
* **Backbone (frozen)** — `DeepChem/ChemBERTa-100M-MLM` (12 layers, hidden 768, 92.1M params), `requires_grad=False`.
  Mean-pooled embeddings are computed once and cached in `outputs/smiles_detector/embeddings.npz` (40,000 × 768,
  ~115 MB; one-time cost ≈ 105 min on CPU).
* **Trainable part** — head on top of the frozen 768-d embedding.
* **Protocol** — seed 42, AdamW, max 50 epochs, batch 512, early stop = **validation accuracy drops in 3
  consecutive epochs**, best-epoch weights always kept.
* **Metric (run 1)** — validation accuracy (chance level = 1/64 = 1.56 %).

## 2. Stage 1 — 3-epoch algorithm search over 5 head configs

Each config trained for exactly 3 epochs, then compared on val accuracy:

| Rank | Config | Architecture | Acc e1 → e2 → e3 | Best val acc | Time |
|---:|---|---|---|---:|---:|
| 1 | **mlp_2x1024_tanh** | 1024-512, tanh | 0.4102 → 0.5005 → 0.5381 | **0.5381** | 7.5 s |
| 2 | mlp_2x512 | 512-256, ReLU | 0.3290 → 0.4318 → 0.4739 | 0.4739 | 3.2 s |
| 3 | mlp_2x512_drop | 512-256, ReLU, drop 0.3 | 0.3044 → 0.4025 → 0.4608 | 0.4608 | 4.6 s |
| 4 | mlp_1x256 | 256, ReLU | 0.3183 → 0.4203 → 0.4599 | 0.4599 | 1.5 s |
| 5 | linear | single layer | 0.3132 → 0.3815 → 0.4250 | 0.4250 | 2.3 s |

**Winner: `mlp_2x1024_tanh`** — +6.4 points over the runner-up. Wider layers, tanh activation and no
dropout performed best on frozen features.

Source: `outputs/smiles_detector/stage1_config_search.csv`

## 3. Stage 2 — 50-epoch run of the winning config with early stopping

Config `mlp_2x1024_tanh`, lr 1e-3, wd 1e-4, batch 512, up to 50 epochs.

* Val accuracy rose 0.410 → 0.639 by epoch 19.
* Epochs 20-22 declined three times in a row (0.6393 → 0.6379 → 0.6368 → 0.6270) → **early stop at epoch 22**.
* **Best val accuracy 0.6393 @ epoch 19.**

| Epoch | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val acc | 0.6272 | 0.6294 | 0.6292 | 0.6308 | **0.6393** | 0.6379 | 0.6368 | 0.6270 ← stop |

Checkpoints saved:

| File | Content |
|---|---|
| `checkpoints/stage2_mlp_2x1024_tanh_best.pt` | best-epoch weights (0.6393 @ e19) |
| `checkpoints/stage2_mlp_2x1024_tanh_early_stopped_SEPARATE.pt` | separate checkpoint written when the 3-drop rule fired |

Source: `outputs/smiles_detector/stage2_history_mlp_2x1024_tanh.csv`

## 4. Stage 3 — hyperparameter search on the same config (8 variants)

Same architecture, constant lr AdamW, max 50 epochs, same early-stop rule:

| Rank | lr | weight decay | batch | Best val acc | Best ep | Stop reason |
|---|---:|---:|---:|---:|---:|---|
| 1 | **5e-4** | **1e-3** | **1024** | **0.6595** | 42 | ran 50 epochs |
| 2 | 3e-4 | 1e-2 | 256 | 0.6562 | 47 | ran 50 epochs |
| 3 | 1e-3 | 0 | 512 | 0.6542 | 49 | ran 50 epochs |
| 4 | 3e-4 | 1e-4 | 512 | 0.6445 | 24 | early stop (e29) |
| 5 | 1e-3 | 1e-4 | 256 | 0.6324 | 22 | early stop (e25) |
| 6 | 1e-3 | 1e-2 | 512 | 0.6307 | 12 | early stop (e17) |
| 7 | 3e-3 | 1e-4 | 512 | 0.6265 | 35 | early stop (e41) |
| 8 | 3e-3 | 0 | 256 | 0.5655 | 25 | ran 50 epochs |

**Best hyperparameters: lr 5e-4, weight decay 1e-3, batch 1024 → val accuracy 0.6595.**
Moderate learning rate with weight decay and a large batch won; lr 3e-3 was clearly too high.

Final model: `checkpoints/BEST_MODEL_final.pt` (config in `outputs/smiles_detector/best_config.json`)

## 5. Accuracy progression (run 1)

| Checkpoint of the run | Val accuracy |
|---|---:|
| Chance level (64 classes) | 0.0156 |
| 3-epoch config search winner | 0.5381 |
| 50-epoch run + early stopping | 0.6393 |
| Hyperparameter-tuned best | **0.6595** |

## 6. Artifacts

```
outputs/smiles_detector/
├── embeddings.npz                       # frozen ChemBERTa embeddings (cache)
├── stage1_config_search.csv             # 3-epoch config comparison
├── stage2_history_mlp_2x1024_tanh.csv   # per-epoch history (acc, loss)
├── stage3_hpo_results.csv               # 8 hyperparameter variants
├── best_config.json                     # winning config + hyperparameters
└── checkpoints/
    ├── stage2_mlp_2x1024_tanh_best.pt
    ├── stage2_mlp_2x1024_tanh_early_stopped_SEPARATE.pt
    └── BEST_MODEL_final.pt              # run-1 best model (0.6595)
```

## 7. Notes and limitations

* Metrics in run 1 were accuracy-only during training; run 2 makes the competition-style MRR@25
  (see section 8 below and `RESULTS_RUN2_hyperparameter_tuning.md`) the primary selection metric and adds
  top-5 accuracy, macro-F1 and balanced accuracy.
* Subsampled, CPU-only run (40k molecules) chosen for speed; scale to the full 277k structures for production.
* Labels are unsupervised chemotype clusters; validation identities are held out, so the numbers are a honest
  check of generalisation to unseen molecules — not the competition MRR@25 metric.

## 8. Re-check of run-1 checkpoints with the 1/25-sum accuracy (MRR@25)

After run 1 finished, the saved checkpoints were re-evaluated with the repository's competition accuracy
logic `mrr_at_25` (`src/nas/runner.py`): for each query, `1/rank` if the true class ranks in the top 25,
else 0, averaged over all validation molecules (`scripts/eval_checkpoint_metrics.py`).

| Checkpoint | MRR@25 (1/25 sum) | Top-1 acc | Top-5 acc | Macro-F1 | Balanced acc |
|---|---:|---:|---:|---:|---:|
| `stage2_..._best.pt` (50-epoch + early stop) | 0.7696 | 0.6393 | 0.9419 | 0.6145 | 0.6279 |
| `stage2_..._early_stopped_SEPARATE.pt` | 0.7696 | 0.6393 | 0.9419 | 0.6145 | 0.6279 |
| `BEST_MODEL_final.pt` (HPO best, lr 5e-4/wd 1e-3/bs 1024) | **0.7830** | **0.6595** | **0.9473** | **0.6520** | **0.6534** |

Run-1 accuracy progression under the 1/25-sum metric: 0.7696 (early-stopped model) → **0.7830**
(final hyperparameter-tuned model). Full table: `outputs/smiles_detector/checkpoint_metrics.csv`.

Note: run 1 selected models by top-1 accuracy; these MRR@25 values are a post-hoc re-check.
Run 2 selects directly by MRR@25 — see `RESULTS_RUN2_hyperparameter_tuning.md`.
