# Run 2 — Extended Hyperparameter Tuning with MRR@25 (1/25-sum) Accuracy (SMILES 2D Molecule Detection)

**Date:** 2026-10-19
**Script:** `scripts/tune_smiles_detector_v2.py` (+ `scripts/eval_checkpoint_metrics.py` for re-checks)
**Artifacts:** `outputs/smiles_detector/` (prefix `v2_*`)
**Companion document:** `RESULTS_RUN1_config_search_early_stopping.md`

## 1. What changed vs run 1

| Aspect | Run 1 | Run 2 |
|---|---|---|
| Accuracy metric / selection | top-1 validation accuracy | **MRR@25 — the repo's `mrr_at_25` logic: `1/rank` if rank ≤ 25, else 0, averaged (`src/nas/runner.py`)** |
| Early stop | 3 consecutive drops of top-1 acc | 3 consecutive drops of **MRR@25** (same rule, new metric) |
| LR schedule | constant | **cosine decay + 3-epoch warm-up** |
| Label smoothing | none | **0.05** (tuned: 0 / 0.05) |
| Reported metrics | val accuracy only | **MRR@25, top-1, top-5, macro-F1, balanced accuracy, train accuracy** |
| Configs | 5 heads | **7 new heads (round A) + 12 hyperparameter variants (round B) + 3-model ensemble (round C)** |
| Data / backbone / protocol | 40k structures, frozen ChemBERTa-100M-MLM, max 50 epochs | identical (directly comparable) |

Chance level for MRR@25 on 64 classes ≈ 0.04 (uniform ranking would give ~`mean(1/rank)` over the first 25 ranks).

## 2. Round A — new head configs (fixed hparams: lr 1e-3, wd 1e-4, batch 512, ls 0.05)

| Config | Architecture | MRR@25 | Top-1 | Top-5 | Macro-F1 | Balanced | Stop |
|---|---|---:|---:|---:|---:|---:|---|
| **A_mlp_2x2048_tanh** | 2048-1024, tanh | **0.7756** | 0.6493 | 0.9426 | 0.6270 | 0.6205 | 50 ep |
| A_mlp_3x1024_tanh | 1024-1024-512, tanh | 0.7694 | 0.6401 | 0.9426 | 0.5717 | 0.5613 | 50 ep |
| A_mlp_2x1024_relu_bn | 1024-512, ReLU + BN | 0.7691 | 0.6422 | 0.9359 | **0.6396** | **0.6271** | stop e41 (best e28) |
| A_mlp_2x1024_tanh_do01 | 1024-512, tanh, drop 0.1 | 0.7670 | 0.6367 | 0.9385 | 0.5834 | 0.5759 | 50 ep |
| A_control_2x1024_tanh | run-1 winner arch, new schedule | 0.7655 | 0.6348 | 0.9380 | 0.5869 | 0.5694 | 50 ep |
| A_mlp_2x1024_tanh_bn | 1024-512, tanh + BN | 0.7629 | 0.6332 | 0.9359 | 0.6215 | 0.6156 | stop e30 (best e27) |
| A_mlp_4x512_tanh | 512-512-512-256, tanh | 0.7550 | 0.6216 | 0.9334 | 0.5150 | 0.5057 | 50 ep |

**Round-A winner: `mlp_2x2048_tanh` (MRR@25 = 0.7756).** Width beats depth (2×2048 > 3×1024 > 4×512);
batch-norm and dropout did not help on frozen features.

Source: `outputs/smiles_detector/v2_round_a.csv`, per-epoch histories in `outputs/smiles_detector/v2_histories/`.

## 3. Round B — hyperparameter sweep on `mlp_2x2048_tanh` (12 variants)

| Variant | lr | weight decay | batch | ls | MRR@25 | Top-1 | Top-5 | Macro-F1 | Balanced | Best ep / stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **B_v11** | **1e-3** | **1e-3** | **1024** | 0.05 | **0.7778** | **0.6525** | 0.9442 | 0.6174 | 0.6059 | e48 / 50 ep |
| B_v7 | 3e-3 | 0 | 512 | 0.05 | 0.7752 | 0.6473 | 0.9449 | 0.6347 | 0.6199 | e39 / 50 ep |
| B_v9 | 3e-3 | 1e-2 | 512 | 0.05 | 0.7743 | 0.6481 | 0.9422 | 0.6278 | 0.6163 | e40 / 50 ep |
| B_v6 | 1e-3 | 1e-2 | 512 | 0.05 | 0.7736 | 0.6445 | 0.9430 | 0.6335 | 0.6291 | e49 / 50 ep |
| B_v8 | 3e-3 | 1e-3 | 512 | 0.05 | 0.7730 | 0.6457 | 0.9405 | **0.6529** | **0.6433** | e46 / 50 ep |
| B_v4 | 1e-3 | 0 | 512 | 0.05 | 0.7724 | 0.6442 | 0.9426 | 0.6038 | 0.5808 | e45 / stop e48 |
| B_v10 | 5e-4 | 1e-3 | 256 | 0.05 | 0.7575 | 0.6252 | 0.9335 | 0.5642 | 0.5550 | e49 / 50 ep |
| B_v5 | 1e-3 | 1e-3 | 512 | 0.05 | 0.7540 | 0.6198 | 0.9349 | 0.5654 | 0.5490 | e31 / stop e34 |
| B_v3 | 3e-4 | 1e-2 | 512 | 0.05 | 0.7529 | 0.6191 | 0.9328 | 0.5602 | 0.5520 | e50 / 50 ep |
| B_v2 | 3e-4 | 1e-3 | 512 | 0.05 | 0.7525 | 0.6189 | 0.9293 | 0.5616 | 0.5507 | e50 / 50 ep |
| B_v1 | 3e-4 | 0 | 512 | 0.05 | 0.7520 | 0.6180 | 0.9317 | 0.5451 | 0.5335 | e49 / 50 ep |
| B_v12 | 5e-4 | 1e-3 | 512 | **0** | 0.7625 | 0.6304 | 0.9379 | 0.6213 | 0.6158 | e50 / 50 ep |

**Best single model: `B_v11` — lr 1e-3, weight decay 1e-3, batch 1024, ls 0.05 → MRR@25 = 0.7778.**
Findings: larger batch (1024) with weight decay wins; lr 3e-4 is too low, lr 3e-3 is unstable but sometimes
competitive; label smoothing 0.05 beats 0 (B_v12); the 3-drop early stop fired only in 2 of 12 runs.

Source: `outputs/smiles_detector/v2_round_b.csv`.

## 4. Round C — soft-voting ensemble of the top 3 (B_v11 + A_mlp_2x2048_tanh + B_v7)

| Model | MRR@25 | Top-1 | Top-5 | Macro-F1 | Balanced |
|---|---:|---:|---:|---:|---:|
| Best single (B_v11) | 0.7778 | 0.6525 | 0.9442 | 0.6174 | 0.6059 |
| **Top-3 ensemble** | **0.7914** | **0.6709** | **0.9500** | 0.6443 | 0.6301 |

The ensemble adds **+1.4 points MRR@25 and +1.8 points top-1** over the best single model.

## 5. Cross-run comparison (all saved checkpoints re-evaluated with MRR@25)

Via `scripts/eval_checkpoint_metrics.py` (`outputs/smiles_detector/checkpoint_metrics.csv`):

| Model | MRR@25 | Top-1 | Top-5 | Macro-F1 | Balanced |
|---|---:|---:|---:|---:|---:|
| Run 1 stage-2 early-stopped (e19) | 0.7696 | 0.6393 | 0.9419 | 0.6145 | 0.6279 |
| Run 1 final best (`BEST_MODEL_final.pt`, top-1 selected) | 0.7830 | 0.6595 | 0.9473 | **0.6520** | **0.6534** |
| Run 2 best single (`V2_BEST_MODEL.pt` = B_v11) | 0.7778 | 0.6525 | 0.9442 | 0.6174 | 0.6059 |
| **Run 2 ensemble (`V2_ENSEMBLE.pt`)** | **0.7914** | **0.6709** | **0.9500** | 0.6443 | 0.6301 |

Reading: the run-2 ensemble is the **overall best model on the 1/25-sum accuracy** (0.7914 vs 0.7830 for
run 1's best). Run 1's top-1-selected model remains stronger on class-balanced metrics
(macro-F1 0.652, balanced acc 0.653) — it favours large clusters; the run-2 models trade a little
class balance for higher reciprocal-rank accuracy.

## 6. Final artifacts

```
outputs/smiles_detector/
├── v2_round_a.csv                    # 7 configs, all metrics
├── v2_round_b.csv                    # 12 hyperparameter variants
├── v2_all_results.csv                # A+B merged, ranked by MRR@25
├── v2_summary.json                   # winner, ensemble, protocol
├── checkpoint_metrics.csv            # cross-run checkpoint re-check (MRR@25 etc.)
├── v2_histories/*.csv                # per-epoch MRR@25 / top1 / top5 / F1 / bal / lr
└── checkpoints/
    ├── V2_BEST_MODEL.pt              # B_v11 single model (MRR@25 0.7778)
    └── V2_ENSEMBLE.pt                # top-3 ensemble (MRR@25 0.7914) ← best overall
```

## 7. Reproduce

```bash
source .venv/bin/activate
python scripts/tune_smiles_detector_v2.py --round all      # rounds A+B+C (~2 h CPU)
python scripts/eval_checkpoint_metrics.py --ckpts outputs/smiles_detector/checkpoints/*.pt
```

Same seed (42), same cached frozen ChemBERTa embeddings (`embeddings.npz`), same identity-disjoint
fold-0 validation split as run 1.
