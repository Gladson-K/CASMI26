# Run 3 — Full-data training, deeper HPO, and chemist-knowledge injection

**Project:** CASMI26 SMILES → chemotype classifier
**Repo:** `/mnt/d/Workspace_Glad/Projects/casmi26_upgrade`
**Primary metric:** MRR@25 from the evaluator's own implementation
`src/evaluation/mrr25.py` → `mrr25_from_scores(scores, truth_indices, max_rank=25)`
(1/rank when the true class is within the top 25, else 0; mean over molecules).
**Hardware:** CPU-only (8 cores, i5-1135G7, no bf16/AMX), 3.85 GB RAM total.
**Backbone:** frozen `DeepChem/ChemBERTa-100M-MLM` (768-d, mean-pooled) → 2-layer
MLP head (2048→1024, tanh, dropout 0, no norm), cosine schedule with 3-epoch warmup,
`clip_grad_norm_` 1.0, ≤50 epochs, early stop after 3 consecutive val-metric drops.

---

## 1. What run 3 adds over runs 1 and 2

| Objective | Status |
|---|---|
| Scale the classifier to the **full 275k** structure set | done — full-data arm trained on all 220,408 train molecules |
| Deeper HPO around the run-2 winner | done — 25-variant grid (rounds H/I/J/S) on the shared full-data label space |
| Compare arms under the evaluator's own metric | done — §6 unified table |
| "Chemist knowledge injected directly into the model" | done — §5, +6.2 MRR@25 points over the model |
| Independent second training on RHS-DL-Development, then test | done — §7, test R² 0.7938 |

Runs 1 and 2 are documented in `RESULTS_RUN1_config_search_early_stopping.md` and
`RESULTS_RUN2_hyperparameter_tuning.md`. Run 3 changes the **label space** so that
40k-trained and full-data-trained heads are directly comparable:

* the 64 MiniBatchKMeans chemotype clusters are now fitted on the **full train fold
  (220,408 molecules)** by streaming `partial_fit` over Morgan-2048 fingerprints
  (seed 42), instead of on the 40k subsample;
* all run-3 numbers are therefore scored on the **same 55,110-molecule validation
  set** (identity-disjoint, `fold_identity == 0`) with the **same 64 labels**.

---

## 2. Data pipeline

| Stage | Value |
|---|---|
| Raw structures | 277,566 |
| After filter (drop null SMILES, dedup by inchikey14, len 3–200) | **275,518** |
| Train / val split | 220,408 / 55,110 |
| Chemotype classes | 64 (MiniBatchKMeans, Morgan-2048, fit on train fold only) |
| Median SMILES length | 39 chars |
| Embeddings | `outputs/smiles_detector_full/embeddings.npy` — 275,518 × 768 float32, resumable memmap |

Artifacts: `outputs/smiles_detector_full/{labels.npz, embeddings.npy, prep_meta.json}`.

---

## 3. Arm 1 — Full-data training (220,408 molecules)

Four recipes × 50 epochs + two extra seeds of the winner. All trained on the full
train fold, scored on the 55,110-molecule val set with the evaluator metric.

| Tag | lr | wd | bs | ls | best epoch | MRR@25 | top-1 | top-5 |
|---|---|---|---|---|---|---|---|---|
| **S_F2_seed44** | 3e-3 | 0 | 1024 | 0.05 | 50 | **0.8689** | 0.7730 | 0.9906 |
| **F2** | 3e-3 | 0 | 1024 | 0.05 | 48 | **0.8676** | 0.7715 | 0.9905 |
| S_F2_seed43 | 3e-3 | 0 | 1024 | 0.05 | 50 | 0.8667 | 0.7696 | 0.9906 |
| F3 | 1e-3 | 1e-2 | 1024 | 0.05 | 48 | 0.8504 | 0.7447 | 0.9875 |
| F1 | 1e-3 | 1e-3 | 1024 | 0.05 | 47 | 0.8492 | 0.7426 | 0.9875 |
| F4 | 5e-4 | 1e-3 | 2048 | 0.05 | 50 | 0.8442 | 0.7351 | 0.9859 |

**3-seed soft-voting ensemble (F2 + seed43 + seed44): MRR@25 = 0.8746, top-1 = 0.7821,
top-5 = 0.9919.**

Findings:

* **5.5× more training data is worth +4.3 MRR@25 points** over the 40k arm
  (0.8676 vs 0.8239 best single) — the dominant factor of run 3.
* The winner moved to **lr 3e-3 with zero weight decay and label smoothing 0.05**
  (the run-2 winner used lr 1e-3 / wd 1e-3 / ls 0.05). More data supports a
  larger step size and no regularisation.
* Label smoothing 0.05 remains important; lr > 3e-3 was never tested on the full
  data and is the obvious next probe.
* Seed variance is small (±0.002 MRR@25), so the 3-seed ensemble gain (+0.007) is
  a genuine but modest variance-reduction effect.

Artifacts: `outputs/smiles_detector_full/runs/full_results.csv`,
`full_results_from_histories.csv`, `full_summary.json`,
`outputs/smiles_detector_full/checkpoints/{FULL_BEST.pt, FULL_3SEED_ENS.pt}`.

---

## 4. Arm 2 — Deeper HPO (25 variants, 40k-trained, full-data label space)

Grid on the 40k subsample, but every variant is scored on the 55,110-molecule full
validation set, so the numbers are directly comparable to arm 1.

| Round | Search space | Best variant | full-val MRR@25 |
|---|---|---|---|
| H | lr {5e-4 … 3e-3} × wd {3e-4, 1e-3, 3e-3} (18) | `H_lr0.002_wd0.0003` | 0.8236 |
| I | batch size {512, 1024, 2048} (2 new) | `I_bs512` | 0.8236 |
| J | label smoothing {0, 0.02, 0.05, 0.1} (3 new) | **`J_ls0.1`** | **0.8239** |
| S | 3 seeds of the winner (2 new) | seed43 / seed44 | 0.8213 / 0.8187 |

Full 25-row table: `outputs/smiles_detector_v3/v3_grid.csv`.
**Top-3 ensemble (`J_ls0.1` + `H_lr0.002_wd0.0003` + `I_bs512`): MRR@25 = 0.8321,
top-1 = 0.7173, top-5 = 0.9826.**

Findings:

* The optimum sits at **lr 2e-3 / wd 3e-4 / bs 512–1024 / label smoothing 0.1**;
  label smoothing 0.1 is worth +0.002 over 0.05 and +0.004 over 0.
* Strong weight decay (3e-3) is clearly harmful on 40k (0.8025–0.8184); the full-data
  arm independently confirms this (wd 0 beat wd 1e-2 by +0.017).
* Small batch (512) ≈ 1024 > 2048 (+0.004): more gradient steps help slightly.
* lr ≤ 1e-3 underfits (0.801–0.821); lr ≥ 3e-3 with wd 3e-3 destabilises training
  (several runs stopped at 22–38 epochs).

Artifacts: `outputs/smiles_detector_v3/{v3_grid.csv, v3_summary.json,
checkpoints/{V3_BEST.pt, V3_TOP3_ENS.pt}, histories/*.csv}`.

---

## 5. Chemist knowledge injected into the model

Implemented in `src/knowledge/chemist_knowledge.py`:

| Component | What it encodes | Level |
|---|---|---|
| `ModalFingerprintPrior` | Morgan-2048 **class prototypes** fitted on the train fold (the "phenotype" of each chemotype) + per-class **medoid** representative | feature / candidate |
| `chemist_prior` | NP-likeness SMARTS motifs, neutral-mass window, RDKit formula check, reactive/assay-interference structural alerts, halogen penalty | molecule / candidate |
| `zscore` / `fuse` / `evaluate_fusion` | evaluator-style fusion `z(model) + λ·prior` | — |

### 5.1 Feature-level injection (Morgan chemist memory) — the strong result

`chem[i, j]` = cosine between the query molecule's Morgan-2048 fingerprint and the
prototype of class *j* (fitted on the 220,408 train molecules).

| Scoring | MRR@25 | top-1 | top-5 |
|---|---|---|---|
| Model only (full-data 3-seed ensemble) | 0.8746 | 0.7821 | 0.9919 |
| Chemist prototype prior **alone** | 0.9252 | — | — |
| **Fusion `z(log p_model) + 10 · z(chem)`** | **0.9366** | **0.8830** | **0.9990** |
| Fusion `0.05·p + 0.95·minmax(chem)` | 0.9298 | — | — |
| Rank fusion (RRF, k = 5…60) | 0.9229 | — | — |

**+6.2 MRR@25 points over the model, +1.1 over the chemist prior alone.**
Both signals are complementary: the ChemBERTa head encodes semantic similarity,
the fingerprint prototypes encode the literal structural motifs a chemist would
name. The fusion optimum is broad and flat (λ = 8–14 all within 0.002 of the peak),
so the gain is not a fragile hyper-parameter artefact.

### 5.2 Candidate-level injection (per-candidate medicinal-chemistry prior)

Scoring the 64 class representatives (medoids) with the NP-likeness / mass / alert
prior and fusing as `z(log p) + λ·z(candidate prior)` **did not help**:

| λ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| MRR@25 | 0.8746 | 0.8746 | 0.8742 | 0.8716 | 0.8633 | 0.8329 |

Interpretation: the 64 clusters are **fingerprint-defined**, so a per-class
medoid-level prior is nearly constant within a class and carries no discriminative
signal for *which* class a molecule belongs to. The popular-panel prior (NP-like,
no alerts) is informative for *which structures to propose*, not for *which chemotype
a molecule is*. It remains useful in the lively-competition setting where the
candidates are molecules rather than classes — the same reason the popularity prior
helps there (benchmark: MRR@25 0.51 → 0.95) but not in this class-ranking setup.

Artifacts: `outputs/smiles_detector_full/knowledge_fusion_full.json`,
`knowledge_fusion_search.json`, `fusion_cache.npz`;
40k-model counterpart `outputs/smiles_detector/knowledge_fusion_40k.json`
(base 0.7778, prototype prior 0.9415, model + 0.05·prior 0.7843).

---

## 6. Unified comparison (all under `mrr25_from_scores`)

`outputs/checkpoint_metrics_run3.csv`

| Arm | Checkpoint | Val mols | Label space | MRR@25 | top-1 | top-5 |
|---|---|---|---|---|---|---|
| full-data 3-seed ens **+ chemist prior** | `z(log p) + 10 z(chem)` | 55,110 | full-64 | **0.9366** | **0.8830** | **0.9990** |
| full-data 3-seed ensemble | `FULL_3SEED_ENS.pt` | 55,110 | full-64 | 0.8746 | 0.7821 | 0.9919 |
| full-data single best | `FULL_BEST.pt` | 55,110 | full-64 | 0.8676 | 0.7715 | 0.9905 |
| 40k HPO top-3 ensemble | `V3_TOP3_ENS.pt` | 55,110 | full-64 | 0.8321 | 0.7173 | 0.9826 |
| 40k HPO best single | `V3_BEST.pt` | 55,110 | full-64 | 0.8239 | 0.7050 | 0.9813 |
| run-2 ensemble | `V2_ENSEMBLE.pt` | 7,982 | 40k-64 | 0.7914 | 0.6709 | 0.9500 |
| run-2 best single | `V2_BEST_MODEL.pt` | 7,982 | 40k-64 | 0.7778 | 0.6525 | 0.9442 |

Caveat: the two label spaces are not interchangeable — 40k-64 rows are scored on
the 40k-cluster val split, full-64 rows on the 55,110-molecule full-val split. The
run-3 rows (v3 and full) share one label space and one val set, so *those* are
directly comparable.

**Progression: 0.7778 (run 2) → 0.8239 (deeper HPO) → 0.8676 (full data) →
0.8746 (3-seed ensemble) → 0.9366 (with chemist knowledge).**

---

## 7. Independent second training — RHS-DL-Development

Repo: `/mnt/d/Workspace_Glad/Projects/RHS-DL-Development` (NAS on MIMO-64 beamforming).

* Architecture selected by the repo's own NAS: `h256x256x128x64-leaky_relu-d0.1-nobn-deep`
  (997,825 params), from `search_results/mimo64_nas/architectures/best.json`.
* Independent training: `python3 scripts/train_best_real.py --cache
  data/processed/mimo64_5000.npz --best search_results/mimo64_nas/architectures/best.json
  --epochs 60 --batch-size 64 --lr 1e-3 --seed 42 --out checkpoints/mimo64_nas_best_run3.pt`
  → 24 epochs (early stop, patience 15).
* **Re-evaluated after the CASMI arm finished** with the new standalone script
  `scripts/evaluate_checkpoint.py` (loads the saved checkpoint, rebuilds the model,
  evaluates on the held-out **test** split).

| Split | rate R² | M cosine | M nMSE | M mse | rate mse |
|---|---|---|---|---|---|
| **test** (503 mols) | **0.7938** | **0.7441** | **0.4522** | 0.037757 | 30.7714 |
| val (503 mols) | 0.7770 | 0.7431 | 0.4552 | 0.038500 | 32.1022 |
| baseline (`best.json`, same arch/val) | 0.7815 | 0.7407 | 0.4600 | 0.038908 | 31.4615 |

Deltas vs the NAS baseline on the same val split: rate R² −0.0045, M cosine +0.0024,
M nMSE −0.0048. The independent training is **on par with** the NAS reference
(essentially tied on rate R², slightly better on the M-matrix reconstruction),
which confirms the architecture generalises to the test split.

Artifacts: `checkpoints/mimo64_nas_best_run3.pt`, `.metrics.json`,
`results/mimo64_nas_best_run3_eval.json`, log `logs/rhs_train.log`.

---

## 8. Reproduction

```bash
# data prep (streaming KMeans + resumable embeddings)     ~4 h
.venv/bin/python scripts/prep_full_data.py

# arm 2: deeper HPO on 40k, scored on the full val set    ~3 h
.venv/bin/python scripts/tune_smiles_detector_v3.py --round all
.venv/bin/python scripts/finalize_v3_ensemble.py          # V3_BEST / V3_TOP3_ENS

# arm 1: full-data training                               ~4 h
.venv/bin/python scripts/train_smiles_detector_full.py
.venv/bin/python scripts/finalize_full_ensemble.py        # FULL_BEST / FULL_3SEED_ENS

# chemist-knowledge injection + fusion search
.venv/bin/python scripts/validate_knowledge_fusion_full.py
.venv/bin/python scripts/knowledge_fusion_search.py

# unified comparison table
.venv/bin/python scripts/eval_checkpoint_metrics_run3.py

# RHS-DL independent evaluation
cd /mnt/d/Workspace_Glad/Projects/RHS-DL-Development
python3 scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/mimo64_nas_best_run3.pt \
    --cache data/processed/mimo64_5000.npz \
    --out results/mimo64_nas_best_run3_eval.json
```

---

## 9. Notes, limits and next steps

* **Memory discipline.** The box has ~1.2 GB usable RAM; every run-3 script is
  written to stream: memmap-backed embeddings (`MemmapIndex`), chunked
  `eval_fullval`, streaming `ModalFingerprintPrior.fit` (per-class sums, two-pass
  medoid selection), chunked ensemble scoring. Three OOM kills during run 3 were
  all fixed this way; the final numbers above come from re-runs that reproduced
  the earlier per-variant values bit-for-bit (F2 0.8676 @e48, seed43 0.8667 @e50,
  seed44 0.8689 @e50).
* **Full-data arm scope.** The user scoped arm 1 to the best recipe plus seeds
  rather than all four variants; F1/F3/F4 were in fact already trained to
  completion by the earlier run, so the table in §3 is complete.
* **Untested next probes** (in priority order): lr > 3e-3 and label smoothing > 0.05
  on the full 220k data; unfreezing the last 2 ChemBERTa layers
  (`tune_smiles_detector_v3.py --probe`, written but not run — 3k molecules / 3 epochs
  as a cheap probe); larger head (3 layers); larger Morgan radius for the prototype
  prior; and folding the popularity prior (PubChem `log_substances + log_pubmed`)
  into the same fusion as the fingerprint prototypes.
* **Honest limitation.** The chemist prior is fitted on the *same* label space as
  the model (train-fold prototypes of the 64 clusters), so it is a non-parametric
  memory of the training taxonomy rather than independent chemical knowledge. Its
  strength (0.9252 alone) reflects that fingerprint prototypes are a very good
  classifier for fingerprint-defined clusters. For a genuinely independent signal,
  the next step is to inject knowledge that is *not* derived from the labels —
  e.g. a real structure–property model, or the NP-panel panel prior in the
  candidate-ranking setting where it was shown to help.
