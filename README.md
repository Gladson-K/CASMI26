# ENVEDA-CASMI26 — CPU-first retrieval, model assets and 10-hour Optuna search

This revision upgrades the supplied CASMI26 project to bootstrap its public repositories, competition files, optional public spectral libraries and pretrained DreaMS weights, and to run a resumable 10-hour CPU hyperparameter search.

## What is included

- `setup.sh`: isolated `.venv`, CPU-only PyTorch, core dependencies, repository checkout and core/full asset modes.
- `scripts/bootstrap_assets.py`: idempotent downloads with a JSON report; Kaggle failures are reported rather than silently ignored.
- `repos/`: DreaMS, Spec2Vec, matchms and MSBuddy source checkouts (downloaded by setup).
- `models/dreams/`: official DreaMS pretrained `embedding_model.ckpt` and `ssl_model.ckpt` in full mode; the downloader verifies published MD5 hashes. Together they are approximately 2.6 GB.
- `models/kaggle_buddy/`: optional attempt to fetch the Buddy model via KaggleHub; if that model handle or authentication is unavailable, the report gives the official page to use manually.
- `data/competition/`: official Kaggle competition files.
- `data/coconut/`: optional COCONUT structure dump.
- `data/gnps/`: optional GNPS MGF download and a chunked annotated-spectra Parquet conversion.
- `configs/cpu_10h.yaml`: CPU-conservative, resumable Optuna TPE search profile.
- `scripts/run_baseline.py`: direct spectral-library retrieval plus neutral-mass candidate fallback. It uses observed spectra; it does **not** fabricate MS/MS peaks from a molecule's mass.
- Existing reference notebooks and reference library are preserved.

## Requirements and disk space

Recommended: Ubuntu 22.04/24.04, Python 3.11, Git, `curl`, `wget`, and several GB of free RAM. Core setup requires enough disk space for the competition data and Python environment. Full setup can require **many additional GB**: two DreaMS checkpoints (~2.6 GB total), COCONUT, and a GNPS library containing millions of spectra. Keep additional free disk space for extracted files and processed Parquet chunks. GNPS metadata and license terms vary by included library; review source terms before redistribution.

## 1. Core setup

First create a Kaggle API token from your Kaggle account settings and store `kaggle.json` at `~/.kaggle/kaggle.json` with restrictive permissions. You must also accept the competition rules on the competition page.

```bash
unzip casmi26_upgrade_v2.zip
cd casmi26_upgrade
bash setup.sh --core
source .venv/bin/activate
python scripts/inspect_data.py
```

Core mode installs the CPU environment, clones the relevant repositories, and attempts the official competition download. It does not download multi-gigabyte pretrained weights or the full GNPS library.

## 2. Full asset bootstrap

```bash
bash setup.sh --full
```

Full mode attempts to download:

1. Competition data and the referenced benchmark dataset through the Kaggle CLI.
2. COCONUT structures.
3. The GNPS MGF reference library, then converts up to 250,000 scanned spectra to chunked Parquet for practical CPU use.
4. The DreaMS source repo and official pretrained weights from Zenodo.
5. The Buddy model through KaggleHub when supported by the current model handle and credentials.

The GNPS preparation cap can be changed before setup:

```bash
GNPS_MAX_SPECTRA=1000000 bash setup.sh --full
# 0 means scan the entire downloaded MGF; this may take a long time.
GNPS_MAX_SPECTRA=0 bash setup.sh --full
```

A source can fail because a dataset slug or URL changed, a Kaggle asset is private, a license needs acceptance, or a remote service is unavailable. Inspect `outputs/asset_download_report.json`; do not treat a failed or partial download as a usable asset.

If you want only one asset later:

```bash
source .venv/bin/activate
python scripts/bootstrap_assets.py --dreams-weights
python scripts/bootstrap_assets.py --gnps
python scripts/bootstrap_assets.py --coconut
python scripts/bootstrap_assets.py --benchmark
python scripts/bootstrap_assets.py --buddy-model
python scripts/prepare_gnps.py --max-spectra 250000
```

### DreaMS installation and checkpoints

The official DreaMS repository documents a Python 3.11 environment and exposes an embedding API; pretrained model files are published separately on Zenodo. This project downloads source and checkpoint files but deliberately does **not** install DreaMS into the main competition environment automatically, because its dependency set can conflict with other tools. Use `repos/DreaMS/README.md` and `https://dreams-docs.readthedocs.io/` to create a separate DreaMS environment and validate inference. Downloading weights alone does not mean the checkpoint has been loaded or that DreaMS embeddings have been integrated into the competition submission.

## 3. Inspect actual schemas before training

```bash
python scripts/inspect_data.py
```

The NAS expects the competition's training Parquet to contain precursor m/z, adduct, observed MS/MS peak arrays (`ms2_mzs`, `ms2_normalized_intensities`) and a structure label such as `inchikey14` or `normalized_smiles`. If the competition schema changes, update the loader after inspecting the columns; do not guess column mappings.

## 4. Run the CPU NAS for up to 10 hours

```bash
source .venv/bin/activate
python run_nas.py --config configs/cpu_10h.yaml --timeout-hours 10
```

For a short installation smoke test:

```bash
python run_nas.py --config configs/cpu_10h.yaml --smoke-test
```

The search uses Optuna's **TPE sampler** (multivariate/group search), a median pruner, one worker, a fixed seed, SQLite persistence, and a 36,000-second timeout. It tunes precursor tolerance, peak alignment tolerance, intensity floor/power, maximum peaks, candidate cap and spectrum aggregation. Re-running the command resumes the same study from SQLite. The `--trials` option caps the number of trials; the timeout is a hard upper budget, not a promise that every trial completes.

Outputs:

- `outputs/nas_best.json` — best score and configuration.
- `outputs/nas_top_configs.json` — top configurations.
- `outputs/nas_history.csv` — trial history.
- `outputs/nas_cpu_10h.sqlite3` — resumable study database.

### Important validation limitation

The current NAS objective tunes the **direct spectral-library retrieval channel** using a spectrum-level split. If the same compound has another spectrum in the library, the correct structure can be retrieved; therefore this is an optimistic library-hit proxy, **not** an unbiased estimate of competition accuracy on unseen compounds and not a neural architecture search. The best MRR@25 from this run must not be reported as a public leaderboard score. For a reliable unseen-compound estimate, use a molecule/InChIKey-grouped holdout and an external library/candidate pool, and report direct-library, analog-propagation and fallback results separately.

## 5. Generate a baseline submission

```bash
python scripts/run_baseline.py
python scripts/validate_submission.py
```

The baseline searches the observed training spectra and, if prepared, annotated GNPS spectra. It then uses precursor neutral mass to generate fallback structures from train SMILES and COCONUT. The mass-only fallback is deliberately labelled as a weak safety net; it is not a confidence-calibrated model. Validate the generated `outputs/submission.csv` against `sample_submission.csv` and evaluate with a leakage-aware validation protocol before submitting.

## Recommended path to stronger accuracy

1. Verify download status and schemas; retain data provenance and licenses.
2. Establish a molecule-grouped validation split and a simple observed-spectrum cosine baseline.
3. Measure GNPS/library coverage and candidate recall@25 separately from ranking quality.
4. Add the pretrained DreaMS embedding channel only after inference works in its own environment; cache embeddings so NAS trials do not recompute them.
5. Add precursor/adduct-aware candidate generation, mass-shifted analog propagation with RDKit fingerprints, and in-silico fragment explainability as independent ranking features.
6. Train a CPU-friendly LightGBM/HistGradientBoosting ranker on candidate features and evaluate MRR@25 across multiple grouped folds.
7. Only tune parameters on validation folds; keep the official test set untouched until final inference.

No accuracy score is promised here: data downloads and full NAS have not been executed as part of creating this ZIP, because they require your Kaggle credentials, network access, disk capacity and potentially several hours of CPU time.

## Upstream sources

- Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
- DreaMS code: https://github.com/pluskal-lab/DreaMS
- DreaMS pretrained checkpoints: https://zenodo.org/records/10997887
- DreaMS documentation: https://dreams-docs.readthedocs.io/
- Spec2Vec: https://github.com/iomega/spec2vec
- matchms: https://github.com/matchms/matchms
- MSBuddy: https://github.com/CompMS-MS2/MSBuddy
- Buddy model reference: https://www.kaggle.com/models/jek1wantaufik/buddy/Other/enveda
- GNPS libraries and formats: https://external.gnps2.org/gnpslibrary
- GNPS download documentation: https://ccms-ucsd.github.io/GNPSDocumentation/downloadlibraries/
- COCONUT: https://coconut.naturalproducts.net/
- Kaggle CLI docs: https://github.com/Kaggle/kaggle-cli

Review upstream licenses and citations before publishing or redistributing any downloaded dataset or checkpoint.
