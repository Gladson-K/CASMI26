"""Optuna objective: full retrieval + ranker + MRR@25 pipeline."""
import numpy as np
import pandas as pd
from rdkit import Chem
from src.retrieval.cosine_search import retrieve_topk
from src.features.ranker_features import build_features
from src.models.rankers import build_ranker
from src.evaluation.mrr25 import mrr25
from src.nas.search_space import define_search_space


def _aggregate(scores_list, method):
    if not scores_list:
        return 0.0
    if method == 'max':
        return float(np.max(scores_list))
    if method == 'mean':
        return float(np.mean(scores_list))
    if method == 'top3_mean':
        return float(np.mean(sorted(scores_list, reverse=True)[:3]))
    return float(np.max(scores_list))


def nas_objective(trial, train_df, val_df, library_specs, library_smiles):
    params = define_search_space(trial)
    tol = params['mz_tolerance']

    # --- Step 1: Retrieve for train & val ---
    def _retrieve(df):
        rows = []
        for i, row in df.iterrows():
            spectra = row['clean_spectra']
            cand_scores = {}
            for spec in spectra:
                top = retrieve_topk(
                    spec, library_specs, library_smiles,
                    metric=params['similarity_metric'],
                    tol=tol,
                    top_k=100,
                    precursor_tol=params['precursor_tolerance_ppm'] / 1e6 * 50,
                )
                for smi, s in top:
                    cand_scores.setdefault(smi, []).append(s)
            aggregated = {smi: _aggregate(v, params['aggregation_method'])
                          for smi, v in cand_scores.items()}
            rows.append(aggregated)
        return rows

    train_cands = _retrieve(train_df)
    val_cands = _retrieve(val_df)

    # --- Step 2: Build features ---
    def _features(df, cands):
        X, y, groups = [], [], []
        for gi, (row_idx, cand_dict) in enumerate(cands.items()):
            truth = df.loc[row_idx, 'smiles']
            spec = df.loc[row_idx, 'clean_spectra'][0]  # first spectrum
            for smi, sc in cand_dict.items():
                f = build_features(spec, smi, sc, params)
                if f is None:
                    continue
                X.append(f)
                y.append(1 if smi == truth else 0)
                groups.append(row_idx)
        return np.asarray(X), np.asarray(y), np.asarray(groups)

    train_cands = {idx: train_cands[i] for i, idx in enumerate(train_df.index)}
    val_cands = {idx: val_cands[i] for i, idx in enumerate(val_df.index)}

    Xtr, ytr, _ = _features(train_df, train_cands)
    if len(ytr) == 0 or ytr.sum() == 0:
        return 0.0

    # --- Step 3: Train ranker ---
    ranker = build_ranker(params)
    ranker.fit(Xtr, ytr)

    # --- Step 4: Rank on validation and score ---
    Xva, yva, groups_va = _features(val_df, val_cands)
    if len(yva) == 0:
        return 0.0
    y_pred = ranker.predict_proba(Xva)[:, 1]

    # --- Step 5: Build prediction dataframe ---
    records = []
    for idx in val_df.index:
        mask = groups_va == idx
        if mask.sum() == 0:
            records.append({
                'molecule_id': idx,
                'smiles_ranked': '',
                'is_truth': val_df.loc[idx, 'smiles'],
            })
            continue
        mol_scores = y_pred[mask]
        mol_cands = [list(val_cands[idx].keys())[j]
                     for j in range(mask.sum())]
        order = np.argsort(-mol_scores)
        ranked = ';'.join([mol_cands[j] for j in order[:25]])
        records.append({
            'molecule_id': idx,
            'smiles_ranked': ranked,
            'is_truth': val_df.loc[idx, 'smiles'],
        })

    pred_df = pd.DataFrame(records)
    return mrr25(pred_df)