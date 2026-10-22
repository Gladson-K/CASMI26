"""Generate final Kaggle submission."""
import numpy as np
import pandas as pd
from src.retrieval.cosine_search import retrieve_topk
from src.features.ranker_features import build_features
from src.models.rankers import build_ranker


def _aggregate(scores, method):
    if not scores:
        return 0.0
    if method == 'max':
        return float(np.max(scores))
    if method == 'mean':
        return float(np.mean(scores))
    if method == 'top3_mean':
        return float(np.mean(sorted(scores, reverse=True)[:3]))
    return float(np.max(scores))


def predict_candidates(df, library_specs, library_smiles, params, K=100):
    """Return {index: {smiles: aggregated_score}} for every row."""
    out = {}
    for idx, row in df.iterrows():
        cand_scores = {}
        for spec in row['clean_spectra']:
            top = retrieve_topk(
                spec, library_specs, library_smiles,
                metric=params['similarity_metric'],
                tol=params['mz_tolerance'],
                top_k=K,
                precursor_tol=params['precursor_tolerance_ppm'] / 1e6 * 50,
            )
            for smi, s in top:
                cand_scores.setdefault(smi, []).append(s)
        out[idx] = {smi: _aggregate(v, params['aggregation_method'])
                    for smi, v in cand_scores.items()}
    return out


def generate_submission(train_df, test_df, library_specs, library_smiles,
                        best_params, top_n=25):
    """Train best ranker on full train set, then produce submission.csv."""

    # Retrieval
    train_cands = predict_candidates(train_df, library_specs, library_smiles,
                                     best_params)
    test_cands = predict_candidates(test_df, library_specs, library_smiles,
                                    best_params)

    # Features (train)
    X, y = [], []
    for idx, cands in train_cands.items():
        truth = train_df.loc[idx, 'smiles']
        spec = train_df.loc[idx, 'clean_spectra'][0]
        for smi, sc in cands.items():
            f = build_features(spec, smi, sc, best_params)
            if f is None:
                continue
            X.append(f)
            y.append(1 if smi == truth else 0)
    X, y = np.asarray(X), np.asarray(y)

    ranker = build_ranker(best_params)
    ranker.fit(X, y)

    # Features (test) & predict
    rows = []
    for idx, cands in test_cands.items():
        if not cands:
            rows.append({'molecule_id': idx, 'smiles': ''})
            continue
        Xte = []
        cand_list = list(cands.keys())
        spec = test_df.loc[idx, 'clean_spectra'][0] if test_df.loc[idx, 'clean_spectra'] else None
        for smi, sc in cands.items():
            f = build_features(spec, smi, sc, best_params) if spec else None
            if f is None:
                Xte.append(np.zeros(15, dtype=np.float32))
            else:
                Xte.append(f)
        Xte = np.asarray(Xte)
        y_pred = ranker.predict_proba(Xte)[:, 1]
        order = np.argsort(-y_pred)[:top_n]
        ranked = ';'.join([cand_list[j] for j in order])
        rows.append({'molecule_id': idx, 'smiles': ranked})

    return pd.DataFrame(rows)