"""Mean Reciprocal Rank @ 25 metric."""
import numpy as np
import pandas as pd

def mrr25(predictions_df, max_rank=25):
    """
    predictions_df must have columns:
        molecule_id, smiles_ranked (semicolon-separated), is_truth
    """
    scores = []
    for _, group in predictions_df.groupby('molecule_id'):
        ranked = str(group['smiles_ranked'].iloc[0]).split(';')
        truth = group['is_truth'].iloc[0]
        rank = None
        for i, smi in enumerate(ranked[:max_rank]):
            if smi == truth:
                rank = i + 1
                break
        scores.append(1.0 / rank if rank else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def mrr25_from_scores(scores, truth_indices, max_rank=25):
    """Utility for scoring a single molecule."""
    ranked = np.argsort(-np.asarray(scores))
    for r, idx in enumerate(ranked[:max_rank]):
        if idx in truth_indices:
            return 1.0 / (r + 1)
    return 0.0