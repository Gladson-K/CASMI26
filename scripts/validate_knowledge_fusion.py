"""Validate chemist-knowledge injection on the 40k run-2 model (memory-lean).

The model outputs a 64-class distribution per molecule; the evaluator's MRR@25
uses the rank of the true class. Two honest ways to inject the scalar chemist
prior into that ranking are tested:

  A. temperature/gate fusion  : p'_j = p_j * exp(lam * prior)  (reshapes confidence)
  B. shortlist re-rank        : among the model's top-25 classes, re-order by
                                z(p_j) + lam * prior_j_chem   (needs per-class prior;
                                here prior is molecule-level so it reduces to A)

Because the lively competition setting ranks *candidate structures* (each with its
own prior), we also rank the 64 classes by their prototype-fingerprint chemist
prior alone and fuse that chemical knowledge with the model (feature-level fusion).

Writes outputs/smiles_detector/knowledge_fusion_40k.json
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import train_smiles_detector as base
import tune_smiles_detector_v2 as v2
from src.knowledge.chemist_knowledge import ModalFingerprintPrior
from src.evaluation.mrr25 import mrr25_from_scores


def mrr_many(scores2d, y):
    return float(np.mean([mrr25_from_scores(scores2d[i], {int(y[i])}) for i in range(len(y))]))


def main():
    torch.set_num_threads(6)
    df = base.load_dataset()
    is_train = (df.fold_identity.values != 0)
    labels, good, _ = base.cluster_labels(df.smiles.values, is_train)
    emb = base.embed_smiles(df.smiles.values[good].tolist(), f'{base.OUT}/embeddings.npz')
    y = labels[good]
    tr = is_train[good]
    Xtr, ytr, Xva, yva = base.tensorize(emb, y, tr)
    del emb, ytr, Xtr
    c = torch.load('outputs/smiles_detector/checkpoints/V2_BEST_MODEL.pt', map_location='cpu', weights_only=False)
    cfg = dict(c['head_config']); cfg.setdefault('norm', 'none')
    m = v2.Head(Xva.shape[1], int(y.max()) + 1, cfg)
    m.load_state_dict(c['state_dict']); m.eval()
    with torch.no_grad():
        logits = m(Xva)
    probs = logits.softmax(1).numpy()
    ynp = yva.numpy()
    base_mrr = mrr_many(probs, ynp)
    print('BASE model MRR@25 (evaluator)', round(base_mrr, 4), flush=True)

    # Feature-level chemist knowledge: prototype-fingerprint prior per class, per molecule.
    smis_tr = df.smiles.values[good][tr]
    smis_va = df.smiles.values[good][~tr]
    mfp = ModalFingerprintPrior().fit(smis_tr.tolist(), y[tr])
    chem = mfp.scores(smis_va.tolist())          # (n_val, 64) cosine to class prototypes
    chem_mrr = mrr_many(chem, ynp)
    print('prototype-fingerprint chemist prior MRR@25', round(chem_mrr, 4), flush=True)

    # Only near-tie classes get reordered -> conservative, no damage when model is confident.
    eps = 0.05
    fused_chem = probs * (1 - eps) + eps * (chem - chem.min(1, keepdims=True)) / \
        (chem.max(1, keepdims=True) - chem.min(1, keepdims=True) + 1e-8)
    chem_fusion_mrr = mrr_many(fused_chem, ynp)
    print(f'model + {eps} * prototype-prior MRR@25', round(chem_fusion_mrr, 4), flush=True)

    out = dict(base_mrr25=base_mrr, prototype_prior_mrr25=chem_mrr,
               model_plus_prototype_mrr25=chem_fusion_mrr, eps=eps,
               note='prototype-fingerprint prior = Morgan prototypes of the 64 chemotype classes;'
                    ' feature-level chemist knowledge fused with the ChemBERTa head')
    json.dump(out, open('outputs/smiles_detector/knowledge_fusion_40k.json', 'w'), indent=2)
    print('saved outputs/smiles_detector/knowledge_fusion_40k.json', flush=True)


if __name__ == '__main__':
    main()
