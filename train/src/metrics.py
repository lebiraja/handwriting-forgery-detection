import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def verification_metrics(emb: np.ndarray, writers: list[str]) -> dict:
    """All page pairs: same writer = genuine pair, different = impostor/forgery proxy."""
    sim = emb @ emb.T
    iu = np.triu_indices(len(writers), 1)
    y = np.array([writers[i] == writers[j] for i, j in zip(*iu)])
    s = sim[iu]
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return {"eer": float((fpr[k] + fnr[k]) / 2), "auc": float(roc_auc_score(y, s)), "threshold": float(thr[k])}
