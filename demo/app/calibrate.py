"""Fit demo thresholds on VALIDATION pages only (never test/Firemaker) and export sample pairs.
Run:  docker compose run --rm demo python -m demo.calibrate
Thresholds follow eval.three_way (99th pct of impostor scores, 1st pct of genuine), computed per dataset and
combined conservatively (higher accept threshold, lower reject threshold) so the inconclusive band is wide."""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from src.data import page_patches
from .engine import CALIB, K, S, load_model

HERE = Path(__file__).parent
MANIFEST = "/data/cache/manifests/master.csv"


def main():
    torch.set_num_threads(3)
    model = load_model()
    m = pd.read_csv(MANIFEST)
    m = m[(m.split == "val") & m.dataset.isin(["iam", "cvl"])].reset_index(drop=True)
    emb = []
    with torch.no_grad():
        for i, r in m.iterrows():
            x = torch.from_numpy(page_patches(r.path, r["view"], K, S, False))[None]
            emb.append(model(x)[0].numpy())
            if i % 50 == 0:
                print(i, len(m), flush=True)
    emb = np.stack(emb)
    bins = np.linspace(-0.4, 1.0, 71)
    G = np.zeros(len(bins) - 1); I = np.zeros(len(bins) - 1)
    lo, hi, per = [], [], {}
    for ds in ("iam", "cvl"):
        idx = np.where(m.dataset.values == ds)[0]
        e, w = emb[idx], m.writer.values[idx]
        iu = np.triu_indices(len(idx), 1)
        sc = (e @ e.T)[iu]; y = (w[iu[0]] == w[iu[1]])
        fpr, tpr, thr = roc_curve(y, sc); k = int(np.argmin(np.abs(fpr - (1 - tpr))))
        per[ds] = {"eer": float((fpr[k] + 1 - tpr[k]) / 2), "tau_eer": float(thr[k]), "pairs": int(len(sc))}
        hi.append(float(np.quantile(sc[~y], 0.99))); lo.append(float(np.quantile(sc[y], 0.01)))
        G += np.histogram(sc[y], bins)[0]; I += np.histogram(sc[~y], bins)[0]
    calib = {"model": "main_s0", "k": K, "tau_high": max(hi), "tau_low": min(lo), "bins": bins.tolist(),
             "genuine": G.astype(int).tolist(), "impostor": I.astype(int).tolist(), "per_dataset": per,
             "n_genuine": int(G.sum()), "n_impostor": int(I.sum()),
             "note": "Validation pages of IAM and CVL, single page vs single page. Thresholds are the more conservative of the two datasets."}
    json.dump(calib, open(CALIB, "w"), indent=1)
    print("tau_low %.3f  tau_high %.3f  %s" % (calib["tau_low"], calib["tau_high"], per))

    # sample pairs: fixed seed, no cherry-picking by score (the score each pair gets is shown honestly in the UI)
    rng = random.Random(7); out = []; sd = HERE / "samples"; sd.mkdir(exist_ok=True)
    def save(row, name):
        img = np.load(row.path); cv2.imwrite(str(sd / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90]); return name
    for ds, n_same, n_diff in (("cvl", 2, 2), ("iam", 1, 1)):
        d = m[m.dataset == ds]
        writers = [w for w, g in d.groupby("writer") if len(g) >= 2]
        for j in range(n_same):
            w = rng.choice(writers); a, b = d[d.writer == w].sample(2, random_state=rng.randint(0, 10**6)).itertuples()
            out.append({"id": f"{ds}_same{j}", "truth": "same writer", "dataset": ds,
                        "reference": save(a, f"{ds}_same{j}_ref.jpg"), "questioned": save(b, f"{ds}_same{j}_q.jpg")})
        for j in range(n_diff):
            w1, w2 = rng.sample(writers, 2)
            a = d[d.writer == w1].sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
            pool = d[(d.writer == w2) & (d.cond == a.cond)] if ds == "cvl" else d[d.writer == w2]  # CVL: same text
            b = (pool if len(pool) else d[d.writer == w2]).sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
            out.append({"id": f"{ds}_diff{j}", "truth": "different writers", "dataset": ds,
                        "reference": save(a, f"{ds}_diff{j}_ref.jpg"), "questioned": save(b, f"{ds}_diff{j}_q.jpg")})
    json.dump(out, open(sd / "samples.json", "w"), indent=1)
    print(len(out), "sample pairs written")


if __name__ == "__main__":
    main()
