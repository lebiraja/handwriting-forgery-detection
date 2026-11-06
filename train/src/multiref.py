"""Controlled multi-reference study on TEST pages (one extra pass; same model, same embeddings as eval.py, no tuning).

The trials in eval.py draw genuine queries only from writers with >= N+1 pages, so the writer set shrinks as N grows and N is
confounded with writer difficulty (see DECISIONS.md). Here every N is evaluated on ONE fixed set of writers (>= 4 pages for
N = 1,2,3; >= 6 pages for the extra N = 5 set), with nested references (N=1 is contained in N=2 is contained in N=3) and the same impostor
queries for every N.   python -m src.multiref --run /app/runs/main_s0 --out /app/runs/main_s0/multiref_test.json
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_curve

from .eval import embed_split, fuse, load_model

FUS = ("mean", "max", "top2", "centroid")


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); k = np.argmin(np.abs(fpr - (1 - tpr))); return float((fpr[k] + 1 - tpr[k]) / 2)


def study(emb, writers, min_pages, ns, T=40, seed=0, boot=200):
    writers = np.array(writers); idx = {w: np.where(writers == w)[0] for w in sorted(set(writers))}
    W = [w for w in idx if len(idx[w]) >= min_pages]; rng = np.random.RandomState(seed)
    raw = {(n, f): {w: ([], []) for w in W} for n in ns for f in FUS}
    for w in W:
        for _ in range(T):
            pages = rng.permutation(idx[w]); v = W[rng.randint(len(W))]
            while v == w: v = W[rng.randint(len(W))]
            imp = emb[rng.choice(idx[v])]
            for n in ns:
                refs = emb[pages[1:n + 1]]
                for f in FUS:
                    g, i = raw[(n, f)][w]; g.append(fuse(emb[pages[0]], refs, f)); i.append(fuse(imp, refs, f))
    out = {"writers": len(W), "min_pages": min_pages, "trials_per_writer": T, "results": {}}
    for (n, f), d in raw.items():
        def e(sel):
            g = np.concatenate([d[w][0] for w in sel]); i = np.concatenate([d[w][1] for w in sel])
            return eer(np.r_[np.ones(len(g)), np.zeros(len(i))], np.r_[g, i])
        point = e(W); bs = np.random.RandomState(1); cis = [e([W[j] for j in bs.randint(0, len(W), len(W))]) for _ in range(boot)]
        out["results"][f"N{n}_{f}"] = {"eer": point, "ci95": [float(np.percentile(cis, 2.5)), float(np.percentile(cis, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="/data/cache/manifests/master.csv"); a = ap.parse_args()
    model, _ = load_model(a.run, "cuda"); e, w, d, _ = embed_split(model, a.manifest, "test", ["iam", "cvl"], 64, 1, "cuda"); res = {}
    for ds in ("iam", "cvl"):
        m = d == ds; res[ds] = {"common_N123": study(e[m], list(w[m]), 4, (1, 2, 3)), "common_N1235": study(e[m], list(w[m]), 6, (1, 2, 3, 5))}
    json.dump(res, open(a.out, "w"), indent=1)
    print({ds: {k: v["writers"] for k, v in r.items()} for ds, r in res.items()})


if __name__ == "__main__":
    main()
