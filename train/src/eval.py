"""Evaluation: single-reference, multi-reference fusion, calibration, 3-way verdict, Firemaker stress tests.

Thresholds / calibrators are fit on VAL only and applied unchanged to TEST and Firemaker.
python -m src.eval --run /app/runs/main_s0 --out /app/runs/main_s0/eval.json
"""
import argparse, json
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from .data import PageDataset
from .model import PageEncoder
from .train import embed_pages

FUSIONS = ("mean", "max", "centroid", "top2")


def eer_auc(y, s):
    fpr, tpr, thr = roc_curve(y, s); fnr = 1 - tpr; k = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2), float(roc_auc_score(y, s))


def far_at_frr(y, s, frr):
    gen = np.sort(s[y == 1]); thr = np.quantile(gen, frr)  # reject the lowest `frr` genuine
    return float((s[y == 0] >= thr).mean())


def fuse(q, refs, how):  # q [D], refs [N,D] (all L2-normalised)
    sims = refs @ q
    if how == "mean": return float(sims.mean())
    if how == "max": return float(sims.max())
    if how == "top2": return float(np.sort(sims)[-2:].mean())
    c = refs.mean(0); c /= np.linalg.norm(c) + 1e-9
    return float(c @ q)


def make_trials(emb, writers, n_ref, how, n_trials, seed):
    """genuine: query + n_ref other pages of the same writer; impostor: query of another writer."""
    rng = np.random.RandomState(seed); writers = np.array(writers); uw = sorted(set(writers))
    idx = {w: np.where(writers == w)[0] for w in uw}; elig = [w for w in uw if len(idx[w]) >= n_ref + 1]
    if not elig: return None
    ys, ss = [], []
    for t in range(n_trials):
        w = elig[rng.randint(len(elig))]; pages = rng.permutation(idx[w])
        refs = emb[pages[1:n_ref + 1]]
        ys.append(1); ss.append(fuse(emb[pages[0]], refs, how))
        v = uw[rng.randint(len(uw))]
        while v == w: v = uw[rng.randint(len(uw))]
        ys.append(0); ss.append(fuse(emb[rng.choice(idx[v])], refs, how))
    return np.array(ys), np.array(ss)


def bootstrap_ci(emb, writers, fn, n=200, seed=0):
    rng = np.random.RandomState(seed); writers = np.array(writers); uw = np.array(sorted(set(writers))); vals = []
    for _ in range(n):
        pick = rng.choice(uw, len(uw), replace=True)
        sel = np.concatenate([np.where(writers == w)[0] for w in pick])
        ww = np.concatenate([[f"{w}#{i}"] * (writers == w).sum() for i, w in enumerate(pick)])
        vals.append(fn(emb[sel], list(ww)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b + (b == 1))
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def three_way(val_y, val_s, test_y, test_s, far=0.01, frr=0.01):
    hi = np.quantile(val_s[val_y == 0], 1 - far)  # accept only above 99th pct of val impostors
    lo = np.quantile(val_s[val_y == 1], frr)       # reject only below 1st pct of val genuine
    acc, rej = test_s >= hi, test_s <= lo
    dec = acc | rej
    err = ((acc & (test_y == 0)) | (rej & (test_y == 1))).sum()
    return {"tau_high": float(hi), "tau_low": float(lo), "coverage": float(dec.mean()),
            "error_on_decided": float(err / max(1, dec.sum())), "accepted": float(acc.mean()), "rejected": float(rej.mean()),
            "inconclusive": float((~dec).mean())}


def load_model(run, dev):
    cfg = json.load(open(f"{run}/cfg.json")) if False else None
    import yaml; cfg = yaml.safe_load(open(f"{run}/config.yaml"))
    m = PageEncoder(cfg["embed_dim"], pool=cfg["pool"]).to(dev); m.load_state_dict(torch.load(f"{run}/best.pt", map_location=dev)); return m.eval(), cfg


def embed_split(model, manifest, split, datasets, k, tta, dev, workers=8, conds=None):
    ds = PageDataset(manifest, split, k, 128, False, datasets)
    if conds: ds.df = ds.df[ds.df.cond.isin(conds)].reset_index(drop=True)
    es = []
    for t in range(tta):
        ds.shift = 1000 * t; e, w, d = embed_pages(model, ds, dev, workers=workers); es.append(e)
    e = np.mean(es, 0); e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    return e, np.array(w), np.array(d), ds.df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="/data/cache/manifests/master.csv"); ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--k", type=int, default=64); ap.add_argument("--test", action="store_true", help="also evaluate test + Firemaker")
    a = ap.parse_args(); dev = "cuda"; model, cfg = load_model(a.run, dev); res = {}; SC = {}
    splits = ["val"] + (["test"] if a.test else [])
    E = {s: embed_split(model, a.manifest, s, ["iam", "cvl"], a.k, a.tta, dev) for s in splits}
    for s in splits:
        e, w, d, _ = E[s]; res[s] = {}
        for name in sorted(set(d)):
            m = d == name; ee, ww = e[m], list(w[m])
            iu = np.triu_indices(len(ww), 1); y = np.array([ww[i] == ww[j] for i, j in zip(*iu)]); sc = (ee @ ee.T)[iu]
            SC[f"{s}_{name}_pair_y"], SC[f"{s}_{name}_pair_s"] = y.astype(np.int8), sc.astype(np.float32)
            r = {"single_ref": {"eer": eer_auc(y, sc)[0], "auc": eer_auc(y, sc)[1], "far@frr1": far_at_frr(y, sc, .01), "far@frr10": far_at_frr(y, sc, .10)}}
            if s == "test":
                r["single_ref"]["eer_ci95"] = bootstrap_ci(ee, ww, lambda x, z: eer_auc(np.array([z[i] == z[j] for i, j in zip(*np.triu_indices(len(z), 1))]), (x @ x.T)[np.triu_indices(len(z), 1)])[0])
            r["multi_ref"] = {}
            for n in (1, 2, 3, 5):
                for how in FUSIONS:
                    t = make_trials(ee, ww, n, how, 1000, 0)
                    if t is not None:
                        n_w = int(sum(1 for w_ in set(ww) if ww.count(w_) >= n + 1))
                        r["multi_ref"][f"N{n}_{how}"] = {**dict(zip(("eer", "auc"), eer_auc(*t))), "writers": n_w}
            res[s][name] = r
    # calibration + 3-way (N=3, centroid) : fit on val, apply to test
    cal = {}
    for name in sorted(set(E["val"][2])):
        vm = E["val"][2] == name; vy, vs = make_trials(E["val"][0][vm], list(E["val"][1][vm]), 3, "centroid", 1000, 1)
        platt = LogisticRegression().fit(vs[:, None], vy); iso = IsotonicRegression(out_of_bounds="clip").fit(vs, vy)
        cal[name] = {"val_trials": (vy, vs), "platt": platt, "iso": iso}
        res.setdefault("calibration", {})[name] = {"platt_coef": float(platt.coef_[0, 0])}
        if a.test:
            tm = E["test"][2] == name; ty, ts = make_trials(E["test"][0][tm], list(E["test"][1][tm]), 3, "centroid", 1000, 2)
            pp, pi = platt.predict_proba(ts[:, None])[:, 1], iso.predict(ts)
            SC.update({f"cal_{name}_val_y": vy, f"cal_{name}_val_s": vs, f"cal_{name}_test_y": ty, f"cal_{name}_test_s": ts, f"cal_{name}_test_platt": pp, f"cal_{name}_test_iso": pi})
            res["calibration"][name].update(ece_platt=ece(pp, ty), ece_iso=ece(pi, ty), ece_raw=ece((ts - ts.min()) / (ts.max() - ts.min()), ty))
            res.setdefault("three_way", {})[name] = three_way(vy, vs, ty, ts)
    if a.test:  # Firemaker cross-dataset + disguise stress test (thresholds from val IAM/CVL calibration)
        fm = {}
        for c in ("p1-copy-normal", "p2-copy-upper", "p3-copy-forged", "p4-self-natural"):
            fm[c] = embed_split(model, a.manifest, "test", ["fm"], a.k, a.tta, dev, conds=[c])
        ref_e, ref_w = fm["p1-copy-normal"][0], fm["p1-copy-normal"][1]
        out = {}
        for c in ("p2-copy-upper", "p3-copy-forged", "p4-self-natural"):
            qe, qw = fm[c][0], fm[c][1]; common = [w for w in ref_w if w in set(qw)]
            ri = {w: i for i, w in enumerate(ref_w)}; qi = {w: i for i, w in enumerate(qw)}
            gen = np.array([ref_e[ri[w]] @ qe[qi[w]] for w in common])
            rng = np.random.RandomState(0); imp = np.array([ref_e[ri[w]] @ qe[qi[common[(j + 1 + rng.randint(len(common) - 1)) % len(common)]]] for j, w in enumerate(common)])
            y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]; s = np.r_[gen, imp]
            SC[f"fm_{c}_gen"], SC[f"fm_{c}_imp"] = gen, imp
            out[f"p1_vs_{c}"] = {"eer": eer_auc(y, s)[0], "auc": eer_auc(y, s)[1], "genuine_mean": float(gen.mean()), "impostor_mean": float(imp.mean()), "n_writers": len(common)}
        res["firemaker"] = out
    np.savez_compressed(a.out.replace(".json", "_scores.npz"), **SC)
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k in ("val", "test", "firemaker")})[:1500])


if __name__ == "__main__":
    main()
