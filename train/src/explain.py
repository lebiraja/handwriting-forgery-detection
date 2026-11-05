"""Attention heatmaps and the deletion test (needs the GPU).

python -m src.explain --run /app/runs/main_s0 --out /app/runs/main_s0/explain --split test
 * heatmap PNGs: patch boxes on the page, opacity = attention weight (genuine pair and impostor pair examples)
 * deletion test: drop the top-attended patches vs the same number of random patches, compare the
   similarity to a same-writer reference page. A real explanation should hurt more than random deletion.
"""
import argparse, json, os
import cv2
import numpy as np
import pandas as pd
import torch

from .data import choose_positions, load_view, page_patches
from .eval import load_model


@torch.no_grad()
def page_att(model, path, view, k, s, dev, seed=0):
    img = load_view(path, view); ys, xs = choose_positions(img, k, s, np.random.RandomState(seed))
    x = np.stack([img[y:y + s, x_:x_ + s] for y, x_ in zip(ys, xs)]); x = torch.from_numpy((1.0 - x.astype(np.float32) / 255.0)[:, None])[None].to(dev)
    e, f, a = model(x, return_all=True)
    return img, ys, xs, e[0].cpu(), f[0].cpu(), a[0].cpu().numpy()


def embed_keep(f, a, keep):
    """Re-pool the kept patches with their original attention weights, renormalised."""
    w = torch.tensor(a)[keep]; w = w / w.sum()
    return torch.nn.functional.normalize((w[:, None] * f[keep]).sum(0), dim=0)


def draw_heat(img, ys, xs, att, size=128):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); amax = float(att.max())
    for y, x, wt in zip(ys, xs, att):
        lay = out.copy(); cv2.rectangle(lay, (int(x), int(y)), (int(x) + size, int(y) + size), (0, 0, 255), -1)
        al = 0.55 * float(wt) / amax; out = cv2.addWeighted(lay, al, out, 1 - al, 0)
    return cv2.resize(out, None, fx=0.5, fy=0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="/data/cache/manifests/master.csv"); ap.add_argument("--split", default="val")
    ap.add_argument("--pairs", type=int, default=60); ap.add_argument("--k", type=int, default=64); ap.add_argument("--n_heat", type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"; model, _ = load_model(a.run, dev)
    m = pd.read_csv(a.manifest); m = m[(m.split == a.split) & m.dataset.isin(["iam", "cvl"])]
    rng = np.random.RandomState(0); writers = [w for w, g in m.groupby("writer") if len(g) >= 2]
    res = {"top_deleted": [], "random_deleted": [], "full": []}; drop_frac = 0.25
    for i in range(a.pairs):
        w = writers[rng.randint(len(writers))]; g = m[m.writer == w].sample(2, random_state=int(rng.randint(1e6)))
        (q, r) = [row for _, row in g.iterrows()]
        img, ys, xs, eq, fq, aq = page_att(model, q.path, q["view"], a.k, 128, dev); _, _, _, er, _, _ = page_att(model, r.path, r["view"], a.k, 128, dev)
        n_del = int(a.k * drop_frac); order = np.argsort(-aq); keep_top = np.setdiff1d(np.arange(a.k), order[:n_del])
        keep_rand = np.setdiff1d(np.arange(a.k), rng.choice(a.k, n_del, replace=False))
        res["full"].append(float(eq @ er)); res["top_deleted"].append(float(embed_keep(fq, aq, keep_top) @ er)); res["random_deleted"].append(float(embed_keep(fq, aq, keep_rand) @ er))
        if i < a.n_heat:
            cv2.imwrite(f"{a.out}/heat_{i}_{w}.png", draw_heat(img, ys, xs, aq))
    s = {k: [float(np.mean(v)), float(np.std(v))] for k, v in res.items()}
    s["drop_top_minus_drop_random"] = float(np.mean(np.array(res["full"]) - np.array(res["top_deleted"])) - np.mean(np.array(res["full"]) - np.array(res["random_deleted"])))
    s["pairs"] = a.pairs; s["deleted_fraction"] = drop_frac
    json.dump(s, open(f"{a.out}/deletion.json", "w"), indent=1); print(json.dumps(s))


if __name__ == "__main__":
    main()
