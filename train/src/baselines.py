"""Baselines evaluated with the same protocol as the model.

B0 random-init CNN, B2 frozen DINOv2-small (mean of patch CLS), B3 HOG.
python -m src.baselines --split val --names b0 b2 b3
"""
import argparse, json, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PageDataset
from .metrics import verification_metrics
from .model import PageEncoder
from .train import load_teacher, teacher_embed


def hog_page(x: np.ndarray) -> np.ndarray:  # x: [K,1,H,W] ink-high floats
    h = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
    d = np.mean([h.compute(((1 - p[0]) * 255).astype(np.uint8)).ravel() for p in x], axis=0)
    return d / (np.linalg.norm(d) + 1e-9)


@torch.no_grad()
def embed(name, ds, dev, workers, seed):
    torch.manual_seed(seed)
    if name == "b0":
        m = PageEncoder(256).to(dev).eval()
    elif name == "b2":
        t = load_teacher("/models/dinov2-small", dev)
    out, ws, dn = [], [], []
    for x, w, d in DataLoader(ds, batch_size=8, num_workers=workers):
        if name == "b0":
            with torch.autocast("cuda", dtype=torch.float16):
                e = m(x.to(dev)).float().cpu()
        elif name == "b2":
            with torch.autocast("cuda", dtype=torch.float16):
                e = teacher_embed(t, x.flatten(0, 1).to(dev)).view(x.shape[0], x.shape[1], -1).mean(1)
            e = F.normalize(e, dim=-1).cpu()
        else:
            e = torch.from_numpy(np.stack([hog_page(p.numpy()) for p in x]))
        out.append(e); ws += list(w); dn += list(d)
    return torch.cat(out).numpy(), np.array(ws), np.array(dn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data/cache/manifests/master.csv")
    ap.add_argument("--split", default="val"); ap.add_argument("--names", nargs="+", default=["b0", "b2", "b3"])
    ap.add_argument("--datasets", nargs="+", default=["iam", "cvl"]); ap.add_argument("--out", default="/app/runs/baselines")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ds = PageDataset(a.manifest, a.split, 64, 128, False, a.datasets)
    for n in a.names:
        res = {}
        for sd in (a.seeds if n == "b0" else [0]):
            e, w, d = embed(n, ds, "cuda", 8, sd)
            res[f"seed{sd}"] = {k: verification_metrics(e[d == k], list(w[d == k])) for k in sorted(set(d))}
        json.dump(res, open(f"{a.out}/{n}_{a.split}.json", "w"), indent=1)
        print(n, json.dumps({s: {k: round(v["eer"], 4) for k, v in r.items()} for s, r in res.items()}), flush=True)


if __name__ == "__main__":
    main()
