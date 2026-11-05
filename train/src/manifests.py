"""Writer-disjoint master manifest from the per-dataset page tables.

Columns: path, writer, split, dataset, cond, view, page
  view = full | top | bottom   (top/bottom = IAM single-page writers, training only)
Run: python -m src.manifests --cache /data/cache
"""
import argparse
import json

import numpy as np
import pandas as pd

SEED = 42
MAX_TRAIN_PAGES = 10


def split_multipage(writers: list[str], n_test: int, n_val: int, rng) -> dict[str, str]:
    writers = sorted(writers)
    rng.shuffle(writers)
    out = {}
    for i, w in enumerate(writers):
        out[w] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def build_iam(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df[df.status == "ok"].copy()
    counts = df.groupby("writer").page.count()
    multi = counts[counts >= 2].index.tolist()
    single = counts[counts == 1].index.tolist()
    n_test = min(100, int(round(len(multi) * 0.33)))
    n_val = min(50, int(round(len(multi) * 0.17)))
    sp = split_multipage(multi, n_test, n_val, rng)
    rows = []
    for w, g in df.groupby("writer"):
        g = g.sort_values("page")
        if w in sp:
            s = sp[w]
            if s == "train" and len(g) > MAX_TRAIN_PAGES:
                g = g.sample(MAX_TRAIN_PAGES, random_state=SEED)
            for _, r in g.iterrows():
                rows.append((r.cache, w, s, "iam", "form", "full", r.page))
        else:  # single-page writer: two half views, training only
            r = g.iloc[0]
            for v in ("top", "bottom"):
                rows.append((r.cache, w, "train", "iam", "form", v, r.page))
    return pd.DataFrame(rows, columns=["path", "writer", "split", "dataset", "cond", "view", "page"])


def build_fm(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.status == "ok"]
    return pd.DataFrame({"path": df.cache, "writer": df.writer, "split": "test", "dataset": "fm",
                         "cond": df.cond, "view": "full", "page": df.page})


def build_cvl(df: pd.DataFrame, rng) -> pd.DataFrame:
    # English only: text 6 is the German Faust excerpt
    df = df[(df.status == "ok") & (df.cond != "text6")]
    counts = df.groupby("writer").page.count()
    ws = counts[counts >= 2].index.tolist()
    n_test = int(round(len(ws) * 0.2)); n_val = int(round(len(ws) * 0.1))
    sp = split_multipage(ws, n_test, n_val, rng)
    d = df[df.writer.isin(sp)]
    return pd.DataFrame({"path": d.cache, "writer": d.writer, "split": d.writer.map(sp), "dataset": "cvl",
                         "cond": d.cond, "view": "full", "page": d.page})


def check(m: pd.DataFrame) -> dict:
    """Leakage guards: raise instead of training on a bad split."""
    sets = {s: set(m[m.split == s].writer) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]), "train/val writer overlap"
    assert not (sets["train"] & sets["test"]), "train/test writer overlap"
    assert not (sets["val"] & sets["test"]), "val/test writer overlap"
    assert not m[(m.dataset == "fm") & (m.split != "test")].shape[0], "Firemaker leaked outside test"
    ev = m[m.split.isin(["val", "test"])]
    assert (ev.view == "full").all(), "half views must be training-only"
    assert ev.groupby("writer").page.nunique().min() >= 2, "eval writer with <2 pages"
    return {s: {"writers": len(w), "pages": int((m.split == s).sum())} for s, w in sets.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    rng = np.random.RandomState(SEED)
    parts = [build_iam(pd.read_csv(f"{a.cache}/manifests/iam_pages.csv"), rng),
             build_fm(pd.read_csv(f"{a.cache}/manifests/fm_pages.csv"))]
    try:
        parts.append(build_cvl(pd.read_csv(f"{a.cache}/manifests/cvl_pages.csv"), rng))
    except FileNotFoundError:
        print("CVL page table not found: skipping CVL")
    m = pd.concat(parts, ignore_index=True)
    stats = check(m)
    m.to_csv(f"{a.cache}/manifests/master.csv", index=False)
    print(json.dumps(stats))
    print(m.groupby(["dataset", "split"]).writer.nunique().to_string())


if __name__ == "__main__":
    main()
