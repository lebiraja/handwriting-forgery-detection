"""Build the normalised page cache and the master manifest.

Per page: grayscale -> (IAM: crop to the handwritten area) -> background normalisation
-> resize to width 1600 -> uint8 .npy.  Unreadable / blank / duplicate pages are logged
in the manifest with status != ok and never reach training.

Run inside the container:
  python -m src.prep_cache --dataset iam --out /data/cache
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)
WIDTH = 1600
MIN_INK = 0.0025  # below this = blank page (CVL short texts with thin pens reach ~0.004)


def find_separator_rows(gray: np.ndarray, klen: int = 400, frac: float = 0.3) -> list[int]:
    """Rows holding a long thin horizontal line (IAM form separators).

    A horizontal morphological opening with a 400 px kernel keeps only long straight runs,
    so it is robust to slight skew and ignores handwriting strokes."""
    h, w = gray.shape
    bw = (gray < 170).astype(np.uint8)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
    rows = np.where(opened.sum(axis=1) > frac * w)[0]
    if len(rows) == 0:
        return []
    groups, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > 8:
            groups.append(int((start + prev) // 2)); start = r
        prev = r
    groups.append(int((start + prev) // 2))
    return groups


def crop_iam(gray: np.ndarray) -> tuple[np.ndarray, str]:
    """Keep only the handwritten area: below the 2nd separator, above the 'Name:' footer line."""
    h, w = gray.shape
    seps = find_separator_rows(gray)
    head = [r for r in seps if 0.14 * h < r < 0.26 * h]
    if not head:  # fainter / slightly skewed line: retry with a shorter kernel
        seps = find_separator_rows(gray, 250, 0.15)
        head = [r for r in seps if 0.14 * h < r < 0.26 * h]
    foot = [r for r in seps if r > 0.65 * h]
    top = head[0] + int(0.012 * h) if head else int(0.22 * h)
    bot = foot[-1] - int(0.012 * h) if foot else int(0.78 * h)
    how = ("H" if head else "h") + ("F" if foot else "f")  # capital = found, lower = fallback
    return gray[top:bot, int(0.05 * w): int(0.97 * w)], how


def crop_cvl(gray: np.ndarray) -> tuple[np.ndarray, str]:
    """CVL pages: printed source text + page number on top, handwriting below the separator."""
    h, w = gray.shape
    seps = find_separator_rows(gray, 300, 0.2)
    head = [r for r in seps if 0.12 * h < r < 0.32 * h]
    top = head[0] + int(0.012 * h) if head else int(0.27 * h)
    return gray[top:int(0.97 * h), int(0.03 * w): int(0.97 * w)], ("H" if head else "h")


def normalise(gray: np.ndarray) -> np.ndarray:
    """Remove paper tone / scanner shading so it cannot identify the writer."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(bg, (0, 0), 20).astype(np.float32) + 1.0
    out = np.clip(gray.astype(np.float32) / bg * 255.0, 0, 255)
    return out.astype(np.uint8)


def process(job: dict) -> dict:
    row = dict(job)
    try:
        gray = cv2.imread(job["src"], cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            raise ValueError("unreadable")
        how = "none"
        if job["dataset"] == "iam":
            gray, how = crop_iam(gray)
        elif job["dataset"] == "cvl":
            gray, how = crop_cvl(gray)
        h, w = gray.shape
        if h < 200 or w < 200 or max(h, w) / min(h, w) > 4:
            raise ValueError(f"bad geometry {h}x{w}")
        gray = normalise(gray)
        gray = cv2.resize(gray, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        ink = float((gray < 0.75 * np.median(gray)).mean())
        row.update(crop=how, h=gray.shape[0], w=WIDTH, ink=round(ink, 5))
        if ink < MIN_INK:
            row["status"] = "blank"
            return row
        row["md5"] = hashlib.md5(gray.tobytes()).hexdigest()
        out = Path(job["cache"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, gray)
        row["status"] = "ok"
    except Exception as e:  # never crash the pool on one bad file
        row["status"] = f"error:{e}"
    return row


def jobs_iam(root: str, out: str) -> list[dict]:
    jobs = []
    for wdir in sorted(os.listdir(f"{root}/data")):
        for f in sorted(os.listdir(f"{root}/data/{wdir}")):
            if f.endswith(".png"):
                jobs.append(dict(dataset="iam", writer=f"iam_{wdir}", page=Path(f).stem, cond="form",
                                 src=f"{root}/data/{wdir}/{f}", cache=f"{out}/iam/{Path(f).stem}.npy"))
    return jobs


def jobs_firemaker(root: str, out: str) -> list[dict]:
    jobs = []
    for cond in sorted(os.listdir(root)):
        for f in sorted(os.listdir(f"{root}/{cond}")):
            if f.endswith(".tif"):
                jobs.append(dict(dataset="fm", writer=f"fm_{f[:3]}", page=Path(f).stem, cond=cond,
                                 src=f"{root}/{cond}/{f}", cache=f"{out}/fm/{Path(f).stem}.npy"))
    return jobs


def jobs_cvl(root: str, out: str) -> list[dict]:
    jobs = []
    for part in ("trainset", "testset"):
        for f in sorted(os.listdir(f"{root}/{part}/pages")):
            if f.endswith(".tif"):
                wid, txt = Path(f).stem.split("-")
                jobs.append(dict(dataset="cvl", writer=f"cvl_{wid}", page=Path(f).stem, cond=f"text{txt}",
                                 src=f"{root}/{part}/pages/{f}", cache=f"{out}/cvl/{Path(f).stem}.npy"))
    return jobs


def jobs_kaggle(root: str, out: str) -> list[dict]:
    """Kaggle real-vs-generated: group = filename without page suffix (pages of one source document)."""
    import re
    jobs = []
    for cond, d in (("real", "actual handwriting"), ("generated", "generated handwriting")):
        for f in sorted(os.listdir(f"{root}/{d}")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                grp = re.sub(r"[_-]?(page)?[-_]?\d+$", "", Path(f).stem)
                jobs.append(dict(dataset="kg", writer=f"kg_{cond}_{grp}", page=Path(f).stem, cond=cond,
                                 src=f"{root}/{d}/{f}", cache=f"{out}/kg/{cond}_{abs(hash(f)) % 10**9}_{Path(f).stem[:40]}.npy"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iam", "fm", "cvl", "kg"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = {"iam": jobs_iam, "fm": jobs_firemaker, "cvl": jobs_cvl, "kg": jobs_kaggle}[a.dataset](a.root, a.out)
    if a.limit:
        jobs = jobs[: a.limit]
    with Pool(a.workers) as p:
        rows = p.map(process, jobs, chunksize=8)
    df = pd.DataFrame(rows)
    # duplicates (same md5) -> keep first, flag the rest
    dup = df.duplicated("md5", keep="first") & df.status.eq("ok")
    df.loc[dup, "status"] = "duplicate"
    os.makedirs(f"{a.out}/manifests", exist_ok=True)
    df.to_csv(f"{a.out}/manifests/{a.dataset}_pages.csv", index=False)
    print(json.dumps(df.status.value_counts().to_dict()))
    if a.dataset in ("iam", "cvl"):
        print(df.crop.value_counts().to_dict())


if __name__ == "__main__":
    sys.exit(main())
