"""Page-level data: a page is a bag of ink-rich patches sampled from the cached page array."""
import hashlib
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)
INK_LEVELS = (0.05, 0.03, 0.015, 0.008)  # progressive fallback, never silent padding
MIN_USABLE_INK = 0.004


def load_view(path: str, view: str) -> np.ndarray:
    img = np.load(path, mmap_mode="r")
    h = img.shape[0]
    if view == "top":
        img = img[: h // 2]
    elif view == "bottom":
        img = img[h // 2:]
    return np.ascontiguousarray(img)


def ink_map(img: np.ndarray) -> np.ndarray:
    return (img < 0.75 * float(np.median(img))).astype(np.uint8)


def window_ink(integral: np.ndarray, ys: np.ndarray, xs: np.ndarray, s: int) -> np.ndarray:
    a = integral[ys + s, xs + s] - integral[ys, xs + s] - integral[ys + s, xs] + integral[ys, xs]
    return a / float(s * s)


def choose_positions(img: np.ndarray, k: int, s: int, rng: np.random.RandomState, tries: int = 1500):
    """k top-left corners of windows centred on real ink pixels, ink fraction >= the first level
    that yields >= k windows (progressive fallback, never silent padding)."""
    h, w = img.shape
    if h < s or w < s:
        raise ValueError(f"page too small {h}x{w}")
    ink = ink_map(img)
    py, px = np.nonzero(ink[:: 2, :: 2])  # subsampled ink pixels (speed)
    if len(py) == 0:
        raise ValueError("no ink in page")
    sel = rng.randint(0, len(py), tries)
    ys = np.clip(py[sel] * 2 - s // 2, 0, h - s); xs = np.clip(px[sel] * 2 - s // 2, 0, w - s)
    frac = window_ink(cv2.integral(ink), ys, xs, s)
    for lvl in INK_LEVELS:
        ok = np.where(frac >= lvl)[0]
        if len(ok) >= k:
            return ys[ok[:k]], xs[ok[:k]]
    ok = np.where(frac >= MIN_USABLE_INK)[0]
    if len(ok) == 0:
        raise ValueError("no usable ink in page")
    pick = ok[rng.randint(0, len(ok), k)]  # resample usable windows (with replacement), still real ink
    return ys[pick], xs[pick]


def augment_patch(p: np.ndarray, s: int, rng: random.Random) -> np.ndarray:
    """p: (1.25*s)^2 crop -> s x s patch with rotation/scale/contrast/blur/noise/JPEG."""
    big = p.shape[0]
    ang = rng.uniform(-3, 3); sc = rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((big / 2, big / 2), ang, sc)
    p = cv2.warpAffine(p, m, (big, big), borderMode=cv2.BORDER_REPLICATE)
    o = (big - s) // 2
    p = p[o:o + s, o:o + s].astype(np.float32)
    p = (p - 128) * rng.uniform(0.8, 1.2) + 128 + rng.uniform(-15, 15)
    if rng.random() < 0.3:
        p = cv2.GaussianBlur(p, (0, 0), rng.uniform(0.4, 1.0))
    if rng.random() < 0.3:
        p = p + np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(1, 5), p.shape).astype(np.float32)
    p = np.clip(p, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 90)])
        p = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return p


def page_patches(path: str, view: str, k: int, s: int, train: bool, seed: int | None = None, shift: int = 0, augment: bool = True) -> np.ndarray:
    img = load_view(path, view)
    if train:
        pad = int(s * 0.25) // 2
        big = s + 2 * pad
        rng_np = np.random.RandomState(random.randint(0, 2**31 - 1))
        ys, xs = choose_positions(img, k, s, rng_np)
        r = random.Random(random.random())
        out = []
        for y, x in zip(ys, xs):
            y0, x0 = max(0, min(y - pad, img.shape[0] - big)), max(0, min(x - pad, img.shape[1] - big))
            crop = img[y0:y0 + big, x0:x0 + big]
            if crop.shape != (big, big):
                crop = cv2.resize(crop, (big, big))
            if augment:
                out.append(augment_patch(crop, s, r))
            else:
                o = (big - s) // 2; out.append(crop[o:o + s, o:o + s])
        arr = np.stack(out)
    else:  # deterministic per page: same patches every evaluation
        sd = (seed if seed is not None else int(hashlib.md5((path + view).encode()).hexdigest()[:8], 16)) + shift
        ys, xs = choose_positions(img, k, s, np.random.RandomState(sd))
        arr = np.stack([img[y:y + s, x:x + s] for y, x in zip(ys, xs)])
    return (1.0 - arr.astype(np.float32) / 255.0)[:, None]  # ink = high


class PageDataset(Dataset):
    def __init__(self, manifest: str, split: str, k: int, size: int, train: bool, datasets=None, augment: bool = True):
        df = pd.read_csv(manifest)
        df = df[df.split == split]
        if datasets:
            df = df[df.dataset.isin(datasets)]
        self.df = df.reset_index(drop=True)
        self.k, self.size, self.train, self.shift, self.augment = k, size, train, 0, augment  # shift = TTA view index
        self.by_writer = defaultdict(list)
        for i, w in enumerate(self.df.writer):
            self.by_writer[w].append(i)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            x = page_patches(r.path, r["view"], self.k, self.size, self.train, shift=self.shift, augment=self.augment)
        except ValueError:
            if r["view"] == "full":
                raise
            x = page_patches(r.path, "full", self.k, self.size, self.train, augment=self.augment)  # empty half -> whole page
        return torch.from_numpy(x), r.writer, r.dataset


class BatchSampler:
    """Each batch = W writers x 2 different pages.

    * IAM batches: random writers; the 2 pages are 2 pages (or the 2 half-views) of the writer.
    * CVL batches (content-matched hard negatives): every writer in the batch writes the SAME two
      texts, so positives differ in content and negatives share content -> style, not text.
    * Optional kNN mining: with probability p_mine a batch is built from writers whose centroids
      are close (set via set_neighbours()).
    """

    def __init__(self, ds: PageDataset, writers_per_batch: int, cvl_frac: float = 0.4, p_mine: float = 0.5, seed: int = 0, content_matched: bool = True):
        self.ds, self.w, self.cvl_frac, self.p_mine, self.content_matched = ds, writers_per_batch, cvl_frac, p_mine, content_matched
        self.rng = random.Random(seed)
        d = ds.df
        self.iam_writers = [w for w, idx in ds.by_writer.items() if len(idx) >= 2 and d.dataset[idx[0]] == "iam"]
        cvl = d[d.dataset == "cvl"]
        self.cvl_text = {w: dict(zip(g.cond, g.index)) for w, g in cvl.groupby("writer")}
        self.neigh = {}
        n_iam = max(1, len(self.iam_writers) // self.w)
        self._n = int(n_iam / max(1e-6, 1 - cvl_frac)) if self.cvl_text else n_iam

    def __len__(self):
        return max(1, self._n)

    def set_neighbours(self, neigh: dict[str, list[str]]):
        self.neigh = neigh

    def _pick(self, pool):
        if self.neigh and self.rng.random() < self.p_mine:
            seed_w = self.rng.choice(pool)
            cand = [x for x in self.neigh.get(seed_w, []) if x in set(pool)]
            ws = ([seed_w] + cand)[: self.w]
            if len(ws) < self.w:
                ws += self.rng.sample([x for x in pool if x not in ws], self.w - len(ws))
            return ws
        return self.rng.sample(pool, self.w)

    def _iam_batch(self):
        batch = []
        for w in self._pick(self.iam_writers):
            batch += self.rng.sample(self.ds.by_writer[w], 2)
        return batch

    def _cvl_batch(self):
        texts = ["text1", "text2", "text3", "text4", "text7", "text8"]
        if not self.content_matched:  # ablation: every writer gets two random texts (no shared-text negatives)
            ws = self._pick(list(self.cvl_text))
            return [i for w in ws for i in (self.cvl_text[w][t] for t in self.rng.sample(sorted(self.cvl_text[w]), 2))]
        for _ in range(20):
            ta, tb = self.rng.sample(texts, 2)
            pool = [w for w, t in self.cvl_text.items() if ta in t and tb in t]
            if len(pool) >= self.w:
                return [i for w in self._pick(pool) for i in (self.cvl_text[w][ta], self.cvl_text[w][tb])]
        return self._iam_batch()

    def __iter__(self):
        for _ in range(len(self)):
            yield self._cvl_batch() if (self.cvl_text and self.rng.random() < self.cvl_frac) else self._iam_batch()
