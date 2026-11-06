"""Inference engine for the demo: same preprocessing, patch sampling and model as train/src (no new logic)."""
import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.data import choose_positions
from src.model import PageEncoder
from src.prep_cache import MIN_INK, WIDTH, normalise

K, S = 64, 128                       # patches per page, patch size (as in evaluation)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
CALIB = Path(__file__).parent / "calibration.json"
MAX_REFS = 5
torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "3")))
_lock = threading.Lock()


class PageError(ValueError):
    """A problem with an uploaded page that the user can fix."""


def load_model() -> PageEncoder:
    cfg = yaml.safe_load(open(MODEL_DIR / "config.yaml"))
    m = PageEncoder(cfg["embed_dim"], pool=cfg["pool"])
    m.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location="cpu"))
    return m.eval()


def preprocess(raw: bytes, label: str) -> tuple[np.ndarray, list[str]]:
    """Upload bytes -> normalised grayscale page of width 1600 (the cache format) + warnings."""
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.size == 0:
        raise PageError(f"{label}: could not read this file as an image (use JPG or PNG).")
    h, w = gray.shape
    warns = []
    if min(h, w) < 200:
        raise PageError(f"{label}: image is too small ({w}x{h}).")
    if max(h, w) / min(h, w) > 4:
        raise PageError(f"{label}: looks like a thin strip. Upload a full page of handwriting.")
    if w < 1000:
        warns.append(f"{label}: low resolution ({w}px wide); a flat 150-300 dpi scan works best.")
    if w > 2400:
        gray = cv2.resize(gray, (2400, int(h * 2400 / w)), interpolation=cv2.INTER_AREA)
    gray = normalise(gray)
    h, w = gray.shape
    gray = cv2.resize(gray, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    ink = float((gray < 0.75 * np.median(gray)).mean())
    if ink < MIN_INK:
        raise PageError(f"{label}: the page looks blank (almost no ink found).")
    if ink > 0.15:
        warns.append(f"{label}: very dark page (shadow, border or printed form?). Crop to the handwriting.")
    return gray, warns


def _embed(model, gray: np.ndarray, label: str):
    seed = int(hashlib.md5(gray.tobytes()).hexdigest()[:8], 16)
    try:
        ys, xs = choose_positions(gray, K, S, np.random.RandomState(seed))
    except ValueError as e:
        raise PageError(f"{label}: {e}.")
    patches = np.stack([gray[y:y + S, x:x + S] for y, x in zip(ys, xs)])
    x = torch.from_numpy((1.0 - patches.astype(np.float32) / 255.0)[:, None])[None]
    with torch.no_grad():
        e, _, a = model(x, return_all=True)
    return e[0].numpy(), a[0].numpy(), ys, xs, patches


def _b64(img: np.ndarray, ext=".jpg", q=82) -> str:
    ok, enc = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/%s;base64,%s" % ("jpeg" if ext == ".jpg" else "png", base64.b64encode(enc).decode())


def heatmap(gray, ys, xs, att) -> str:
    """Attention overlay: lavender boxes on the page, opacity = attention weight."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    amax = float(att.max())
    for y, x, wt in sorted(zip(ys, xs, att), key=lambda t: t[2]):
        lay = out.copy()
        cv2.rectangle(lay, (int(x), int(y)), (int(x) + S, int(y) + S), (241, 150, 173), -1)  # lavender #AD96F1 in BGR
        al = 0.6 * float(wt) / amax
        out = cv2.addWeighted(lay, al, out, 1 - al, 0)
    h, w = out.shape[:2]
    return _b64(cv2.resize(out, (900, int(h * 900 / w)), interpolation=cv2.INTER_AREA))


def montage(patches, att, n=6) -> str:
    top = np.argsort(-att)[:n]
    return _b64(np.hstack([patches[i] for i in top]), ".png")


class Engine:
    def __init__(self):
        self.model = load_model()
        self.calib = json.load(open(CALIB)) if CALIB.exists() else None
        self.model_name = MODEL_DIR.name if MODEL_DIR.name != "model" else "main_s0"
        with _lock:  # warm-up so the first request is fast
            self.model(torch.zeros(1, K, 1, S, S))

    def _calibrated(self, score: float) -> dict:
        c = self.calib
        if not c:
            return {"verdict": "uncalibrated", "tau_low": None, "tau_high": None}
        edges = np.array(c["bins"])
        i = int(np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2))
        g, im = np.array(c["genuine"], float), np.array(c["impostor"], float)
        verdict = ("consistent" if score >= c["tau_high"] else "likely_forged" if score <= c["tau_low"] else "inconclusive")
        return {"verdict": verdict, "tau_low": c["tau_low"], "tau_high": c["tau_high"],
                "genuine_below": float(g[:i].sum() / g.sum()), "impostor_above": float(im[i + 1:].sum() / im.sum())}

    def verify(self, refs: list[bytes], questioned: bytes) -> dict:
        t0 = time.time()
        if not 1 <= len(refs) <= MAX_REFS:
            raise PageError(f"Give between 1 and {MAX_REFS} reference pages.")
        warns, pages = [], []
        for i, raw in enumerate(refs):
            g, w = preprocess(raw, f"Reference {i + 1}" if len(refs) > 1 else "Reference")
            pages.append(g); warns += w
        qg, w = preprocess(questioned, "Questioned page"); warns += w
        with _lock:
            R = [_embed(self.model, g, "Reference") for g in pages]
            Q = _embed(self.model, qg, "Questioned page")
        refs_e = np.stack([r[0] for r in R])
        cen = refs_e.mean(0); cen /= np.linalg.norm(cen) + 1e-9      # centroid rule, as in eval.py
        score = float(cen @ Q[0])
        per_ref = [float(r @ Q[0]) for r in refs_e]
        res = {"score": score, "per_reference": per_ref, "n_refs": len(refs), "warnings": warns,
               **self._calibrated(score),
               "ref_heat": heatmap(pages[0], R[0][2], R[0][3], R[0][1]), "ref_patches": montage(R[0][4], R[0][1]),
               "q_heat": heatmap(qg, Q[2], Q[3], Q[1]), "q_patches": montage(Q[4], Q[1]),
               "ms": int((time.time() - t0) * 1000)}
        return res
