import numpy as np
import pandas as pd
import torch

from src.data import page_patches
from src.manifests import check
from src.metrics import verification_metrics
from src.model import supcon_loss

M = "/data/cache/manifests/master.csv"


def test_manifest_leak_free():
    m = pd.read_csv(M)
    stats = check(m)  # raises on any overlap / Firemaker leak / <2 pages in eval
    assert stats["train"]["writers"] > 300 and stats["test"]["writers"] > 100
    assert m.path.map(lambda p: __import__("os").path.exists(p)).all()


def test_patch_sampler_real_pages():
    m = pd.read_csv(M); m = m[m.split == "val"].sample(50, random_state=0)
    for _, r in m.iterrows():
        x = page_patches(r.path, r["view"], 16, 128, train=False)
        assert x.shape == (16, 1, 128, 128)
        assert (x > 0.02).mean() > 0.003  # real ink present
        assert np.isfinite(x).all()


def test_eval_patches_deterministic():
    r = pd.read_csv(M).query("split=='val'").iloc[0]
    a = page_patches(r.path, r["view"], 16, 128, train=False)
    b = page_patches(r.path, r["view"], 16, 128, train=False)
    assert np.array_equal(a, b)


def test_train_patches_vary():
    r = pd.read_csv(M).query("split=='train'").iloc[0]
    assert not np.array_equal(page_patches(r.path, r["view"], 8, 128, True), page_patches(r.path, r["view"], 8, 128, True))


def test_eer_extremes():
    w = ["a"] * 5 + ["b"] * 5
    good = np.r_[np.tile([1.0, 0.0], (5, 1)), np.tile([0.0, 1.0], (5, 1))]
    assert verification_metrics(good, w)["eer"] == 0.0
    rnd = np.random.RandomState(0).randn(400, 16); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    assert 0.4 < verification_metrics(rnd, [str(i // 2) for i in range(400)])["eer"] < 0.6


def test_supcon_decreases():
    torch.manual_seed(0)
    x = torch.randn(16, 8, requires_grad=True); lab = torch.arange(8).repeat_interleave(2)
    opt = torch.optim.Adam([x], 0.1); first = None
    for _ in range(60):
        e = torch.nn.functional.normalize(x, dim=-1); l = supcon_loss(e, lab, 0.1)
        first = first or l.item(); opt.zero_grad(); l.backward(); opt.step()
    assert l.item() < first * 0.6
