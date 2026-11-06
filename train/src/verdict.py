"""Recompute the three-way verdict from the saved test scores (numpy only, no model, no new test data).

Why: eval.three_way as first run could output tau_high < tau_low when a corpus is separable on validation (IAM), so one score
was both accepted and rejected. The fixed rule (see DECISIONS.md) uses a single midpoint threshold in that case. Thresholds still
come from VALIDATION trials only.   python -m src.verdict --runs runs
"""
import argparse, glob, json
import numpy as np


def three_way(vy, vs, ty, ts, far=0.01, frr=0.01):
    hi = float(np.quantile(vs[vy == 0], 1 - far)); lo = float(np.quantile(vs[vy == 1], frr)); sep = hi < lo
    if sep: hi = lo = (hi + lo) / 2
    acc, rej = ts > hi if sep else ts >= hi, ts <= lo if not sep else ts <= lo
    if sep: acc = ts > hi; rej = ~acc
    dec = acc | rej; err = ((acc & (ty == 0)) | (rej & (ty == 1))).sum()
    return {"tau_high": hi, "tau_low": lo, "separable_on_val": bool(sep), "coverage": float(dec.mean()), "error_on_decided": float(err / max(1, dec.sum())),
            "accepted": float(acc.mean()), "rejected": float(rej.mean()), "inconclusive": float((~dec).mean()),
            "false_accept_all": float((acc & (ty == 0)).sum() / max(1, (ty == 0).sum())), "false_reject_all": float((rej & (ty == 1)).sum() / max(1, (ty == 1).sum()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); a = ap.parse_args()
    for f in sorted(glob.glob(f"{a.runs}/main_s*/eval_test_scores.npz")):
        z = np.load(f); out = {d: three_way(z[f"cal_{d}_val_y"], z[f"cal_{d}_val_s"], z[f"cal_{d}_test_y"], z[f"cal_{d}_test_s"]) for d in ("iam", "cvl")}
        json.dump(out, open(f.replace("eval_test_scores.npz", "verdict_test.json"), "w"), indent=1); print(f, json.dumps(out)[:400])


if __name__ == "__main__":
    main()
