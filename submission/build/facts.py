"""Every number used in the PRS report and the review deck, read from train/runs (no hand-typed results)."""
import json, glob
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2] / "train" / "runs"


def _j(p): return json.load(open(p))
def ms(v, scale=100, nd=2):
    v = np.array(v) * scale; return f"{v.mean():.{nd}f} ± {v.std():.{nd}f}"
def mean(v, scale=100, nd=2): return f"{np.mean(v) * scale:.{nd}f}"


ev = [_j(R / f"main_s{s}/eval_test.json") for s in (0, 1, 2)]
vd = [_j(R / f"main_s{s}/verdict_test.json") for s in (0, 1, 2)]
mr = [_j(R / f"main_s{s}/multiref_test.json") for s in (0, 1, 2)]
base = {n: _j(R / f"baselines/{n}_test.json") for n in ("b0", "b2", "b3")}
F = {}
for ds in ("iam", "cvl"):
    F[f"{ds}_eer"] = ms([e["test"][ds]["single_ref"]["eer"] for e in ev])
    F[f"{ds}_eer_m"] = float(np.mean([e["test"][ds]["single_ref"]["eer"] for e in ev]) * 100)
    F[f"{ds}_auc"] = ms([e["test"][ds]["single_ref"]["auc"] for e in ev], 1, 3)
    F[f"{ds}_val_eer"] = ms([e["val"][ds]["single_ref"]["eer"] for e in ev])
    F[f"{ds}_far1"] = ms([e["test"][ds]["single_ref"]["far@frr1"] for e in ev], 100, 1)
    F[f"{ds}_far10"] = ms([e["test"][ds]["single_ref"]["far@frr10"] for e in ev], 100, 1)
    for n in base:
        F[f"{n}_{ds}_eer"] = float(np.mean([base[n][k][ds]["eer"] for k in base[n]]) * 100)
        F[f"{n}_{ds}_auc"] = float(np.mean([base[n][k][ds]["auc"] for k in base[n]]))
    for n in (1, 2, 3):
        for how in ("centroid", "max", "mean", "top2"):
            F[f"mr_{ds}_{how}_{n}"] = float(np.mean([m[ds]["common_N123"]["results"][f"N{n}_{how}"]["eer"] for m in mr]) * 100)
    F[f"mr_{ds}_writers"] = mr[0][ds]["common_N123"]["writers"]
    v = [x[ds] for x in vd]
    for k in ("accepted", "rejected", "inconclusive", "error_on_decided", "coverage"):
        F[f"v_{ds}_{k}"] = float(np.mean([x[k] for x in v]) * 100)
    F[f"ece_raw_{ds}"] = float(np.mean([e["calibration"][ds]["ece_raw"] for e in ev]))
    F[f"ece_platt_{ds}"] = float(np.mean([e["calibration"][ds]["ece_platt"] for e in ev]))
    F[f"ece_iso_{ds}"] = float(np.mean([e["calibration"][ds]["ece_iso"] for e in ev]))
    for cond in ("generated", "real"):
        F[f"syn_{ds}_{cond}"] = float(np.mean([e["synthetic"][ds][cond]["false_accept"] for e in ev]) * 100)
F["iam_n5"] = {n: float(np.mean([m["iam"]["common_N1235"]["results"][f"N{n}_centroid"]["eer"] for m in mr]) * 100) for n in (1, 2, 3, 5)}
for k, name in (("p4", "p1_vs_p4-self-natural"), ("p2", "p1_vs_p2-copy-upper"), ("p3", "p1_vs_p3-copy-forged")):
    F[f"fm_{k}_eer"] = float(np.mean([e["firemaker"][name]["eer"] for e in ev]) * 100)
    F[f"fm_{k}_eer_s"] = ms([e["firemaker"][name]["eer"] for e in ev], 100, 2)
    F[f"fm_{k}_gen"] = float(np.mean([e["firemaker"][name]["genuine_mean"] for e in ev]))
    F[f"fm_{k}_imp"] = float(np.mean([e["firemaker"][name]["impostor_mean"] for e in ev]))
names = [("abl_main40", "Full model (40 epochs)"), ("abl_no_distill", "No distillation"), ("abl_mean_pool", "Mean pooling"), ("abl_no_cmatch", "No content-matched batches"),
         ("abl_no_mining", "No writer mining"), ("abl_k8", "8 patches per page"), ("abl_no_aug", "No augmentation")]
F["abl"] = [(l, *(_j(R / f"{n}/best_val.json")[k]["eer"] * 100 if k != "mean" else _j(R / f"{n}/best_val.json")["mean_eer"] * 100 for k in ("iam", "cvl", "mean"))) for n, l in names]
F["main_val_mean"] = [_j(R / f"main_s{s}/best_val.json")["mean_eer"] * 100 for s in (0, 1, 2)]
d = _j(R / "main_s0/explain/deletion.json")
F["del_full"], F["del_top"], F["del_rand"] = d["full"][0], d["top_deleted"][0], d["random_deleted"][0]
if __name__ == "__main__":
    for k, v in F.items(): print(k, v)
