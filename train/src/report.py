"""Paper figures and tables, generated ONLY from files under runs/ (no hand-typed numbers).

Runs without torch:  python src/report.py --runs runs --out ../paper
Figures that need data not produced yet (test evaluation, ablations) are skipped with a note.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# House style copied from the sem5 deeplearning paper (scripts/make_figures.py): STIX serif 8 pt, fixed categorical palette,
# markers/dashes duplicate the colour for grayscale print, light grid, no top/right spines, vector PDF.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
C = {"iam": BLUE, "cvl": ORANGE, "fm": GREEN}
NAMES = {"iam": "IAM", "cvl": "CVL"}
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.grid": True, "grid.color": "#e4e2dc", "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
FUSION = {"mean": (BLUE, "o", "-"), "max": (ORANGE, "s", "-"), "top2": (AQUA, "^", "--"), "centroid": (VIOLET, "D", "-")}
FUSION_NAME = {"mean": "mean", "max": "max", "top2": "top-2", "centroid": "centroid"}


def jl(p):
    return [json.loads(l) for l in open(p) if l.startswith("{")]


def save(fig, out, name):
    os.makedirs(f"{out}/figures", exist_ok=True)
    fig.savefig(f"{out}/figures/{name}.pdf"); plt.close(fig); print("wrote", name)


def finished(runs, name):
    """A run counts only if its log ends with the done record (never mix in a run that is still training)."""
    p = f"{runs}/{name}.log"
    return os.path.exists(p) and '"done": true' in open(p).read()


def main_runs(runs):
    return [f for f in sorted(glob.glob(f"{runs}/main_s[0-9]*/metrics.jsonl")) if finished(runs, f.split("/")[-2])]


def top_legend(fig, ax, ncol=4, y=1.08):
    h, lab = ax.get_legend_handles_labels(); fig.legend(h, lab, frameon=False, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y))


def fig_training(runs, out):
    fs = main_runs(runs)
    if not fs: return print("skip training curves: no finished main runs")
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for f in fs:
        v = [r for r in jl(f) if "val" in r]; seed = int(f.split("main_s")[1].split("/")[0])
        for d, mk in (("iam", "o"), ("cvl", "s")):
            ax[0].plot([r["epoch"] for r in v], [100 * r["val"][d]["eer"] for r in v], color=C[d], marker=mk, ms=2, lw=0.9, alpha=0.8, ls=["-", "--", ":"][seed % 3], label=f"{NAMES[d]}" if seed == 0 else None)
        t = jl(f); ax[1].plot([r["epoch"] for r in t], [r["loss"] for r in t], color=[BLUE, ORANGE, AQUA][seed % 3], lw=1.1, label=f"seed {seed}")
    ax[0].set(xlabel="epoch", ylabel="validation EER (%)", yscale="log", title="validation EER (line style = seed)"); ax[0].legend(frameon=False)
    ax[1].set(xlabel="epoch", ylabel="training loss", title="SupCon + distillation loss"); ax[1].legend(frameon=False)
    save(fig, out, "fig_training_curves")


def fig_baselines(runs, out):
    rows = {}
    for key, label in (("b0", "random\nCNN"), ("b3", "HOG"), ("b2", "DINOv2\nfrozen")):
        p = f"{runs}/baselines/{key}_val.json"
        if os.path.exists(p):
            d = json.load(open(p)); seeds = [d[s] for s in d]
            rows[label] = {k: (np.mean([s[k]["eer"] for s in seeds]), np.std([s[k]["eer"] for s in seeds])) for k in ("iam", "cvl")}
    p = f"{runs}/shuffle/metrics.jsonl"
    if os.path.exists(p):
        v = [r for r in jl(p) if "val" in r][-1]["val"]; rows["random\nfeat.+BN"] = {k: (v[k]["eer"], 0.0) for k in ("iam", "cvl")}
    ours = [json.load(open(f)) for f in sorted(glob.glob(f"{runs}/main_s[0-9]*/best_val.json")) if finished(runs, f.split("/")[-2])]
    if ours: rows["ours\n(student)"] = {k: (np.mean([o[k]["eer"] for o in ours]), np.std([o[k]["eer"] for o in ours])) for k in ("iam", "cvl")}
    if not rows: return print("skip baselines")
    order = [k for k in ("random\nCNN", "random\nfeat.+BN", "HOG", "DINOv2\nfrozen", "ours\n(student)") if k in rows]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for d, mk, off in (("iam", "o", -0.1), ("cvl", "s", 0.1)):
        ax.errorbar([i + off for i in range(len(order))], [100 * rows[k][d][0] for k in order], yerr=[100 * rows[k][d][1] for k in order], color=C[d], marker=mk, ls="none", capsize=2, label=NAMES[d])
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
    ax.set_xticks(range(len(order)), [k.replace("\\n", "\n") for k in order], fontsize=6.5); ax.margins(x=0.12)
    ax.set(ylabel="validation EER (%), log scale"); top_legend(fig, ax, 2)
    save(fig, out, "fig_baselines")


def fig_multiref(evals, out, split, tag):
    if not evals: return print("skip multi-reference: no eval files")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for ax, ds in zip(axes, ("iam", "cvl")):
        ends = []
        for how, (col, mk, ls) in FUSION.items():
            xs, ys, es = [], [], []
            for n in (1, 2, 3, 5):
                vals = [e[split][ds]["multi_ref"][f"N{n}_{how}"] for e in evals if f"N{n}_{how}" in e[split][ds]["multi_ref"]]
                if vals and min(v.get("writers", 99 if n < 5 else 0) for v in vals) >= 15:  # skip points built from < 15 writers
                    xs.append(n); ys.append(100 * np.mean([v["eer"] for v in vals])); es.append(100 * np.std([v["eer"] for v in vals]))
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ls=ls, capsize=1.5, elinewidth=0.6, label=FUSION_NAME[how]); ends.append((ys[-1], FUSION_NAME[how]))
        ax.set(xlabel="number of reference pages $N$", title=f"{NAMES[ds]} ({split})", xticks=[1, 2, 3, 5]); ax.margins(x=0.12)
    axes[0].set_ylabel(f"{split} EER (%), mean $\\pm$ sd over seeds"); top_legend(fig, axes[0], 4)
    save(fig, out, f"fig_multiref_{tag}")


def fig_ablation(runs, out):
    names = [("abl_main40", "full"), ("abl_no_distill", "no\ndistill."), ("abl_mean_pool", "mean\npooling"), ("abl_no_cmatch", "no content-\nmatching"),
             ("abl_no_mining", "no writer\nmining"), ("abl_k8", "8 patches\nper page"), ("abl_no_aug", "no\naugment.")]
    vals = [(l, json.load(open(f"{runs}/{n}/best_val.json"))) for n, l in names if os.path.exists(f"{runs}/{n}/best_val.json") and finished(runs, n)]
    if len(vals) < 2: return print("skip ablation: runs not finished")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for ax, d in zip(axes, ("iam", "cvl")):
        y = [100 * v[d]["eer"] for _, v in vals]; ax.plot(range(len(vals)), y, color=C[d], marker="o", ls="none")
        ax.axhline(y[0], color="#999", lw=0.7, ls=":"); ax.set_xticks(range(len(vals)), [l.replace("\\n", "\n") for l, _ in vals], fontsize=6.5, rotation=45, ha="right")
        ax.set_title(f"{NAMES[d]} validation EER (40 epochs, 1 seed)"); ax.margins(x=0.1)
    axes[0].set_ylabel("EER (%)")
    save(fig, out, "fig_ablation")


def fig_det_and_calibration(scores, out):
    if not os.path.exists(scores): return print("skip DET/calibration: no test scores yet")
    z = np.load(scores); from sklearn.metrics import roc_curve; from scipy.stats import norm
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for d, mk in (("iam", "o"), ("cvl", "s")):
        y, s = z[f"test_{d}_pair_y"], z[f"test_{d}_pair_s"]; fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
        ok = (fpr > 0) & (fnr > 0); ax[0].plot(norm.ppf(fpr[ok]), norm.ppf(fnr[ok]), color=C[d], lw=1.3, label=NAMES[d])
    t = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8]); ax[0].set_xticks(norm.ppf(t), [f"{100*v:g}" for v in t]); ax[0].set_yticks(norm.ppf(t), [f"{100*v:g}" for v in t])
    ax[0].set(xlabel="false accept rate (%)", ylabel="false reject rate (%)", title="DET curve, test pairs"); ax[0].legend(frameon=False)
    for d, mk in (("iam", "o"), ("cvl", "s")):
        y, p = z[f"cal_{d}_test_y"], z[f"cal_{d}_test_platt"]; idx = np.digitize(p, np.linspace(0, 1, 11)) - 1
        pts = [(p[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 10]
        if pts: ax[1].plot(*zip(*pts), color=C[d], marker=mk, label=NAMES[d])
    ax[1].plot([0, 1], [0, 1], color="#555", lw=0.8, ls="--"); ax[1].set(xlabel="predicted probability (Platt, fitted on validation)", ylabel="observed same-writer rate", title="reliability, $N{=}3$ centroid"); ax[1].legend(frameon=False)
    save(fig, out, "fig_det_calibration")


def fig_verdict(evals, out):
    if not evals or "three_way" not in evals[0]: return print("skip verdict figure: no test calibration yet")
    fig, ax = plt.subplots(figsize=(3.45, 2.3)); parts = (("accepted", "consistent", BLUE), ("inconclusive", "inconclusive", "#bbbbbb"), ("rejected", "likely forged", ORANGE)); bottom = np.zeros(2)
    for k, lab, col in parts:
        v = np.array([100 * np.mean([e["three_way"][d][k] for e in evals]) for d in ("iam", "cvl")]); ax.bar([0, 1], v, bottom=bottom, color=col, label=lab, width=0.55); bottom += v
    ax.set_xticks([0, 1], ["IAM", "CVL"]); ax.set(ylabel="share of test trials (%)"); top_legend(fig, ax, 3)
    save(fig, out, "fig_verdict")


def fig_firemaker(scores, out):
    if not os.path.exists(scores): return print("skip Firemaker: no test scores yet")
    z = np.load(scores); keys = [k for k in z.files if k.startswith("fm_") and k.endswith("_gen")]
    if not keys: return print("skip Firemaker: no Firemaker scores")
    fig, ax = plt.subplots(figsize=(3.45, 2.5)); lab = {"p4-self-natural": "p4 natural", "p2-copy-upper": "p2 upper case", "p3-copy-forged": "p3 disguised"}; col = {"p4-self-natural": BLUE, "p2-copy-upper": AQUA, "p3-copy-forged": RED}
    imp = np.concatenate([z[k.replace("_gen", "_imp")] for k in keys]); bins = np.linspace(min(imp.min(), min(z[k].min() for k in keys)), 1, 40)
    ax.hist(imp, bins, color="#bbbbbb", density=True, label="other writers")
    for k in keys:
        c = k[3:-4]; ax.hist(z[k], bins, histtype="step", lw=1.3, density=True, color=col.get(c), label=lab.get(c, c))
    ax.set(xlabel="cosine similarity to the p1 page of the same writer", ylabel="density"); top_legend(fig, ax, 2)
    save(fig, out, "fig_firemaker")


def fig_attention(explain_dir, out):
    import glob as g
    from matplotlib import image as mim
    fs = sorted(g.glob(f"{explain_dir}/heat_*.png"))[:4]
    if not fs: return print("skip attention figure: run src.explain first")
    fig, axes = plt.subplots(1, len(fs), figsize=(7.1, 2.6))
    for ax, f in zip(np.atleast_1d(axes), fs):
        ax.imshow(mim.imread(f)); ax.axis("off")
    save(fig, out, "fig_attention")


def ms(vals, scale=100, nd=2):
    v = np.array(vals) * scale
    return f"{v.mean():.{nd}f}" if len(v) == 1 else f"{v.mean():.{nd}f}$\\pm${v.std():.{nd}f}"


def write_table(out, name, lines):
    os.makedirs(f"{out}/tables", exist_ok=True); open(f"{out}/tables/{name}.tex", "w").write("\n".join(lines) + "\n"); print("wrote table", name)


def seed_evals(runs, fname):
    fs = [f"{runs}/{n}/{fname}" for n in sorted(os.listdir(runs)) if n.startswith("main_s") and os.path.isdir(f"{runs}/{n}") and os.path.exists(f"{runs}/{n}/{fname}")]
    return [json.load(open(f)) for f in fs]


def table_data(manifest, out):
    import csv, collections
    rows = list(csv.DictReader(open(manifest)))
    w = collections.defaultdict(set); n = collections.Counter()
    for r in rows:
        k = (r["dataset"], r["split"]); w[k].add(r["writer"]); n[k] += 1
    name = {"iam": "IAM", "cvl": "CVL (English)", "fm": "Firemaker"}
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "& \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Validation} & \\multicolumn{2}{c}{Test} \\\\", "Corpus & Writers & Pages & Writers & Pages & Writers & Pages \\\\", "\\midrule"]
    for d in ("iam", "cvl", "fm"):
        cells = []
        for sp in ("train", "val", "test"):
            cells += [str(len(w[(d, sp)])) if w[(d, sp)] else "--", str(n[(d, sp)]) if n[(d, sp)] else "--"]
        L.append(f"{name[d]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_data", L)


def tab_main(runs, out, split, fname):
    ev = seed_evals(runs, fname)
    if not ev: return print("skip tab_main: no", fname)
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "& \\multicolumn{2}{c}{IAM} & \\multicolumn{2}{c}{CVL} \\\\", "Model & EER & AUC & EER & AUC \\\\", "\\midrule"]
    for key, label in (("b0", "Random CNN"), ("b3", "HOG"), ("b2", "DINOv2-small (frozen)")):
        p = f"{runs}/baselines/{key}_{split}.json"
        if os.path.exists(p):
            d = json.load(open(p)); sd = [d[k] for k in d]
            cells = [ms([x[ds]["eer"] for x in sd]) if m == "eer" else ms([x[ds]["auc"] for x in sd], 1, 3) for ds in ("iam", "cvl") for m in ("eer", "auc")]
            L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    cells = [ms([e[split][ds]["single_ref"]["eer"] for e in ev]) if m == "eer" else ms([e[split][ds]["single_ref"]["auc"] for e in ev], 1, 3) for ds in ("iam", "cvl") for m in ("eer", "auc")]
    L += ["\\midrule", f"Ours, {len(ev)} seed(s) & " + " & ".join(f"\\textbf{{{c}}}" for c in cells) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    write_table(out, "tab_main", L)


def tab_firemaker(runs, out, fname):
    ev = seed_evals(runs, fname)
    if not ev or "firemaker" not in ev[0]: return print("skip tab_fm: no Firemaker results")
    names = {"p1_vs_p4-self-natural": "p1 vs p4 (natural, self-written)", "p1_vs_p2-copy-upper": "p1 vs p2 (upper case)", "p1_vs_p3-copy-forged": "p1 vs p3 (disguised)"}
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Pair (same writer) & EER & AUC & Same-writer cos. & Other-writer cos. \\\\", "\\midrule"]
    for k, lab in names.items():
        if k in ev[0]["firemaker"]:
            f = [e["firemaker"][k] for e in ev]
            L.append(f"{lab} & {ms([x['eer'] for x in f])} & {ms([x['auc'] for x in f], 1, 3)} & {ms([x['genuine_mean'] for x in f], 1, 3)} & {ms([x['impostor_mean'] for x in f], 1, 3)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; write_table(out, "tab_fm", L)


def tab_verdict(runs, out, fname):
    ev = seed_evals(runs, fname)
    if not ev or "three_way" not in ev[0]: return print("skip tab_verdict: no test calibration yet")
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "Corpus & Consistent & Likely forged & Inconclusive & Error (decided) & ECE (Platt) \\\\", "\\midrule"]
    for ds in ("iam", "cvl"):
        t = [e["three_way"][ds] for e in ev]; c = [e["calibration"][ds]["ece_platt"] for e in ev]
        L.append(f"{NAMES[ds]} & {ms([x['accepted'] for x in t], 100, 1)} & {ms([x['rejected'] for x in t], 100, 1)} & {ms([x['inconclusive'] for x in t], 100, 1)} & {ms([x['error_on_decided'] for x in t], 100, 2)} & {ms(c, 1, 3)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; write_table(out, "tab_verdict", L)


def tab_ablation(runs, out):
    names = [("abl_main40", "Full model (40 epochs)"), ("abl_no_distill", "No distillation"), ("abl_mean_pool", "Mean pooling"), ("abl_no_cmatch", "No content-matched batches"),
             ("abl_no_mining", "No writer mining"), ("abl_k8", "8 patches per page"), ("abl_no_aug", "No augmentation")]
    rows = [(l, json.load(open(f"{runs}/{n}/best_val.json"))) for n, l in names if os.path.exists(f"{runs}/{n}/best_val.json")]
    if len(rows) < 2: return print("skip tab_abl: runs not finished")
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Variant & IAM EER & CVL EER & Mean \\\\", "\\midrule"]
    for l, v in rows: L.append(f"{l} & {100*v['iam']['eer']:.2f} & {100*v['cvl']['eer']:.2f} & {100*v['mean_eer']:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; write_table(out, "tab_abl", L)


TEMPLATE = ("Disputed handwritten scripts require deciding whether a questioned page matches a writer's earlier pages, yet most deep verifiers compare words or signatures. "
            "We embed a full page as an attention-pooled bag of ink patches with a ResNet-18 trained from scratch, using content-matched hard negatives and distillation from a frozen DINOv2 teacher. "
            "On held-out writers it reaches {iam} EER on IAM and {cvl} on CVL ({n} seeds), against {d_iam} and {d_cvl} for frozen DINOv2 features. "
            "On Firemaker, never used for tuning, EER is {fm}. "
            "Fusing {nref} reference pages lowers CVL EER to {mr}. "
            "A calibrated three-way verdict decides {cov}% of trials with {err}% error. "
            "No public data show one writer imitating another, so these are not forgery detection rates.")


def write_abstract(runs, out, limit=120):
    ev = seed_evals(runs, "eval_test.json")
    b2 = f"{runs}/baselines/b2_test.json"
    if not ev or not os.path.exists(b2) or "firemaker" not in ev[0] or "three_way" not in ev[0]: return print("skip abstract: final test evaluation not complete")
    d2 = json.load(open(b2))["seed0"]; m = lambda f: np.mean([f(e) for e in ev])
    best = {k: m(lambda e, k=k: e["test"]["cvl"]["multi_ref"][k]["eer"]) for k in ev[0]["test"]["cvl"]["multi_ref"] if k.endswith("_centroid") and e_ok(ev, k)}
    bk = min(best, key=best.get)
    cov = np.mean([m(lambda e, d=d: e["three_way"][d]["coverage"]) for d in ("iam", "cvl")]); err = np.mean([m(lambda e, d=d: e["three_way"][d]["error_on_decided"]) for d in ("iam", "cvl")])
    txt = TEMPLATE.format(iam=f"{100*m(lambda e: e['test']['iam']['single_ref']['eer']):.1f}%", cvl=f"{100*m(lambda e: e['test']['cvl']['single_ref']['eer']):.1f}%", n=len(ev),
                          d_iam=f"{100*d2['iam']['eer']:.1f}%", d_cvl=f"{100*d2['cvl']['eer']:.1f}%", fm=f"{100*m(lambda e: e['firemaker']['p1_vs_p4-self-natural']['eer']):.1f}%",
                          nref=bk.split("_")[0][1:], mr=f"{100*best[bk]:.1f}%", cov=f"{100*cov:.0f}", err=f"{100*err:.1f}")
    words = len(txt.split())
    if words > limit: raise SystemExit(f"abstract has {words} words, limit {limit}")
    os.makedirs(f"{out}/sections", exist_ok=True); open(f"{out}/sections/abstract.tex", "w").write(txt.replace("%", "\\%") + "\n"); print(f"wrote abstract ({words} words)")


def e_ok(ev, k):
    return all(k in e["test"]["cvl"]["multi_ref"] and e["test"]["cvl"]["multi_ref"][k].get("writers", 0) >= 15 for e in ev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="../paper")
    ap.add_argument("--run", default="main_s0"); ap.add_argument("--split", default="test"); ap.add_argument("--manifest", default="../datasets/cache/manifests/master.csv"); a = ap.parse_args()
    if os.path.exists(a.manifest): table_data(a.manifest, a.out)
    fig_training(a.runs, a.out); fig_baselines(a.runs, a.out); fig_ablation(a.runs, a.out)
    ev_val, ev_test = seed_evals(a.runs, "eval_val.json"), seed_evals(a.runs, "eval_test.json")
    fig_multiref(ev_val, a.out, "val", "val")
    if ev_test: fig_multiref(ev_test, a.out, "test", "test"); fig_verdict(ev_test, a.out)
    sc = f"{a.runs}/{a.run}/eval_test_scores.npz"; fig_det_and_calibration(sc, a.out); fig_firemaker(sc, a.out); fig_attention(f"{a.runs}/{a.run}/explain", a.out)
    tab_main(a.runs, a.out, a.split, f"eval_{a.split}.json"); tab_firemaker(a.runs, a.out, "eval_test.json"); tab_verdict(a.runs, a.out, "eval_test.json"); tab_ablation(a.runs, a.out); write_abstract(a.runs, a.out)


if __name__ == "__main__":
    main()
