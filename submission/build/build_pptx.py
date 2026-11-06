"""Fill 'Review -PPT Template.pptx' from the project results. Run: /tmp/office/bin/python build_pptx.py deck0.pptx out.pptx
deck0.pptx = the template with slide 3 duplicated 18 times (scripts/add_slide.py). All numbers come from facts.py."""
import re, subprocess, sys
from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from facts import F  # noqa: E402

IMG = HERE / "img"
NAVY, RED, GREEN, BAD = RGBColor(0x00, 0x20, 0x60), RGBColor(0xA6, 0x19, 0x2E), RGBColor(0x2E, 0x7D, 0x32), RGBColor(0xC6, 0x28, 0x28)
TINT, MID, GRAY, BLACK, WHITE = RGBColor(0xEE, 0xF2, 0xF9), RGBColor(0xD0, 0xD7, 0xE5), RGBColor(0x59, 0x59, 0x59), RGBColor(0, 0, 0), RGBColor(255, 255, 255)
FONT = "Times New Roman"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"

prs = Presentation(sys.argv[1])
slides = list(prs.slides)


def para_set(p, text):
    """Replace a paragraph's text keeping the first run's formatting."""
    runs = p.runs
    if not runs:
        p.add_run().text = text; return
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def style(run, size, bold=False, color=BLACK, italic=False):
    f = run.font; f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color


def textbox(slide, x, y, w, h, paras, size=14, color=BLACK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullets=False, space=6, name=None):
    """paras: list of str | list of (text, bold) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, item in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        runs = [(item, bold)] if isinstance(item, str) else item
        for t, b in runs:
            r = p.add_run(); r.text = t; style(r, size, b, color)
        if bullets:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(0.25)))); pPr.set("indent", str(-int(Inches(0.25))))
            bu = etree.SubElement(pPr, f"{{{A}}}buChar"); bu.set("char", "•")
    return tb


def panel(slide, x, y, w, h, fill=TINT, line=None, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.04; s.fill.solid(); s.fill.fore_color.rgb = fill
    if line: s.line.color.rgb = line; s.line.width = Pt(1)
    else: s.line.fill.background()
    s.shadow.inherit = False
    if name: s.name = name
    return s


def stat(slide, x, y, w, h, value, label, color=NAVY):
    panel(slide, x, y, w, h)
    textbox(slide, x, y + 0.08, w, h * 0.55, [value], size=28, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    textbox(slide, x, y + h * 0.58, w, h * 0.4, [label], size=12, color=GRAY, align=PP_ALIGN.CENTER, space=0)


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def table(slide, x, y, w, col_w, rows, size=12, row_h=0.32, header=True, bold_last=False, colors=None):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"): tblPr.set(a, "0")
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            hdr = header and i == 0
            col = WHITE if hdr else (colors.get((i, j), BLACK) if colors else BLACK)
            style(r, size, bold=hdr or (bold_last and i == len(rows) - 1) or bool(colors and (i, j) in colors), color=col)
            c.fill.solid(); c.fill.fore_color.rgb = NAVY if hdr else (TINT if i % 2 == 0 else WHITE)
    return shp


def bottom(sh):
    return (sh.top + sh.height) / 914400


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def slide_setup(idx, title, title_w=7.6):
    s = slides[idx - 1]
    for sh in list(s.shapes):
        if sh.is_placeholder: sh._element.getparent().remove(sh._element)
    for sh in s.shapes:
        if sh.name == "Rectangle 4":
            sh.width = Inches(title_w); para_set(sh.text_frame.paragraphs[0], title)
            sh.text_frame.word_wrap = True
    return s


# ---------------------------------------------------------------- slide 1: title
s1 = slides[0]
for sh in s1.shapes:
    if sh.name == "Rectangle 4":
        sh.height = Inches(1.15)
        para_set(sh.text_frame.paragraphs[0], "HANDWRITING FORGERY DETECTION USING DEEP LEARNING: PAGE-LEVEL, MULTI-REFERENCE VERIFICATION")
        for r in sh.text_frame.paragraphs[0].runs: r.font.size = Pt(18)
        para_set(sh.text_frame.paragraphs[1], "SDG Addressed: SDG 16 – Peace, Justice and Strong Institutions")
        for r in sh.text_frame.paragraphs[1].runs: r.font.size = Pt(14)
    elif sh.name == "Rectangle 6":
        sh.width = Inches(4.6); para_set(sh.text_frame.paragraphs[2], "S. D. Lalitha, Associate Professor")
    elif sh.name == "Rectangle 7":
        sh.left = Inches(5.2); sh.width = Inches(4.6); para_set(sh.text_frame.paragraphs[2], "Lincy Jenita J (URK23CS5079)")
notes(s1, "Introduce the problem: deciding whether a questioned handwritten page belongs to a writer whose earlier pages are available.")

# ---------------------------------------------------------------- slide 3 abstract
s = slide_setup(3, "ABSTRACT", 3)
textbox(s, 0.5, 1.75, 9.0, 3.15, ["Deciding whether a questioned handwritten page matches a writer's earlier pages is a recurring problem in disputed exams, wills and contracts. We built a page-level verifier: a ResNet-18 trained from scratch embeds a whole page as an attention-pooled bag of ink patches, and one or several reference pages are fused into a calibrated verdict (consistent, inconclusive, likely forged).",
                                  "Tested on held-out writers and on a corpus never used for tuning, it roughly halves the error of frozen DINOv2 features, but it is fooled by deliberate style change. No public data show one person imitating another, so these are not forgery detection rates."], size=15, align=PP_ALIGN.JUSTIFY, space=8)
for i, (v, l, c) in enumerate([(f"{F['iam_eer'].split(' ')[0]}%", "IAM test EER", NAVY), (f"{F['cvl_eer'].split(' ')[0]}%", "CVL test EER", NAVY),
                               (f"{F['fm_p4_eer']:.1f}%", "Firemaker, natural writing", GREEN), (f"{F['fm_p3_eer']:.0f}%", "Firemaker, disguised", BAD)]):
    stat(s, 0.5 + i * 2.28, 5.1, 2.1, 1.35, v, l, c)
notes(s, "Numbers are mean over three seeds on held-out test writers. EER = equal error rate; lower is better.")

# ---------------------------------------------------------------- slide 4 problem definition
s = slide_setup(4, "PROBLEM DEFINITION")
textbox(s, 0.5, 1.75, 4.5, 4.9, [
    "A questioned answer sheet: the owner offers earlier genuine scripts as references.",
    "The unit of evidence is a complete script, not a word, sentence or signature.",
    "Task: given N reference pages of a claimed writer and one questioned page, return a similarity score and a verdict.",
    "Hard because a writer varies (pen, mood, text), forgers imitate, and no public data show one person copying another."], size=15, bullets=True, space=9)
for i, (t, sub, fill, col) in enumerate([("N reference pages", "claimed writer", TINT, NAVY), ("Questioned page", "one full script", TINT, NAVY), ("Verifier", "page embedding + fusion", NAVY, WHITE), ("Verdict", "consistent / inconclusive / likely forged", TINT, NAVY)]):
    y = 1.85 + i * 1.2
    panel(s, 5.4, y, 4.1, 0.95, fill=fill)
    textbox(s, 5.4, y + 0.06, 4.1, 0.45, [t], size=16, bold=True, color=col, align=PP_ALIGN.CENTER, space=0)
    textbox(s, 5.4, y + 0.5, 4.1, 0.4, [sub], size=12, color=(WHITE if fill == NAVY else GRAY), align=PP_ALIGN.CENTER, space=0)
notes(s, "The example in the original proposal: another person copies answers and imitates the owner's handwriting; the owner supplies previous exam papers as references.")

# ---------------------------------------------------------------- slide 5 scope and objectives
s = slide_setup(5, "SCOPE AND OBJECTIVES")
panel(s, 0.5, 1.75, 4.4, 2.15); panel(s, 5.1, 1.75, 4.4, 2.15)
textbox(s, 0.6, 1.8, 4.2, 0.4, ["In scope"], size=16, bold=True, color=NAVY)
textbox(s, 0.6, 2.2, 4.2, 1.7, ["English full pages (IAM, CVL, Firemaker)", "1 to 5 reference pages per writer", "Writer-disjoint test, cross-corpus test", "Calibrated three-way verdict, CPU demo"], size=14, bullets=True, space=3)
textbox(s, 5.2, 1.8, 4.2, 0.4, ["Out of scope (first paper)"], size=16, bold=True, color=RED)
textbox(s, 5.2, 2.2, 4.2, 1.7, ["Four Indian languages (data collection paused)", "Real impersonation data (none public)", "Learned imitation generators", "Word-level and signature verification"], size=14, bullets=True, space=3)
textbox(s, 0.5, 4.05, 9, 0.4, ["Research questions"], size=16, bold=True, color=NAVY)
rq = [("RQ1", "How well does a page-level model separate writers, and does it beat untrained and frozen-feature baselines?"),
      ("RQ2", "Do extra reference pages help, and which fusion rule is best?"),
      ("RQ3", "Does it transfer to a new scanner and protocol (Firemaker), including disguised writing?"),
      ("RQ4", "Which training ingredients matter: distillation, pooling, content matching, mining, patches, augmentation?")]
for i, (k, t) in enumerate(rq):
    y = 4.5 + i * 0.6
    panel(s, 0.5, y, 0.8, 0.5, fill=NAVY)
    textbox(s, 0.5, y, 0.8, 0.5, [k], size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    textbox(s, 1.4, y, 8.1, 0.5, [t], size=13, anchor=MSO_ANCHOR.MIDDLE, space=0)
notes(s, "Indian languages are a planned second phase; downloads were paused by decision until the English paper is done.")

# ---------------------------------------------------------------- slide 6 literature survey
s = slide_setup(6, "LITERATURE SURVEY")
rows = [["Work", "Level", "What it does", "Gap for our task"],
        ["DeepWriter (Xing, Qiao 2016)", "text patches", "CNN for text-independent writer identification", "identifies, does not verify a claim"],
        ["FragNet (He, Schomaker 2020)", "fragments", "fragment network for writer identification", "closed-set identification"],
        ["Self-supervised ViT (Raven et al. 2024)", "pages", "unsupervised writer retrieval", "ranks pages, no verdict"],
        ["Attention verifier (Shaikh et al. 2020)", "pairs", "writer-independent verification", "one pair, not many references"],
        ["Self-supervised verification (Chauhan et al. 2024)", "pairs", "pre-training for verification", "pairs, no calibration"],
        ["SigNet (Dey et al. 2017)", "signatures", "Siamese signature verification", "signatures, not full pages"]]
table(s, 0.5, 1.75, 9.0, [2.9, 1.1, 2.6, 2.4], rows, size=12, row_h=0.52)
textbox(s, 0.5, 5.65, 9, 1.2, ["Writer identification answers \"which writer\"; verification work compares pairs of words, patches or signatures. Little of it combines a whole questioned page with several references, reports calibrated support, or is allowed to say \"inconclusive\"."], size=14, align=PP_ALIGN.JUSTIFY)
notes(s, "Only abstracts of the cited arXiv papers were read; no numbers from other papers are quoted.")

# ---------------------------------------------------------------- slide 7 gap and position
s = slide_setup(7, "RESEARCH GAP AND OUR POSITION")
cols = [("Page-level", "A complete script is the unit of evidence, embedded as an attention-pooled bag of 64 ink patches."),
        ("Multi-reference", "One to five reference pages are fused (mean, max, top-2, centroid) and compared on one fixed set of writers."),
        ("Calibrated verdict", "Thresholds fixed on validation data; the system may answer inconclusive.")]
for i, (h, t) in enumerate(cols):
    x = 0.5 + i * 3.05
    panel(s, x, 1.85, 2.9, 2.9)
    textbox(s, x + 0.1, 1.95, 2.7, 0.5, [h], size=18, bold=True, color=NAVY)
    textbox(s, x + 0.1, 2.55, 2.7, 2.2, [t], size=15, space=0)
textbox(s, 0.5, 5.0, 9, 1.8, ["To our knowledge no prior writer-verification work combines these elements; we do not claim any single component is new. Distillation and likelihood-ratio reporting each have precedent.",
                              "Honest protocol: writer-disjoint splits, test sets used once, a cross-corpus test never used for tuning."], size=14, align=PP_ALIGN.JUSTIFY, space=8)
notes(s, "A web search on 2026-11-05 found precedent for distillation in signature verification and for likelihood-ratio reporting in forensics; the wording stays 'to our knowledge'.")

# ---------------------------------------------------------------- slide 8 architecture
s = slide_setup(8, "PROPOSED SYSTEM ARCHITECTURE")
picture(s, IMG / "architecture.png", 0.5, 1.8, w=9.0)
textbox(s, 0.5, 5.75, 9, 1.1, ["(a) Training: each page becomes ink-centred 128 × 128 patches, encoded by a ResNet-18 trained from scratch and pooled by attention into a 256-d page embedding. A frozen DINOv2-small teacher is used only during training.",
                               "(b) Inference: the questioned page is compared with N references with the same encoder, scores are fused, calibrated and turned into a verdict."], size=12, space=4)
notes(s, "The shipped model is the student alone; the teacher is not needed at inference.")

# ---------------------------------------------------------------- slide 9 model and objective
s = slide_setup(9, "MODEL AND TRAINING OBJECTIVE")
blocks = [("Encoder", "ResNet-18 (1 channel, random init) → 256-d patch vectors; attention pooling a_i = softmax(wᵀ tanh(V z_i)); page embedding is L2-normalised. 16 patches per page in training, 64 in evaluation."),
          ("Loss", "Supervised contrastive loss (τ = 0.1) on batches of 12 writers × 2 pages. 40% of batches are content-matched CVL batches (all writers copy the same texts); half use nearest-neighbour writers. Plus feature and relational distillation from frozen DINOv2-small (weights 0.5 + 0.5)."),
          ("Verdict", "Fuse N reference scores (mean, max, top-2, centroid), calibrate with Platt or isotonic regression fitted on validation only. Consistent if the score is at or above the 99th percentile of validation impostors; likely forged at or below the 1st percentile of genuine scores; otherwise inconclusive.")]
for i, (h, t) in enumerate(blocks):
    y = 1.8 + i * 1.7
    panel(s, 0.5, y, 1.7, 1.5, fill=NAVY)
    textbox(s, 0.5, y, 1.7, 1.5, [h], size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    panel(s, 2.3, y, 7.2, 1.5)
    textbox(s, 2.4, y, 7.0, 1.5, [t], size=13, anchor=MSO_ANCHOR.MIDDLE, space=0)
notes(s, "Distillation is described as an adopted technique, not a contribution; the ablation later shows it did not clearly help at 40 epochs.")

# ---------------------------------------------------------------- slide 10 datasets
s = slide_setup(10, "DATASETS AND PREPROCESSING")
dat = (HERE.parents[1] / "paper/tables/tab_data.tex").read_text()
rows = [["Corpus", "Train writers / pages", "Validation", "Test"]]
for line in dat.splitlines():
    m = re.match(r"(IAM|CVL \(English\)|Firemaker) & (.*) \\\\", line)
    if m:
        c = [x.strip() for x in m.group(2).split("&")]
        f = lambda w, p: "–" if w == "--" else f"{w} / {p}"
        rows.append([m.group(1), f(c[0], c[1]), f(c[2], c[3]), f(c[4], c[5])])
table(s, 0.5, 1.8, 9.0, [2.4, 2.4, 2.1, 2.1], rows, size=14, row_h=0.42)
textbox(s, 0.5, 3.75, 9, 3.1, [
    [("IAM: ", True), ("forms by many writers; the copy used is a third-party Kaggle one (official registration pending).", False)],
    [("CVL: ", True), ("all writers copy the same texts, which gives content-matched hard negatives. English texts only.", False)],
    [("Firemaker: ", True), ("test only, never used for tuning. Page 3 is self-disguise, not imitation of another person.", False)],
    [("Preprocessing: ", True), ("crop printed form parts, remove paper tone and scanner shading, resize to width 1600, drop blank and duplicate pages. Splits are writer-disjoint.", False)]], size=14, bullets=True, space=6)
notes(s, "Kaggle real-vs-generated pages are font-rendered text and are used only for a synthetic-page false-accept test.")

# ---------------------------------------------------------------- slide 11 implementation
s = slide_setup(11, "IMPLEMENTATION")
panel(s, 0.5, 1.8, 4.4, 3.6); panel(s, 5.1, 1.8, 4.4, 3.6)
textbox(s, 0.6, 1.85, 4.2, 0.4, ["Software"], size=16, bold=True, color=NAVY)
textbox(s, 0.6, 2.3, 4.2, 4.4, ["Docker image handwriting-train:0.1 (PyTorch 2.6, CUDA 12.4)", "timm ResNet-18, DINOv2-small teacher, OpenCV, scikit-learn", "One module per stage: prep_cache, manifests, data, model, train, eval, multiref, verdict, report", "Unit tests (pytest) including a label-shuffle leak test", "FastAPI demo (CPU) and a live training dashboard"], size=13, bullets=True, space=4)
textbox(s, 5.2, 1.85, 4.2, 0.4, ["Training and hardware"], size=16, bold=True, color=NAVY)
textbox(s, 5.2, 2.3, 4.2, 4.4, ["Laptop: RTX 4050 (6 GB), i7-13650HX, 15 GB RAM", "AdamW, lr 3e-4, cosine, up to 120 epochs, mixed precision, EMA 0.999", "Three seeds, about 50 min and 26 s per epoch each", "Seven 40-epoch ablations, one GPU job at a time", "Every table and figure is generated from the run files by report.py"], size=13, bullets=True, space=4)
steps = ["raw pages", "prep_cache", "manifests", "train", "eval", "report", "paper"]
for i, t in enumerate(steps):
    x = 0.5 + i * 1.3
    panel(s, x, 5.75, 1.15, 0.65, fill=NAVY if i in (3, 4) else TINT)
    textbox(s, x, 5.75, 1.15, 0.65, [t], size=12, bold=True, color=WHITE if i in (3, 4) else NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
textbox(s, 0.5, 6.5, 9, 0.4, ["Pipeline: each stage reads the previous stage's files, so any table can be regenerated."], size=12, color=GRAY, space=0)
notes(s, "Everything heavy ran on one laptop GPU, niced and serialised with a lock.")

# ---------------------------------------------------------------- slide 12 demo
s = slide_setup(12, "LIVE DEMONSTRATION")
picture(s, IMG / "demo_inputs.jpg", 0.5, 1.8, w=4.4)
picture(s, IMG / "demo_result.jpg", 5.1, 1.8, w=4.4)
textbox(s, 0.5, 4.4, 4.4, 0.4, ["Upload a reference page and a questioned page"], size=12, color=GRAY, space=0)
textbox(s, 5.1, 4.4, 4.4, 0.4, ["Verdict, score and where it falls among validation pairs"], size=12, color=GRAY, space=0)
textbox(s, 0.5, 4.95, 9, 1.9, ["Runs on CPU in about 2 to 3 seconds per comparison; one to five reference pages.", "Shows the calibrated verdict, similarity score, validation score distributions and attention heatmaps.", "Limits are printed on the page: it scores style similarity, not forgery directly."], size=14, bullets=True, space=5)
notes(s, "Demo URL on the presenting machine: http://localhost:8010. Sample pairs come from the validation set, so they look better than unseen data.")

# ---------------------------------------------------------------- slide 13 main results
s = slide_setup(13, "RESULTS: MAIN COMPARISON")
cd = CategoryChartData(); cd.categories = ["Random CNN", "HOG", "DINOv2 (frozen)", "Ours"]
cd.add_series("IAM", [round(F["b0_iam_eer"], 2), round(F["b3_iam_eer"], 2), round(F["b2_iam_eer"], 2), round(F["iam_eer_m"], 2)])
cd.add_series("CVL", [round(F["b0_cvl_eer"], 2), round(F["b3_cvl_eer"], 2), round(F["b2_cvl_eer"], 2), round(F["cvl_eer_m"], 2)])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.75), Inches(5.6), Inches(3.9), cd); ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12); ch.legend.font.name = FONT
ch.has_title = True; ch.chart_title.text_frame.text = "Test EER (%), lower is better"
for r in ch.chart_title.text_frame.paragraphs[0].runs: style(r, 13, True)
pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True; dl = pl.data_labels; dl.font.size = Pt(11); dl.font.name = FONT; dl.number_format = "0.0"; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, col in zip(pl.series, (NAVY, RED)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
ch.value_axis.has_major_gridlines = True; ch.value_axis.major_gridlines.format.line.color.rgb = MID; ch.value_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.size = Pt(11)
stat(s, 6.4, 1.8, 3.1, 1.25, F["iam_eer"], f"IAM EER (%), AUC {F['iam_auc'].split(' ')[0]}")
stat(s, 6.4, 3.2, 3.1, 1.25, F["cvl_eer"], f"CVL EER (%), AUC {F['cvl_auc'].split(' ')[0]}")
textbox(s, 6.4, 4.6, 3.1, 1.1, ["Mean ± sd over three seeds, 99 IAM and 62 CVL test writers."], size=12, color=GRAY)
textbox(s, 0.5, 5.8, 9, 1.1, [f"The student halves the error of the frozen teacher on both corpora. Test is worse than validation (IAM {F['iam_val_eer']}%), so validation numbers were optimistic.",
                              f"Strict operating point: at 1% false rejects, {F['cvl_far1'].split(' ')[0]}% of CVL impostors are accepted, so one page pair is weak evidence."], size=13, bullets=True, space=3)
notes(s, "Baselines HOG and frozen DINOv2 are deterministic, the random CNN is averaged over three seeds.")

# ---------------------------------------------------------------- slide 14 multi-reference
s = slide_setup(14, "RESULTS: MORE REFERENCE PAGES")
pic = picture(s, IMG / "fig_multiref_test.png", 1.2, 1.75, w=7.6)
textbox(s, 0.5, bottom(pic) + 0.1, 9, 6.95 - bottom(pic) - 0.1, [
    f"Same writers for every N ({F['mr_iam_writers']} IAM, {F['mr_cvl_writers']} CVL). Centroid fusion: IAM {F['mr_iam_centroid_1']:.2f} → {F['mr_iam_centroid_2']:.2f} → {F['mr_iam_centroid_3']:.2f}; CVL {F['mr_cvl_centroid_1']:.2f} → {F['mr_cvl_centroid_2']:.2f} → {F['mr_cvl_centroid_3']:.2f} (N = 1, 2, 3).",
    f"With five references, 17 IAM writers: {F['iam_n5'][5]:.2f} against {F['iam_n5'][1]:.2f} for one.",
    f"On CVL the max rule is best at N = 3 ({F['mr_cvl_max_3']:.2f}), but the centroid was the rule chosen on validation, so this is a post-hoc observation.",
    "Bootstrap intervals overlap: the gain is consistent in direction but not established."], size=13, bullets=True, space=4)
notes(s, "An earlier version let the writer set shrink with N and was discarded because it confounded N with writer difficulty.")

# ---------------------------------------------------------------- slide 15 firemaker
s = slide_setup(15, "RESULTS: CROSS-CORPUS AND DISGUISE")
rows = [["Firemaker pair", "EER (%)", "Same-writer cos."],
        ["p1 vs p4 (natural)", F["fm_p4_eer_s"], f"{F['fm_p4_gen']:.3f}"],
        ["p1 vs p2 (upper case)", F["fm_p2_eer_s"], f"{F['fm_p2_gen']:.3f}"],
        ["p1 vs p3 (disguised)", F["fm_p3_eer_s"], f"{F['fm_p3_gen']:.3f}"]]
table(s, 0.5, 1.8, 4.7, [2.0, 1.4, 1.3], rows, size=12, row_h=0.45, colors={(1, 1): GREEN, (3, 1): BAD, (2, 1): BAD})
textbox(s, 0.5, 3.8, 4.7, 3.1, [f"Natural writing transfers across corpora ({F['fm_p4_eer']:.1f}% EER) on 250 writers never used for tuning.",
                                f"Upper case and deliberate disguise break it: same-writer cosine falls from {F['fm_p4_gen']:.2f} to {F['fm_p3_gen']:.2f} while other writers stay near {F['fm_p3_imp']:.2f}.",
                                "A writer who changes style can be rejected, and a forger who copies a style is helped."], size=13, bullets=True, space=5)
pic = picture(s, IMG / "fig_firemaker.png", 5.4, 1.8, w=4.1)
textbox(s, 5.4, bottom(pic) + 0.05, 4.1, 0.4, ["Seed 0, 250 writers"], size=11, color=GRAY)
notes(s, "Firemaker page 3 is self-disguise, not impersonation, so it is a stress test of intra-writer variation, not a forgery rate.")

# ---------------------------------------------------------------- slide 16 verdicts
s = slide_setup(16, "RESULTS: CALIBRATED VERDICTS")
pic = picture(s, IMG / "fig_verdict.png", 0.5, 1.8, h=2.9)
rows = [["", "CVL", "IAM"], ["Consistent (%)", f"{F['v_cvl_accepted']:.1f}", f"{F['v_iam_accepted']:.1f}"], ["Inconclusive (%)", f"{F['v_cvl_inconclusive']:.1f}", f"{F['v_iam_inconclusive']:.1f}"],
        ["Likely forged (%)", f"{F['v_cvl_rejected']:.1f}", f"{F['v_iam_rejected']:.1f}"], ["Error on decided (%)", f"{F['v_cvl_error_on_decided']:.1f}", f"{F['v_iam_error_on_decided']:.1f}"],
        ["ECE raw / Platt / isotonic", f"{F['ece_raw_cvl']:.2f} / {F['ece_platt_cvl']:.3f} / {F['ece_iso_cvl']:.3f}", f"{F['ece_raw_iam']:.2f} / {F['ece_platt_iam']:.3f} / {F['ece_iso_iam']:.3f}"]]
table(s, 4.9, 1.8, 4.6, [1.9, 1.35, 1.35], rows, size=11, row_h=0.42)
textbox(s, 0.5, max(bottom(pic), 4.4) + 0.15, 9, 2.0, [f"Three references, thresholds from validation only. On CVL, {F['v_cvl_inconclusive']:.1f}% of trials are inconclusive and {F['v_cvl_error_on_decided']:.1f}% of the decided ones are wrong.",
                               "IAM is separable on validation, so one threshold is used and there is no inconclusive band; do not read this as evidence that the problem is easy.",
                               "Raw cosine scores are not probabilities (ECE about 0.22); isotonic regression is the best calibrator."], size=13, bullets=True, space=4)
notes(s, "The first IAM three-way numbers had overlapping thresholds because of a bug in my rule; it was fixed and the verdict recomputed from the saved scores (see FIXES.md).")

# ---------------------------------------------------------------- slide 17 ablations
s = slide_setup(17, "ABLATIONS AND TRAINING BEHAVIOUR")
rows = [["Variant (40 epochs, 1 seed)", "IAM", "CVL", "Mean"]] + [[l, f"{a:.2f}", f"{c:.2f}", f"{m:.2f}"] for l, a, c, m in F["abl"]]
table(s, 0.5, 1.75, 5.3, [2.9, 0.8, 0.8, 0.8], rows, size=11, row_h=0.3)
textbox(s, 6.0, 1.75, 3.5, 2.6, ["Validation EER (%).", "No ingredient shows a clear benefit; three ablations beat the full 40-epoch model, but one seed cannot tell if that is real.", f"Main 120-epoch runs reach {min(F['main_val_mean']):.2f} to {max(F['main_val_mean']):.2f} mean val EER."], size=12, bullets=True, space=3)
pic = picture(s, IMG / "fig_training_curves.png", 1.2, 4.3, h=2.75)
notes(s, "We do not claim that distillation, content-matched batches or mining help; the paper says so.")

# ---------------------------------------------------------------- slide 18 limitations
s = slide_setup(18, "LIMITATIONS")
textbox(s, 0.5, 1.75, 9, 3.4, [
    [("No real impersonation data. ", True), ("Impostors are other writers, an easier case than a skilled forger.", False)],
    [("Disguise breaks the model. ", True), ("Firemaker p3 is self-disguise, reported as a stress test.", False)],
    [("Synthetic pages are fonts. ", True), (f"Accepted for {F['syn_iam_generated']:.2f}% of pairs on IAM thresholds; real unknown pages behave alike, so it says little about imitation.", False)],
    [("Corpus effects. ", True), ("IAM pages share scanner and pen, so IAM is optimistic; the IAM copy is third-party.", False)],
    [("Scale. ", True), ("English only, three seeds, one-seed ablations, one laptop GPU.", False)]], size=15, bullets=True, space=7)
pic = picture(s, IMG / "fig_attention.png", 1.2, 4.55, w=7.6)
textbox(s, 1.2, bottom(pic) + 0.05, 7.6, 0.4, [f"Attention maps are a visualisation only: dropping top-attended patches lowers the score by {F['del_full'] - F['del_top']:.4f} vs {F['del_full'] - F['del_rand']:.4f} for random."], size=11, color=GRAY, space=0)

# ---------------------------------------------------------------- slide 19 conclusion
s = slide_setup(19, "CONCLUSION AND FUTURE WORK")
panel(s, 0.5, 1.8, 4.4, 3.9); panel(s, 5.1, 1.8, 4.4, 3.9)
textbox(s, 0.6, 1.85, 4.2, 0.4, ["Conclusion"], size=16, bold=True, color=NAVY)
textbox(s, 0.6, 2.3, 4.2, 4.4, [f"A page-level verifier trained from scratch halves the error of frozen DINOv2 ({F['iam_eer'].split(' ')[0]}% IAM, {F['cvl_eer'].split(' ')[0]}% CVL EER).", f"It transfers to a new corpus for ordinary writing ({F['fm_p4_eer']:.1f}%).", "Extra references and a calibrated, abstaining verdict help.", "It does not survive deliberate style change, and no ablated ingredient is shown to help."], size=14, bullets=True, space=6)
textbox(s, 5.2, 1.85, 4.2, 0.4, ["Future work"], size=16, bold=True, color=NAVY)
textbox(s, 5.2, 2.3, 4.2, 4.4, ["Volunteer-copied impersonation set per language", "Handwriting generators to test learned imitation", "Multi-seed ablations at the full schedule", "A larger validation set to choose the fusion rule", "Four Indian languages (data collection paused)"], size=14, bullets=True, space=6)
panel(s, 0.5, 5.9, 9.0, 0.9, fill=NAVY)
textbox(s, 0.6, 5.9, 8.8, 0.9, ["Page-level verification works for ordinary writing, but real impersonation data are needed before it can be called forgery detection."], size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)

# ---------------------------------------------------------------- slide 20 references
s = slide_setup(20, "REFERENCES")
refs = ["L. Xing, Y. Qiao, “DeepWriter: A Multi-Stream Deep CNN for Text-independent Writer Identification,” arXiv:1606.06472, 2016.",
        "S. He, L. Schomaker, “FragNet: Writer Identification using Deep Fragment Networks,” arXiv:2003.07212, 2020.",
        "S. Dey et al., “SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification,” arXiv:1707.02131, 2017.",
        "M. A. Shaikh et al., “Attention based Writer Independent Handwriting Verification,” arXiv:2009.04532, 2020.",
        "M. Chauhan et al., “Self-Supervised Learning Based Handwriting Verification,” arXiv:2405.18320, 2024.",
        "T. Raven, A. Matei, G. A. Fink, “Self-Supervised Vision Transformers for Writer Retrieval,” arXiv:2409.00751, 2024.",
        "M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” arXiv:2304.07193, 2023.",
        "K. He et al., “Deep Residual Learning for Image Recognition,” arXiv:1512.03385, 2015.",
        "P. Khosla et al., “Supervised Contrastive Learning,” arXiv:2004.11362, 2020.",
        "U.-V. Marti, H. Bunke, “The IAM-database,” Int. J. Document Analysis and Recognition, 2002."]
textbox(s, 0.5, 1.75, 9.0, 5.1, [f"[{i + 1}] {r}" for i, r in enumerate(refs)], size=13, space=6)

# ---------------------------------------------------------------- slide 21 supporting evidence
s = slide_setup(21, "OUTCOME SPECIFIC SUPPORTING EVIDENCE", 8.0)
subprocess.run(["pdftoppm", "-r", "90", "-f", "1", "-l", "1", "-png", "-singlefile", str(HERE.parents[1] / "paper/main.pdf"), str(IMG / "paper_p1")], check=True)
picture(s, IMG / "paper_p1.png", 0.5, 1.8, h=4.3)
textbox(s, 3.9, 1.8, 5.6, 0.4, ["Selected outcome: Journal publication"], size=16, bold=True, color=NAVY)
textbox(s, 3.9, 2.3, 5.6, 4.5, ["Paper draft: 8 pages, IEEEtran, 270-word abstract, six result figures and eight tables; submission venue not yet chosen.",
                                "Every number is generated from the run files and mapped to its evidence file.", "Code, unit tests, run logs and Docker setup are in the project repository.",
                                "Working demo and live training dashboard.", "Pending: citation check, author block confirmation, official IAM registration."], size=13, bullets=True, space=5)
textbox(s, 0.5, 6.15, 3.2, 0.5, ["First page of the paper draft"], size=11, color=GRAY)

prs.save(sys.argv[2])
print("saved", sys.argv[2])
