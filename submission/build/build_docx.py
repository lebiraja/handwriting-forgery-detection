"""Fill 'PRS-Front Page Template.docx' and append the PRS report body (all numbers from facts.py).
Run: /tmp/office/bin/python build_docx.py tpl_merged.docx out.docx [pages.json]
pages.json = {"1": page, ...} body page numbers for the contents table (second pass, read from the rendered PDF)."""
import copy, json, re, sys
from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from facts import F  # noqa: E402

IMG = HERE / "img"
TITLE = "HANDWRITING FORGERY DETECTION USING DEEP LEARNING: PAGE-LEVEL, MULTI-REFERENCE VERIFICATION"
STUDENT, REG, GUIDE, DESIG = "Lincy Jenita J", "URK23CS5079", "S. D. Lalitha", "Associate Professor"
SDG = "SDG 16 – Peace, Justice and Strong Institutions"
BODY = "Times New Roman"

doc = Document(sys.argv[1])
pages = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}


# ------------------------------------------------------------------ front matter
def para_set(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text); return
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


for p in doc.paragraphs:
    t = p.text
    if "Project Title in CAPS" in t: para_set(p, TITLE)
    elif t.startswith("SDG :"): para_set(p, "SDG : " + SDG)
    elif "<< Name >>" in t: para_set(p, STUDENT.upper())
    elif "Register Number" in t: para_set(p, f"({REG})")
    elif "Title of your Project work" in t:
        for r in p.runs:
            if "<<" in r.text: r.text = f"“{TITLE}”"
    elif "Name (Reg.No)" in t: para_set(p, f"{STUDENT} ({REG})")
    elif t.startswith("Dr/Mr. Guide Name"): para_set(p, GUIDE)
    elif t.strip() == "Designation": para_set(p, DESIG)
    elif t.startswith("(Students are required"): p._p.getparent().remove(p._p)

# contents table: serial numbers (+ page numbers on the second pass)
toc = doc.tables[0]
for i, row in enumerate(toc.rows[1:], 1):
    for ci, val in ((2, str(pages.get(str(i), ""))),):
        c = row.cells[ci]; p = c.paragraphs[0]
        if p.runs: p.runs[0].text = val
        else:
            r = p.add_run(val); r.font.name = "Bookman Old Style"; r.font.size = Pt(12)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# drop everything after the contents table (blank lines and the "Formatting specifications" instructions)
body = doc.element.body
after = False
for el in list(body):
    if el is toc._tbl: after = True; continue
    if after and el.tag == qn("w:p"): body.remove(el)

# ------------------------------------------------------------------ numbering for bullets
numbering = doc.part.numbering_part.element
absid, numid = "20", "20"
an = OxmlElement("w:abstractNum"); an.set(qn("w:abstractNumId"), absid)
lvl = OxmlElement("w:lvl"); lvl.set(qn("w:ilvl"), "0")
for tag, val in (("w:start", "1"), ("w:numFmt", "bullet"), ("w:lvlText", "•"), ("w:lvlJc", "left")):
    e = OxmlElement(tag); e.set(qn("w:val"), val); lvl.append(e)
ppr = OxmlElement("w:pPr"); ind = OxmlElement("w:ind"); ind.set(qn("w:left"), "720"); ind.set(qn("w:hanging"), "360"); ppr.append(ind); lvl.append(ppr)
rpr = OxmlElement("w:rPr"); rf = OxmlElement("w:rFonts"); rf.set(qn("w:ascii"), BODY); rf.set(qn("w:hAnsi"), BODY); rpr.append(rf); lvl.append(rpr)
an.append(lvl)
first_num = numbering.find(qn("w:num"))
(first_num.addprevious(an) if first_num is not None else numbering.append(an))
nm = OxmlElement("w:num"); nm.set(qn("w:numId"), numid); aid = OxmlElement("w:abstractNumId"); aid.set(qn("w:val"), absid); nm.append(aid); numbering.append(nm)

# ------------------------------------------------------------------ new section for the body (page numbers restart)
sec = doc.add_section(WD_SECTION.NEW_PAGE)
sec.footer.is_linked_to_previous = False
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for kind, txt in (("begin", None), (None, " PAGE "), ("end", None)):
    r = fp.add_run(); r.font.name = BODY; r.font.size = Pt(11)
    if kind:
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)


# ------------------------------------------------------------------ writing helpers
def fmt(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, before=0, keep=False):
    pf = p.paragraph_format; pf.line_spacing = 1.15; pf.space_after = Pt(after); pf.space_before = Pt(before); p.alignment = align
    if keep: pf.keep_with_next = True


def run(p, text, size=12, bold=False, italic=False):
    r = p.add_run(text); r.font.name = BODY; r._element.rPr.rFonts.set(qn("w:eastAsia"), BODY)
    r.font.size = Pt(size); r.bold = bold; r.italic = italic; r.font.color.rgb = RGBColor(0, 0, 0)
    return r


def H1(num, text, first=False):
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, after=8, keep=True)
    if not first: p.paragraph_format.page_break_before = True
    run(p, f"{num}. {text.upper()}" if num else text.upper(), 13, True)
    p._p.get_or_add_pPr().append(_outline(0))
    return p


def _outline(level):
    o = OxmlElement("w:outlineLvl"); o.set(qn("w:val"), str(level)); return o


def H2(text):
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.LEFT, after=4, before=8, keep=True); run(p, text, 12, True); p._p.get_or_add_pPr().append(_outline(1)); return p


def P(text="", runs=None):
    p = doc.add_paragraph(); fmt(p)
    for t, b in (runs or [(text, False)]): run(p, t, 12, b)
    return p


def B(items):
    for it in items:
        p = doc.add_paragraph(); fmt(p, after=3)
        numPr = p._p.get_or_add_pPr().get_or_add_numPr(); numPr.get_or_add_ilvl().val = 0; numPr.get_or_add_numId().val = int(numid)
        for t, b in ([(it, False)] if isinstance(it, str) else it): run(p, t, 12, b)


counters = {"Figure": 0, "Table": 0}


def caption(kind, text):
    counters[kind] += 1
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=10, before=3)
    run(p, f"{kind} {counters[kind]}: ", 11, True); run(p, text, 11, False, True)


def table_caption(text):  # captions above tables
    counters["Table"] += 1
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=4, before=6, keep=True)
    run(p, f"Table {counters['Table']}: ", 11, True); run(p, text, 11, False, True)


def figure(path, width, text):
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=2, before=6, keep=True)
    p.add_run().add_picture(str(path), width=Inches(width))
    caption("Figure", text)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill); tcPr.append(s)


def table(rows, widths, size=10.5, header=True, align_first_left=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    tblPr = t._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:color"), "808080"); b.append(x)
    tblPr.insert_element_before(b, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription", "w:tblPrChange")
    for j, wj in enumerate(widths): t.columns[j].width = Inches(wj)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Inches(widths[j]); p = c.paragraphs[0]; fmt(p, WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and align_first_left) or len(str(val)) > 28 else WD_ALIGN_PARAGRAPH.CENTER, after=2)
            p.paragraph_format.line_spacing = 1.0
            run(p, str(val), size, bold=(header and i == 0))
            if len(rows) <= 12 and i < len(rows) - 1: p.paragraph_format.keep_with_next = True
            if header and i == 0: shade(c, "D9E2F3")
    sp = doc.add_paragraph(); fmt(sp, after=4); sp.paragraph_format.line_spacing = 1.0
    return t


f2 = lambda v, nd=2: f"{v:.{nd}f}"

# ================================================================== 1 INTRODUCTION
H1(1, "Introduction", first=True)
H2("1.1 Background")
P("Handwritten examination scripts, wills, contracts and notes are routinely disputed. A typical case is an answer sheet whose ownership is questioned because another person is suspected of copying the answers in an imitation of the owner's handwriting. The owner can then offer earlier genuine scripts as references, and an examiner must decide whether the questioned script is consistent with them.")
P("This is a page-level, multi-reference verification problem. The evidence is a complete script, not a single word, sentence or signature, and several known pages of the claimed writer are available. The system described in this report takes the previous handwriting samples as reference images and the questioned handwriting as input, compares letter shapes, strokes, spacing and other writing characteristics with a deep neural network, and reports whether the questioned page is consistent with the claimed writer or potentially forged.")
H2("1.2 Motivation")
P("Most deep-learning work on handwriting is organised differently. Writer identification and retrieval systems assign a page to one of many known writers or rank a collection by similarity [1]–[5]. Verification work compares a pair of samples, usually words, text patches or signatures [6]–[9]. Little of it combines a whole questioned page with several reference pages, reports a calibrated degree of support, or allows the system to answer “inconclusive”.")
P("A second difficulty is the threat model. Public datasets contain genuine pages by many writers, but almost none contain a person deliberately imitating another. At the same time, generative models can now imitate a given writing style [10]–[13]. A useful system must therefore be honest about what it can and cannot show.")
H2("1.3 Overview of the work")
B(["A page-level verifier: a ResNet-18 trained from scratch embeds a full script as an attention-pooled bag of ink patches; one or several reference pages are fused into a calibrated three-way verdict (consistent, inconclusive, likely forged).",
   "A data pipeline over three English corpora (IAM, CVL, Firemaker) with writer-disjoint splits, leakage checks and a test protocol in which test sets are used once.",
   "An evaluation of seven baselines and ablations, multi-reference fusion, cross-corpus transfer, disguised writing, calibration and synthetic pages.",
   "A working demonstration service (CPU) and a live training dashboard, and a journal-style paper draft generated from the result files."])
H2("1.4 Organisation of the report")
P("Chapter 2 defines the problem, scope and objectives. Chapter 3 surveys the literature. Chapter 4 analyses the requirements. Chapter 5 presents the methodology and system design, and Chapter 6 the implementation. Chapter 7 reports and discusses the results, Chapter 8 concludes, and Chapter 9 lists the references. Annexure I gives the outcome-specific supporting evidence.")

# ================================================================== 2 PROBLEM DEFINITION
H1(2, "Problem Definition, Scope and Objectives")
H2("2.1 Problem statement")
P("Given a claimed writer with N ≥ 1 reference pages R = {r1, …, rN} and one questioned page q, the system must return a similarity score s(q, R) and a verdict: consistent, inconclusive or likely forged. The unit of evidence is a full page. All pages are embedded by one network f into unit vectors, so that s is a function of cosine similarities.")
H2("2.2 Scope")
table([["In scope", "Out of scope for this report (first paper)"],
       ["English full pages from IAM, CVL and Firemaker", "The four Indian languages planned as a second phase (data collection paused by decision)"],
       ["One to five reference pages per claimed writer", "Real impersonation data (none is public)"],
       ["Writer-disjoint testing and a cross-corpus test", "Generators of learned imitation (DiffusionPen and similar)"],
       ["Calibrated three-way verdict; CPU demonstration", "Word-level and signature verification"]], [3.2, 3.2], size=11)
H2("2.3 Objectives and research questions")
B([[("O1. ", True), ("Build and evaluate a page-level, multi-reference handwriting verifier trained from scratch.", False)],
   [("O2. ", True), ("Measure how reliable the verdict is, including its abstention (inconclusive) behaviour and calibration.", False)],
   [("O3. ", True), ("Test transfer to a corpus never used for training or tuning, and the effect of disguised writing.", False)],
   [("O4. ", True), ("Deliver a reproducible pipeline, a working demonstration and a journal-style paper.", False)]])
P("The objectives are studied through four research questions. RQ1: how well does a page-level model separate writers, and does it beat untrained and frozen-feature baselines? RQ2: does adding reference pages help, and which fusion rule is best? RQ3: does the model transfer to a corpus recorded with a different scanner and protocol (Firemaker), including pages written in a disguised style? RQ4: which training ingredients matter: distillation, attention pooling, content-matched batches, writer mining, patch count and augmentation?")
H2("2.4 Expected outcome and sustainable development goals")
P(f"The outcome selected for this project is a journal publication (Annexure I). The work supports {SDG}, because reliable document verification helps courts, examination bodies and institutions settle disputes about authorship, and, secondarily, SDG 4 (quality education) through the integrity of handwritten examinations.")

# ================================================================== 3 LITERATURE SURVEY
H1(3, "Literature Survey")
P("Only the abstracts of the cited arXiv papers and the dataset papers were read for this survey; no performance numbers from other papers are quoted.")
H2("3.1 Writer identification and retrieval")
P("Text-independent writer identification with CNNs was established by DeepWriter [1], and later work introduced fragment-based and attention-based models [2], [16]. Unsupervised and self-supervised representations have become competitive for retrieval on historical and modern pages [3]–[5], [17]. A recent study analyses which pipeline factors matter for historical writer identification [18]. These systems answer “which writer” or “which pages are similar”; they do not output a verification decision for a claimed writer.")
H2("3.2 Handwriting verification")
P("Adak et al. study writer identification and verification from intra-writer variation [6], [19]. Shaikh et al. propose an attention-based writer-independent verifier [7], and self-supervised pre-training has been applied to the same task [9], [20]. These compare one pair of samples; our setting differs in using complete pages and several references.")
H2("3.3 Signature verification")
P("Signature verification uses twin networks and metric learning [8], [21], [22]. The methods are designed for signatures, not for whole pages, but they motivate the use of distance learning and of the cross-dataset generalisation check that we apply to Firemaker.")
H2("3.4 Generated handwriting as a threat")
P("Diffusion and transformer generators produce handwriting in a target style from short style samples [10]–[12], including paragraphs [13]. Synthetic data has also been used to pre-train handwriting models [23]. Such generators are the foreseeable forgery tool, but they were not run in this work (Chapter 7, Limitations).")
H2("3.5 Distillation, fusion and calibration")
P("Knowledge distillation [24] has been used in offline signature verification [14]. Self-supervised ViTs such as DINOv2 [25] provide strong frozen features. Forensic work argues that the strength of evidence should be expressed as a likelihood ratio [15], and modern networks are often poorly calibrated [26]. We adopt these ideas for page-level handwriting verification.")
H2("3.6 Summary and research gap")
table_caption("Representative related work and the gap for a page-level, multi-reference task.")
table([["Work", "Level", "What it does", "Gap for this task"],
       ["DeepWriter [1]", "text patches", "CNN for text-independent writer identification", "identifies; does not verify a claim"],
       ["FragNet [2]", "fragments", "fragment network for writer identification", "closed-set identification"],
       ["Self-supervised ViT [4]", "pages", "unsupervised writer retrieval", "ranks pages; no verdict"],
       ["Attention verifier [7]", "pairs", "writer-independent verification", "one pair, not many references"],
       ["Self-supervised verification [9]", "pairs", "pre-training for verification", "pairs; no calibration"],
       ["SigNet [8]", "signatures", "Siamese signature verification", "signatures, not full pages"]], [1.7, 1.0, 2.1, 1.9], size=10)
P("To our knowledge, no prior writer-verification work combines a whole questioned page, several reference pages, content-matched hard negatives, a calibrated verdict that may abstain, a never-tuned cross-corpus test and explicit reporting of what the data cannot show. We do not claim that any single component is new: distillation and likelihood-ratio reporting each have precedent [14], [15].")
H2("3.7 Datasets")
P("IAM [27] provides forms by many writers; CVL [28] provides pages by writers who copy the same texts; Firemaker [29] provides four pages for each of 250 writers, with page 3 written in a deliberately changed style.")

# ================================================================== 4 REQUIREMENT ANALYSIS
H1(4, "Requirement Analysis")
H2("4.1 Functional requirements")
table_caption("Functional requirements.")
table([["ID", "Requirement", "Where met"],
       ["FR1", "Accept one questioned page and one to five reference pages as images", "Demo API /api/verify; train/src/eval.py"],
       ["FR2", "Normalise pages (crop printed parts, remove background, width 1600)", "train/src/prep_cache.py; demo engine"],
       ["FR3", "Return a similarity score from a page embedding", "train/src/model.py"],
       ["FR4", "Fuse several references (mean, max, top-2, centroid)", "train/src/eval.py, multiref.py"],
       ["FR5", "Return consistent / inconclusive / likely forged with thresholds fixed on validation data", "train/src/verdict.py; demo calibration"],
       ["FR6", "Show where the model looked (attention heatmaps)", "train/src/explain.py; demo"],
       ["FR7", "Reproduce every table and figure from stored run files", "train/src/report.py"]], [0.5, 3.7, 2.5], size=10)
H2("4.2 Non-functional requirements")
B([[("Correctness and leakage control: ", True), ("writer-disjoint splits with assertions; test sets used once; Firemaker never used for tuning; a label-shuffle test to detect leaks.", False)],
   [("Reproducibility: ", True), ("seeded runs, Docker image, configuration saved with each run, no hand-typed numbers in the paper.", False)],
   [("Performance: ", True), ("the demonstration answers in about 2 to 3 seconds on CPU; a 120-epoch training run takes about 50 minutes on the laptop GPU.", False)],
   [("Honest reporting: ", True), ("results are not described as forgery detection rates; limits are printed in the demo and the paper.", False)],
   [("Privacy and security: ", True), ("all data and models stay on the local machine; secrets are kept in a git-ignored environment file.", False)]])
H2("4.3 Data requirements")
P("Three English corpora are used (Table 7 in Chapter 6). IAM and CVL require registration; the IAM copy used here is a third-party Kaggle copy, so official registration is needed before IAM is cited in a submitted paper. Firemaker is test-only. A Kaggle real-versus-generated set of font-rendered pages is used only for a synthetic-page false-accept test.")
H2("4.4 Software requirements")
table_caption("Software environment.")
table([["Component", "Version / choice"],
       ["Operating system", "Linux (CachyOS), Docker with NVIDIA container runtime"],
       ["Base image", "handwriting-train:0.1 built on a local PyTorch image (torch 2.6.0, CUDA 12.4)"],
       ["Python packages", "timm 1.0.11, opencv-python-headless 4.10.0.84, scikit-learn 1.6.1, pandas 2.2.3, pytest 8.3.3, matplotlib 3.9.2"],
       ["Teacher model", "DINOv2-small (frozen, Apache-2.0), used during training only"],
       ["Demo service", "FastAPI, uvicorn, python-multipart (CPU only)"],
       ["Paper", "TeX Live 2026, IEEEtran, biblatex with biber"]], [1.8, 4.9], size=10)
H2("4.5 Hardware requirements")
table_caption("Hardware used and measured resource use.")
table([["Item", "Value"],
       ["GPU", "NVIDIA RTX 4050 Laptop, 6 GB (about 3.5 GB used in training)"],
       ["CPU / memory", "Intel i7-13650HX (20 threads), 15 GB RAM"],
       ["Training speed", "about 25 to 28 s per epoch (70 steps) with distillation"],
       ["Demo", "CPU only, 3 cores; about 2 to 3 s per comparison"]], [1.8, 4.9], size=10)
H2("4.6 Constraints, assumptions and risks")
table_caption("Main risks and how they are handled.")
table([["Risk", "Handling"],
       ["Writer or text leakage between splits", "writer-disjoint splits; assertions in manifests.py; unit test; label-shuffle test"],
       ["Scanner or pen cues inflate IAM results", "report CVL and Firemaker as the main evidence; state IAM is optimistic"],
       ["No real impersonation data", "other-writer impostors, Firemaker disguise as a separate stress test; no forgery-rate claim"],
       ["Disguised writing breaks the model", "reported as a limitation; inconclusive outcome retained"],
       ["Third-party IAM copy", "official registration pending (BLOCKERS.md)"],
       ["Single laptop GPU, 10-hour budget", "small student model, early stopping, one GPU job at a time"]], [2.6, 4.1], size=10)
H2("4.7 Feasibility")
P("Technically the work is feasible: a ResNet-18 student trains in under an hour on a 6 GB GPU, the teacher is open-weight, and the demonstration runs on CPU. Economically it uses only free, open-source software and public or registered datasets. Operationally the pipeline is a set of scripted stages that rebuild every number from stored files.")

# ================================================================== 5 METHODOLOGY AND DESIGN
H1(5, "Proposed Methodology and System Design")
H2("5.1 Overall design")
figure(IMG / "architecture.png", 6.4, "System overview. (a) Training: ink-centred patches are encoded by a ResNet-18 trained from scratch and pooled by attention into a page embedding; a frozen DINOv2 teacher is used only during training. (b) Inference: the questioned page is compared with N reference pages, the scores are fused and calibrated, and a verdict is returned.")
H2("5.2 Preprocessing")
P("Each page is converted to gray scale and the printed parts of the forms are removed: for IAM the printed header and footer are cropped using long horizontal separator lines, and for CVL the printed source text above the first separator is cropped. The paper tone and scanner shading are removed by dividing by a smoothed morphological-closing background estimate. Pages are resized to a width of 1600 px, which keeps stroke scale comparable across 300 dpi scans. Pages with an ink fraction below 0.25% are discarded as blank, and duplicate pages are removed by hash.")
H2("5.3 Patch sampling and page encoder")
P("A page is treated as a bag of K patches of 128 × 128 pixels. Patches are centred on randomly chosen ink pixels and kept only if their ink fraction passes a threshold that is lowered progressively (5%, 3%, 1.5%, 0.8%) when a page has too few dense windows; a page with no usable ink is rejected instead of being padded. A ResNet-18 [30] with a one-channel stem, randomly initialised, maps each patch to a 256-dimensional vector z_i. Attention pooling [31] computes weights a_i = softmax(wᵀ tanh(V z_i)) and the page embedding e = normalise(Σ a_i z_i). Training uses K = 16 random patches per page with augmentation (rotation ±3°, scale ±10%, contrast and brightness, blur, noise and JPEG artefacts); evaluation uses K = 64 deterministic patches per page.")
H2("5.4 Training objective")
P("The main loss is the supervised contrastive loss [32] with temperature 0.1 on page embeddings. Each batch contains 12 writers with two different pages each. IAM batches use random writers; content-matched CVL batches (40% of steps) choose two texts and use only writers who wrote both, so that all negatives share text with the positives' texts and the network must separate writers by style instead of text. Every 10 epochs the writer centroids are computed with the EMA model and, with probability 0.5, a batch is built from a writer and its nearest neighbours (hard-writer mining).")
H2("5.5 Distillation from a frozen teacher")
P("A frozen DINOv2-small teacher [25] embeds the same augmented patches. A feature term, 1 − cos(g(z_i), t_i), and a relational term, the KL divergence between the patch-to-patch similarity matrices of teacher and student, are added with weights 0.5 each to the contrastive loss. The teacher is not used at inference; the deployed model is the student alone.")
H2("5.6 Multi-reference scoring, calibration and verdict")
P("For N references we compare four fusion rules: the mean cosine, the maximum cosine, the mean of the two largest cosines, and the cosine to the renormalised centroid of the reference embeddings. Scores are calibrated by Platt and isotonic regression fitted on validation trials only. A page is consistent if the score is at or above the 99th percentile of validation impostor scores, likely forged if it is at or below the 1st percentile of validation genuine scores, and inconclusive otherwise. When the two thresholds would overlap because a corpus is separable on validation (IAM), one midpoint threshold is used and there is no inconclusive band.")
H2("5.7 Evaluation protocol and metrics")
P("For single-reference evaluation every pair of pages from the same dataset and split is scored; same-writer pairs are genuine and different-writer pairs are impostors, so impostor pairs on CVL include pairs that copy the same text. For multi-reference evaluation every N is tested on one fixed set of writers with nested reference sets and the same impostor queries for every N. We report the equal error rate (EER), the area under the ROC curve (AUC), the false accept rate at 1% and 10% false reject rate, and 95% bootstrap intervals over writers. Thresholds and calibrators are fitted on validation data and applied unchanged to the test sets. Main results are the mean and standard deviation over three seeds.")
H2("5.8 Baselines")
P("We compare with (i) a randomly initialised encoder, (ii) the same encoder with batch-normalisation statistics adapted to the data but no label training (validation only), (iii) frozen DINOv2-small features averaged over patches, and (iv) HOG features averaged over patches.")
H2("5.9 Demonstration service")
P("The demonstration is a FastAPI service with three endpoints: /api/health, /api/info (model, calibration and sample pairs) and /api/verify (one to five reference images and one questioned image). It repeats the preprocessing, patch sampling and centroid fusion of the evaluation code, so its scores match the evaluation pipeline, and its thresholds are fitted on validation pages only. It runs on CPU so that it never competes with training for the GPU.")

# ================================================================== 6 IMPLEMENTATION
H1(6, "Implementation")
H2("6.1 Code organisation")
table_caption("Main modules (train/src/).")
table([["Module", "Role"],
       ["prep_cache.py", "page cache builder: crops, background normalisation, blank and duplicate filtering"],
       ["manifests.py", "writer-disjoint splits and leakage assertions; writes master.csv"],
       ["data.py", "ink-centred patch sampler, augmentation, content-matched batch sampler with writer mining"],
       ["model.py", "page encoder, supervised contrastive loss, distillation losses"],
       ["train.py", "training loop: mixed precision, EMA, resume, OOM and NaN handling, JSONL log"],
       ["eval.py, multiref.py, verdict.py", "single and multi-reference evaluation, calibration, verdicts, Firemaker, synthetic pages"],
       ["baselines.py, explain.py", "baselines; attention heatmaps and deletion test"],
       ["report.py", "all tables, figures and the abstract, generated from the run files"]], [2.1, 4.6], size=10)
H2("6.2 Environment")
P("Everything runs in the Docker image handwriting-train:0.1 (PyTorch 2.6, CUDA 12.4), with source and tests bind-mounted so code changes need no rebuild. Training jobs are launched through run_train.sh, which serialises GPU jobs with a lock, lowers priority and records GPU temperature and memory. Exact commands for GPU and CPU use are in docs/HOW_TO_RUN.md.")
H2("6.3 Data preparation")
table_caption("Writers and pages after preprocessing.")
table([["Corpus", "Train (writers / pages)", "Validation", "Test"],
       ["IAM", "507 / 1262", "50 / 190", "99 / 392"],
       ["CVL (English)", "217 / 909", "31 / 127", "62 / 254"],
       ["Firemaker", "–", "–", "250 / 1000"]], [1.8, 1.8, 1.6, 1.5], size=10.5)
P("IAM writers with a single page contribute two half-page views to training only. A leakage check asserts that no writer appears in two splits, that Firemaker is test-only and that every evaluation writer has at least two pages. A label-shuffle test, which shuffles writer labels inside each batch, keeps the loss at chance (ln 24 = 3.135) and nothing is learned, which confirms that the training signal is the writer identity.")
H2("6.4 Training")
table_caption("Training and evaluation settings.")
table([["Setting", "Value"],
       ["Patch size / patches per page (train, eval)", "128 × 128 / 16, 64"],
       ["Embedding size, backbone", "256, ResNet-18 (1 channel, random initialisation)"],
       ["Batch", "12 writers × 2 pages; 40% CVL content-matched"],
       ["Mining", "probability 0.5, centroids refreshed every 10 epochs"],
       ["Optimiser and schedule", "AdamW, lr 3e-4, weight decay 1e-4, 5 warm-up epochs, cosine, 120 epochs, patience 30"],
       ["Loss", "supervised contrastive (τ = 0.1) + 0.5 feature + 0.5 relational distillation"],
       ["Stabilisation", "mixed precision, gradient clip 1.0, EMA 0.999"]], [2.8, 3.9], size=10)
P("The main model was trained with three seeds; each run took about 50 minutes (about 26 s per epoch) and stopped at epoch 109, 116 and 119, with the best validation epoch at 79, 93 and 105. Seven further 40-epoch runs form the ablation study.")
figure(IMG / "fig_training_curves.png", 6.4, "Validation EER (left, log scale; line style = seed) and training loss (right) of the three main runs.")
H2("6.5 Final evaluation")
P("Test and Firemaker data were evaluated once per seed after all training had finished (final_eval.sh). Two analysis errors were found afterwards and corrected without touching the model: the IAM verdict thresholds overlapped, and the multi-reference curve let the writer set shrink with the number of references. Both are recorded in FIXES.md and DECISIONS.md, and the corrected analyses (verdict.py and multiref.py) are the ones used in this report.")
H2("6.6 Demonstration and dashboard")
figure(IMG / "demo_inputs.jpg", 5.6, "Demonstration: a reference page and a questioned page are uploaded or chosen from validation samples.")
figure(IMG / "demo_result.jpg", 5.6, "Demonstration result: verdict, similarity score, where the score falls among validation pairs, and attention heatmaps.")
P("A separate local web dashboard shows the progress, estimated remaining time, GPU status and live loss and validation curves of the training queue. Both tools are stdlib or FastAPI services that only read run files or run inference.")
H2("6.7 Testing")
P("Six unit tests run in the container: leak-free manifests, patch sampling on real pages, deterministic evaluation patches, varying training patches, EER on extreme inputs, and a decreasing contrastive loss. No linter is configured. The demonstration was also checked end to end by calling its API with sample pairs and with a corrupt file.")
H2("6.8 Problems met and how they were fixed")
B([[("Resume bug. ", True), ("a second --set argument silently overwrote the first, so a seed-1 run resumed the seed-0 checkpoint; fixed with an extending argument and a seed check on resume.", False)],
   [("Invalid leak test. ", True), ("the first label-shuffle test kept the pairing between pages and writers and was replaced by per-batch shuffling.", False)],
   [("Overlapping verdict thresholds. ", True), ("on separable corpora the accept threshold fell below the reject threshold; fixed with a single midpoint threshold.", False)],
   [("Confounded multi-reference curve. ", True), ("the writer set shrank as N grew; replaced by a controlled study on one writer set.", False)]])

# ================================================================== 7 RESULTS AND DISCUSSION
H1(7, "Results and Discussion")
P("All numbers are test results, in percent, as mean ± standard deviation over three seeds, unless stated. The three seeds share the same writer-disjoint splits.")
H2("7.1 RQ1: separating writers")
table_caption("Single-reference test results over all page pairs.")
table([["Model", "IAM EER", "IAM AUC", "CVL EER", "CVL AUC"],
       ["Random CNN", f2(F["b0_iam_eer"]), f2(F["b0_iam_auc"], 3), f2(F["b0_cvl_eer"]), f2(F["b0_cvl_auc"], 3)],
       ["HOG", f2(F["b3_iam_eer"]), f2(F["b3_iam_auc"], 3), f2(F["b3_cvl_eer"]), f2(F["b3_cvl_auc"], 3)],
       ["DINOv2-small (frozen)", f2(F["b2_iam_eer"]), f2(F["b2_iam_auc"], 3), f2(F["b2_cvl_eer"]), f2(F["b2_cvl_auc"], 3)],
       ["Ours (3 seeds)", F["iam_eer"], F["iam_auc"], F["cvl_eer"], F["cvl_auc"]]], [2.2, 1.1, 1.1, 1.1, 1.1], size=10.5)
P(f"On held-out writers the student reaches an EER of {F['iam_eer']} on IAM and {F['cvl_eer']} on CVL. Frozen DINOv2-small features, the teacher without any training on handwriting, give {F['b2_iam_eer']:.2f} and {F['b2_cvl_eer']:.2f}, so the student roughly halves the error of the teacher on both corpora; HOG and an untrained CNN are far behind. The 95% bootstrap intervals over writers are wide (1.7 to 4.2 on IAM, 4.7 to 12.0 on CVL, per seed) because only 99 and 62 writers are available. Test error is higher than validation error (IAM {F['iam_val_eer']}, CVL {F['cvl_val_eer']}), which shows how optimistic the validation numbers were.")
P(f"At a strict operating point the picture is weaker: with the false reject rate fixed at 1%, the false accept rate is {F['iam_far1']}% on IAM and {F['cvl_far1']}% on CVL, and at 10% it is {F['iam_far10'].split(' ')[0]} and {F['cvl_far10']}%. The content-matched CVL pairs are therefore the hard case, and a single page pair is not enough evidence for a high-stakes decision. On validation, an untrained encoder with batch-normalisation statistics adapted to the data already reaches 8.5% on IAM and 21.3% on CVL, which suggests that corpus-level cues such as scanner and paper carry part of the IAM signal.")
table_caption("Per-seed test results (EER, %) and the share of inconclusive CVL verdicts.")
import json as _j
_ev = [_j.load(open(HERE.parents[1] / f"train/runs/main_s{s}/eval_test.json")) for s in (0, 1, 2)]
_vd = [_j.load(open(HERE.parents[1] / f"train/runs/main_s{s}/verdict_test.json")) for s in (0, 1, 2)]
table([["Seed", "IAM EER", "CVL EER", "Firemaker p4", "Firemaker p3", "CVL inconclusive"]] +
      [[str(i), f2(100 * e["test"]["iam"]["single_ref"]["eer"]), f2(100 * e["test"]["cvl"]["single_ref"]["eer"]), f2(100 * e["firemaker"]["p1_vs_p4-self-natural"]["eer"]), f2(100 * e["firemaker"]["p1_vs_p3-copy-forged"]["eer"]), f"{100 * _vd[i]['cvl']['inconclusive']:.1f}"] for i, e in enumerate(_ev)],
      [0.6, 1.0, 1.0, 1.4, 1.4, 1.3], size=10)
H2("7.2 RQ2: more reference pages and fusion rules")
figure(IMG / "fig_multiref_test.png", 6.5, "Test EER against the number of reference pages for the four fusion rules, on one fixed set of writers for every N (mean ± sd over three seeds).")
P(f"Figure 5 evaluates N = 1, 2, 3 references on one fixed set of writers ({F['mr_iam_writers']} on IAM, {F['mr_cvl_writers']} on CVL). With the centroid rule, EER falls from {F['mr_iam_centroid_1']:.2f} to {F['mr_iam_centroid_2']:.2f} and {F['mr_iam_centroid_3']:.2f} on IAM, and from {F['mr_cvl_centroid_1']:.2f} to {F['mr_cvl_centroid_2']:.2f} and {F['mr_cvl_centroid_3']:.2f} on CVL. On the 17 IAM writers with at least six pages, five references give {F['iam_n5'][5]:.2f} against {F['iam_n5'][1]:.2f} for one; CVL has too few such writers for N = 5. The fusion rule matters: on CVL the maximum is best at N = 3 ({F['mr_cvl_max_3']:.2f}, against {F['mr_cvl_mean_3']:.2f} for the mean and {F['mr_cvl_top2_3']:.2f} for the two largest), while on IAM the centroid is best at three references. The centroid was fixed on validation, where it was best; that the maximum is better on test CVL is a post-hoc observation that has not been confirmed on separate data. The seed-0 bootstrap intervals for N = 1 and N = 3 overlap, so the gain from extra references is consistent in direction but not established for the population of writers.")
H2("7.3 RQ3: cross-corpus transfer and disguise")
table_caption("Firemaker (250 writers): page 1 against pages 4, 2 and 3 of the same writers.")
table([["Pair (same writer)", "EER", "Same-writer cosine", "Other-writer cosine"],
       ["p1 vs p4 (natural)", F["fm_p4_eer_s"], f2(F["fm_p4_gen"], 3), f2(F["fm_p4_imp"], 3)],
       ["p1 vs p2 (upper case)", F["fm_p2_eer_s"], f2(F["fm_p2_gen"], 3), f2(F["fm_p2_imp"], 3)],
       ["p1 vs p3 (disguised)", F["fm_p3_eer_s"], f2(F["fm_p3_gen"], 3), f2(F["fm_p3_imp"], 3)]], [2.2, 1.4, 1.6, 1.5], size=10.5)
figure(IMG / "fig_firemaker.png", 3.8, "Cosine similarity to page 1 of the same writer on Firemaker (seed 0).")
P(f"Firemaker differs in scanner and protocol and was never used for training, validation or thresholds. For ordinary writing the EER is {F['fm_p4_eer_s']} (AUC 0.994), close to IAM, so the representation transfers. It does not survive a deliberate change of writing: upper-case copying raises the EER to {F['fm_p2_eer_s']} and the disguised page to {F['fm_p3_eer_s']}. The mean same-writer cosine falls from {F['fm_p4_gen']:.3f} to {F['fm_p2_gen']:.3f} and {F['fm_p3_gen']:.3f} while the mean other-writer cosine stays at about 0.79: disguise moves genuine pairs towards impostors. A writer who simply changes style can therefore be rejected by the system, and the same effect would help a forger who copies a style.")
H2("7.4 RQ4: training ingredients")
table_caption("Ablations on validation (EER, %; 40 epochs, one seed).")
table([["Variant", "IAM", "CVL", "Mean"]] + [[l, f2(a), f2(c), f2(m)] for l, a, c, m in F["abl"]], [3.0, 1.0, 1.0, 1.0], size=10.5)
P(f"No ingredient shows a clear benefit. Removing distillation ({F['abl'][1][3]:.2f} against {F['abl'][0][3]:.2f} mean EER), content-matched batches ({F['abl'][3][3]:.2f}) or augmentation ({F['abl'][6][3]:.2f}) gives a lower error than the full model, and mean pooling, no mining and eight patches ({F['abl'][2][3]:.2f}, {F['abl'][4][3]:.2f}, {F['abl'][5][3]:.2f}) are within 0.2 of it. The seed-to-seed spread of the main runs is about 0.2 in mean validation EER ({', '.join(f'{v:.2f}' for v in F['main_val_mean'])}), so only larger differences deserve attention, and with one seed per variant we cannot tell whether the lower values of the three variants are real. The 120-epoch main model is better than every 40-epoch variant, which suggests that training length matters more than any single ingredient, although the two schedules also differ in warm-up and patience.")
H2("7.5 Calibrated verdicts")
table_caption("Three-way verdict on test trials with three references (centroid rule), thresholds from validation only.")
table([["Corpus", "Consistent", "Likely forged", "Inconclusive", "Error (decided)", "ECE raw / Platt / isotonic"]] +
      [[n, f"{F[f'v_{k}_accepted']:.1f}", f"{F[f'v_{k}_rejected']:.1f}", f"{F[f'v_{k}_inconclusive']:.1f}", f"{F[f'v_{k}_error_on_decided']:.1f}", f"{F[f'ece_raw_{k}']:.2f} / {F[f'ece_platt_{k}']:.3f} / {F[f'ece_iso_{k}']:.3f}"] for n, k in (("IAM", "iam"), ("CVL", "cvl"))],
      [0.7, 1.0, 1.0, 1.0, 1.1, 1.9], size=10)
figure(IMG / "fig_verdict.png", 3.3, "Share of test trials judged consistent, inconclusive and likely forged (mean over three seeds, three references).")
P(f"Raw cosine scores are not probabilities: their expected calibration error is about {F['ece_raw_cvl']:.2f}. Platt scaling reduces it to {F['ece_platt_iam']:.3f} and {F['ece_platt_cvl']:.3f}, and isotonic regression to {F['ece_iso_iam']:.3f} and {F['ece_iso_cvl']:.3f}, all fitted on validation only (Figure 8). On CVL, {F['v_cvl_inconclusive']:.1f}% of trials are inconclusive and of the {F['v_cvl_coverage']:.1f}% that are decided, {F['v_cvl_error_on_decided']:.1f}% are wrong. IAM is separable on validation, so one threshold and no inconclusive band is used (error {F['v_iam_error_on_decided']:.1f}%), and the IAM verdict should not be read as evidence that the problem is easy.")
figure(IMG / "fig_det_calibration.png", 6.5, "Left: DET curves on test page pairs. Right: reliability of the Platt-calibrated probability for three references; the sigmoid is over-confident in the middle range and under-confident at the top.")
H2("7.6 Synthetic pages")
P(f"The 1362 font-rendered pages of the Kaggle set (and 990 real pages by unknown writers as a control) were scored against every real test page at the validation threshold. Font renderings are accepted for {F['syn_iam_generated']:.2f}% of page pairs on IAM and for none on CVL, but unrelated real pages are accepted at a similar rate ({F['syn_iam_real']:.2f}% and {F['syn_cvl_real']:.3f}%), and all score far below same-writer pairs. The test shows that fonts are not mistaken for real writers in this scan domain; it says nothing about learned imitation.")
H2("7.7 Where the model looks")
figure(IMG / "fig_attention.png", 6.5, "Attention weights (red, opacity proportional to weight) on four IAM validation pages.")
P(f"Removing the 25% most attended patches lowers the cosine to a same-writer reference by {F['del_full'] - F['del_top']:.4f} on average, against {F['del_full'] - F['del_rand']:.4f} for random patches (60 validation pairs, pair-to-pair standard deviation about 0.04). The difference is too small to call attention an explanation, so the maps are shown only as a description of where the pooling puts weight.")
H2("7.8 Discussion and threats to validity")
B([[("What the evidence supports. ", True), ("A small student trained from scratch separates writers better than the foundation model it was distilled from. IAM errors are low and optimistic; CVL (content-matched) and Firemaker (new scanner) are the better guides.", False)],
   [("Disguise is the weak point. ", True), ("A verdict of likely forged means “unlike this writer's ordinary pages”, which a forger and a disguising writer can both produce; this is why the inconclusive outcome is kept.", False)],
   [("No impersonation data. ", True), ("Impostors are other writers, an easier case than a skilled forger; the reported error rates are not forgery detection rates.", False)],
   [("Statistical limits. ", True), ("three seeds, one-seed ablations, wide intervals because few writers are available.", False)],
   [("Data limits. ", True), ("English only, third-party IAM copy, adults writing under instruction rather than under exam pressure.", False)]])

# ================================================================== 8 CONCLUSION
H1(8, "Conclusion")
P(f"We built a page-level handwriting verifier that compares a questioned page with one or several reference pages and returns a calibrated score and a three-way verdict. Trained from scratch on about 2,200 pages, it halves the error of frozen DINOv2 features ({F['iam_eer'].split(' ')[0]}% EER on IAM, {F['cvl_eer'].split(' ')[0]}% on CVL), transfers to a new corpus for ordinary writing ({F['fm_p4_eer']:.2f}% on Firemaker), and gains from extra reference pages, but it is fooled by deliberate style change ({F['fm_p2_eer']:.0f} to {F['fm_p3_eer']:.0f}% EER), and none of the training ingredients we ablated is demonstrated to help in a short schedule. Because no public data show one person imitating another, our error rates are not forgery detection rates.")
H2("Future work")
B(["Collect a real impersonation set with volunteers copying each other's handwriting, per language.", "Use a handwriting generator to test robustness to learned imitation.", "Repeat the ablations with several seeds at the full schedule.", "Use a larger validation set to choose the fusion rule.", "Extend to the four Indian languages for which data collection was paused."])

# ================================================================== 9 REFERENCES
H1(9, "References")
refs = json.load(open("/tmp/work/refs.json"))
for r in refs:
    r = re.sub(r"([a-z])- ([a-z])", r"\1\2", r); r = re.sub(r"- (?=[A-Z])", "-", r); r = r.replace("arXiv preprint arXiv:", "arXiv:")
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, after=4); p.paragraph_format.left_indent = Inches(0.4); p.paragraph_format.first_line_indent = Inches(-0.4)
    run(p, r, 11)

# ================================================================== ANNEXURE
H1(None, "Annexure - I: Outcome-Specific Supporting Evidence")
P(runs=[("Selected outcome: ", True), ("Journal publication.", False)])
table_caption("Evidence for the selected outcome.")
table([["Evidence", "Details", "Status"],
       ["Paper draft", "Page-Level Handwriting Forgery Detection: Multi-Reference Verification with Calibrated Verdicts. IEEEtran, 8 pages including references, 270-word abstract, six result figures, eight tables (paper/main.pdf; LaTeX archive supplied)", "Draft complete"],
       ["Results traceability", "docs/paper-claims-map.md maps every number to its evidence file; all tables and figures are generated by report.py", "Done"],
       ["Code and tests", "train/ (modules, six unit tests, run logs, Docker setup)", "Done; no remote repository yet"],
       ["Working demonstration", "demo/ FastAPI service (CPU) and train/dashboard/ training dashboard", "Done"],
       ["Venue and submission", "Journal not yet chosen", "Pending"],
       ["Citation check and AI-use disclosure", "Planned with the ARS citation-check and disclosure commands", "Pending"],
       ["Author block and IAM registration", "Confirm the guide's title and co-authorship; register officially for IAM", "Pending (BLOCKERS.md)"]], [1.5, 3.8, 1.4], size=10)
H2("Abstract of the paper draft")
P(open(HERE.parents[1] / "paper/sections/abstract.tex").read().replace("\\%", "%").strip())
H2("AI-assistance disclosure")
P("Claude (Anthropic) assisted with code, analysis scripts and drafting. The authors are responsible for the content. The frozen DINOv2-small teacher is used during training only.")

doc.save(sys.argv[2])
print("saved", sys.argv[2])
