# What we did (project summary)

**Project:** Handwriting Forgery Detection Using Deep Learning, Lincy Jenita J (URK23CS5079), guide S. D. Lalitha, Karunya Institute of Technology and Sciences. Question: given reference pages of a claimed writer, is a questioned **full page** consistent with that writer?

## Work done
1. **Research pack:** 66 open-access papers, 15 reference code bases, model and dataset lists (`papers/`, `code/`, `models/MODELS.md`, `datasets/DATASETS.md`).
2. **Data:** IAM, CVL (English), Firemaker (test only) and a Kaggle real-vs-font-generated set; cached and split writer-disjointly with leakage checks (`docs/data.md`).
3. **Model:** ResNet-18 trained from scratch on ink-centred patches, attention pooling to a 256-d page embedding; supervised contrastive loss with content-matched CVL batches and writer mining; distillation from a frozen DINOv2-small teacher (`docs/architecture.md`).
4. **Training:** 3 seeds (about 50 min each) plus 7 ablations on one laptop GPU.
5. **Evaluation (run once):** baselines, multi-reference fusion (controlled), Firemaker cross-corpus and disguise, calibration, three-way verdict, synthetic pages, attention deletion test.
6. **Paper:** IEEEtran draft, 8 pages, generated tables and figures, 270-word abstract (`paper/`, `docs/paper-plan.md`, `docs/paper-claims-map.md`).
7. **Tools:** CPU demo service (`demo/`), live training dashboard (`train/dashboard/`).
8. **Deliverables:** PRS report (.docx), review deck (.pptx), paper LaTeX zip in `submission/`.

## Key results (test, mean of 3 seeds)
- EER: IAM 2.35%, CVL 7.30% (frozen DINOv2: 5.44%, 15.02%; HOG 11.14%, 23.06%; untrained CNN 23.20%, 30.65%).
- Firemaker (never tuned on): natural 3.33%, upper case 23.87%, disguised 32.00%.
- Three references, same writers, centroid fusion: IAM 2.25 to 1.22, CVL 6.76 to 5.19 (max rule 3.31 on CVL, post-hoc).
- CVL verdict with three references: 8.5% inconclusive, 2.8% error on decided.

## Honest limits
No real impersonation data, so these are **not forgery rates**; disguise breaks the model; no ablated ingredient is shown to help (single seed); IAM is optimistic and the copy is third-party; English only. Two analysis errors were found and fixed after the test run (`FIXES.md`, `DECISIONS.md`).

## Open items (`BLOCKERS.md`)
Confirm author block (guide's title, co-authorship); official IAM registration; citation check and disclosure; choose a venue; Indian-language phase.

## Where things are
`HANDOVER.md` (start here), `STATUS.md`, `docs/README.md`, `docs/HOW_TO_RUN.md`, `submission/`.
