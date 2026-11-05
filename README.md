# Handwriting Forgery Detection Using Deep Learning: research pack

Lincy Jenita J (URK23CS5079). Guide: S D Lalitha, Associate Professor.

**Question:** given several reference pages from a known writer, is a questioned *full page* of handwriting consistent with that writer, or likely forged?

## Contents
| Path | What |
|---|---|
| `papers/` | 66 open-access arXiv PDFs in six topic folders. See `papers/INDEX.md` |
| `references.bib` | Stub BibTeX for all 66. Titles may be truncated and authors are missing, so complete them before submission |
| `code/` | 15 shallow-cloned reference implementations (git metadata stripped, so check licenses upstream) |
| `models/MODELS.md` | Open-weight models, licenses, and the suggested pipeline |
| `models/weights/` | Downloaded: TrOCR base and small handwritten, DINOv2 small and base |
| `datasets/DATASETS.md` | Page-level datasets and how to get each one. No data is downloaded, because IAM, CVL and Firemaker need registration or a request |

## Suggested reading order (chosen from titles and abstracts; I have not read the full texts)
1. 1708.03361: Writer identification and verification, Siamese on page-level features
2. 2009.04532: Attention-based writer-independent handwriting verification
3. 2405.18320: Self-supervised handwriting verification
4. 1606.06472: DeepWriter, text-independent writer ID
5. 2003.07212: FragNet
6. 2409.00751: Self-supervised ViT for writer retrieval
7. 2305.05358: Writer retrieval for historical datasets
8. 1707.02131: SigNet, the Siamese baseline for forgery
9. 2409.00786: Paragraph-level handwriting imitation (the AI-forger threat)
10. 2409.06065: DiffusionPen
11. 2510.18671: Key factors in end-to-end historical writer identification

## Current focus (English model, then paper)
Status, plan and run commands: `docs/` (start at `docs/README.md`), `STATUS.md`, `HANDOVER.md`. Code in `train/`, paper in `paper/`, runs in `train/runs/`.

## Known gaps
- No paper found that does exactly page-level genuine vs. forged verification with multiple reference pages. This is the project's opening.
- Paywalled work (IEEE, Elsevier, Springer journals) was not downloaded. Only arXiv preprints are included.
- Two ICDAR 2026 papers (FormWR, X-VLAD) were seen only as announcements, so their PDFs are not here.
- Firemaker, CERUG-EN, IAM and CVL must be requested or registered for by Lincy.
