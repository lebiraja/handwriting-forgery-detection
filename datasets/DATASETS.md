# Datasets for Handwriting Forgery / Writer Verification

Page-level (full-script) datasets matter most for this project, since the questioned sample is a complete script, not a word or sentence. Nothing here is redistributed. Each one must be fetched from the official source under its license.

| Dataset | Level | Writers | Content | Access / license | Link |
|---|---|---|---|---|---|
| **IAM Handwriting DB** | page / line / word | 657 | English forms (~1,539 pages) | Free for non-commercial use, registration required | https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database |
| **CVL Database** | page | 310 (311 in the paper) | English and German, 7 texts per writer | CC BY-NC 3.0 | https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/ |
| **Firemaker** | page | 250 | Dutch, 4 pages per writer (normal, upper-case, forged, natural). **Page 3 is a "forged" condition, but in the Firemaker README that means the writer deliberately tries to write in a different style than their own (disguise), not imitation of another person.** It is useful for disguised-writing tests, not for impersonation. | Request from the University of Groningen (Schomaker) | https://www.ai.rug.nl/~sheng/writeridataset.html |
| **CERUG-EN** | page / paragraph | 105 | English, 2 paragraphs | Request from the University of Groningen | https://www.ai.rug.nl/~sheng/writeridataset.html |
| CERUG / Firemaker word-level splits | word | - | Segmented word images and train/test lists | Open | https://zenodo.org/records/13258163 |
| **ICDAR 2013 writer ID/retrieval** | page | 250 | Greek and English | Competition data, request from the organizers | (ICDAR 2013 competition page) |
| **HisIR19 / Historical-WI** | page | 720 | Historical documents | Open (ICDAR 2017 / 2019 competitions) | https://arxiv.org/abs/1912.03713 |
| **BanglaWriting** | page | 260 | Bangla, with line and word segmentation | Open (arXiv 2011.07499) | arXiv 2011.07499 (dataset link in the paper) |
| **DohaScript** (2026) | page | multi-writer | Hindi | See paper 2602.18089 | arXiv 2602.18089 |
| **BullingerDB** (2026) | page | multi | Historical, with HTR and writer retrieval | See paper 2605.30235 | arXiv 2605.30235 |
| **FormWR** (ICDAR 2026) | page | ~100,000 | ~400k pages, large-scale writer retrieval | See the Dortmund group's ICDAR 2026 paper | https://patrec.cs.tu-dortmund.de/ |
| **NIDS** (2024) | page | - | Non-destructive inspection of handwritten documents (forensic) | See paper 2401.04448 | arXiv 2401.04448 |
| IAM on HuggingFace | line | - | `Voxel51/iam_handwriting_finevision`, `Adarsh203/IAM_Handwriting_Line` | Check each card | https://huggingface.co/datasets/Voxel51/iam_handwriting_finevision |

## Signature datasets (related forgery task, useful for baselines)
CEDAR, GPDS-960/Synthetic, BHSig260 (Bengali and Hindi), MCYT-75, SVC-onGoing, MSDS (Chinese, NeurIPS 2022, https://github.com/HCIILAB/MSDS).
HuggingFace mirrors: `indra-inc/signature_genuine_forged_cedar`, `rakshitdabral/Signature-Verification-Dataset`.

## Gap worth stating in the paper
Few public datasets contain **genuine vs. skilled-forgery full-page** pairs. Firemaker page 3 (self-disguise) is the closest, and it is not impersonation. The usual workaround is to build forgeries synthetically with handwriting-generation models (DiffusionPen, WordStylist, VATr++ and paragraph-level imitation, arXiv 2409.00786). Say clearly in the paper that this is a limitation.

## Update 2026-11-05
New since the first version: `datasets/cache/` (normalised page arrays and manifests), the Kaggle real-vs-generated set (font-rendered, not AI imitation), cached CVL, Firemaker (unpacked), IAM Kaggle copy. Firemaker page 3 is self-disguise. See `../docs/data.md` for the authoritative description. Paused: IIIT Indic zips, PHDIndic_11, CEDAR.
