# Data

Last updated: 2026-11-05 23:55 IST. Downloaded data is never redistributed; each source keeps its own licence.

## Sources in use (English)
| Source | Location | Licence / access | Used for |
|---|---|---|---|
| IAM forms (Kaggle copy `naderabdelghany/iam-handwritten-forms-dataset`) | `datasets/kaggle_iam-handwritten-forms-dataset/` | IAM: non-commercial research, official registration at the FKI site. The Kaggle copy is third-party, **register officially before publishing** | train / val / test |
| CVL Database 1.1 (Zenodo 1492267) | `datasets/cvl_database/cvl-database-1-1/` | CC BY-NC (readme says 3.0, Zenodo 4.0); md5 verified `9cf3cdcd...` | train / val / test |
| Firemaker (Zenodo 1194612) | `datasets/firemaker_full_pages/firemaker/` | research use only, do not redistribute (README) | **test only** (cross-dataset) |
| Kaggle real vs generated (`rutwikkumarm/handwriting-dataset-real-and-generated`) | `datasets/kaggle_handwriting-dataset-real-and-generated/` | licence unknown | synthetic-page false-accept test only |

Also on disk, not used yet: `cerug_firemaker_words/` (word-level), `banglawriting.zip`, `trigraphslant/` (partial), `kaggle_*` signature sets were never fetched, HF signature sets and the Indic zips were **paused** (see BLOCKERS.md).

## Facts that matter
- **IAM:** 657 writer folders, 1,539 forms. 356 writers have 1 page, 142 have 2, 119 have 3 or more, one has 59. Printed header and a printed "Name:" footer are cropped (header found on 199/200 sampled pages; fallback crop otherwise).
- **CVL:** 310 writers, 1,604 pages, texts 1,2,3,4,(6),7,8. Text 6 is the German Faust excerpt and is excluded (English scope). Every writer copies the same texts, which makes content-matched negatives possible. Writer 0431 has blank pages.
- **Firemaker:** 250 writers x 4 conditions: p1 copy-normal, p2 copy-UPPER, p3 copy-forged, p4 self-natural. **p3 means the writer tries to write in a different style than their own (self-disguise). It is not imitation of another person.**
- **Kaggle generated pages** are font-rendered text (folders such as `ccursive`, `schoolbell2`, `better grade`), not AI imitation of a writer.

## Preprocessing (per page)
Grayscale -> crop printed parts (IAM, CVL) -> divide by a smoothed morphological-closing background -> resize to width 1600 -> uint8 `.npy`. Blank if ink fraction < 0.0025 (lowered from 0.005 after 34 CVL pages with thin pens were wrongly dropped). Duplicates by md5. Result: IAM 1,538 ok, CVL 1,597 ok, Firemaker 1,000 ok. Cache: `datasets/cache/` (about 16 GB with the Kaggle pages).

## Splits (writer-disjoint, seed 42, `datasets/cache/manifests/master.csv`)
| Split | Writers | Pages | Composition |
|---|---|---|---|
| train | 724 | 2,171 | IAM 507 (151 multi-page + 356 single-page as top/bottom halves), CVL 217 |
| val | 81 | 317 | IAM 50, CVL 31 (whole pages) |
| test | 411 | 1,646 | IAM 99, CVL 62, Firemaker 250 |
Rules: writer ids are prefixed per dataset (`iam_`, `cvl_`, `fm_`); at most 10 pages per training writer; eval uses whole pages from writers with >= 2 pages; checks in `manifests.check()` and `tests/test_pipeline.py` raise on any overlap.

## Known data issues
- IAM pages of the same writer share scanner, pen and session, so IAM EER is optimistic; Firemaker is the independent check.
- CVL fallback crop (521 pages) can clip the first handwritten line.
- One IAM writer lost a page to the blank filter, so 299 (not 301) writers have >= 2 pages.
- No genuine "one person copies another" data exists in these sets.
