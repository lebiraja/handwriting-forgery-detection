# Paper claim -> evidence map

Same rule as the Tamil transliteration paper (`/home/lebi/Projects/sem5/deeplearning/docs/paper-claims-map.md`): **no number is written into the paper by hand.** Every number comes from `train/runs/*/eval*.json` or `best_val.json` through `train/src/report.py`; a number that is not in this table must be added here with its source before it goes into the paper. Rows marked PENDING are filled when the final evaluation (`src.eval --test`) has run.

| Claim (paper section) | Number | Evidence file | Where | Status |
|---|---|---|---|---|
| Dataset sizes after preprocessing (Data) | IAM 507/50/99 writers, CVL 217/31/62, Firemaker 250 test | `datasets/cache/manifests/master.csv` | `tab:data` | DONE |
| Untrained CNN is a poor baseline (Results, RQ1) | val EER IAM 22.3 / CVL 34.3 | `runs/baselines/b0_val.json` | `fig:baselines` | DONE (val) |
| Random features with adapted BatchNorm already reach 8.5 / 21.3 | | `runs/shuffle/metrics.jsonl` (last val) | text | DONE (val) |
| Frozen DINOv2-small 5.0 / 16.0; HOG 14.1 / 23.6 | | `runs/baselines/b2_val.json`, `b3_val.json` | `fig:baselines` | DONE (val) |
| Seed-0 student: val EER IAM 0.48, CVL 5.9 (best epoch 79) | | `runs/main_s0/best_val.json` | text | DONE (val) |
| Leak test: shuffled labels give chance loss (ln 24 = 3.135) | 3.135 | `runs/shuffle/metrics.jsonl` | Experiments | DONE |
| Student vs baselines on TEST (RQ1) | EER, AUC, FAR@FRR, 95% CI | `runs/main_s*/eval_test.json` | `tab:main` | PENDING |
| More references help, centroid vs mean vs max vs top-2 (RQ2) | EER per N and rule | `eval_test.json` `multi_ref` | `fig:multiref` | PENDING (val preliminary) |
| Cross-dataset on Firemaker p1 vs p4, p2 (RQ3) | EER, AUC | `eval_test.json` `firemaker` | `tab:fm` | PENDING |
| Disguise (p3) shifts same-writer scores | genuine mean, EER | `eval_test.json` `firemaker` | `tab:fm` | PENDING |
| Distillation, pooling, content-matched, mining, k, augmentation ablations (RQ4) | val EER | `runs/abl_*/best_val.json` | `fig:ablation` | PENDING (queue) |
| Calibration (ECE) and 3-way verdict coverage / error | ECE, coverage, error on decided | `eval_test.json` `calibration`, `three_way` | `tab:verdict` | PENDING |
| Deletion test: removing top-attended patches hurts more than random | mean score drops | `runs/main_s0/explain/deletion.json` | text | PENDING (`explain.py` not run yet) |
| Abstract numbers (EER IAM/CVL, frozen DINOv2, Firemaker, best fusion, verdict coverage/error) | all generated, <= 120 words | `report.py write_abstract` from `eval_test.json`, `baselines/b2_test.json` | Abstract | PENDING |
| Reference for ablations: full model at 40 epochs, val EER IAM 1.72 / CVL 8.98 | | `runs/abl_main40/best_val.json` | `fig:ablation` | DONE (val); other 6 ablations PENDING |
| Seed variation | mean and sd over 3 seeds | `runs/main_s{0,1,2}/best_val.json` | tables | PENDING |
| Synthetic-page false-accept rate (font-rendered Kaggle pages) | rate at validation thresholds | to be added to `eval.py` | Results | PENDING |

Literature statements: only papers whose **abstract** was read (all cited arXiv papers, checked 2026-11-06) or that are dataset papers named in the dataset READMEs (IAM, CVL, Firemaker). Full texts were not read; no numbers from other papers are quoted.
