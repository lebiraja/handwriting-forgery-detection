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

## Final results (filled 2026-11-06; mean +- sd over 3 seeds, test)
| Claim (paper section) | Number | Evidence file | Status |
|---|---|---|---|
| Student single-reference EER (RQ1) | IAM 2.35 +- 0.12, CVL 7.30 +- 0.29; AUC 0.996 / 0.968 | `runs/main_s*/eval_test.json` `test.*.single_ref` | DONE `tab:main` |
| Baselines on test (RQ1) | DINOv2 5.44 / 15.02; HOG 11.14 / 23.06; random CNN 23.20 / 30.65 | `runs/baselines/b{0,2,3}_test.json` | DONE `tab:main` |
| Test vs validation gap | val IAM 0.90 +- 0.31, CVL 5.81 +- 0.15 | `eval_test.json` `val.*.single_ref` | DONE text |
| FAR at FRR 1% / 10% | IAM 10.1 / 0.0, CVL 80.2 / 3.5 | `eval_test.json` `far@frr1`, `far@frr10` | DONE text |
| Bootstrap 95% CI per seed | IAM 1.7-4.2, CVL 4.7-12.0 | `eval_test.json` `eer_ci95` | DONE text |
| Random features + adapted BN (val only) | 8.5 / 21.3 | `runs/shuffle/metrics.jsonl` | DONE text |
| Controlled multi-reference, centroid (RQ2) | IAM 2.25, 1.57, 1.22; CVL 6.76, 5.22, 5.19 for N = 1, 2, 3 | `runs/main_s*/multiref_test.json` `common_N123` | DONE `fig:multiref` |
| Fusion rules at N=3 on CVL | max 3.31, mean 5.99, top-2 5.13 (post-hoc, not chosen on val) | `multiref_test.json` | DONE text |
| IAM five references (17 writers) | N=5 0.81 vs N=1 1.96 | `multiref_test.json` `common_N1235` | DONE text |
| Firemaker (RQ3) | p4 3.33, p2 23.87, p3 32.00 EER; cosine 0.978 / 0.906 / 0.874 vs 0.79 | `eval_test.json` `firemaker` | DONE `tab:fm`, `fig:fm` |
| Ablations (RQ4, val, 1 seed, 40 epochs) | full 5.35; no distill 4.38; mean pool 5.29; no c-match 4.57; no mining 5.25; k8 5.15; no aug 4.88 | `runs/abl_*/best_val.json` | DONE `tab:abl` |
| Calibration ECE | raw 0.22, Platt 0.070 / 0.076, isotonic 0.009 / 0.020 | `eval_test.json` `calibration` | DONE text |
| Three-way verdict (3 refs, corrected rule) | CVL 47.8 / 43.7 / 8.5, error 2.8; IAM separable, error 0.7 | `runs/main_s*/verdict_test.json` (from `src.verdict`) | DONE `tab:verdict` |
| Synthetic pages accepted | fonts 0.03% IAM, 0 CVL; real Kaggle 0.14% / 0.003%; mean cosine 0.40-0.53 | `eval_test.json` `synthetic` | DONE text |
| Attention deletion test | drop 0.0043 (top 25%) vs 0.0022 (random); 60 val pairs | `runs/main_s0/explain/deletion.json` | DONE text |
| Abstract (exactly 270 words, as in the sem5 deeplearning paper) | generated; also quotes HOG 11.1/23.1, untrained CNN 23.2/30.6, IAM centroid 2.25 -> 1.22 | `report.py write_abstract` | DONE |

Not used in the paper (flawed, see FIXES.md): `eval_test.json["three_way"]`, `eval_test.json["multi_ref"]`, `fig_multiref_val`.
