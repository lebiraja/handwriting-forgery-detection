# Open-weight / open-source models for the project

Only models with open licenses are listed. Download status is noted at the bottom of this file.

## A. Backbones for a page-level embedding (Siamese / metric learning)
| Model | License | Why it fits | Link |
|---|---|---|---|
| **DINOv2** (small / base / large) | Apache-2.0 | Strong self-supervised ViT features. Writer retrieval results with a ViT + VLAD (arXiv 2409.00751) came from the same family of ideas. | https://huggingface.co/facebook/dinov2-base |
| **SigLIP 2** | Apache-2.0 | Alternative strong image encoder | https://huggingface.co/google/siglip2-base-patch16-224 |
| ResNet-50/18, EfficientNet, ConvNeXt (timm) | Apache-2.0 | CNN baselines (SigNet, DeepWriter and FragNet use CNNs) | https://huggingface.co/timm |
| **ViT-base-patch16-224-in21k** | Apache-2.0 | Fine-tuned for writer ID in community models | https://huggingface.co/Imene/vit-base-patch16-224-in21k-Writer-Identification |

## B. Handwriting recognition (HTR/OCR). Optional: text-content features, or check that the script is genuine
| Model | License | Link |
|---|---|---|
| **TrOCR base/small/large handwritten** (IAM fine-tuned) | MIT | https://huggingface.co/microsoft/trocr-base-handwritten |
| TrOCR stage-1 | MIT | https://huggingface.co/microsoft/trocr-base-stage1 |
| Riksarkivet TrOCR (historical Swedish) | check card | https://huggingface.co/Riksarkivet/trocr-base-handwritten-hist-swe-2 |
| Surya OCR (layout and line detection, useful for splitting a page into lines) | check repo (GPL-3.0 code, custom weights terms) | https://github.com/VikParuchuri/surya |
| PaddleOCR / docTR (line detection) | Apache-2.0 | https://github.com/mindee/doctr |

## C. Handwriting generation models. These are the "forger" side: use them to build synthetic forgeries and test robustness
| Model | Venue / license | Code | Weights |
|---|---|---|---|
| **DiffusionPen** | ECCV 2024, MIT | https://github.com/koninik/DiffusionPen | see the repo README |
| **WordStylist** | ICDAR 2023, MIT | https://github.com/koninik/WordStylist | see the repo README |
| **VATr / VATr++** | ICDAR 2023 / 2024, MIT | https://github.com/aimagelab/VATr | see the repo README |
| **HWD** (Handwriting Distance metric) | - | https://github.com/aimagelab/HWD | - |
| Zero-shot paragraph-level imitation (2409.00786) | arXiv 2024 | see paper | - |
| ScriptViT, One-Shot Diffusion Mimicker, Layout Stroke Imitation | arXiv 2024-25 | see papers in `papers/04_*` | - |
| **InkShield** (2026), defense against mimicry | arXiv 2607.26976 | see paper | - |

## D. Writer identification / retrieval / verification reference implementations (in `code/`)
| Repo | Paper | License |
|---|---|---|
| marco-peer/icdar23 | Towards Writer Retrieval for Historical Datasets (2305.05358) | see repo |
| marco-peer/icdar24 | SAGHOG (2404.17221) | see repo |
| Traven16/SSL_ViT_WR | Self-Supervised ViT for Writer Retrieval (2409.00751) | see repo |
| Traven16/writer-retrieval-at-scale | Writer Retrieval at Scale, ICDAR 2026 | see repo |
| VChristlein/icdar17code | Unsupervised Feature Learning for Writer ID (1705.09369) | GPL-3.0 |
| QuwsarOhi/DeepWriter | DeepWriter (1606.06472) | see repo |
| mshaikh2/AttentionHandwritingVerification | Attention-based writer-independent handwriting verification (ICFHR 2020, 2009.04532) | see repo |
| Mihir2/ssl-hv | Self-Supervised Handwriting Verification (2405.18320) | Apache-2.0 |
| HCIILAB/MSDS | MSDS signature/handwriting verification (NeurIPS 2022) | GPL-3.0 |
| NiceRingNode/SPECTRUM | Multi-domain online handwriting verification (ACM MM 2025) | GPL-3.0 |
| Lab41/d-script | Writer ID of handwritten documents | see repo |
| koninik/DiffusionPen, koninik/WordStylist, aimagelab/VATr | handwriting generation | MIT |
| rsommerfeld/trocr | TrOCR fine-tuning | MIT |

Licenses marked "see repo" were not checked: the `.git` metadata was stripped after cloning, so check each upstream repo before reusing code.

## E. Suggested pipeline for the project (page-level, as Lincy's guide asked)
1. **Preprocess:** binarize, deskew, remove ruled lines and noise; segment the page into lines or tile it into patches (doctr or Surya for line detection).
2. **Encode:** an embedding network per patch (ResNet/DINOv2), then aggregate the patch embeddings into **one page-level descriptor** (VLAD / NetVLAD / attention pooling). This is what makes the verdict "full script", not word or sentence.
3. **Compare:** a Siamese network (contrastive / triplet / ArcFace loss) maps a questioned page and the reference pages (all of the previous papers) to a cosine similarity score.
4. **Decide:** thresholds calibrated on EER, with a three-way verdict (consistent, inconclusive, likely forged). Report FAR, FRR, EER, AUC and mAP.
5. **Stress test:** generate forgeries with DiffusionPen, WordStylist and VATr++ for robustness evaluation.

## Download status
Downloaded to `models/weights/`: microsoft/trocr-base-handwritten (MIT), microsoft/trocr-small-handwritten, facebook/dinov2-small and dinov2-base (Apache-2.0). Everything else is linked only.

## Update 2026-11-05
In use: DINOv2-small as a frozen distillation teacher (training only, `models/weights/dinov2-small`); TrOCR and DINOv2-base weights are downloaded but unused. DiffusionPen, WordStylist and VATr++ were not run (generation step dropped, see `../DECISIONS.md`). The deployed model is our from-scratch ResNet-18 student, `train/runs/<run>/best.pt`.
