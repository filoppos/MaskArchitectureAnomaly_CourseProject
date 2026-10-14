# Comprehensive Road Scene Understanding for Autonomous Driving

> University course project — Politecnico di Torino, Master's in Data Science and Engineering  
> Built on a starting repository provided by the course instructors.

This project tackles two interconnected computer vision tasks: **semantic segmentation** and **anomaly segmentation** in autonomous driving road scenes. The core challenge is evaluating how well state-of-the-art segmentation models can not only classify known objects, but also detect Out-of-Distribution (OoD) objects never seen during training — a critical safety requirement for autonomous vehicles.

---

## What We Built

### 1. COCO-to-Cityscapes Class Mapping
The project compares two pre-trained EoMT checkpoints: one trained on **COCO** (133 panoptic classes, general-purpose) and one on **Cityscapes** (19 semantic classes, road-specific). A direct comparison is non-trivial because the two models have incompatible output spaces and were trained on different tasks.

We designed a **score-level aggregation mapping** that:
- Switches the COCO model's inference pipeline from panoptic (resize + pad) to semantic (sliding window), preserving spatial detail on high-resolution Cityscapes images
- Aggregates per-pixel logit scores of all COCO classes sharing the same Cityscapes target class before the argmax decision, rather than mapping after the argmax
- Assigns unmappable COCO classes to void (excluded from mIoU), making the comparison metric-consistent

This improved the COCO model's mIoU from ~0.40 (no mapping) to **54.52%** on the Cityscapes validation set.

### 2. Progressive Fine-tuning of EoMT on Cityscapes
Since the mapping alone was insufficient to close the gap with the Cityscapes-pretrained model (81.68% mIoU), we fine-tuned the COCO model on the Cityscapes training set using a **progressive unfreezing strategy** across four configurations of increasing trainable parameters:

| Configuration | Unfrozen Components | mIoU |
|---|---|---|
| `head-only` | Classification + mask heads | 62.81% |
| `head-upscale` | + Upscaling layers | 72.90% |
| `head-upscale-query` | + Decoder queries | 73.05% |
| `config-blocks-10-11` | + Last 2 encoder blocks + LayerNorm | **75.20%** |

The best configuration uses **discriminative learning rates** (10⁻⁴ for task-specific modules, 10⁻⁵ for encoder blocks) to reduce catastrophic forgetting, combined with AdamW, Cosine Annealing over 10 epochs, AMP, and gradient clipping.

### 3. Anomaly Segmentation Evaluation
We evaluated both **ERFNet** (pixel-based baseline) and **EoMT** (mask-based) on five anomaly benchmarks using post-hoc scoring methods — no retraining required:

- **MSP** — Maximum Softmax Probability
- **MaxLogit** — Maximum raw logit score
- **MaxEntropy** — Entropy of the softmax distribution
- **RbA** (Rejected by All) — mask-based method exploiting EoMT's region structure

For EoMT, the mask-based output is converted to a per-pixel logit map via:

$$\ell_c(x) = \sum_i p_i(c) \cdot m_i(x)$$

where $m_i(x)$ is the sigmoid mask of query $i$ and $p_i(c)$ its class probability.

We also investigated **Temperature Scaling** as a post-hoc calibration technique, finding marginal improvements and suggesting the model's predictions are already well-calibrated.

---

## Key Results

| Model | mIoU | Best anomaly benchmark |
|---|---|---|
| ERFNet | 69.7% | Weak across all benchmarks |
| EoMT COCO | 54.5% | Surprisingly strong on FS Static (AuPRC ~87%) |
| EoMT fine-tuned | 75.2% | Strong on RO-21 (AuPRC ~88%) |
| EoMT Cityscapes | **81.68%** | Most balanced across all benchmarks |

The Cityscapes-pretrained model is the most balanced overall. Interestingly, the COCO-pretrained model achieves competitive anomaly scores despite its lower mIoU — likely because its broader 133-class representations produce higher uncertainty on road-specific OoD objects.

Full results across all five benchmarks (SMIYC RA-21, SMIYC RO-21, FS L&F, FS Static, Road Anomaly) are reported in the project report.

---

## Repository Structure

```
├── eval/                        # ERFNet evaluation and anomaly scoring tools
├── trained_models/              # ERFNet pretrained weights
├── eomt/                        # EoMT model code and pretrained checkpoints
├── step4_inference.ipynb        # COCO-to-Cityscapes mapping and qualitative comparison
├── step5_config_1_2_3.ipynb     # Fine-tuning configs 1–3 (head-only, head-upscale, head-upscale-query)
├── step5_config_blocks_10_11.ipynb  # Fine-tuning config 4 (encoder blocks unfrozen)
├── step7.ipynb                  # ERFNet anomaly scoring (MSP, MaxLogit, MaxEntropy)
└── step8.ipynb                  # EoMT anomaly scoring (MSP, MaxLogit, MaxEntropy, RbA, Temperature Scaling)
```

---

## Benchmarks Used

- **Cityscapes** — semantic segmentation evaluation (19 classes, urban road scenes)
- **Fishyscapes Lost&Found** and **Fishyscapes Static** — OoD detection in driving scenes
- **SegmentMeIfYouCan RA-21** and **RO-21** — anomaly and obstacle detection benchmarks

---

## Report
The full technical report is available [here](project_report.pdf).

---

## Team

Filippo Manca, Leonardo Recchia, Giovanni Massetti, Livia Scoppola  
Politecnico di Torino — Master's in Data Science and Engineering
