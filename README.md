# Face Anti-Spoofing System

> 6-class face liveness detection · DINOv3 ConvNeXt-Large · **96.6% accuracy**

[![Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-orange)](https://huggingface.co/spaces/KevinJonathanR/face-anti-spoofing-system)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.56%2B-yellow)](https://huggingface.co)

---

## Live Demo

**→ [Try it on HuggingFace Spaces](https://huggingface.co/spaces/KevinJonathanR/face-anti-spoofing-system)**

Upload any face image — the model instantly classifies it as a real person or one of 5 spoofing attack types.

| Metric | Score |
|---|---|
| Accuracy | **96.6%** |
| Macro F1 | **96.1%** |

> Built for **FIND IT DAC UGM 2026** (national data analytics competition) — **Top 13 finish**.

---

## What It Does

Face anti-spoofing detects **presentation attacks**: attempts to fool a face recognition system using a printed photo, screen replay, mask, or mannequin instead of a real face. This is a core security component in access control and identity verification systems.

**6 classes:**

| Class | Description |
|---|---|
| `realperson` | Genuine live face |
| `fake_printed` | Printed photo attack |
| `fake_screen` | Screen / digital photo replay |
| `fake_mask` | Physical face mask |
| `fake_mannequin` | 3D mannequin / doll |
| `fake_unknown` | Other spoofing method |

---

## Architecture

```
Input Image
    │
    ▼
DINOv3 ConvNeXt-Large  ←  pretrained on 1.68B images (self-supervised)
    │  fine-tuned end-to-end
    ▼
Global Average Pooling  →  1536-dim embedding
    │
    ▼
Residual Adapter:  x + 0.5 × Adapter(x)
  └─ Linear → LayerNorm → GELU → Dropout(0.1) → Linear
    │
    ▼
Classifier Head
  └─ Linear(1536→512) → LayerNorm → GELU → Dropout(0.3) → Linear(512→6)
    │
    ▼
Logits (6 classes)
```

**At inference** — Test-Time Augmentation (TTA):
- 3 input scales × (original + H-flip) = **6 forward passes**, probabilities averaged

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **DINOv3 ConvNeXt-Large backbone** | Self-supervised pretraining on 1.68B images gives strong general visual features; ConvNeXt captures fine-grained local textures critical for spoofing detection |
| **Residual Adapter** `x + 0.5·f(x)` | Allows task-specific adaptation without catastrophically forgetting pretrained features |
| **Focal Loss** (γ=2, label smoothing=0.05) | Dataset is imbalanced (324 real vs 85 printed); focal loss down-weights easy examples |
| **Domain augmentation** | Custom transforms simulate real spoofing artifacts: JPEG compression, pixelation, low-resolution blur, Moiré patterns |
| **Data cleaning** | Found and corrected 219 mislabeled images via JSON manifest; removed 178 exact duplicates via MD5 hash |

---

## Project Structure

```
face-anti-spoofing-system/
├── notebooks/
│   └── TheGacors.ipynb      # Full competition notebook (Kaggle-ready)
├── src/
│   ├── model.py             # DINOv3ConvNeXtClassifier + FocalLoss
│   ├── transforms.py        # Domain-specific augmentation pipeline
│   ├── dataset.py           # Data cleaning & deduplication utilities
│   ├── train.py             # Training CLI
│   └── inference.py         # TTA inference CLI → submission.csv
├── app/
│   └── app.py               # Gradio demo (deployed on HuggingFace Spaces)
├── configs/
│   └── config.yaml          # All hyperparameters in one place
└── misplaced_images.json    # Label correction manifest (219 fixes)
```

---

## Quick Start

```bash
git clone https://github.com/KevinJonathanR/face-anti-spoofing-system
cd face-anti-spoofing-system
pip install -r requirements.txt
```

**Train:**
```bash
python src/train.py --config configs/config.yaml
```

**Inference (competition submission):**
```bash
python src/inference.py \
    --model-path model_traced.pt \
    --test-dir data/test \
    --sample-sub samplesubmission.csv \
    --output submission.csv
```

**Run demo locally:**
```bash
pip install -r app/requirements.txt
python app/app.py
```

> **Note**: Competition dataset is private and not redistributable. Model weights available on the live demo.

---

## Team

**Team The Gacors** — FIND IT DAC UGM 2026

Team Member:
1. Kevin Jonathan - Team Lead [@KevinJonathanR](https://github.com/KevinJonathanR)
2. Faaris Khairrudin - Team Member [@FaarisKhairrudin](https://github.com/FaarisKhairrudin)
3. Fauzan Ahsanudin - Team Member [@Fauzan-A25](https://github.com/Fauzan-A25)