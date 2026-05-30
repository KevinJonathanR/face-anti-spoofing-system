# Face Anti-Spoofing System

> **FIND IT DAC UGM 2026** | Team The Gacors  
> 6-class face liveness detection using DINOv3 ConvNeXt-Large

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-blue)](https://huggingface.co/spaces/<your-username>/face-anti-spoofing)

---

## Live Demo

**→ [Try it on HuggingFace Spaces](https://huggingface.co/spaces/<your-username>/face-anti-spoofing)**

Upload any face image to detect whether it's a real person or a spoofing attack — no setup needed.

To run locally:

```bash
pip install -r app/requirements.txt
MODEL_PATH=best_model.pth python app/app.py
```

---

## Overview

This project develops a face anti-spoofing model that distinguishes between a **real face** and various **presentation attacks** in images. The system was built for the Data Analytics Competition FIND IT UGM 2026.

**Task**: 6-class image classification

| Class | Description |
|---|---|
| `realperson` | Genuine face |
| `fake_mannequin` | 3D mannequin / doll attack |
| `fake_mask` | Physical face mask |
| `fake_printed` | Printed photo attack |
| `fake_screen` | Screen replay / digital photo |
| `fake_unknown` | Other / unknown spoofing method |

**Validation Results**

| Metric | Score |
|---|---|
| Accuracy | **96.6%** |
| Macro F1 | **96.1%** |

---

## Model Architecture

```
Input Image
    │
    ▼
DINOv3 ConvNeXt-Large (frozen backbone)
    │
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

**At inference**, Test-Time Augmentation (TTA) is applied:
- 5 input scales: `[224, 256, 288, 320, 384]`
- 2 spatial augmentations per scale: original + horizontal flip
- **10 forward passes** per image, probabilities averaged

---

## Key Design Decisions

- **Backbone**: `facebook/dinov3-convnext-large-pretrain-lvd1689m` — strong visual features from self-supervised pretraining on 1.68B images.
- **Residual Adapter**: Preserves pretrained representations while allowing task-specific adaptation.
- **Focal Loss** (γ=2): Handles class imbalance (realperson 324 vs fake_printed 85 samples).
- **Data Cleaning**: 219 mislabeled images corrected via `misplaced_images.json`; 178 duplicates removed by MD5 hash.
- **Domain Augmentation**: Custom transforms simulating spoofing artifacts — JPEG compression, pixelation, low-resolution blur, and Moiré patterns.

---

## Project Structure

```
face-anti-spoofing-system/
├── notebooks/
│   └── TheGacors.ipynb         # Full competition notebook (Kaggle)
├── src/
│   ├── model.py                # DINOv3ConvNeXtClassifier + FocalLoss
│   ├── transforms.py           # Custom augmentation pipeline
│   ├── dataset.py              # Data cleaning & deduplication
│   ├── train.py                # Training CLI
│   └── inference.py            # TTA inference CLI
├── app/
│   ├── app.py                  # Gradio demo (HuggingFace Spaces)
│   └── requirements.txt
├── configs/
│   └── config.yaml             # All hyperparameters
├── misplaced_images.json       # Label correction manifest
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/<your-username>/face-anti-spoofing-system
cd face-anti-spoofing-system
pip install -r requirements.txt
```

Set your HuggingFace token (required to download the private backbone checkpoint):

```bash
export HF_TOKEN="your_token_here"
# or use: huggingface-cli login
```

---

## Training

```bash
# Configure paths in configs/config.yaml first
python src/train.py --config configs/config.yaml
```

Key hyperparameters (see `configs/config.yaml`):

| Parameter | Value |
|---|---|
| Learning rate | 1e-5 |
| Scheduler | Cosine |
| Batch size | 32 |
| Epochs | 10 (early stop patience=3) |
| Weight decay | 0.02 |
| FP16 | Yes |

---

## Inference

```bash
python src/inference.py \
    --model-path best_model.pth \
    --test-dir data/test \
    --sample-sub samplesubmission.csv \
    --output submission.csv
```

---

## Dataset

The dataset is from the **FIND IT DAC UGM 2026** competition (private, not redistributable).

Structure after preprocessing:

```
data/
├── train_clean/
│   ├── fake_mannequin/   (142 images)
│   ├── fake_mask/        (209 images)
│   ├── fake_printed/     (85 images)
│   ├── fake_screen/      (155 images)
│   ├── fake_unknown/     (264 images)
│   └── realperson/       (324 images)
└── test/                 (404 images)
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `transformers` | DINOv3 backbone + Trainer |
| `datasets` | Efficient data loading |
| `torch` / `torchvision` | Model training |
| `gradio` | Demo interface |
| `evaluate` | F1 & accuracy metrics |
| `scikit-learn` | Confusion matrix |

---

## Team

**Team The Gacors** — FIND IT DAC UGM 2026
