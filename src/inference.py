"""
TTA inference script for face anti-spoofing.

Usage:
    python src/inference.py \\
        --model-path best_model.pth \\
        --test-dir /path/to/test \\
        --sample-sub samplesubmission.csv \\
        --output submission.csv
"""

import argparse
import contextlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image as PILImage
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from model import DINOv3ConvNeXtClassifier

LABEL_NAMES = ["fake_mannequin", "fake_mask", "fake_printed", "fake_screen", "fake_unknown", "realperson"]
VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
TTA_SCALES = [224, 256, 288, 320, 384]
TEMPERATURE = 1.0


def find_image(image_id: str, base_dir: Path) -> str | None:
    for ext in VALID_EXTS:
        p = base_dir / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def load_model(model_path: str, checkpoint: str, device: str) -> DINOv3ConvNeXtClassifier:
    id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    label2id = {n: i for i, n in id2label.items()}
    backbone = AutoModel.from_pretrained(checkpoint)
    model = DINOv3ConvNeXtClassifier(backbone, len(LABEL_NAMES), id2label, label2id)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, processor, image_paths: list, device: str, batch_size: int = 16) -> list:
    """Run TTA inference: 5 scales × (original + H-flip) = 10 forward passes per image."""
    pred_labels = []
    autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else contextlib.nullcontext

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [
            PILImage.open(p).convert("RGB") if p else PILImage.new("RGB", (224, 224))
            for p in batch_paths
        ]

        probs_tta = None
        count = 0

        with autocast_ctx():
            for size in TTA_SCALES:
                resized = [img.resize((size, size)) for img in imgs]

                inp = processor(images=resized, return_tensors="pt").to(device)
                p = torch.softmax(model(**inp)["logits"] / TEMPERATURE, dim=-1)
                probs_tta = p if probs_tta is None else probs_tta + p
                count += 1

                flipped = [img.transpose(PILImage.FLIP_LEFT_RIGHT) for img in resized]
                inp_f = processor(images=flipped, return_tensors="pt").to(device)
                p_f = torch.softmax(model(**inp_f)["logits"] / TEMPERATURE, dim=-1)
                probs_tta = probs_tta + p_f
                count += 1

        probs_tta /= count
        preds = torch.argmax(probs_tta, dim=-1).cpu().numpy().tolist()
        pred_labels.extend([LABEL_NAMES[p] for p in preds])

    return pred_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--sample-sub", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--checkpoint", default="facebook/dinov3-convnext-large-pretrain-lvd1689m")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = load_model(args.model_path, args.checkpoint, device)

    submission = pd.read_csv(args.sample_sub)
    test_dir = Path(args.test_dir)
    image_paths = submission["id"].map(lambda x: find_image(x, test_dir)).tolist()

    missing = sum(1 for p in image_paths if p is None)
    if missing:
        print(f"Warning: {missing} images not found.")

    pred_labels = predict(model, processor, image_paths, device, args.batch_size)
    submission["label"] = pred_labels
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
