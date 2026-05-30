"""
Training script for DINOv3 ConvNeXt face anti-spoofing classifier.

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoImageProcessor,
    AutoModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

from dataset import deduplicate, prepare_and_clean_dataset
from model import DINOv3ConvNeXtClassifier
from transforms import get_train_transforms, get_valid_transforms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in examples]),
        "labels": torch.tensor([x["label"] for x in examples], dtype=torch.long),
    }


def compute_metrics_fn(eval_pred):
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }


def main(cfg: dict):
    set_seed(cfg["seed"])

    data_dir = Path(cfg["data"]["train_dir"])
    clean_dir = Path(cfg["data"]["clean_train_dir"])
    json_file = cfg["data"]["misplaced_json"]
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_and_clean_dataset(json_file, str(data_dir), str(clean_dir))

    cache_path = output_dir / "deduped_split"
    meta_path = output_dir / "dedup_meta.json"

    if cache_path.exists() and meta_path.exists():
        print("Loading cached split from disk.")
        split = load_from_disk(str(cache_path))
        train_ds, valid_ds = split["train"], split["test"]
        meta = json.loads(meta_path.read_text())
        print(f"train={meta['n_train']}, valid={meta['n_valid']}, removed={meta['n_removed_exact']}")
    else:
        full_ds = load_dataset("imagefolder", data_dir=str(clean_dir))["train"]
        deduped, n_removed = deduplicate(full_ds)
        print(f"Removed {n_removed} duplicates from {len(full_ds)} images.")
        split = deduped.train_test_split(
            test_size=cfg["data"]["val_split"], seed=cfg["seed"], stratify_by_column="label"
        )
        train_ds, valid_ds = split["train"], split["test"]
        cache_path.mkdir(parents=True, exist_ok=True)
        split.save_to_disk(str(cache_path))
        meta_path.write_text(json.dumps({"n_train": len(train_ds), "n_valid": len(valid_ds), "n_removed_exact": n_removed}))

    label_names = train_ds.features["label"].names
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(cfg["model"]["checkpoint"])
    image_size = processor.size.get("shortest_edge") or processor.size.get("height") or 224
    mean, std = processor.image_mean, processor.image_std

    train_tfms = get_train_transforms(image_size, mean, std)
    valid_tfms = get_valid_transforms(image_size, mean, std)

    def preprocess_train(batch):
        batch["pixel_values"] = [train_tfms(img.convert("RGB")) for img in batch["image"]]
        return batch

    def preprocess_valid(batch):
        batch["pixel_values"] = [valid_tfms(img.convert("RGB")) for img in batch["image"]]
        return batch

    counts = Counter(train_ds["label"])
    total = sum(counts.values())
    class_weights = torch.tensor(
        [(total / counts[i]) ** 0.4 for i in range(len(counts))], dtype=torch.float
    )

    backbone = AutoModel.from_pretrained(cfg["model"]["checkpoint"])
    model = DINOv3ConvNeXtClassifier(backbone, len(label_names), id2label, label2id, class_weights)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=cfg["training"]["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["training"]["warmup_steps"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"] // 2,
        num_train_epochs=cfg["training"]["epochs"],
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds.with_transform(preprocess_train),
        eval_dataset=valid_ds.with_transform(preprocess_valid),
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["training"]["patience"])],
    )

    trainer.train()
    torch.save(trainer.model.state_dict(), output_dir / "best_model.pth")
    print(f"Model saved to {output_dir / 'best_model.pth'}")

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
