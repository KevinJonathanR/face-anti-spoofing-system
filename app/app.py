"""
Real-time face anti-spoofing demo using Gradio.
Deploy on HuggingFace Spaces for free hosting.

To run locally:
    pip install -r app/requirements.txt
    python app/app.py
"""

import os
import sys
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import DINOv3ConvNeXtClassifier

CHECKPOINT = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
# Set MODEL_REPO as a Space variable, e.g. "your-username/face-antispoofing"
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_model_dinov3_convnext.pth")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = ["fake_mannequin", "fake_mask", "fake_printed", "fake_screen", "fake_unknown", "realperson"]
LABEL_DISPLAY = {
    "fake_mannequin": "Mannequin Attack",
    "fake_mask": "Mask Attack",
    "fake_printed": "Printed Photo Attack",
    "fake_screen": "Screen Replay Attack",
    "fake_unknown": "Unknown Attack",
    "realperson": "Real Person",
}
TTA_SCALES = [224, 256, 288]
TEMPERATURE = 1.0


def resolve_model_path() -> str:
    # Priority 1: local file (useful when running locally)
    local = Path(MODEL_FILENAME)
    if local.exists():
        print(f"Using local weights: {local}")
        return str(local)

    # Priority 2: download from HuggingFace Hub
    if MODEL_REPO:
        print(f"Downloading weights from Hub: {MODEL_REPO}/{MODEL_FILENAME}")
        return hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            token=HF_TOKEN or None,
        )

    raise FileNotFoundError(
        "Model weights not found. Set MODEL_REPO env variable or place "
        f"'{MODEL_FILENAME}' in the working directory."
    )


def load_model():
    id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    label2id = {n: i for i, n in id2label.items()}
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT, token=HF_TOKEN or None)
    backbone = AutoModel.from_pretrained(CHECKPOINT, token=HF_TOKEN or None)
    model = DINOv3ConvNeXtClassifier(backbone, len(LABEL_NAMES), id2label, label2id)
    weights_path = resolve_model_path()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    print("Weights loaded.")
    return model.to(DEVICE).eval(), processor


model, processor = load_model()


@torch.no_grad()
def predict(image: Image.Image) -> dict:
    if image is None:
        return {}

    image = image.convert("RGB")
    probs_tta = None
    count = 0

    for size in TTA_SCALES:
        resized = image.resize((size, size))
        inp = processor(images=[resized], return_tensors="pt").to(DEVICE)
        p = torch.softmax(model(**inp)["logits"] / TEMPERATURE, dim=-1)[0]
        probs_tta = p if probs_tta is None else probs_tta + p
        count += 1

        flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
        inp_f = processor(images=[flipped], return_tensors="pt").to(DEVICE)
        p_f = torch.softmax(model(**inp_f)["logits"] / TEMPERATURE, dim=-1)[0]
        probs_tta = probs_tta + p_f
        count += 1

    probs = (probs_tta / count).cpu().tolist()
    return {LABEL_DISPLAY[LABEL_NAMES[i]]: round(probs[i], 4) for i in range(len(LABEL_NAMES))}


examples = [
    ["assets/example_real.jpg"],
    ["assets/example_printed.jpg"],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Label(num_top_classes=6, label="Prediction"),
    title="Face Anti-Spoofing System",
    description=(
        "Upload a face image to detect spoofing attacks.\n\n"
        "**Model**: DINOv3 ConvNeXt-Large + Custom Adapter\n"
        "**Classes**: Real Person vs. Mannequin / Mask / Printed / Screen / Unknown attack\n\n"
        "Built for FIND IT DAC UGM 2026 | Team The Gacors"
    ),
    examples=examples if any(Path(e[0]).exists() for e in examples) else None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=False)
