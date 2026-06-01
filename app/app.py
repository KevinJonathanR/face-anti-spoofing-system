"""
Real-time face anti-spoofing demo using Gradio.
Deploy on HuggingFace Spaces for free hosting.

To run locally:
    pip install -r app/requirements.txt
    python app/app.py
"""

import os
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download, login
from PIL import Image

MODEL_REPO     = os.environ.get("MODEL_REPO", "")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model_traced.pt")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

if HF_TOKEN:
    login(token=HF_TOKEN)

LABEL_NAMES = ["fake_mannequin", "fake_mask", "fake_printed", "fake_screen", "fake_unknown", "realperson"]
LABEL_DISPLAY = {
    "fake_mannequin": "Mannequin Attack",
    "fake_mask":      "Mask Attack",
    "fake_printed":   "Printed Photo Attack",
    "fake_screen":    "Screen Replay Attack",
    "fake_unknown":   "Unknown Attack",
    "realperson":     "Real Person",
}
TTA_SCALES  = [224, 256, 288]
TEMPERATURE = 1.0

# Processor params from DINOv3 preprocessor_config.json (hardcoded to avoid Hub download)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def _make_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def resolve_model_path() -> str:
    local = Path(MODEL_FILENAME)
    if local.exists():
        print(f"Using local model: {local}")
        return str(local)
    if MODEL_REPO:
        print(f"Downloading model from Hub: {MODEL_REPO}/{MODEL_FILENAME}")
        return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=HF_TOKEN or None)
    raise FileNotFoundError(
        f"'{MODEL_FILENAME}' not found locally and MODEL_REPO is not set."
    )


def load_model() -> torch.jit.ScriptModule:
    path = resolve_model_path()
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    print("TorchScript model loaded.")
    return model


model = load_model()


@torch.no_grad()
def predict(image: Image.Image) -> dict:
    if image is None:
        return {}

    image = image.convert("RGB")
    probs_tta = None
    count = 0

    for size in TTA_SCALES:
        transform = _make_transform(size)
        resized = image.resize((size, size))

        inp = transform(resized).unsqueeze(0).to(DEVICE)
        p = torch.softmax(model(inp) / TEMPERATURE, dim=-1)[0]
        probs_tta = p if probs_tta is None else probs_tta + p
        count += 1

        flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
        inp_f = transform(flipped).unsqueeze(0).to(DEVICE)
        p_f = torch.softmax(model(inp_f) / TEMPERATURE, dim=-1)[0]
        probs_tta = probs_tta + p_f
        count += 1

    probs = (probs_tta / count).cpu().tolist()
    return {LABEL_DISPLAY[LABEL_NAMES[i]]: round(probs[i], 4) for i in range(len(LABEL_NAMES))}


EXAMPLES_DIR = Path(__file__).parent / "examples"
CLASS_EXAMPLES = [
    ("realperson.jpg",     "Real Person"),
    ("fake_mask.jpg",      "Mask Attack"),
    ("fake_printed.jpg",   "Printed Photo Attack"),
    ("fake_screen.jpg",    "Screen Replay Attack"),
    ("fake_mannequin.jpg", "Mannequin Attack"),
    ("fake_unknown.jpg",   "Unknown Attack"),
]
available_examples = [
    [str(EXAMPLES_DIR / fname)]
    for fname, _ in CLASS_EXAMPLES
    if (EXAMPLES_DIR / fname).exists()
]
available_labels = [
    label
    for fname, label in CLASS_EXAMPLES
    if (EXAMPLES_DIR / fname).exists()
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Label(num_top_classes=6, label="Prediction"),
    title="Face Anti-Spoofing · Liveness Detection",
    description=(
        "Detects whether a face is **real** or a presentation attack "
        "(printed photo, screen replay, mask, mannequin).\n\n"
        "**Model** — DINOv3 ConvNeXt-Large · Focal Loss · Domain Augmentation\n\n"
        "**96.6% accuracy · 96.1% macro F1** — FIND IT DAC UGM 2026 · Top 13 · Team The Gacors\n\n"
        "💡 Click an example below or upload your own image."
    ),
    examples=available_examples if available_examples else None,
    example_labels=available_labels if available_labels else None,
    flagging_mode="never",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(share=False)