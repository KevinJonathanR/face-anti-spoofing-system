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
import torchvision.transforms as T
from huggingface_hub import hf_hub_download, login
from PIL import Image
from transformers import AutoConfig, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import DINOv3ConvNeXtClassifier

# MODEL_REPO must be set as a Space variable, e.g. "your-username/face-antispoofing"
# It should contain: config.json (backbone architecture) + best_model_dinov3_convnext.pth
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_model_dinov3_convnext.pth")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN not set.")

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

# Processor parameters are fixed from the DINOv3 preprocessor_config.json:
# size=224, mean/std = ImageNet defaults. Hardcoded to avoid fetching the
# gated model's config file from Hub on every Space startup.
_IMAGE_SIZE = 224
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
        print(f"Using local weights: {local}")
        return str(local)

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


def load_backbone_config():
    """
    Load backbone config with no outbound network calls.
    Priority:
      1. config.json next to app.py  (uploaded directly to the Space)
      2. MODEL_REPO on HuggingFace Hub (fallback, requires network)
    """
    local_config = Path(__file__).parent / "config.json"
    if local_config.exists():
        print(f"Loading backbone config from local file: {local_config}")
        return AutoConfig.from_pretrained(str(local_config.parent), trust_remote_code=True)

    if MODEL_REPO:
        print(f"Loading backbone config from Hub: {MODEL_REPO}")
        return AutoConfig.from_pretrained(MODEL_REPO, token=HF_TOKEN or None, trust_remote_code=True)

    raise FileNotFoundError(
        "config.json not found locally and MODEL_REPO is not set. "
        "Upload config.json to the Space or set MODEL_REPO."
    )


def load_model():
    id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    label2id = {n: i for i, n in id2label.items()}

    config = load_backbone_config()
    backbone = AutoModel.from_config(config, trust_remote_code=True)

    model = DINOv3ConvNeXtClassifier(backbone, len(LABEL_NAMES), id2label, label2id)

    weights_path = resolve_model_path()
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Ignored keys (not needed at inference): {unexpected}")
    if missing:
        print(f"Warning - missing keys: {missing}")
    print("All weights loaded.")
    return model.to(DEVICE).eval()


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
        p = torch.softmax(model(pixel_values=inp)["logits"] / TEMPERATURE, dim=-1)[0]
        probs_tta = p if probs_tta is None else probs_tta + p
        count += 1

        flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
        inp_f = transform(flipped).unsqueeze(0).to(DEVICE)
        p_f = torch.softmax(model(pixel_values=inp_f)["logits"] / TEMPERATURE, dim=-1)[0]
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
