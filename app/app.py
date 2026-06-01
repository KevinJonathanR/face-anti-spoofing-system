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

# render=False defers placement so gr.Examples can be rendered above the upload box
img_input    = gr.Image(type="pil", label="Upload Face Image", render=False)
label_output = gr.Label(num_top_classes=6, label="Prediction", render=False)

with gr.Blocks(title="Face Anti-Spoofing · Liveness Detection") as demo:
    gr.Markdown(
        "# Face Anti-Spoofing · Liveness Detection\n\n"
        "Detects whether a face is **real** or a presentation attack "
        "(printed photo, screen replay, mask, mannequin).\n\n"
        "💡 Click an example below or upload your own image."
    )

    if available_examples:
        gr.Examples(
            examples=available_examples,
            inputs=img_input,
            label="Examples — click to try",
            examples_per_page=6,
        )

    with gr.Row():
        img_input.render()
        label_output.render()

    with gr.Row():
        gr.ClearButton([img_input, label_output], value="Clear")
        submit_btn = gr.Button("Submit", variant="primary")

    submit_btn.click(fn=predict, inputs=img_input, outputs=label_output)

    gr.Markdown(
        "### Class Guide — What image to upload?\n\n"
        "| Class | What it looks like |\n"
        "|---|---|\n"
        "| 🟢 **Real Person** | A genuine selfie or live face photo |\n"
        "| 🖨️ **Printed Photo** | A printed photo held in front of the camera — including ID cards |\n"
        "| 🖥️ **Screen Replay** | A face displayed on a phone screen, monitor, or tablet |\n"
        "| 🎭 **Mask Attack** | A person wearing a printed face mask or physical face cover |\n"
        "| 🪆 **Mannequin** | A photo of a mannequin, doll, or face sculpture |\n"
        "| 🎨 **Unknown** | Other spoofing methods — e.g., painting, sketch, or digital illustration |\n"
    )

if __name__ == "__main__":
    demo.launch(share=False)