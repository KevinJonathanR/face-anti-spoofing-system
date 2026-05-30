import math
import random
from io import BytesIO

import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


class AddGaussianNoise:
    def __init__(self, std: float = 0.005, p: float = 0.03):
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return tensor + torch.randn_like(tensor) * self.std
        return tensor


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts from cameras or messaging apps."""

    def __init__(self, quality_range=(70, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality_range))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomLowResolution:
    """Downscale then upscale to simulate blur or low-quality captures."""

    def __init__(self, scale_range=(0.6, 0.9), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        nw, nh = int(w * scale), int(h * scale)
        return img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class RandomPixelation:
    """Simulate pixelation artifacts."""

    def __init__(self, scale_range=(0.3, 0.5), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        sw, sh = int(w * scale), int(h * scale)
        return img.resize((sw, sh), Image.NEAREST).resize((w, h), Image.NEAREST)


class MoirePattern:
    """Add a subtle sinusoidal pattern to simulate screen/print moire effect."""

    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        freq = random.uniform(10, 30)
        h = tensor.shape[-2]
        y = torch.arange(h, dtype=torch.float32) / h
        moire = (torch.sin(2 * math.pi * freq * y) * 0.04).view(-1, 1)
        return (tensor + moire).clamp(0, 1)


class RandomSharpening:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return ImageEnhance.Sharpness(img).enhance(random.uniform(1.5, 3.0))


def get_train_transforms(image_size: int, mean: list, std: list) -> Compose:
    return Compose([
        Resize((image_size, image_size)),
        RandomSharpening(p=0.3),
        RandomJPEGCompression(quality_range=(70, 95), p=0.5),
        RandomLowResolution(scale_range=(0.6, 0.9), p=0.5),
        RandomPixelation(scale_range=(0.3, 0.5), p=0.25),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        T.RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
        T.RandomGrayscale(p=0.02),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ToTensor(),
        AddGaussianNoise(std=0.005, p=0.03),
        MoirePattern(p=0.08),
        Normalize(mean=mean, std=std),
    ])


def get_valid_transforms(image_size: int, mean: list, std: list) -> Compose:
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
