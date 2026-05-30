import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight is not None:
            weight = weight.to(logits.device)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


class DINOv3ConvNeXtClassifier(nn.Module):
    """
    DINOv3 ConvNeXt-Large backbone with a residual adapter and classifier head.

    Architecture:
        backbone -> GAP (mean over tokens) -> residual adapter -> classifier

    The adapter uses a skip connection: x + 0.5 * adapter(x)
    This allows fine-grained control over the backbone representations without
    catastrophically forgetting pretrained features.
    """

    HIDDEN_DIM = 1536

    def __init__(self, backbone, num_classes: int, id2label=None, label2id=None, class_weights=None):
        super().__init__()
        self.backbone = backbone

        self.adapter = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.loss_fn = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=0.05)
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, pixel_values: torch.Tensor, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        x = outputs.last_hidden_state.mean(dim=1)
        x = x + 0.5 * self.adapter(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
