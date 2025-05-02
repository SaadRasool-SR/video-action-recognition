"""
models/r2plus1d_net.py

Loads the R(2+1)D‑18 architecture from torchvision,
swaps in a custom classification head, and (optionally) freezes the
back‑bone.  Works with inputs of shape [B, C, T, H, W].
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DNet(nn.Module):
    """
    Wrapper around torchvision.models.video.r2plus1d_18.

    Args
    ----
    num_classes : int
        How many output classes (e.g. 101 for UCF‑101).
    pretrained  : bool | str | None
        • True / "kinetics" / "DEFAULT"  →  Kinetics‑400 weights
        • False / None                   →  random init
    freeze      : bool
        If True, keeps the backbone frozen and trains only the new
        classification layer.  Useful for fast fine‑tuning.
    """

    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool | str | None = True,
        freeze: bool = True,
    ):
        super().__init__()

        # Weights
        if pretrained in (True, "kinetics", "DEFAULT"):
            weights = R2Plus1D_18_Weights.DEFAULT
        elif pretrained:
            raise ValueError(f"Unknown pretrained option: {pretrained}")
        else:
            weights = None

        # Backbone
        self.backbone = r2plus1d_18(weights=weights, progress=True)

        # Replace classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Freeze everything except the new head
        if freeze:
            for name, param in self.backbone.named_parameters():
                # leave the new classification layer trainable
                param.requires_grad = name.startswith("fc")

    # Forward expects [B, C, T, H, W] just like torchvision’s implementation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
