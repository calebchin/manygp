from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 ViT encoder loaded via torch.hub.

    Satisfies the FrozenBackboneSNGPClassifier encoder contract:
      - self.output_dim: int  (384 for vits14, 768 for vitb14)
      - forward(x) -> (B, output_dim)  [CLS token embedding]

    All parameters are frozen at construction. If the input spatial size is
    not 224×224, forward() bicubic-upsamples before passing to the ViT — this
    lets unchanged 32×32 OOD loaders (SVHN, CIFAR-C) work transparently.
    """

    _EMBED_DIMS: dict[str, int] = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    _TARGET_SIZE = 224

    def __init__(self, model_name: str = "dinov2_vits14", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.vit = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

        self.output_dim: int = self._EMBED_DIMS.get(model_name, self.vit.embed_dim)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.vit.eval()
        if x.shape[-1] != self._TARGET_SIZE or x.shape[-2] != self._TARGET_SIZE:
            x = F.interpolate(
                x,
                size=(self._TARGET_SIZE, self._TARGET_SIZE),
                mode="bicubic",
                align_corners=False,
            )
        return self.vit(x)
