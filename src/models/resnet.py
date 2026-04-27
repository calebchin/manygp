import torch
import torch.nn as nn
import torch.nn.functional as F


class WideResNetPreActBlock(nn.Module):
    """
    Pre-activation residual block for Wide ResNet.

    Order: BN -> ReLU -> Conv -> BN -> ReLU -> Conv, with shortcut on raw input.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class CifarResNetEncoder(nn.Module):
    """
    Wide ResNet-28-10 encoder for 32x32 inputs (CIFAR).

    Architecture (Liu et al., 2020):
      - Depth 28 = 6 * 4 + 4  (4 blocks per group, 3 groups)
      - Widen factor 10: channels [16*10, 32*10, 64*10] = [160, 320, 640]
      - Pre-activation residual blocks
      - Final projection: 640 -> embedding_dim
    """

    _N_BLOCKS = 4
    _BASE_CHANNELS = 16

    def __init__(self, widen_factor: int = 10, embedding_dim: int = 128):
        super().__init__()
        base = self._BASE_CHANNELS
        w = [base * widen_factor, 2 * base * widen_factor, 4 * base * widen_factor]

        self.stem = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = self._make_group(base,  w[0], self._N_BLOCKS, stride=1)
        self.group2 = self._make_group(w[0],  w[1], self._N_BLOCKS, stride=2)
        self.group3 = self._make_group(w[1],  w[2], self._N_BLOCKS, stride=2)
        self.bn_final = nn.BatchNorm2d(w[2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w[2], embedding_dim)
        self.output_dim = embedding_dim

    def _make_group(
        self, in_channels: int, out_channels: int, n_blocks: int, stride: int
    ) -> nn.Sequential:
        blocks = [WideResNetPreActBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(WideResNetPreActBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn_final(x), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SupConResNet(nn.Module):
    """Encoder + projection head for supervised contrastive learning."""

    def __init__(
        self,
        widen_factor: int = 10,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = CifarResNetEncoder(widen_factor=widen_factor, embedding_dim=embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        projections = self.projector(embeddings)
        return F.normalize(projections, dim=1)


class CifarResNetClassifier(nn.Module):
    """CIFAR ResNet encoder with a linear classification head."""

    def __init__(
        self,
        widen_factor: int = 10,
        embedding_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.encoder = CifarResNetEncoder(widen_factor=widen_factor, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))
