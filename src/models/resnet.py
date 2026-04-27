import torch
import torch.nn as nn
import torch.nn.functional as F

from .sngp import SpectralConv2d, SpectralLinear


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


class SpectralBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, coeff: float = 0.95):
        super().__init__()
        self.conv1 = SpectralConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            coeff=coeff,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SpectralConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            coeff=coeff,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                SpectralConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    coeff=coeff,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SpectralCifarResNetEncoder(nn.Module):
    """Spectrally normalized ResNet encoder adapted to 32x32 inputs."""

    def __init__(self, width: int = 64, embedding_dim: int = 128, coeff: float = 0.95):
        super().__init__()
        self.stem = nn.Sequential(
            SpectralConv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, width, blocks=2, stride=1, coeff=coeff)
        self.layer2 = self._make_layer(width, width * 2, blocks=2, stride=2, coeff=coeff)
        self.layer3 = self._make_layer(width * 2, width * 4, blocks=2, stride=2, coeff=coeff)
        self.layer4 = self._make_layer(width * 4, width * 8, blocks=2, stride=2, coeff=coeff)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpectralLinear(width * 8, embedding_dim, bias=True, coeff=coeff)
        self.output_dim = embedding_dim

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        coeff: float,
    ) -> nn.Sequential:
        layers = [SpectralBasicBlock(in_channels, out_channels, stride=stride, coeff=coeff)]
        for _ in range(1, blocks):
            layers.append(SpectralBasicBlock(out_channels, out_channels, stride=1, coeff=coeff))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SupConResNet(nn.Module):
    """SupCon model with an SNGP-style spectral-normalized CIFAR ResNet backbone."""

    def __init__(
        self,
        widen_factor: int = 10,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
        width: int = 64,
        spec_norm_bound: float = 0.95,
        use_projection_head: bool = True,
        spectral: bool = True,
    ):
        super().__init__()
        if spectral:
            self.encoder = SpectralCifarResNetEncoder(
                width=width,
                embedding_dim=embedding_dim,
                coeff=spec_norm_bound,
            )
        else:
            self.encoder = CifarResNetEncoder(widen_factor=widen_factor, embedding_dim=embedding_dim)
        self.encoder_dim = self.encoder.output_dim
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder_dim, projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_dim, projection_dim),
            )
        else:
            self.projector = nn.Identity()

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode_backbone(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_backbone(x)
        projections = self.projector(features)
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
