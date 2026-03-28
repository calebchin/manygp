import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarResNetEncoder(nn.Module):
    """ResNet-style encoder adapted to 32x32 inputs."""

    def __init__(self, width: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, width, blocks=2, stride=1)
        self.layer2 = self._make_layer(width, width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(width * 2, width * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(width * 4, width * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8, embedding_dim)
        self.output_dim = embedding_dim

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
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
    """Encoder + projection head for supervised contrastive learning."""

    def __init__(
        self,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
        width: int = 64,
    ):
        super().__init__()
        self.encoder = CifarResNetEncoder(width=width, embedding_dim=embedding_dim)
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
        embedding_dim: int = 128,
        num_classes: int = 10,
        width: int = 64,
    ):
        super().__init__()
        self.encoder = CifarResNetEncoder(width=width, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))
