import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG11Small(nn.Module):
    """VGG11-like model adapted for small images (e.g., 28x28 or 32x32)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, min_input_size: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.min_input_size = min_input_size

        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        layers = []
        curr_in = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(curr_in, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                curr_in = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.shape[-1] < self.min_input_size or x.shape[-2] < self.min_input_size:
            x = F.interpolate(x, size=(self.min_input_size, self.min_input_size), mode="bilinear", align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetSmall(nn.Module):
    """AlexNet-like model adapted for small images (e.g., 28x28 or 32x32)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
