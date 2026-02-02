import torch
import torch.nn as nn
import torch.nn.functional as F


class ModerateCNN(nn.Module):
    """A small/moderate CNN for small image classification (e.g., CIFAR-10, FashionMNIST)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        if input_size < 8:
            raise ValueError("input_size must be >= 8")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)

        feat_spatial = input_size // 8  # 3x MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * feat_spatial * feat_spatial, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
