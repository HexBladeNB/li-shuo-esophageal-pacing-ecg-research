# -*- coding: utf-8 -*-
"""
Original-architecture ResNet model (matching Keras specs from 总体数据预测结果.md).
Key differences from our previous model.py:
  - BatchNorm (not GroupNorm)
  - Dropout after every activation
  - ReLU + he_normal init (not LeakyReLU)
  - No CBAM attention
  - Channel progression: 64 → 128 → 200 → 196 → 100 → 320
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class ResBlock(nn.Module):
    """Residual block matching the original Keras architecture."""

    def __init__(self, in_ch, out_ch, kernel_size=16, stride=1, dropout=0.2):
        super().__init__()
        # Main path: Conv → BN → ReLU → Dropout → Conv
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)

        # Shortcut path: MaxPool (if stride > 1) + 1x1 Conv
        self.shortcut = nn.Sequential()
        if stride > 1 or in_ch != out_ch:
            layers = []
            if stride > 1:
                layers.append(nn.MaxPool1d(kernel_size=stride, stride=stride))
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))
            self.shortcut = nn.Sequential(*layers)

        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        # Align lengths (padding can cause ±1 difference)
        min_len = min(out.shape[2], shortcut.shape[2])
        out = out[:, :, :min_len] + shortcut[:, :, :min_len]
        out = self.dropout2(self.relu(self.bn2(out)))
        return out


class OriginalResNet(nn.Module):
    """
    1D ResNet matching the original Keras model architecture.
    Channel progression: input → 64 → 128 → 200 → 196 → 100 → 320 → Dense(num_classes)
    """

    def __init__(self, in_channels=13, num_classes=4, dropout=0.2):
        super().__init__()

        # Initial conv (kernel=16, matching 13312 params for 13 input channels)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16, stride=1, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Residual blocks (matching original Keras layer progression)
        self.block1 = ResBlock(64, 128, kernel_size=16, stride=4, dropout=dropout)
        self.block2 = ResBlock(128, 200, kernel_size=16, stride=5, dropout=dropout)
        self.block3 = ResBlock(200, 196, kernel_size=16, stride=1, dropout=dropout)
        self.block4 = ResBlock(196, 100, kernel_size=16, stride=1, dropout=dropout)
        self.block5 = ResBlock(100, 320, kernel_size=16, stride=8, dropout=dropout)

        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = None  # Lazy init based on actual tensor size
        self.num_classes = num_classes

        # He normal initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.flatten(x)

        # Lazy init classifier on first forward pass
        if self.classifier is None:
            self.classifier = nn.Linear(x.shape[1], self.num_classes).to(x.device)
            init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        x = self.classifier(x)
        return x
