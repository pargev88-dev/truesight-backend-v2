# app/detectors/model_loader.py
import os
from functools import lru_cache

import torch
import torch.nn as nn

from app.core.config import get_settings


class DummyDeepfakeModel(nn.Module):
    """
    Placeholder model.

    Later you can replace this with a real architecture and load real
    weights from settings.VIDEO_MODEL_PATH. For now it just outputs a
    “fake probability” between 0 and 1.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)          # [B, 16, 1, 1]
        x = x.view(x.size(0), -1) # [B, 16]
        x = self.fc(x)            # [B, 1]
        x = self.sigmoid(x)
        return x                  # fake probability in [0, 1]


def _load_model(device: torch.device) -> nn.Module:
    settings = get_settings()
    model_path = settings.VIDEO_MODEL_PATH

    if os.path.exists(model_path):
        # When you have a real model file, this will load it.
        model = torch.load(model_path, map_location=device)
    else:
        # For now, use the dummy model.
        model = DummyDeepfakeModel()

    model.to(device)
    model.eval()
    return model


@lru_cache()
def get_video_model() -> nn.Module:
    """
    Returns a cached instance of the video deepfake model on the right device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)
    # attach device info so we can use it in video.py
    model.device = device  # type: ignore[attr-defined]
    return model
