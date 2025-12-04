# app/detectors/model_loader.py

import os
from functools import lru_cache

import torch
import torch.nn as nn
from torchvision import models

from app.core.config import get_settings


class VideoDeepfakeNet(nn.Module):
    """
    ResNet18 backbone + binary head for deepfake detection.

    Later you can train this model (or a compatible one) and save weights to
    models/video/deepfake_model.pt using:

        torch.save(model.state_dict(), "models/video/deepfake_model.pt")

    This loader will then pick them up automatically.
    """
    def __init__(self):
        super().__init__()
        # Use ResNet18 backbone (no auto-download of ImageNet weights)
        backbone = models.resnet18(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()       # remove original classifier

        self.backbone = backbone
        self.head = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # [B, num_features]
        logits = self.head(feats)         # [B, 1]
        probs = self.sigmoid(logits)      # [B, 1] in [0, 1]
        return probs


class DummyDeepfakeModel(nn.Module):
    """
    Very small fallback model used when no real weights are available.
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


def _try_load_state_dict(model: nn.Module, model_path: str, device: torch.device) -> nn.Module:
    """
    Try to load a state_dict from model_path into the given model.
    If anything fails, the caller will decide how to fall back.
    """
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Common pattern: {"state_dict": ...}
        state_dict = ckpt["state_dict"]
        print("[TrueSight] ℹ️ Found 'state_dict' key in checkpoint.")
    elif isinstance(ckpt, dict):
        # Raw state_dict
        state_dict = ckpt
        print("[TrueSight] ℹ️ Checkpoint looks like a raw state_dict.")
    else:
        # File was saved as a full model; just return that
        print(f"[TrueSight] ℹ️ Checkpoint is a full model object: {type(ckpt)}")
        return ckpt.to(device)

    model.load_state_dict(state_dict)
    return model


def _load_model(device: torch.device) -> nn.Module:
    """
    Try to load a real model from VIDEO_MODEL_PATH.
    If the file is missing or invalid, fall back to a small dummy model.
    """
    settings = get_settings()
    model_path = settings.VIDEO_MODEL_PATH

    print(f"[TrueSight] 🔍 VIDEO_MODEL_PATH = {model_path}")

    # Default: ResNet-based detector (untrained until you add weights)
    model: nn.Module = VideoDeepfakeNet()

    if os.path.exists(model_path):
        try:
            print(f"[TrueSight] 🧪 Attempting to load model/weights from {model_path}")
            model = _try_load_state_dict(model, model_path, device)
            print(f"[TrueSight] ✅ Loaded video model of type {type(model)} from {model_path}")
        except Exception as e:
            # Placeholder / invalid .pt file → use dummy instead of crashing
            print(f"[TrueSight] ❌ Failed to load model from {model_path}: {e}")
            model = DummyDeepfakeModel()
            print("[TrueSight] ⚠️ Falling back to DummyDeepfakeModel().")
    else:
        # No file yet → dummy
        print(f"[TrueSight] ❌ Model file not found at {model_path}; using DummyDeepfakeModel().")
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
