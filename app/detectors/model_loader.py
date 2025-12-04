# app/detectors/model_loader.py

import os
from functools import lru_cache

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from app.core.config import get_settings


class EfficientNetDeepfake(nn.Module):
    """
    EfficientNet-B0 backbone (from torchvision) for binary deepfake detection.

    - Expects RGB images normalized like ImageNet, size ~224x224.
    - Outputs a single-column tensor [B, 1] with fake probability in [0, 1].

    Your checkpoint deepfake_model.pt was trained on torchvision's
    efficientnet_b0, which uses keys like:
        - features.*
        - classifier.1.weight (shape [2, 1280])
    """

    def __init__(self):
        super().__init__()

        # Create EfficientNet-B0 with default architecture
        backbone = efficientnet_b0(weights=None)  # no ImageNet weights

        # Replace classifier to have 2 outputs (e.g., [FAKE, REAL])
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 2)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] → logits [B, 2] → probs [B, 1] (fake)
        """
        logits = self.backbone(x)         # [B, 2]
        probs = torch.softmax(logits, 1)  # [B, 2]

        # The DeepfakeDetector repo uses class 0 = FAKE, class 1 = REAL
        fake_prob = probs[:, 0].unsqueeze(1)  # [B, 1]
        return fake_prob


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


def _load_model(device: torch.device) -> nn.Module:
    """
    Try to load EfficientNet weights from VIDEO_MODEL_PATH.
    If anything fails, fall back to DummyDeepfakeModel.
    """
    settings = get_settings()
    model_path = settings.VIDEO_MODEL_PATH

    print(f"[TrueSight] 🔍 VIDEO_MODEL_PATH = {model_path}")

    # Start with EfficientNet-based detector
    try:
        model: nn.Module = EfficientNetDeepfake()
    except Exception as e:
        print(f"[TrueSight] ❌ Could not initialize EfficientNetDeepfake: {e}")
        print("[TrueSight] ⚠️ Falling back to DummyDeepfakeModel().")
        model = DummyDeepfakeModel()
        model.to(device)
        model.eval()
        return model

    if os.path.exists(model_path):
        try:
            print(f"[TrueSight] 🧪 Attempting to load EfficientNet state_dict "
                  f"from {model_path}")
            ckpt = torch.load(model_path, map_location=device)

            # Handle Lightning-style checkpoints with a "state_dict" key
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            # Sometimes keys are prefixed like "model." or "net."
            # Strip any leading prefix before "features" or "classifier".
            cleaned_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith("features.") or k.startswith("classifier."):
                    cleaned_ckpt[k] = v
                elif ".features." in k or ".classifier." in k:
                    # e.g., "model.features.0.0.weight" -> "features.0.0.weight"
                    idx = k.find("features.")
                    if idx == -1:
                        idx = k.find("classifier.")
                    new_k = k[idx:]
                    cleaned_ckpt[new_k] = v
                else:
                    # Ignore other keys (optimizer, schedulers, etc.)
                    continue

            missing, unexpected = model.backbone.load_state_dict(
                cleaned_ckpt, strict=False
            )
            if missing:
                print(f"[TrueSight] ⚠️ Missing keys when loading: {missing}")
            if unexpected:
                print(f"[TrueSight] ⚠️ Unexpected keys when loading: {unexpected}")

            print("[TrueSight] ✅ Loaded torchvision EfficientNet-B0 deepfake model.")
        except Exception as e:
            print(f"[TrueSight] ❌ Failed to load state_dict from {model_path}: {e}")
            print("[TrueSight] ⚠️ Falling back to DummyDeepfakeModel().")
            model = DummyDeepfakeModel()
    else:
        print(f"[TrueSight] ❌ Model file not found at {model_path}; "
              "using DummyDeepfakeModel().")
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
