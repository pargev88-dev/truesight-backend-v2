# app/detectors/video.py

from typing import List, Literal, Tuple, Optional
import base64
import io

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from mtcnn import MTCNN

from app.detectors.model_loader import get_video_model

_detector = MTCNN()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _decode_base64_image(data_url: str) -> Image.Image:
    """Handles both full data URLs and raw base64 strings."""
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url

    b64data = "".join(b64data.split())
    missing_padding = len(b64data) % 4
    if missing_padding:
        b64data += "=" * (4 - missing_padding)

    img_bytes = base64.b64decode(b64data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# -----------------------------
# FACE DETECTION (MTCNN)
# -----------------------------
def _extract_face(img: Image.Image) -> Optional[Image.Image]:
    """
    Use MTCNN to detect faces and return the largest face crop.
    If no face is found, return None.
    """
    rgb = np.array(img)  # PIL -> numpy
    results = _detector.detect_faces(rgb)

    if not results:
        return None

    # Pick largest face
    best = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = best["box"]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = x1 + max(1, w)
    y2 = y1 + max(1, h)

    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return Image.fromarray(face)


# -----------------------------
# AGGREGATION (ANY FRAME FAKE => VIDEO FAKE)
# -----------------------------
def _aggregate_scores(scores: List[float]) -> Tuple[Literal["REAL", "FAKE", "UNCLEAR"], int]:
    if not scores:
        return "UNCLEAR", 0

    max_fake = max(scores)

    # FAKE if ANY frame looks strongly fake
    if max_fake > 0.80:
        return "FAKE", int(max_fake * 100)

    # REAL if ALL frames look non-fake
    if all(s < 0.30 for s in scores):
        return "REAL", int((1.0 - max_fake) * 100)

    # Otherwise UNCLEAR
    conf = int(40 + (max_fake - 0.30) / 0.50 * 30)
    conf = max(0, min(conf, 100))
    return "UNCLEAR", conf


# -----------------------------
# MAIN ANALYSIS LOGIC
# -----------------------------
def analyze_frames(frames_base64: List[str]) -> Tuple[List[float], Literal["REAL", "FAKE", "UNCLEAR"], int]:
    if not frames_base64:
        return [], "UNCLEAR", 0

    model = get_video_model()
    device = getattr(model, "device", torch.device("cpu"))

    face_tensors = []
    full_tensors = []

    # Try to extract faces from each frame
    for data_url in frames_base64:
        try:
            img = _decode_base64_image(data_url)

            face = _extract_face(img)
            if face is not None:
                face_tensors.append(_transform(face))
            else:
                full_tensors.append(_transform(img))

        except Exception:
            continue

    # Prefer faces if any detected
    if face_tensors:
        batch_tensors = face_tensors
    elif full_tensors:
        batch_tensors = full_tensors
    else:
        return [], "UNCLEAR", 0

    batch = torch.stack(batch_tensors).to(device)

    with torch.inference_mode():
        outputs = model(batch)      # [batch, 1]
        probs_fake = outputs.view(-1).cpu().numpy().tolist()

    frame_scores = [float(p) for p in probs_fake]

    verdict, confidence = _aggregate_scores(frame_scores)
    return frame_scores, verdict, confidence
