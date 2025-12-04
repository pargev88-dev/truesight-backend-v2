# app/detectors/video.py

from typing import List, Literal, Tuple, Optional
import base64
import io

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from retinaface import RetinaFace
import cv2

from app.detectors.model_loader import get_video_model


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
    """
    Handles both full data URLs and raw base64 strings.
    """
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
# FACE DETECTION (RETINAFACE)
# -----------------------------
def _extract_face(img: Image.Image) -> Optional[Image.Image]:
    """
    Use RetinaFace to detect faces and return the largest face crop.
    If no face is found, return None.
    """
    # Convert PIL -> numpy BGR (as expected by many CV libs)
    rgb = np.array(img)  # H, W, 3 (RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        # detect_faces returns a dict of face_id -> info
        detections = RetinaFace.detect_faces(bgr)
    except Exception:
        return None

    if not isinstance(detections, dict) or len(detections) == 0:
        return None

    # Pick largest face by bounding box area
    best_face = None
    best_area = 0

    for _, det in detections.items():
        facial_area = det.get("facial_area")
        if not facial_area or len(facial_area) != 4:
            continue
        x1, y1, x2, y2 = facial_area
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_face = (x1, y1, x2, y2)

    if best_face is None or best_area <= 0:
        return None

    x1, y1, x2, y2 = best_face

    # Clip to image bounds
    h, w, _ = bgr.shape
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    face_bgr = bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_img = Image.fromarray(face_rgb)
    return face_img


# -----------------------------
# AGGREGATION (ANY FRAME FAKE => FAKE)
# -----------------------------
def _aggregate_scores(scores: List[float]) -> Tuple[Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    Convert per-frame fake probabilities into a global verdict.

    Policy:
      - If ANY frame > 0.80 → FAKE
      - Else if ALL frames < 0.30 → REAL
      - Else → UNCLEAR

    confidence:
      - For FAKE  → max_fake * 100
      - For REAL  → (1 - max_fake) * 100
      - For UNCLEAR → mid-range mapping (~40–70)
    """
    if not scores:
        return "UNCLEAR", 0

    max_fake = max(scores)

    # FAKE: any strongly fake-looking frame
    if max_fake > 0.80:
        verdict: Literal["REAL", "FAKE", "UNCLEAR"] = "FAKE"
        confidence = int(max_fake * 100)
        return verdict, confidence

    # REAL: all frames look strongly non-fake
    if all(s < 0.30 for s in scores):
        verdict = "REAL"
        confidence = int((1.0 - max_fake) * 100)
        return verdict, confidence

    # UNCLEAR: in-between zone
    # Map e.g. max_fake 0.30–0.80 roughly to 40–70 confidence
    confidence = int(40 + (max_fake - 0.30) / 0.50 * 30)
    confidence = max(0, min(confidence, 100))

    return "UNCLEAR", confidence


# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------
def analyze_frames(
    frames_base64: List[str],
) -> Tuple[List[float], Literal["REAL", "FAKE", "UNCLEAR"], int]:
    """
    Main entry point used by the /api/v1/scan endpoint.

    Steps:
      1. Decode base64 frames to PIL images.
      2. Run RetinaFace to get face crops (if any).
      3. Preprocess faces (or full frame if no faces across all frames).
      4. Run through the PyTorch model.
      5. Aggregate per-frame fake probabilities into verdict + confidence.

    Face logic:
      - Prefer face crops; if at least one face is found across frames,
        ONLY those face crops are used.
      - If no faces are found in ANY frame, fall back to full-frame analysis.
    """
    if not frames_base64:
        return [], "UNCLEAR", 0

    model = get_video_model()
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    face_tensors: List[torch.Tensor] = []
    full_tensors: List[torch.Tensor] = []

    # 1) Decode and attempt face extraction
    for data_url in frames_base64:
        try:
            img = _decode_base64_image(data_url)

            # Try to get face crop
            face = _extract_face(img)
            if face is not None:
                face_tensors.append(_transform(face))
            else:
                # store full frame as fallback
                full_tensors.append(_transform(img))

        except Exception:
            continue  # skip this frame if anything fails

    # 2) Decide whether to use faces or full frames
    if face_tensors:
        batch_tensors = face_tensors
    elif full_tensors:
        batch_tensors = full_tensors
    else:
        # No usable frames at all
        return [], "UNCLEAR", 0

    batch = torch.stack(batch_tensors).to(device, non_blocking=True)

    # 3) Run model
    with torch.inference_mode():
        outputs = model(batch)          # expect shape [batch, 1]
        probs_fake = outputs.view(-1).detach().cpu().numpy().tolist()

    frame_scores = [float(p) for p in probs_fake]

    verdict, confidence = _aggregate_scores(frame_scores)
    return frame_scores, verdict, confidence
