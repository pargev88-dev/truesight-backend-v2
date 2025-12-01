# app/api/v1/routes_scan.py
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.schemas import FrameScanRequest, FrameScanResponse
from app.core.config import get_settings
from app.core.heuristics import apply_heuristics
from app.detectors.video import analyze_frames
from app.services.explanation import generate_explanation

router = APIRouter(tags=["scan"])


@router.post("/scan", response_model=FrameScanResponse)
async def scan_frames(
    req: FrameScanRequest,
    settings = Depends(get_settings),
):
    if not req.frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No frames provided.",
        )

    # 1. Run video detector (dummy for now)
    frame_scores, base_verdict, base_confidence = analyze_frames(req.frames)

    # 2. Apply heuristics (placeholder)
    final_verdict, final_confidence, extra = apply_heuristics(
        base_verdict,
        base_confidence,
        metadata=req.metadata or {},
        heuristics_info=req.heuristicsInfo or {},
    )

    # 3. Generate explanation (simple template for now)
    explanation = generate_explanation(
        verdict=final_verdict,
        confidence=final_confidence,
        frame_scores=frame_scores,
        url=req.url,
        mode=req.mode,
    )

    return FrameScanResponse(
        verdict=final_verdict,
        confidence=final_confidence,
        frame_scores=frame_scores,
        explanation=explanation,
        extra=extra,
    )
