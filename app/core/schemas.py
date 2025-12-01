# app/core/schemas.py
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel


class FrameScanRequest(BaseModel):
    url: Optional[str] = None
    mode: Literal["standard", "pro"] = "standard"
    metadata: Optional[Dict[str, Any]] = None
    frames: List[str]  # base64 data URLs from extension
    heuristicsInfo: Optional[Dict[str, Any]] = None


class FrameScanResponse(BaseModel):
    verdict: Literal["REAL", "FAKE", "UNCLEAR"]
    confidence: int
    frame_scores: List[float]
    explanation: str
    extra: Dict[str, Any] = {}
