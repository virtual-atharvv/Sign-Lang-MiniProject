from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class KeypointFrame(BaseModel):
    pose: Optional[List[dict]] = []
    leftHand: Optional[List[dict]] = []
    rightHand: Optional[List[dict]] = []

class PredictRequest(BaseModel):
    keypoints: List[KeypointFrame] = Field(..., description="Sequence of landmark frames from browser")
    mode: Optional[str] = Field("continuous", description="Inference mode: continuous or isolated")

class PredictResponse(BaseModel):
    glosses: List[str] = Field(..., description="Ordered list of recognized sign glosses")
    sentence: str = Field(..., description="Grammatically corrected spoken language output")
    confidence: float = Field(..., description="Average confidence score")
    top_k: Optional[List[Tuple[str, float]]] = None
    processing_time_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    recognizer_type: str
    neural_path_active: bool = False
    encoder_loaded: bool = False
    knn_loaded: bool = False
    class_count: int = 0
