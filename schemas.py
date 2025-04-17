from pydantic import BaseModel
from typing import Optional

# Hyperparameters (can be moved to a config file later if needed)
DEFAULT_PROMPT = "스스로를 자랑해 봐"
DEFAULT_MAX_NEW_TOKENS = 128
TEXT_DETECTOR_THRESHOLD = 0.5

# --- Request & Response Models ---
class TextGenerationRequest(BaseModel):
    prompt: str = DEFAULT_PROMPT # Default prompt in Korean
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS # Default max_new_tokens
    watermark: bool = False # Add watermark flag, default to False

class TextGenerationResponse(BaseModel):
    response: str

class TextDetectionRequest(BaseModel):
    text: str # Text to detect watermark in
    threshold: float = TEXT_DETECTOR_THRESHOLD # Threshold for z-score based prediction

class TextDetectionResponse(BaseModel):
    # Based on typical output of WatermarkDetector.detect()
    num_tokens_scored: int
    num_green_tokens: int
    green_fraction: float
    z_score: float
    p_value: float
    prediction: bool # Prediction based on the *requested* threshold used during detection
    confidence: Optional[float] = None # Optional로 변경, prediction이 True일 때만 값 가짐 