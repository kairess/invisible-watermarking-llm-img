from pydantic import BaseModel, Field
from typing import Optional, List

# Hyperparameters (can be moved to a config file later if needed)
DEFAULT_PROMPT = "스스로를 자랑해 봐"
DEFAULT_MAX_NEW_TOKENS = 128
TEXT_DETECTOR_THRESHOLD = 0.5
DEFAULT_WATERMARK_MESSAGE = "헬로월드"

# --- Text Request & Response Models ---
class TextGenerationRequest(BaseModel):
    prompt: str = Field(default=DEFAULT_PROMPT)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS)
    watermark: bool = False

class TextGenerationResponse(BaseModel):
    response: str

class TextDetectionRequest(BaseModel):
    text: str
    threshold: float = Field(default=TEXT_DETECTOR_THRESHOLD)

class TextDetectionResponse(BaseModel):
    # Based on typical output of WatermarkDetector.detect()
    num_tokens_scored: int
    num_green_tokens: int
    green_fraction: float
    z_score: float
    p_value: Optional[float] = None # p_value는 Optional이 더 적합할 수 있음 (예: prediction False일 때)
    prediction: bool # Prediction based on the *requested* threshold used during detection
    confidence: Optional[float] = None # Optional로 변경, prediction이 True일 때만 값 가짐

# --- Image Watermarking Models ---

class ImageEmbedRequest(BaseModel):
    """Request model for embedding image watermark."""
    image_base64: str = Field(..., description="Base64 encoded image string (with optional data URL prefix)")
    message: str = Field(default=DEFAULT_WATERMARK_MESSAGE, description="The string message to embed as a watermark.")
    mask_proportion: float = Field(0.1, ge=0.0, le=1.0, description="Maximum proportion of pixels for each watermark mask (0.0 to 1.0).")

class ImageEmbedResponse(BaseModel):
    """Response model for embedding image watermark."""
    image_url: str = Field(..., description="URL of the saved watermarked image file")
    difference_image_url: str = Field(..., description="URL of the saved difference visualization image file")

class ImageDetectRequest(BaseModel):
    """Request model for detecting image watermark."""
    image_base64: str = Field(..., description="Base64 encoded image string (with optional data URL prefix)")
    epsilon: float = Field(1.0, description="DBSCAN epsilon parameter for detection clustering")
    min_samples: int = Field(500, description="DBSCAN min_samples parameter for detection clustering")


class DetectedMessageInfo(BaseModel):
    message: str = Field(..., description="Detected watermark message chunk (string representation of binary)")

class ImageDetectResponse(BaseModel):
    num_chunks_found: int = Field(..., description="Number of distinct watermark message chunks found")
    detected_messages: List[DetectedMessageInfo] = Field(..., description="List of detected watermark message chunks (binary string format, sorted by position)")
    decoded_message: Optional[str] = Field(None, description="The fully decoded watermark message string.")
    predicted_position_url: Optional[str] = Field(None, description="URL of the saved cluster visualization image (colored by message chunk)") 