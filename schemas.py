from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List

# Hyperparameters (can be moved to a config file later if needed)
DEFAULT_PROMPT = "스스로를 자랑해 봐"
DEFAULT_MAX_NEW_TOKENS = 128
TEXT_DETECTOR_THRESHOLD = 0.5
DEFAULT_MESSAGE_LENGTH = 32 # 메시지 길이 상수 추가
# 기본 메시지 정의: [ [0]*32, [1]*32 ]
DEFAULT_MESSAGES = [[0] * DEFAULT_MESSAGE_LENGTH, [1] * DEFAULT_MESSAGE_LENGTH]

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
    messages: List[List[int]] = Field(default=DEFAULT_MESSAGES, description=f"List of watermark messages. Each message must be a list of {DEFAULT_MESSAGE_LENGTH} bits (0 or 1). Defaults to one all-zero and one all-one message.")
    mask_proportion: float = Field(0.1, ge=0.0, le=1.0, description="Maximum proportion of pixels for each watermark mask (0.0 to 1.0).")

    @validator('messages')
    def validate_messages(cls, v):
        if not v: # 빈 리스트 방지
            raise ValueError("Messages list cannot be empty.")
        for i, message in enumerate(v):
            if not isinstance(message, list):
                 raise ValueError(f"Message at index {i} must be a list.")
            if len(message) != DEFAULT_MESSAGE_LENGTH:
                raise ValueError(f"Message at index {i} must have length {DEFAULT_MESSAGE_LENGTH}, got {len(message)}")
            if not all(bit in [0, 1] for bit in message):
                raise ValueError(f"Message bits at index {i} must be 0 or 1.")
        return v

class ImageEmbedResponse(BaseModel):
    """Response model for embedding image watermark."""
    image_url: str = Field(..., description="URL of the saved watermarked image file")

class ImageDetectRequest(BaseModel):
    """Request model for detecting image watermark."""
    image_base64: str = Field(..., description="Base64 encoded image string (with optional data URL prefix)")
    epsilon: float = Field(1.0, description="DBSCAN epsilon parameter for detection clustering")
    min_samples: int = Field(500, description="DBSCAN min_samples parameter for detection clustering")


class DetectedMessageInfo(BaseModel):
    message: str = Field(..., description="Detected watermark message (string representation)")
    # 필요시 여기에 추가 정보 필드 정의 (예: cluster_id, confidence 등)

class ImageDetectResponse(BaseModel):
    num_messages_found: int = Field(..., description="Number of distinct watermark messages found")
    detected_messages: List[DetectedMessageInfo] = Field(..., description="List of detected watermark messages") 