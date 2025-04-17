import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import sys
sys.path.append("lm_watermarking")
from lm_watermarking.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from typing import Optional

# Hyperparameters
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
TEXT_DETECTOR_THRESHOLD = 0.5
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_PROMPT = "스스로를 자랑해 봐"

# --- Model & Tokenizer Loading ---
# Load the model and tokenizer once when the application starts
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"  # Automatically uses CUDA if available
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Determine the device the model is loaded on
    device = model.device
    print(f"Model loaded successfully on device: {device}")

    # Initialize Watermark Processor after tokenizer is loaded
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=2.0,
        seeding_scheme="selfhash" # or other schemes as needed
    )
    print("Watermark processor initialized.")

    # Initialize Watermark Detector
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25, # Should match processor setting
        seeding_scheme="selfhash", # Should match processor setting
        device=device, # Use the same device as the model
        tokenizer=tokenizer,
        z_threshold=TEXT_DETECTOR_THRESHOLD,
        normalizers=[],
        ignore_repeated_ngrams=True
    )
    print("Watermark detector initialized.")

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Handle the error appropriately, maybe exit or raise
    model = None
    tokenizer = None
    device = "cpu" # Default to CPU if model loading fails
    watermark_processor = None # Initialize as None if loading failed
    print("Watermark processor could not be initialized.")
    watermark_detector = None # Initialize detector as None if loading failed
    print("Watermark detector could not be initialized.")

# --- API Definition ---
app = FastAPI(
    title="Watermark API",
    description="An API to generate watermark using the LLM and the image generation model.",
    version="1.0.0",
)

# --- Request & Response Models ---
class GenerationRequest(BaseModel):
    prompt: str = DEFAULT_PROMPT # Default prompt in Korean
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS # Default max_new_tokens
    watermark: bool = False # Add watermark flag, default to False

class GenerationResponse(BaseModel):
    response: str

class DetectionRequest(BaseModel):
    text: str # Text to detect watermark in
    threshold: float = TEXT_DETECTOR_THRESHOLD # Threshold for z-score based prediction

class DetectionResponse(BaseModel):
    # Based on typical output of WatermarkDetector.detect()
    num_tokens_scored: int
    num_green_tokens: int
    green_fraction: float
    z_score: float
    p_value: float
    prediction: bool # Prediction based on the *requested* threshold used during detection
    confidence: Optional[float] = None # Optional로 변경, prediction이 True일 때만 값 가짐

# --- API Endpoint ---
@app.post("/generate",
          response_model=GenerationResponse,
          summary="Generate text based on a prompt, optionally with watermark",
          description="Takes a user prompt and returns the model's generated response. Allows specifying max_new_tokens and enabling watermark.")
async def generate_text(request: GenerationRequest):
    """
    Generates text using the preloaded EXAONE model, optionally applying a watermark.

    - **prompt**: The input text prompt for the model.
    - **max_new_tokens**: The maximum number of new tokens to generate (default: 128).
    - **watermark**: Boolean flag to enable watermark generation (default: False).
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    # Check if watermark is requested but processor is not available
    if request.watermark and watermark_processor is None:
        raise HTTPException(status_code=503, detail="Watermark processor not loaded. Cannot generate watermarked text.")

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device) # Move input tensors to the correct device

        # Ensure model is on the correct device (if not already handled by device_map)
        model.to(device)

        # Prepare logits processor based on the watermark flag
        logits_processor_list = None
        if request.watermark:
            logits_processor_list = LogitsProcessorList([watermark_processor])

        generated_ids = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=request.max_new_tokens, # Use max_new_tokens from request
            logits_processor=logits_processor_list, # Apply watermark processor if requested
            do_sample=False, # Consider setting do_sample based on use case
        )

        # Extract only the generated tokens, excluding the input prompt
        # Adjust slicing based on potential padding or variations in `apply_chat_template` output
        # This assumes the input_ids are the prefix of generated_ids[0]
        output_ids = generated_ids[0][input_ids.shape[1]:]

        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        return GenerationResponse(response=response_text)
    except Exception as e:
        error_type = "watermarked generation" if request.watermark else "generation"
        print(f"Error during {error_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during text {error_type}: {e}")


# --- Watermark Detection Endpoint ---
@app.post("/detect_watermark",
          response_model=DetectionResponse,
          summary="Detect watermark in a given text using a specific threshold",
          description="Takes a text string and a mandatory threshold. Modifies the detector's shared threshold for prediction and returns detection scores. P-value is only returned if prediction is True. **Warning: Modifying shared state without a lock can lead to race conditions.**")
async def detect_watermark(request: DetectionRequest):
    """
    Detects the presence of a watermark in the input text using the requested threshold.
    Modifies the shared detector's z-score threshold for the 'prediction' calculation.
    Returns scores. p_value is only included if the prediction is True.

    **Warning**: Modifying the shared detector's threshold per request **without a lock** can lead to race conditions
                 under concurrent load, where one request might interfere with another's threshold setting.

    - **text**: The input text string to analyze.
    - **threshold**: (Required) The z-score threshold to use for determining the 'prediction' field.
    """
    if watermark_detector is None:
        raise HTTPException(status_code=503, detail="Watermark detector is not initialized. Service unavailable.")

    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    score_dict = None
    # 요청된 임계값으로 detector의 임계값 직접 변경
    watermark_detector.z_threshold = request.threshold

    # 탐지 수행 - prediction은 설정된 임계값을 사용하여 계산됨
    score_dict = watermark_detector.detect(request.text)

    # 응답 데이터 준비
    response_data = {
        "num_tokens_scored": score_dict.get("num_tokens_scored"),
        "num_green_tokens": score_dict.get("num_green_tokens"),
        "green_fraction": score_dict.get("green_fraction"),
        "z_score": score_dict.get("z_score"),
        "prediction": score_dict.get("prediction"),
        "p_value": score_dict.get("p_value"),
        "confidence": score_dict.get("confidence")
    }

    try:
        # Pydantic 모델을 사용하여 응답 반환 (유효성 검사 포함)
        return DetectionResponse(**response_data)
    except Exception as e: # Pydantic 유효성 검사 오류 등 처리
        print(f"Error preparing/validating detection response: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error processing detection results: {e}")

# --- Running the App (Optional) ---
if __name__ == "__main__":
    # Make sure to run with uvicorn for production: uvicorn main:app --reload
    # Example: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=7860)