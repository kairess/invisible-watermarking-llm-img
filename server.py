import torch
from transformers import LogitsProcessorList # Removed AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
import uvicorn

# Import from new modules
from model_loader import load_models_and_processors
from schemas import (
    TextGenerationRequest,
    TextGenerationResponse,
    TextDetectionRequest,
    TextDetectionResponse,
    DEFAULT_PROMPT, # Import defaults if needed here
    DEFAULT_MAX_NEW_TOKENS,
    TEXT_DETECTOR_THRESHOLD
)
# Removed direct import of WatermarkProcessor, WatermarkDetector, BaseModel, Optional, sys, lm_watermarking


# Hyperparameters (keep main constants or move to config)
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# --- Load Models and Processors ---
# Call the loading function from model_loader
model, tokenizer, watermark_processor, watermark_detector, device = load_models_and_processors(
    MODEL_NAME, TEXT_DETECTOR_THRESHOLD
)

# Check if loading was successful
if model is None:
    print("Failed to load model. Exiting or running in degraded mode.")
    # Depending on requirements, you might want to exit or prevent the API from starting fully.
    # For now, endpoints will raise 503 if model is None.

# --- API Definition ---
app = FastAPI(
    title="Multimodal Watermark API",
    description="An API to generate and detect watermarks in text, images and potentially other modalities (e.g., audio, video).",
    version="1.0.0",
)

# --- Request & Response Models ---
# Definitions are now moved to schemas.py

# --- API Endpoint ---
@app.post("/generate_text",
          response_model=TextGenerationResponse,
          summary="Generate text based on a prompt, optionally with watermark",
          description="Takes a user prompt and returns the model's generated text response. Allows specifying max_new_tokens and enabling watermark.")
async def generate_text_endpoint(request: TextGenerationRequest):
    """
    Generates text using the preloaded EXAONE model, optionally applying a watermark.

    - **prompt**: The input text prompt for the model. Defaults from schemas.
    - **max_new_tokens**: The maximum number of new tokens to generate. Defaults from schemas.
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
        output_ids = generated_ids[0][input_ids.shape[1]:]

        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        return TextGenerationResponse(response=response_text)
    except Exception as e:
        error_type = "watermarked text generation" if request.watermark else "text generation"
        print(f"Error during {error_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during {error_type}: {e}")


# --- Watermark Detection Endpoint ---
@app.post("/detect_text_watermark",
          response_model=TextDetectionResponse,
          summary="Detect watermark in a given text using a specific threshold",
          description="Takes a text string and a mandatory threshold. Modifies the detector's shared threshold for prediction and returns text watermark detection scores. P-value is only returned if prediction is True. **Warning: Modifying shared state without a lock can lead to race conditions.**")
async def detect_text_watermark_endpoint(request: TextDetectionRequest):
    """
    Detects the presence of a watermark in the input text using the requested threshold.
    Modifies the shared detector's z-score threshold for the 'prediction' calculation.
    Returns text watermark scores. p_value is only included if the prediction is True.

    **Warning**: Modifying the shared detector's threshold per request **without a lock** can lead to race conditions
                 under concurrent load, where one request might interfere with another's threshold setting.

    - **text**: The input text string to analyze.
    - **threshold**: (Required) The z-score threshold to use for determining the 'prediction' field. Defaults from schemas.
    """
    if watermark_detector is None:
        raise HTTPException(status_code=503, detail="Watermark detector is not initialized. Service unavailable.")

    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    original_threshold = watermark_detector.z_threshold # 원래 임계값 저장 (선택 사항)
    watermark_detector.z_threshold = request.threshold

    score_dict = None
    try:
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
        # Pydantic 모델을 사용하여 응답 반환 (유효성 검사 포함)
        return TextDetectionResponse(**response_data)

    except Exception as e: # 탐지 또는 응답 준비 중 오류 처리
        print(f"Error during text detection or response preparation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during text detection: {e}")
    finally:
        # 원래 임계값으로 복원 (선택 사항, finally 블록에서 실행 보장)
        # 하지만 동시 요청 환경에서는 여전히 문제가 발생할 수 있습니다.
        watermark_detector.z_threshold = original_threshold


# --- Running the App (Optional) ---
if __name__ == "__main__":
    # Make sure to run with uvicorn for production: uvicorn main:app --reload
    # Example: uvicorn server:app --host 0.0.0.0 --port 7860 (파일 이름이 server.py라고 가정)
    uvicorn.run(app, host="0.0.0.0", port=7860)