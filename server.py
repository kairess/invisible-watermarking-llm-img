import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
import uvicorn
import traceback
import os
import uuid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import from new modules
from model_loader import load_models_and_processors, load_image_watermark_model
from schemas import (
    TextGenerationRequest,
    TextGenerationResponse,
    TextDetectionRequest,
    TextDetectionResponse,
    ImageDetectResponse,
    DetectedMessageInfo,
    ImageEmbedRequest,
    ImageEmbedResponse, 
    ImageDetectRequest,
    DEFAULT_PROMPT,
    DEFAULT_MAX_NEW_TOKENS,
    TEXT_DETECTOR_THRESHOLD
)

# 헬퍼 함수 import 추가
from utils import load_image_from_base64, unnormalize_img

# watermark_anything 유틸리티 함수 import
from watermark_anything.notebooks.inference_utils import (
    default_transform,
    multiwm_dbscan,
    msg2str,
    create_random_mask,
)


# Hyperparameters
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
IMAGE_CHECKPOINT_DIR = "watermark_anything/checkpoints"

# --- Static File Configuration ---
STATIC_DIR = "static"
IMAGE_DIR = os.path.join(STATIC_DIR, "generated_images")
os.makedirs(IMAGE_DIR, exist_ok=True) # 이미지 저장 디렉토리 생성

# --- Load Models and Processors ---
text_model, tokenizer, watermark_processor, watermark_detector, device = load_models_and_processors(
    MODEL_NAME, TEXT_DETECTOR_THRESHOLD
)
image_watermark_model = load_image_watermark_model(checkpoint_dir=IMAGE_CHECKPOINT_DIR, device=device)

# --- API Definition ---
app = FastAPI(
    title="Multimodal Watermark API",
    description="An API to generate/detect watermarks. Embed endpoint saves image and returns URL.",
    version="1.1.0",
)

# Mount static directory AFTER FastAPI app initialization
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Text Generation Endpoint ---
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
    if text_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Text model not loaded. Service unavailable.")
    if request.watermark and watermark_processor is None:
        raise HTTPException(status_code=503, detail="Watermark processor not loaded. Cannot generate watermarked text.")

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        logits_processor_list = None
        if request.watermark:
            logits_processor_list = LogitsProcessorList([watermark_processor])

        generated_ids = text_model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=request.max_new_tokens,
            logits_processor=logits_processor_list,
            do_sample=False,
        )
        output_ids = generated_ids[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return TextGenerationResponse(response=response_text)
    except Exception as e:
        error_type = "watermarked text generation" if request.watermark else "text generation"
        print(f"Error during {error_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during {error_type}: {e}")


# --- Text Watermark Detection Endpoint ---
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

    original_threshold = watermark_detector.z_threshold
    watermark_detector.z_threshold = request.threshold
    score_dict = None
    try:
        score_dict = watermark_detector.detect(request.text)
        response_data = {
            "num_tokens_scored": score_dict.get("num_tokens_scored"),
            "num_green_tokens": score_dict.get("num_green_tokens"),
            "green_fraction": score_dict.get("green_fraction"),
            "z_score": score_dict.get("z_score"),
            "p_value": score_dict.get("p_value"),
            "prediction": score_dict.get("prediction"),
            "confidence": score_dict.get("confidence")
        }
        return TextDetectionResponse(**response_data)
    except Exception as e:
        print(f"Error during text detection or response preparation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during text detection: {e}")
    finally:
        watermark_detector.z_threshold = original_threshold

# --- Image Watermark Embedding Endpoint ---
@app.post("/embed_image",
          response_model=ImageEmbedResponse, # 응답 모델 확인
          summary="Embed watermarks, save image, and return URL",
          description="Takes Base64 image, messages, mask proportion. Embeds watermarks, saves the result on the server, and returns the URL to the saved image.")
async def embed_image_endpoint(embed_request: ImageEmbedRequest, http_request: Request): # embed_request 이름 변경, http_request 추가
    """
    Embeds watermarks into the image, saves it locally, and returns its URL.

    - **embed_request**: Contains `image_base64`, `messages`, `mask_proportion`.
    - **http_request**: Used to construct the base URL for the response.
    """
    if image_watermark_model is None:
        raise HTTPException(status_code=503, detail="Image watermark model (WAM) not loaded.")
    if default_transform is None or create_random_mask is None or unnormalize_img is None:
         raise HTTPException(status_code=500, detail="Required image processing functions not available.")

    try:
        # 1. Load image from Base64 (utils 함수 사용)
        pil_image = load_image_from_base64(embed_request.image_base64)
        img_pt = default_transform(pil_image).unsqueeze(0).to(device)

        # 2. Prepare watermark messages
        try:
            wm_msgs = torch.tensor(embed_request.messages, dtype=torch.int64).to(device)
            num_messages = wm_msgs.shape[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid message format: {e}")

        # 3. Create random masks
        masks = create_random_mask(img_pt, num_masks=num_messages, mask_percentage=embed_request.mask_proportion).to(device)

        # 4. Embed watermarks iteratively
        multi_wm_img_tensor = img_pt.clone()
        for i in range(num_messages):
            wm_msg = wm_msgs[i].unsqueeze(0)
            mask = masks[i].unsqueeze(0) # Ensure mask is [1, 1, H, W]
            with torch.no_grad():
                 outputs = image_watermark_model.embed(img_pt, wm_msg)
            watermarked_segment = outputs['imgs_w']
            multi_wm_img_tensor = watermarked_segment * mask + multi_wm_img_tensor * (1 - mask)

        # 5. Postprocess the final watermarked tensor (utils 함수 사용)
        final_img_tensor_unnormalized = unnormalize_img(multi_wm_img_tensor.squeeze(0))
        result_image = Image.fromarray((final_img_tensor_unnormalized.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8'))

        # 6. Save the result image to the static directory
        filename = f"{uuid.uuid4()}.png" # 고유 파일명 생성
        save_path = os.path.join(IMAGE_DIR, filename)
        try:
            result_image.save(save_path, format="PNG")
        except Exception as e:
            print(f"Error saving image file: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error saving image: {e}")

        # 6.1. Calculate and save the difference image
        original_img_tensor_unnormalized = unnormalize_img(img_pt.squeeze(0)) # 원본 이미지 텐서
        original_np = (original_img_tensor_unnormalized.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        watermarked_np = (final_img_tensor_unnormalized.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')

        # 차이 계산 및 시각화 (inference_utils.py 참고)
        delta = watermarked_np.astype(np.float32) - original_np.astype(np.float32)
        delta_visual = np.clip(np.abs(10 * delta), 0, 255).astype('uint8') # 스케일링 및 클리핑

        # Matplotlib을 사용하여 흑백 이미지로 저장
        diff_filename = f"diff_{uuid.uuid4()}.png"
        diff_save_path = os.path.join(IMAGE_DIR, diff_filename)
        try:
            plt.imsave(diff_save_path, delta_visual.mean(axis=2), cmap='hot', format='png') # 흑백 'hot' colormap 사용
        except Exception as e:
            print(f"Error saving difference image file: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error saving difference image: {e}")

        # 7. Construct the full URLs for the saved images
        base_url = str(http_request.base_url)
        static_path = os.path.join("static", "generated_images", filename).replace("\\", "/")
        image_url = f"{base_url.rstrip('/')}/{static_path.lstrip('/')}"

        diff_static_path = os.path.join("static", "generated_images", diff_filename).replace("\\", "/")
        difference_image_url = f"{base_url.rstrip('/')}/{diff_static_path.lstrip('/')}"

        # 8. Return the URLs in the response
        return ImageEmbedResponse(image_url=image_url, difference_image_url=difference_image_url)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during image embedding: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during image embedding: {e}")


# --- Image Watermark Detection Endpoint ---
@app.post("/detect_image_watermark",
          response_model=ImageDetectResponse,
          summary="Detect watermark messages in an image provided as Base64",
          description="Takes a Base64 encoded image file and DBSCAN parameters, detects watermark messages using WAM, and returns the found messages.")
async def detect_image_watermark_endpoint(request: ImageDetectRequest):
    """
    Detects watermark messages embedded in the image provided as Base64.

    - **request**: Contains `image_base64`, `epsilon`, and `min_samples`.
    """
    if image_watermark_model is None:
        raise HTTPException(status_code=503, detail="Image watermark model (WAM) not loaded. Service unavailable.")
    if default_transform is None or multiwm_dbscan is None or msg2str is None:
         raise HTTPException(status_code=500, detail="Required image processing/detection functions not available.")

    try:
        # 1. Load image from Base64 string (utils 함수 사용)
        pil_image = load_image_from_base64(request.image_base64)
        img_pt = default_transform(pil_image).unsqueeze(0).to(device) # [1, 3, H, W]

        # 2. Detect watermark signals
        with torch.no_grad(): # 탐지 시에도 그래디언트 계산 비활성화
            preds = image_watermark_model.detect(img_pt)["preds"].to(device)

        # 3. Extract predictions
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]

        # 4. Use DBSCAN to find message centroids
        centroids, positions = multiwm_dbscan(
            bit_preds,
            mask_preds,
            epsilon=request.epsilon, # 요청에서 값 사용
            min_samples=request.min_samples # 요청에서 값 사용
        )

        # 5. Format results
        detected_messages_info = []
        if centroids:
             centroids_pt = torch.stack(list(centroids.values())).detach().cpu()
             for msg_tensor in centroids_pt:
                 msg_string = msg2str(msg_tensor)
                 detected_messages_info.append(DetectedMessageInfo(message=msg_string))

        return ImageDetectResponse(
            num_messages_found=len(detected_messages_info),
            detected_messages=detected_messages_info
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during image watermark detection: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during image watermark detection: {e}")


# --- Running the App (Optional) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)