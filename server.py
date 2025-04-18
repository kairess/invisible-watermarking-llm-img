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
import random
import matplotlib.patches as mpatches
from torchvision.transforms.functional import InterpolationMode, resize as TF_resize

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
          summary="Detect watermark messages and visualize clusters with legend",
          description="Takes a Base64 encoded image, detects watermark messages, and returns the messages along with a URL to a visualization image where detected clusters are colored and include a legend mapping colors to messages.")
async def detect_image_watermark_endpoint(request: ImageDetectRequest, http_request: Request):
    """
    Detects watermark messages, saves a visualization of the detected message clusters
    colored by message including a legend, and returns the URL.

    - **request**: Contains `image_base64`, `epsilon`, and `min_samples`.
    - **http_request**: Used to construct the base URL for the response.
    """
    if image_watermark_model is None:
        raise HTTPException(status_code=503, detail="Image watermark model (WAM) not loaded. Service unavailable.")
    if default_transform is None or multiwm_dbscan is None or msg2str is None or plt is None:
         raise HTTPException(status_code=500, detail="Required image processing/detection/plotting functions not available.")

    cluster_visualization_url = None # URL 초기화
    try:
        # 1. Load image from Base64 string
        pil_image = load_image_from_base64(request.image_base64)
        img_pt = default_transform(pil_image).unsqueeze(0).to(device) # [1, 3, H, W]
        _, _, H, W = img_pt.shape # Get image dimensions

        # 2. Detect watermark signals
        with torch.no_grad():
            preds = image_watermark_model.detect(img_pt)["preds"].to(device)

        # 3. Extract predictions
        mask_preds = F.sigmoid(preds[:, 0, :, :]) # [1, H_pred, W_pred]
        bit_preds = preds[:, 1:, :, :] # [1, C-1, H_pred, W_pred]

        # 4. Use DBSCAN to find message centroids and pixel positions
        centroids, positions = multiwm_dbscan(
            bit_preds,
            mask_preds,
            epsilon=request.epsilon,
            min_samples=request.min_samples
        )

        # 4.1 Visualize and save cluster image with legend if centroids are found
        if centroids: # Check if centroids dictionary is not empty
            fig = None # Initialize fig to ensure it's closed in finally block
            try:
                # Ensure positions is a tensor we can work with
                if not isinstance(positions, torch.Tensor) or positions.dim() < 2:
                     print(f"Warning: 'positions' returned by multiwm_dbscan is not a valid tensor. Skipping visualization.")
                else:
                    # Resize positions tensor
                    positions_pred_h, positions_pred_w = positions.shape[-2:]
                    if positions_pred_h != H or positions_pred_w != W:
                        print(f"Resizing positions tensor from {positions.shape} to target size ({H}, {W})")
                        if positions.dim() == 2: positions = positions.unsqueeze(0)
                        if positions.dim() == 3: positions = positions.unsqueeze(0)
                        positions_resized = TF_resize(
                            positions.float(), size=[H, W], interpolation=InterpolationMode.NEAREST, antialias=None
                        ).long()
                    else:
                        positions_resized = positions.long()
                    positions_np = positions_resized.squeeze().cpu().numpy()

                    if positions_np.shape != (H, W):
                        print(f"Error: Resized positions_np shape {positions_np.shape} does not match target ({H}, {W}). Skipping visualization.")
                    else:
                        # Create an empty RGB image
                        cluster_image_np = np.zeros((H, W, 3), dtype=np.uint8)
                        legend_elements = [] # For storing legend handles and labels

                        # Generate distinct colors and prepare legend elements
                        sorted_centroids = dict(sorted(centroids.items())) # Use sorted for consistent legend order
                        colors = {}
                        for cluster_id, msg_tensor in sorted_centroids.items():
                            if cluster_id < 0: continue # Skip background label if present

                            # Generate random bright color
                            r = random.randint(100, 255)
                            g = random.randint(100, 255)
                            b = random.randint(100, 255)
                            color_rgb = (r, g, b)
                            colors[cluster_id] = color_rgb

                            # Color the pixels
                            cluster_mask = (positions_np == cluster_id)
                            cluster_image_np[cluster_mask] = color_rgb

                            # Prepare legend element (normalized color for matplotlib)
                            color_normalized = (r/255.0, g/255.0, b/255.0)
                            msg_string = msg2str(msg_tensor.detach().cpu()) # Get message string
                            legend_elements.append(mpatches.Patch(color=color_normalized, label=f"{msg_string}"))


                        # Create plot and save with legend using Matplotlib
                        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100) # Adjust figsize based on image dimensions if needed
                        ax.imshow(cluster_image_np)
                        ax.axis('off') # Turn off axis labels and ticks

                        # Add legend outside the plot area
                        if legend_elements:
                            ax.legend(handles=legend_elements, loc='upper right', fancybox=True, shadow=True) # Adjust ncol

                        cluster_filename = f"cluster_{uuid.uuid4()}.png"
                        cluster_save_path = os.path.join(IMAGE_DIR, cluster_filename)
                        # Save the figure, bbox_inches='tight' removes extra whitespace
                        plt.savefig(cluster_save_path, format="PNG", bbox_inches='tight', pad_inches=0.1)

                        # Construct URL
                        base_url = str(http_request.base_url)
                        cluster_static_path = os.path.join("static", "generated_images", cluster_filename).replace("\\", "/")
                        cluster_visualization_url = f"{base_url.rstrip('/')}/{cluster_static_path.lstrip('/')}"

            except Exception as e:
                print(f"Error saving cluster visualization image: {e}")
                traceback.print_exc()
                cluster_visualization_url = None
            finally:
                 if fig: # Close the figure to free memory
                      plt.close(fig)


        # 5. Format results
        detected_messages_info = []
        if centroids:
             sorted_centroids = dict(sorted(centroids.items())) # Ensure consistent order with legend
             centroids_pt = torch.stack([t for i, t in sorted_centroids.items() if i >= 0]).detach().cpu() # Filter background if needed
             for msg_tensor in centroids_pt:
                 msg_string = msg2str(msg_tensor)
                 detected_messages_info.append(DetectedMessageInfo(message=msg_string))

        return ImageDetectResponse(
            num_messages_found=len(detected_messages_info),
            detected_messages=detected_messages_info,
            predicted_position_url=cluster_visualization_url
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