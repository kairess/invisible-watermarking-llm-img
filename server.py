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
    ImageEmbedRequest,
    ImageEmbedResponse,
    ImageDetectRequest,
    ImageDetectResponse as ImageDetectResponseSchema,
    DEFAULT_PROMPT,
    DEFAULT_MAX_NEW_TOKENS,
    TEXT_DETECTOR_THRESHOLD,
    DetectedMessageInfo,
)

# 헬퍼 함수 import 추가
from utils import load_image_from_base64, unnormalize_img, BinaryStringConverter, sort_masks_by_position, sort_centroids_by_position

# watermark_anything 유틸리티 함수 import
from watermark_anything.notebooks.inference_utils import (
    default_transform,
    multiwm_dbscan,
    msg2str, # 범례 표시에 계속 사용
    create_random_mask,
)

# --- Binary String Converter Instance ---
converter = BinaryStringConverter()

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
    version="1.2.0", # 버전 업데이트
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
          response_model=ImageEmbedResponse,
          summary="Embed string watermark, save image, and return URL",
          description="Takes Base64 image, a string message, mask proportion. Converts string to binary chunks, embeds them sequentially based on mask position, saves the result, and returns URLs.")
async def embed_image_endpoint(embed_request: ImageEmbedRequest, http_request: Request):
    """
    Embeds a string watermark into the image, saves it locally, and returns its URL.
    The string is converted to binary chunks, and masks are sorted by x-coordinate
    for sequential embedding.

    - **embed_request**: Contains `image_base64`, `message` (string), `mask_proportion`.
    - **http_request**: Used to construct the base URL for the response.
    """
    if image_watermark_model is None:
        raise HTTPException(status_code=503, detail="Image watermark model (WAM) not loaded.")
    if default_transform is None or create_random_mask is None or unnormalize_img is None:
         raise HTTPException(status_code=500, detail="Required image processing functions not available.")

    try:
        # 1. Load image from Base64
        pil_image = load_image_from_base64(embed_request.image_base64)
        img_pt = default_transform(pil_image).unsqueeze(0).to(device)

        # 2. Convert string message to binary tensor chunks
        try:
            # converter 사용
            wm_msgs = converter.string_to_binary(embed_request.message).to(device)
            if wm_msgs.shape[0] == 0: # 빈 문자열 등 변환 결과가 없을 경우
                 raise HTTPException(status_code=400, detail="Input message resulted in empty binary data.")
            num_messages = wm_msgs.shape[0] # 청크 수
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid message or error during conversion: {e}")

        # 3. Create random masks and sort them by x-coordinate
        # 마스크 생성 시 device 명시
        masks = create_random_mask(img_pt, num_masks=num_messages, mask_percentage=embed_request.mask_proportion).to(device)
        # 마스크 정렬 함수 사용
        sorted_masks, sorted_indices = sort_masks_by_position(masks)

        # 4. Embed watermarks iteratively using sorted masks
        multi_wm_img_tensor = img_pt.clone()
        # 정렬된 순서로 임베딩
        for i in range(num_messages):
            # wm_msgs도 정렬된 인덱스에 맞춰야 하는가? -> 아니오, 메시지 청크는 순서대로, 마스크만 정렬된 것을 사용
            wm_msg = wm_msgs[i].unsqueeze(0) # i번째 청크 사용
            mask = sorted_masks[i].unsqueeze(0) # i번째 정렬된 마스크 사용
            with torch.no_grad():
                 # 원본 이미지(img_pt)와 메시지 청크, 마스크를 사용해 임베딩
                 # 주의: embed 함수가 mask 인자를 직접 받는지 확인 필요.
                 # watermark-anything의 embed는 mask를 직접 받지 않음.
                 # 따라서, embed 결과와 원본 이미지를 마스크를 이용해 결합해야 함.
                 outputs = image_watermark_model.embed(img_pt, wm_msg) # 원본 이미지에 각 청크 임베딩 시도
            watermarked_segment = outputs['imgs_w']
            # 마스크를 이용해 해당 영역만 업데이트
            multi_wm_img_tensor = watermarked_segment * mask + multi_wm_img_tensor * (1 - mask)


        # 5. Postprocess the final watermarked tensor
        final_img_tensor_unnormalized = unnormalize_img(multi_wm_img_tensor.squeeze(0))
        result_image = Image.fromarray((final_img_tensor_unnormalized.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8'))

        # 6. Save the result image
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join(IMAGE_DIR, filename)
        try:
            result_image.save(save_path, format="PNG")
        except Exception as e:
            print(f"Error saving image file: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error saving image: {e}")

        # 6.1. Calculate and save the difference image
        original_img_tensor_unnormalized = unnormalize_img(img_pt.squeeze(0))
        original_np = (original_img_tensor_unnormalized.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        watermarked_np = (final_img_tensor_unnormalized.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        delta = watermarked_np.astype(np.float32) - original_np.astype(np.float32)
        delta_visual = np.clip(np.abs(10 * delta), 0, 255).astype('uint8')
        diff_filename = f"diff_{uuid.uuid4()}.png"
        diff_save_path = os.path.join(IMAGE_DIR, diff_filename)
        try:
            plt.imsave(diff_save_path, delta_visual.mean(axis=2), cmap='hot', format='png')
        except Exception as e:
            print(f"Error saving difference image file: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error saving difference image: {e}")

        # 7. Construct the full URLs
        base_url = str(http_request.base_url)
        static_path = os.path.join("static", "generated_images", filename).replace("\\", "/")
        image_url = f"{base_url.rstrip('/')}/{static_path.lstrip('/')}"
        diff_static_path = os.path.join("static", "generated_images", diff_filename).replace("\\", "/")
        difference_image_url = f"{base_url.rstrip('/')}/{diff_static_path.lstrip('/')}"

        # 8. Return the URLs
        return ImageEmbedResponse(image_url=image_url, difference_image_url=difference_image_url)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during image embedding: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during image embedding: {e}")


# --- Image Watermark Detection Endpoint ---
@app.post("/detect_image_watermark",
          response_model=ImageDetectResponseSchema,
          summary="Detect watermark, decode message string, visualize clusters",
          description="Takes Base64 image, detects watermark chunks, sorts them by position, decodes the full string message, and returns the message, individual chunks, and visualization URL.")
async def detect_image_watermark_endpoint(request: ImageDetectRequest, http_request: Request):
    """
    Detects watermark message chunks, sorts them by estimated position,
    decodes the full string, saves a visualization of the detected clusters,
    and returns the decoded string and URL.

    - **request**: Contains `image_base64`, `epsilon`, `min_samples`.
    - **http_request**: Used to construct the base URL for the response.
    """
    if image_watermark_model is None:
        raise HTTPException(status_code=503, detail="Image watermark model (WAM) not loaded. Service unavailable.")
    if default_transform is None or multiwm_dbscan is None or msg2str is None or plt is None:
         raise HTTPException(status_code=500, detail="Required image processing/detection/plotting functions not available.")

    cluster_visualization_url = None
    decoded_message = None
    num_chunks_found = 0
    detected_messages_list = []
    try:
        # 1. Load image
        pil_image = load_image_from_base64(request.image_base64)
        img_pt = default_transform(pil_image).unsqueeze(0).to(device)
        _, _, H, W = img_pt.shape

        # 2. Detect watermark signals
        with torch.no_grad():
            # 모델의 detect 결과에서 preds 가져오기
            preds = image_watermark_model.detect(img_pt)["preds"].to(device) # GPU로 이동 확인

        # 3. Extract predictions
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]

        # 4. Use DBSCAN to find message centroids and pixel positions
        # multiwm_dbscan은 centroids(dict)와 positions(tensor) 반환
        centroids, positions = multiwm_dbscan(
            bit_preds,
            mask_preds,
            epsilon=request.epsilon,
            min_samples=request.min_samples
        )
        # positions 텐서를 이후 처리를 위해 device 유지 또는 필요시 cpu() 호출

        # 5. Sort detected centroids by position and decode message
        sorted_centroids_tensor = None
        if centroids:
            # centroids 정렬 함수 사용 (positions 텐서는 device에 있을 수 있음)
            # sort_centroids_by_position은 내부에서 cpu 처리 후 tensor 반환
            sorted_centroids_tensor = sort_centroids_by_position(centroids, positions.squeeze(0)) # positions는 [1, H, W] -> [H, W]

            if sorted_centroids_tensor is not None and sorted_centroids_tensor.nelement() > 0: # Check if tensor is not empty
                num_chunks_found = sorted_centroids_tensor.shape[0]
                # BinaryStringConverter를 사용하여 정렬된 텐서 디코딩
                decoded_message = converter.binary_to_string(sorted_centroids_tensor)

                # --- detected_messages 리스트 생성 ---
                for i in range(num_chunks_found):
                    chunk_tensor = sorted_centroids_tensor[i]
                    # msg2str을 사용하여 이진 문자열 표현 생성
                    chunk_str = msg2str(chunk_tensor.cpu()) # CPU로 이동 후 변환
                    detected_messages_list.append(DetectedMessageInfo(message=chunk_str))
                # ------------------------------------

            else:
                 num_chunks_found = 0
                 decoded_message = "" # 빈 문자열 또는 None
                 detected_messages_list = [] # 빈 리스트
        else:
            num_chunks_found = 0
            decoded_message = "" # 빈 문자열 또는 None
            detected_messages_list = [] # 빈 리스트


        # 6. Visualize and save cluster image with legend if centroids are found
        if centroids: # Check if centroids dictionary is not empty
            fig = None
            try:
                # positions 텐서 유효성 검사 및 리사이즈
                positions_squeezed = positions.squeeze(0) # [H_pred, W_pred]
                if positions_squeezed.dim() != 2:
                     print(f"Warning: 'positions' tensor has unexpected shape {positions.shape}. Skipping visualization.")
                else:
                    positions_pred_h, positions_pred_w = positions_squeezed.shape
                    # 리사이즈 로직 (필요한 경우)
                    if positions_pred_h != H or positions_pred_w != W:
                        # 리사이즈 시에는 Float Tensor로 변환 후 수행
                        positions_resized = TF_resize(
                            positions_squeezed.unsqueeze(0).float(), # [1, H_pred, W_pred] 형태로 변환
                            size=[H, W],
                            interpolation=InterpolationMode.NEAREST,
                            antialias=None # NEAREST 사용 시 antialias=None 권장
                        ).long().squeeze(0) # 다시 [H, W] 형태로
                    else:
                        positions_resized = positions_squeezed.long()

                    positions_np = positions_resized.cpu().numpy()

                    if positions_np.shape != (H, W):
                        print(f"Error: Resized positions_np shape {positions_np.shape} does not match target ({H}, {W}). Skipping visualization.")
                    else:
                        # 클러스터 시각화 로직 (이전과 유사하게 진행)
                        cluster_image_np = np.zeros((H, W, 3), dtype=np.uint8)
                        legend_elements = []

                        # cluster_centers 계산 추가 (sort_centroids_by_position 내부 로직 참고)
                        cluster_centers = {}
                        positions_cpu = positions.cpu() # CPU로 이동
                        H_pos, W_pos = positions_cpu.shape[-2:] # positions 형태가 [1, H, W] 또는 [H, W] 일 수 있음

                        for cluster_id in centroids.keys():
                             if cluster_id < 0: continue # 배경 레이블 (-1) 무시
                             y, x = torch.where(positions_cpu.squeeze() == cluster_id) # squeeze()로 차원 축소
                             if len(x) > 0:
                                 cluster_centers[cluster_id] = x.float().mean().item()
                             else:
                                 cluster_centers[cluster_id] = float('inf')

                        # 범례 순서를 위해 정렬된 ID 사용 (이제 cluster_centers 사용 가능)
                        sorted_cluster_ids = sorted([k for k in centroids.keys() if k >= 0], key=lambda k: cluster_centers.get(k, float('inf')))

                        colors = {}
                        # 정렬된 ID 순서대로 색상 할당 및 범례 생성
                        for cluster_id in sorted_cluster_ids:
                            r, g, b = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
                            color_rgb = (r, g, b)
                            colors[cluster_id] = color_rgb

                            cluster_mask = (positions_np == cluster_id)
                            cluster_image_np[cluster_mask] = color_rgb

                            color_normalized = (r/255.0, g/255.0, b/255.0)
                            # 범례에는 msg2str 사용 (개별 청크 정보)
                            msg_string = msg2str(centroids[cluster_id].cpu()) # CPU로 이동 후 변환
                            legend_elements.append(mpatches.Patch(color=color_normalized, label=f"C{int(cluster_id)}: {msg_string}"))

                        # Matplotlib으로 저장
                        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
                        ax.imshow(cluster_image_np)
                        ax.axis('off')

                        if legend_elements:
                             # 범례 위치 조정 (겹치지 않도록)
                             ax.legend(handles=legend_elements, loc='upper right', fancybox=True, shadow=True)

                        cluster_filename = f"cluster_{uuid.uuid4()}.png"
                        cluster_save_path = os.path.join(IMAGE_DIR, cluster_filename)
                        plt.savefig(cluster_save_path, format="PNG", bbox_inches='tight', pad_inches=0.1)

                        # URL 생성
                        base_url = str(http_request.base_url)
                        cluster_static_path = os.path.join("static", "generated_images", cluster_filename).replace("\\", "/")
                        cluster_visualization_url = f"{base_url.rstrip('/')}/{cluster_static_path.lstrip('/')}"

            except Exception as e:
                print(f"Error saving cluster visualization image: {e}")
                traceback.print_exc()
                cluster_visualization_url = None # 에러 시 URL None
            finally:
                 if fig:
                      plt.close(fig)


        # 7. Format and return results
        return ImageDetectResponseSchema(
            num_chunks_found=num_chunks_found, # 감지된 청크 수
            detected_messages=detected_messages_list, # 생성된 리스트 반환
            decoded_message=decoded_message,   # 디코딩된 최종 문자열
            predicted_position_url=cluster_visualization_url # 시각화 URL
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