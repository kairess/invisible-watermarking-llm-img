import torch
import base64
import io
import re
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException

def unnormalize_img(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """이미지 텐서를 정규화 해제합니다."""
    unnormalized = img_tensor.clone()
    for t, m, s in zip(unnormalized, mean, std):
        t.mul_(s).add_(m)
    return unnormalized.clamp(0, 1)

def load_image_from_base64(base64_string: str) -> Image.Image:
    """Base64 문자열(선택적 데이터 URL 접두사 포함)을 PIL 이미지로 디코딩합니다."""
    try:
        # 데이터 URL 프리픽스 제거 (예: "data:image/png;base64,")
        base64_data = re.sub('^data:image/.+;base64,', '', base64_string)
        # Base64 디코딩
        image_data = base64.b64decode(base64_data)
        # BytesIO를 사용하여 이미지 열기
        image = Image.open(io.BytesIO(image_data))
        # 이미지 형식이 RGB가 아니면 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except (base64.binascii.Error, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"잘못된 Base64 문자열입니다: {e}")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Base64 문자열에 잘못된 이미지 데이터가 있습니다.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64에서 이미지를 로드하지 못했습니다: {e}") 