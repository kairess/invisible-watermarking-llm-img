import torch
import base64
import io
import re
import math
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException


DEFAULT_MESSAGE_LENGTH = 32


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


class BinaryStringConverter:
    def __init__(self, chunk_size: int = DEFAULT_MESSAGE_LENGTH):
        self.chunk_size = chunk_size

    def string_to_binary(self, text: str):
        """
        문자열을 지정된 청크 크기의 이진수 텐서로 변환
        """
        byte_array = text.encode('utf-8')
        bit_string = ''.join([format(byte, '08b') for byte in byte_array])
        num_chunks = math.ceil(len(bit_string) / self.chunk_size)
        chunks = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = bit_string[start:end].ljust(self.chunk_size, '0')
            chunks.append([int(bit) for bit in chunk])
        # GPU 사용 시 device 설정 필요할 수 있음
        return torch.tensor(chunks, dtype=torch.int64)

    def binary_to_string(self, binary_tensor):
        """
        이진수 텐서를 문자열로 변환
        """
        full_bit_string = ''
        for chunk in binary_tensor:
            # .item() 호출은 개별 스칼라 텐서에만 가능하므로 수정
            chunk_bits = ''.join(str(bit) for bit in chunk.tolist()) # tolist() 사용
            full_bit_string += chunk_bits

        bytes_data = []
        # 마지막 패딩된 0을 제거하기 위해 실제 비트 길이 계산 시도 (선택적 개선)
        # 여기서는 간단하게 8비트씩 처리
        for i in range(0, len(full_bit_string), 8):
            byte_str = full_bit_string[i:i+8]
            # 8비트가 안되는 마지막 부분 처리 및 모든 비트가 0인 경우 무시
            if len(byte_str) == 8 and '1' in byte_str:
                try:
                    bytes_data.append(int(byte_str, 2))
                except ValueError: # 잘못된 비트 문자열 처리
                    print(f"Warning: Could not convert byte string '{byte_str}' to int.")
                    continue
            elif len(byte_str) < 8 and '1' in byte_str:
                 # 패딩된 부분일 수 있으므로 주의, 여기서는 일단 변환 시도
                 try:
                     bytes_data.append(int(byte_str.ljust(8, '0'), 2)) # 필요시 0으로 패딩 후 변환
                 except ValueError:
                     print(f"Warning: Could not convert padded byte string '{byte_str}' to int.")
                     continue


        try:
            # errors='ignore' 또는 'replace' 사용
            return bytes(bytes_data).decode('utf-8', errors='replace')
        except Exception as e: # 일반적인 예외 처리 추가
            print(f"Error decoding bytes to string: {e}")
            return '' # 디코딩 실패 시 빈 문자열 반환


# --- Helper Functions for Sorting ---
def get_mask_centroid(mask):
    """마스크의 중심점 좌표를 계산 (x 좌표만)"""
    H, W = mask.shape
    y, x = torch.where(mask > 0)
    if len(x) == 0:
        return 0 # 마스크가 비어있으면 0 반환
    return x.float().mean().item()

def sort_masks_by_position(masks):
    """마스크를 x좌표 기준으로 정렬"""
    # masks 텐서가 GPU에 있을 수 있으므로 cpu로 이동 후 계산 고려
    centroids = [get_mask_centroid(mask.squeeze().cpu()) for mask in masks]
    # argsort는 tensor에서 직접 수행
    sorted_indices = torch.tensor(centroids).argsort()
    sorted_masks = masks[sorted_indices]
    return sorted_masks, sorted_indices

def sort_centroids_by_position(centroids, positions):
    """centroids 딕셔너리를 감지된 클러스터의 x좌표 기준으로 정렬된 텐서로 반환"""
    if not centroids: # centroids가 비어있으면 빈 텐서 반환
        return torch.empty((0, DEFAULT_MESSAGE_LENGTH), dtype=torch.int64)

    cluster_centers = {}
    # positions 텐서가 GPU에 있을 수 있음
    positions_cpu = positions.cpu() # CPU로 이동
    H, W = positions_cpu.shape[-2:] # positions 형태가 [1, H, W] 또는 [H, W] 일 수 있음

    for cluster_id in centroids.keys():
        if cluster_id < 0: continue # 배경 레이블 (-1) 무시
        # positions_cpu에서 클러스터 위치 찾기
        y, x = torch.where(positions_cpu.squeeze() == cluster_id) # squeeze()로 차원 축소
        if len(x) > 0:
            cluster_centers[cluster_id] = x.float().mean().item()
        else:
             cluster_centers[cluster_id] = float('inf') # 해당 클러스터 ID의 픽셀이 없으면 맨 뒤로

    # cluster_id를 x 좌표 기준으로 정렬
    # cluster_centers에 없는 cluster_id 처리 필요 -> centroids.items()를 순회하며 확인
    sorted_clusters = sorted(
        [(k, v) for k, v in centroids.items() if k >= 0], # 배경 제외하고 실제 클러스터만 정렬
        key=lambda item: cluster_centers.get(item[0], float('inf')) # get으로 안전하게 접근
    )

    # 정렬된 순서대로 centroid 텐서만 추출하여 스택
    if not sorted_clusters:
        return torch.empty((0, DEFAULT_MESSAGE_LENGTH), dtype=torch.int64)

    # .to(device) 불필요, CPU에서 처리 후 반환
    return torch.stack([centroid for _, centroid in sorted_clusters])