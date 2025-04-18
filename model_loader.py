import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import sys
sys.path.append("lm_watermarking")
from lm_watermarking.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
sys.path.append("watermark_anything")
from watermark_anything.notebooks.inference_utils import load_model_from_checkpoint


# --- Model & Tokenizer Loading Function ---
def load_models_and_processors(model_name: str, text_detector_threshold: float):
    """
    언어 모델, 토크나이저, 텍스트 워터마크 프로세서 및 탐지기를 로드합니다.

    Args:
        model_name (str): 로드할 사전 훈련된 모델의 이름.
        text_detector_threshold (float): 텍스트 워터마크 탐지기의 z-점수 임계값.

    Returns:
        tuple: 로드된 모델, 토크나이저, 워터마크 프로세서, 워터마크 탐지기, 디바이스를 포함합니다.
               실패 시 (None, None, None, None, "cpu")를 반환합니다.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"  # 사용 가능한 경우 자동으로 CUDA 사용
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 모델이 로드된 디바이스 결정
        device = model.device
        print(f"Text model loaded successfully on device: {device}")

        # 토크나이저 로드 후 워터마크 프로세서 초기화
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25,
            delta=2.0,
            seeding_scheme="selfhash" # 또는 필요에 따라 다른 스키마
        )
        print("Text watermark processor initialized.")

        # 워터마크 탐지기 초기화
        watermark_detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25, # 프로세서 설정과 일치해야 함
            seeding_scheme="selfhash", # 프로세서 설정과 일치해야 함
            device=device, # 모델과 동일한 디바이스 사용
            tokenizer=tokenizer,
            z_threshold=text_detector_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
        print("Text watermark detector initialized.")

        return model, tokenizer, watermark_processor, watermark_detector, device

    except Exception as e:
        print(f"Error loading text model or related components: {e}")
        # 오류를 적절하게 처리합니다. 예를 들어 종료하거나 예외를 발생시킵니다.
        return None, None, None, None, "cpu" # 모델 로딩 실패 시 CPU로 기본 설정


# --- 이미지 워터마크 모델 로딩 함수 ---
def load_image_watermark_model(checkpoint_dir: str = "checkpoints", device: torch.device = torch.device("cpu")):
    """
    지정된 체크포인트에서 이미지 워터마킹 모델(WAM)을 로드합니다.

    Args:
        checkpoint_dir (str): 체크포인트 파일(`wam_mit.pth`)과 설정 파일(`params.json`)이 있는 디렉토리 경로.
        device (torch.device): 모델을 로드할 디바이스 (예: torch.device("cuda")).

    Returns:
        torch.nn.Module or None: 로드된 이미지 워터마킹 모델 또는 실패 시 None.
    """
    if load_model_from_checkpoint is None:
        print("Error: 'load_model_from_checkpoint' function not available due to import error.")
        return None

    try:
        seed = 42 # 필요시 시드 설정
        torch.manual_seed(seed)

        json_path = os.path.join(checkpoint_dir, "params.json")
        ckpt_path = os.path.join(checkpoint_dir, 'wam_mit.pth')

        if not os.path.exists(json_path) or not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint files not found in {checkpoint_dir}. Looked for {json_path} and {ckpt_path}")
            return None

        # 체크포인트에서 모델 로드
        wam_model = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
        print(f"Image watermark model (WAM) loaded successfully on device: {device}")
        return wam_model

    except Exception as e:
        print(f"Error loading image watermark model: {e}")
        return None 