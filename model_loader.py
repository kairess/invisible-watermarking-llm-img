import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import sys
sys.path.append("lm_watermarking")
from lm_watermarking.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# --- Model & Tokenizer Loading Function ---
def load_models_and_processors(model_name: str, text_detector_threshold: float):
    """
    Loads the language model, tokenizer, watermark processor, and detector.

    Args:
        model_name (str): The name of the pre-trained model to load.
        text_detector_threshold (float): The z-score threshold for the watermark detector.

    Returns:
        tuple: Contains the loaded model, tokenizer, watermark processor, watermark detector, and device.
               Returns (None, None, None, None, "cpu") on failure.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"  # Automatically uses CUDA if available
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            z_threshold=text_detector_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
        print("Watermark detector initialized.")

        return model, tokenizer, watermark_processor, watermark_detector, device

    except Exception as e:
        print(f"Error loading model or related components: {e}")
        # Handle the error appropriately, maybe exit or raise
        return None, None, None, None, "cpu" # Default to CPU if model loading fails 