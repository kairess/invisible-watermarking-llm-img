import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Model & Tokenizer Loading ---
# Load the model and tokenizer once when the application starts
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
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
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Handle the error appropriately, maybe exit or raise
    model = None
    tokenizer = None
    device = "cpu" # Default to CPU if model loading fails

# --- API Definition ---
app = FastAPI(
    title="Watermark API",
    description="An API to generate watermark using the LLM and the image generation model.",
    version="1.0.0",
)

# --- Request & Response Models ---
class GenerationRequest(BaseModel):
    prompt: str = "스스로를 자랑해 봐" # Default prompt in Korean as in the original code

class GenerationResponse(BaseModel):
    response: str

# --- API Endpoint ---
@app.post("/generate",
          response_model=GenerationResponse,
          summary="Generate text based on a prompt",
          description="Takes a user prompt and returns the model's generated response.")
async def generate_text(request: GenerationRequest):
    """
    Generates text using the preloaded EXAONE model.

    - **prompt**: The input text prompt for the model.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

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

        generated_ids = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            do_sample=False,
        )

        # Extract only the generated tokens, excluding the input prompt
        # Adjust slicing based on potential padding or variations in `apply_chat_template` output
        # This assumes the input_ids are the prefix of generated_ids[0]
        output_ids = generated_ids[0][input_ids.shape[1]:]

        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        return GenerationResponse(response=response_text)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during text generation: {e}")

# --- Running the App (Optional) ---
if __name__ == "__main__":
    # Make sure to run with uvicorn for production: uvicorn main:app --reload
    # Example: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=7860)