# Taking too long to give output and still got no output


"""
Moondream test: vision-language model for book spine text extraction
----------------------------------------------------------------------
Purpose: test Moondream's raw ability to read Urdu Nastaliq text from
a bookshelf photo, as a candidate for eventual local/self-hosted
deployment (per your MVP -> production roadmap).

This does NOT use YOLO cropping first -- it feeds the whole image (or
a manually cropped region) directly to Moondream and asks it to
describe/read what it sees. This tests Moondream's raw perception,
separate from any detection-stage questions.

Requirements:
    pip install transformers torch torchvision pillow einops

Note: first run will download the model weights (~a few GB),
requires internet access and disk space. No GPU required, but
inference will be slow on CPU -- expect it to take a while per image.
"""

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "vikhyatk/moondream2"   # official Moondream repo on Hugging Face
REVISION = "2024-08-26"             # pin a revision for reproducibility; check HF page for latest
IMAGE_PATH = "urdu.jpg"

# Prompts to try -- Moondream responds to natural language questions
PROMPTS = [
    "List every book title visible on the spines in this image, in the original script.",
    "Read and transcribe all the Urdu text visible in this image.",
    "Describe what text you can see on each book spine, left to right.",
]


def load_model():
    print("Loading Moondream model (first run will download weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=REVISION,
        torch_dtype=torch.float32,   # float32 for CPU; use torch.float16 if running on GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    return model, tokenizer


def run_moondream(image_path, model, tokenizer, prompts=PROMPTS):
    image = Image.open(image_path).convert("RGB")
    encoded_image = model.encode_image(image)

    results = {}
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        answer = model.answer_question(encoded_image, prompt, tokenizer)
        print(f"Response: {answer}")
        results[prompt] = answer

    return results


def main():
    model, tokenizer = load_model()
    results = run_moondream(IMAGE_PATH, model, tokenizer)
    return results


if __name__ == "__main__":
    main()