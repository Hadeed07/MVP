## use your Gemini API after prompts

"""
Gemini vision pipeline: one API call per bookshelf image -> structured JSON
------------------------------------------------------------------------------
Purpose: free-tier alternative to the Claude vision LLM pipeline, for
comparing outputs at zero cost before deciding what to use in production.

NOTE: this uses the new unified `google-genai` SDK. The old
`google.generativeai` package is fully deprecated and no longer maintained.

Requirements:
    pip uninstall google-generativeai -y
    pip install google-genai pillow

Get a free API key (no credit card required) at:
    https://aistudio.google.com/
Then set it as an environment variable:
    Windows (PowerShell):  $env:GEMINI_API_KEY = "your-key-here"
    Windows (CMD):         set GEMINI_API_KEY=your-key-here
    Linux/macOS:            export GEMINI_API_KEY="your-key-here"

Note on free tier: rate limits apply (a handful of requests per minute,
capped requests per day). Fine for testing a few images, not for
production-scale batch processing.
"""

import json
import os
from pathlib import Path

from google import genai
from google.genai import types

IMAGE_PATH = "urdu.jpg"
MODEL = "gemini-3.5-flash"   # current fast/free-tier-friendly model as of mid-2026
                              # check https://ai.google.dev/gemini-api/docs/models
                              # if this errors as unavailable by the time you run it

PROMPT = """You are a precise visual extraction system. You will be shown a photo \
of a bookshelf with books stacked or shelved, spines visible. Your job is to identify \
every distinct book spine visible in the image and extract any readable text from it.

Respond ONLY with a valid JSON array, and nothing else -- no preamble, no explanation, \
no markdown code fences. Each element in the array should be an object with these fields:

- "position": your best estimate of left-to-right or top-to-bottom order (integer, starting at 1)
- "text_raw": the exact text you can read on the spine, in its original script (Urdu, English, etc.) -- \
if partially obscured or unclear, transcribe what you can and note uncertainty in "confidence_note"
- "script": the script/language you detected ("urdu_nastaliq", "urdu_naskh", "english", "mixed", "unknown")
- "confidence_note": brief note on legibility (e.g. "fully clear", "partially obscured", "blurry", "guessed from partial characters")

If NO text is legible on a spine at all, still include it with "text_raw": null and an appropriate confidence_note.
If you cannot make out any book spines at all in the image, return an empty array: []
"""

client = genai.Client(api_key="your-api")


def call_gemini(image_path, model_name=MODEL):
    """Send the image to Gemini and return the parsed JSON response."""

    image_bytes = Path(image_path).read_bytes()
    suffix = Path(image_path).suffix.lower()
    mime_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ],
    )

    raw_text = response.text.strip()

    # Defensive parsing: strip markdown fences if the model adds them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("WARNING: could not parse model output as JSON.")
        print("Raw output was:\n", raw_text)
        raise e

    return parsed


def main():
    print(f"Sending {IMAGE_PATH} to {MODEL}...")
    results = call_gemini(IMAGE_PATH)

    print(f"\nExtracted {len(results)} book entries:\n")
    for entry in results:
        print(json.dumps(entry, ensure_ascii=False, indent=2))

    output_path = "gemini_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {output_path}")

    return results


if __name__ == "__main__":
    main()