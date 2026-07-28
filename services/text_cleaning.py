import re
import json

def clean_ocr(book: dict) -> str:
    """Clean and merge OCR text tokens for a single detected book spine."""
    words = []

    for text, score in zip(book['text'], book['scores']):
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Separate CamelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Trim
        text = text.strip()

        # Remove garbage tokens: drop any single-character token,
        # ASCII or not (a lone stray character is noise either way)
        filtered = [token for token in text.split() if len(token) > 1]

        if filtered:
            words.append(' '.join(filtered).lower())

    return ' '.join(words)