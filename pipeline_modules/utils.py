"""Shared utilities.

This module contains BOTH text-cleaning functions from the working pipeline.
"""

import re
import numpy as np


def normalize_text(text: str) -> str:
    """
    Cleans OCR text for OCR output and catalog matching.

    Operations:
        - remove punctuation
        - split CamelCase
        - collapse multiple spaces
        - remove one-character tokens
        - convert to lowercase
    """
    text = str(text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)

    tokens = [
        token
        for token in text.strip().split()
        if len(token) > 1
    ]

    return " ".join(tokens).lower()


def remove_publisher_numbers(title: str | None) -> str:
    """
    Removes likely publisher/edition numbers from OCR queries.

    Numbers with >4 digits are always removed.
    Short titles keep numbers according to the working pipeline's rule.
    """
    if not title:
        return ""

    words = normalize_text(title).split()
    filtered = []

    for i, word in enumerate(words):
        if word.isdigit():
            if len(word) > 4:
                continue

            if len(words) < 3 or i <= 1:
                filtered.append(word)

            continue

        filtered.append(word)

    return " ".join(filtered)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
