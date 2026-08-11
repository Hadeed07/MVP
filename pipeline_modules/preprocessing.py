"""Default OCR preprocessing used by the working pipeline."""

import cv2
import numpy as np
from .config import (
    PREPROCESS_TARGET_HEIGHT,
    PREPROCESS_TARGET_WIDTH,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    SHARPEN_SIGMA_X,
    SHARPEN_WEIGHT,
    SHARPEN_BLUR_WEIGHT,
)


def preprocess(crop):
    """Apply grayscale + mild CLAHE + sharpening + resize/padding."""
    target_height = PREPROCESS_TARGET_HEIGHT
    target_width = PREPROCESS_TARGET_WIDTH

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )

    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=SHARPEN_SIGMA_X,
    )

    gray = cv2.addWeighted(
        gray,
        SHARPEN_WEIGHT,
        blurred,
        SHARPEN_BLUR_WEIGHT,
        0,
    )

    h, w = gray.shape

    scale = min(
        target_width / w,
        target_height / h,
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC
    )

    resized = cv2.resize(
        gray,
        (new_w, new_h),
        interpolation=interpolation,
    )

    canvas = np.full(
        (target_height, target_width),
        255,
        dtype=np.uint8,
    )

    x = (target_width - new_w) // 2
    y = (target_height - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR,
    )
