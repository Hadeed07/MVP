# pipeline_modules/image_loader.py

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from .config import SUPPORTED_IMAGE_EXTENSIONS 

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True

except ImportError:
    HEIF_AVAILABLE = False

def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return it as an OpenCV BGR image.

    Supported formats:
        JPG, JPEG, PNG, WEBP, BMP,
        TIFF, HEIC, HEIF, AVIF

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Image in OpenCV BGR format.
    """

    path = Path(image_path)

    # Check that the file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    # Check extension
    extension = path.suffix.lower()

    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_IMAGE_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported image format: {extension}"
        )
    
    # HEIC / HEIF
    if extension in {".heic", ".heif"}:

        if not HEIF_AVAILABLE:
            raise ImportError(
                "HEIC/HEIF support requires pillow-heif. "
                "Install it with: pip install pillow-heif"
            )

        pil_image = Image.open(path).convert("RGB")

        image = np.array(pil_image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # All other supported formats
    else:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not decode the image: {path}"
            )
    
    return image