## very bad results

"""
Baseline pipeline: YOLOv8 (spine detection) + Tesseract (OCR)
--------------------------------------------------------------
Purpose: quick baseline/comparison tool to see how far Tesseract's
Urdu OCR is from usable on Nastaliq book spines, before relying
on the vision-LLM MVP path or a future UTRNet pipeline.

Requirements (already installed in your venv):
    pip install ultralytics pytesseract opencv-python pillow numpy
Also requires the Tesseract binary installed system-wide, with
'urd' (and ideally 'Arabic' script data) in tessdata.
"""

import cv2
import pytesseract
from ultralytics import YOLO

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------

# If you're on Windows and tesseract.exe isn't on PATH, uncomment
# and set this to your actual install path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = "urdu.jpg"          # path to your test bookshelf photo
YOLO_MODEL = "yolov8n.pt"             # generic pretrained model (auto-downloads on first run)
                                       # NOTE: this is NOT trained on book spines specifically —
                                       # it's just for structural testing until you have a
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                                   # spine-specific model
OCR_LANG = "urd"                      # Urdu language pack; try "urd+ara" to also use Arabic script data
CONFIDENCE_THRESHOLD = 0.25            # lower threshold since generic model won't be tuned for spines

def detect_regions(image_path, model_path=YOLO_MODEL, conf=CONFIDENCE_THRESHOLD):
    """
    Run YOLOv8 on the image and return bounding boxes.
    Returns a list of (x1, y1, x2, y2, confidence) tuples.
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)

    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, confidence))
    return boxes


def crop_region(image, box):
    """Crop a region from the image given a bounding box."""
    x1, y1, x2, y2, _ = box
    return image[y1:y2, x1:x2]


def preprocess_for_ocr(crop):
    """
    Basic preprocessing to help Tesseract: grayscale + threshold.
    Real Nastaliq spines will likely need more (deskew, contrast
    enhancement, etc.) but this is a reasonable starting point.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def run_ocr(crop, lang=OCR_LANG):
    """Run Tesseract OCR on a cropped image region."""
    processed = preprocess_for_ocr(crop)
    text = pytesseract.image_to_string(processed, lang=lang)
    return text.strip()


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")

    print(f"Running YOLOv8 detection on {IMAGE_PATH}...")
    boxes = detect_regions(IMAGE_PATH)
    print(f"Detected {len(boxes)} regions.\n")

    results = []
    for i, box in enumerate(boxes):
        crop = crop_region(image, box)
        if crop.size == 0:
            continue

        text = run_ocr(crop)
        results.append({
            "region_id": i,
            "bbox": box[:4],
            "confidence": round(box[4], 3),
            "ocr_text": text,
        })

        print(f"Region {i} (conf={box[4]:.2f}): '{text}'")

    return results


if __name__ == "__main__":
    main()