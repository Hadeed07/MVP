from ultralytics import YOLO
from paddleocr import PaddleOCR
from config import YOLO_WEIGHTS


#Initialize YOLO
model = YOLO(YOLO_WEIGHTS)

# Initialize PaddleOCR
paddle_ocr = PaddleOCR(
    enable_mkldnn=False,
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True
)