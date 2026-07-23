from ultralytics import YOLO
from paddleocr import PaddleOCR


#Initialize YOLO
model = YOLO("models/best.pt")

# Initialize PaddleOCR
paddle_ocr = PaddleOCR(
    enable_mkldnn=False,
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True
)