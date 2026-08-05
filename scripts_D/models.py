from ultralytics import YOLO
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer

#Initialize YOLO
model = YOLO("models/best_obb.pt")

# Initialize PaddleOCR
paddle_ocr = PaddleOCR(
    enable_mkldnn=False,
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True
)

# Initialize Sentence Tranformer
embedding_model = SentenceTransformer('all-mpnet-base-v2')