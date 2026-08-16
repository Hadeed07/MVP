from ultralytics import YOLO
from rapidocr import RapidOCR, EngineType, LangDet, LangRec, ModelType, OCRVersion
from sentence_transformers import SentenceTransformer

# Initialize YOLO
model = YOLO(r"..\models\best_obb.pt")

# Initialize RapidOCR
rapidocr_engine = RapidOCR(
    params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.EN,
        "Det.model_type": ModelType.SMALL,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.EN,
        "Rec.model_type": ModelType.SMALL,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
    }
)

# Initialize Sentence Tranformer
embedding_model = SentenceTransformer('all-mpnet-base-v2')