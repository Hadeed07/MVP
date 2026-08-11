"""RapidOCR processing with explicit preprocessing and threshold dependencies."""

import time
from .utils import normalize_text, remove_publisher_numbers


def run_ocr(
    crop_records,
    rapidocr_engine,
    preprocess_fn,
    ocr_result_score_threshold,
):
    """Run RapidOCR while preserving crop index and OCR metadata."""
    results = []

    for record in crop_records:
        start = time.perf_counter()

        crop = record["crop"]
        record["ocr_status"] = "failed"
        record["ocr_text"] = []
        record["ocr_scores"] = []
        record["ocr_query"] = ""

        if crop is None:
            record["ocr_time"] = time.perf_counter() - start
            results.append(record)
            continue

        image = preprocess_fn(crop)
        prediction = rapidocr_engine(image)

        if prediction.txts:
            for text, score in zip(
                prediction.txts,
                prediction.scores,
            ):
                if score >= ocr_result_score_threshold:
                    cleaned = normalize_text(text)

                    if cleaned:
                        record["ocr_text"].append(cleaned)
                        record["ocr_scores"].append(float(score))

        if record["ocr_text"]:
            record["ocr_status"] = "success"

            ocr_query = normalize_text(
                " ".join(record["ocr_text"])
            )

            record["ocr_query"] = remove_publisher_numbers(
                ocr_query
            )

        record["ocr_time"] = time.perf_counter() - start
        results.append(record)

    return results
