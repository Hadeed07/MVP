"""Per-crop progress and pipeline timing tables."""

import pandas as pd


def build_crop_tracking_dataframe(crop_records):
    columns = [
        "Crop Index",
        "BBox",
        "OCR Status",
        "OCR Text",
        "OCR Scores",
        "OCR Query",
        "Local Match Status",
        "Local Match Score",
        "Google Books Status",
        "Google Books Score",
        "Match Source",
        "Final Status",
        "Matched Title",
        "Matched Author",
        "ISBN13",
        "OCR Time (s)",
        "Local Match Time (s)",
        "Google Books Time (s)",
        "Book Recommendation Time (s)",
        "Crop Total Time (s)",
    ]

    rows = []

    for record in crop_records:
        book = record.get("final_book") or {}

        rows.append({
            "Crop Index": record["crop_idx"],
            "BBox": record["bbox"],
            "OCR Status": record["ocr_status"],
            "OCR Text": " | ".join(record["ocr_text"]),
            "OCR Scores": record["ocr_scores"],
            "OCR Query": record["ocr_query"],
            "Local Match Status": record["local_match_status"],
            "Local Match Score": record["local_match_score"],
            "Google Books Status": record["google_status"],
            "Google Books Score": record["google_match_score"],
            "Match Source": record["match_source"],
            "Final Status": record["final_status"],
            "Matched Title": book.get("Title", ""),
            "Matched Author": book.get("Author", ""),
            "ISBN13": book.get("ISBN13", ""),
            "OCR Time (s)": record.get("ocr_time"),
            "Local Match Time (s)": record.get("local_match_time"),
            "Google Books Time (s)": record.get("google_books_time"),
            "Book Recommendation Time (s)": record.get(
                "book_recommendation_time"
            ),
            "Crop Total Time (s)": record.get("crop_total_time"),
        })

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def build_timing_dataframe(timing):
    return pd.DataFrame(
        list(timing.items()),
        columns=["Step", "Time (s)"],
    )
