"""Local catalog matching."""

import time
from rapidfuzz import process, fuzz
import pandas as pd
from typing import Any, Callable


def match_books(
    crop_records: list[dict[str, Any]],
    catalog_df: pd.DataFrame,
    choices: list[str],
    local_match_score_cutoff: float,
    normalize_text_fn: Callable[[str], str],
) -> list[dict[str, Any]]:

    """Match each OCR query against the normalized local catalog."""
    for record in crop_records:
        start = time.perf_counter()

        record["local_match_status"] = "not_attempted"
        record["local_match_score"] = None
        record["local_match"] = None
        record["match_source"] = None

        query = record.get("ocr_query", "")

        if not query:
            record["local_match_time"] = time.perf_counter() - start
            continue

        result = process.extractOne(
            query,
            choices,
            scorer=fuzz.token_set_ratio,
            score_cutoff=local_match_score_cutoff,
        )

        if result is None:
            record["local_match_status"] = "unmatched"
            record["local_match_time"] = time.perf_counter() - start
            continue

        _, score, idx = result
        row = catalog_df.iloc[idx]

        book = {
            "Title": row["title"],
            "Author": row["authors"],
            "ISBN13": row["isbn13"],
            "Description": row.get("description", ""),
            "Thumbnail": row.get("thumbnail", ""),
            "Source": "Local Catalog",
            "Match Score": float(score),
        }

        record["local_match_status"] = "matched"
        record["local_match_score"] = float(score)
        record["local_match"] = book
        record["match_source"] = "Local Catalog"
        record["local_match_time"] = time.perf_counter() - start

    return crop_records
