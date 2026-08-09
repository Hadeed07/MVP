import argparse
import json
import time
from pathlib import Path
import cv2
import pandas as pd

from scripts_D.test_pipeline_paddleocr import SpinePipeline as PaddleOCRPipeline
from scripts_D.test_pipeline_rapidocr import SpinePipeline as RapidOCRPipeline


def serialize_output(output):
    books = output.get("matched_books")
    return {
        "ocr_results": output.get("ocr_results", []),
        "query_strings": output.get("query_strings", []),
        "matched_books": [] if books is None or books.empty else books.to_dict(orient="records"),
        "unmatched_queries": output.get("unmatched_queries", []),
        "num_detected_spines": output.get("num_detected_spines", 0),
        "num_ocr_results": output.get("num_ocr_results", 0),
        "num_matched_books": output.get("num_matched_books", 0),
        "num_unmatched_books": output.get("num_unmatched_books", 0),
    }


def run_pipeline(pipeline, image):
    start = time.perf_counter()
    output = pipeline.results(image)
    runtime = time.perf_counter() - start
    result = serialize_output(output)
    result["runtime_sec"] = round(runtime, 4)
    return result


def compare_pipelines(image_dir, image_names, image_ext=".jpg",
                      catalog_path=r"..\MVP\Dataset\books_cleaned.csv",
                      chroma_path=r"..\MVP\chroma_db",
                      google_books_api_key=None):

    pipelines = [
        ("PaddleOCR", PaddleOCRPipeline(
            catalog_path=catalog_path,
            chroma_path=chroma_path,
            google_books_api_key=google_books_api_key)),
        ("RapidOCR", RapidOCRPipeline(
            catalog_path=catalog_path,
            chroma_path=chroma_path,
            google_books_api_key=google_books_api_key)),
    ]

    rows = []

    for image_name in image_names:
        image_path = Path(image_dir) / f"{image_name}{image_ext}"
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read: {image_path}")
            continue

        for name, pipeline in pipelines:
            print(f"\n[{image_name}] {name}")

            try:
                result = run_pipeline(pipeline, image)
                rows.append({
                    "image": image_name,
                    "pipeline": name,
                    **result,
                    "error": "",
                })

                print(f"  Runtime         : {result['runtime_sec']} sec")
                print(f"  Detected spines : {result['num_detected_spines']}")
                print(f"  OCR results     : {result['num_ocr_results']}")
                print(f"  OCR queries     : {result['query_strings']}")
                print(f"  Matched books   : {result['num_matched_books']}")
                print(f"  Unmatched books : {result['num_unmatched_books']}")

            except Exception as exc:
                print(f"  [ERROR] {exc}")
                rows.append({
                    "image": image_name,
                    "pipeline": name,
                    "runtime_sec": None,
                    "ocr_results": [],
                    "query_strings": [],
                    "matched_books": [],
                    "unmatched_queries": [],
                    "num_detected_spines": None,
                    "num_ocr_results": None,
                    "num_matched_books": None,
                    "num_unmatched_books": None,
                    "error": repr(exc),
                })

    return pd.DataFrame(rows)


def json_columns(df):
    df = df.copy()
    for col in ["ocr_results", "query_strings", "matched_books", "unmatched_queries"]:
        df[col] = df[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False, default=str)
        )
    return df


def build_summary(df):
    if df.empty:
        return pd.DataFrame()

    numeric = [
        "num_detected_spines",
        "num_ocr_results",
        "num_matched_books",
        "num_unmatched_books",
    ]

    runtime = df.groupby("pipeline")["runtime_sec"].agg(
        avg_runtime_sec="mean",
        total_runtime_sec="sum",
        min_runtime_sec="min",
        max_runtime_sec="max",
    )

    counts = df.groupby("pipeline")[numeric].agg(["mean", "sum"])
    counts.columns = [f"{a}_{b}" for a, b in counts.columns]

    return pd.concat([runtime, counts], axis=1).round(4)


def build_side_by_side(df):
    if df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        records.append({
            "image": row["image"],
            "pipeline": row["pipeline"],
            "runtime_sec": row["runtime_sec"],
            "ocr_results": json.dumps(row["ocr_results"], ensure_ascii=False, default=str),
            "ocr_queries": json.dumps(row["query_strings"], ensure_ascii=False, default=str),
            "matched_books": json.dumps(row["matched_books"], ensure_ascii=False, default=str),
            "unmatched_books": json.dumps(row["unmatched_queries"], ensure_ascii=False, default=str),
            "num_detected_spines": row["num_detected_spines"],
            "num_ocr_results": row["num_ocr_results"],
            "num_matched_books": row["num_matched_books"],
            "num_unmatched_books": row["num_unmatched_books"],
            "error": row["error"],
        })

    return pd.DataFrame(records)


def build_query_table(df):
    rows = []
    for _, row in df.iterrows():
        for i, query in enumerate(row["query_strings"]):
            rows.append({
                "image": row["image"],
                "pipeline": row["pipeline"],
                "spine_index": i,
                "ocr_query": query,
            })
    return pd.DataFrame(rows)


def build_unmatched_table(df):
    rows = []
    for _, row in df.iterrows():
        for i, query in enumerate(row["unmatched_queries"]):
            rows.append({
                "image": row["image"],
                "pipeline": row["pipeline"],
                "unmatched_index": i,
                "unmatched_query": query,
            })
    return pd.DataFrame(rows)


def build_matched_table(df):
    rows = []
    for _, row in df.iterrows():
        for i, book in enumerate(row["matched_books"]):
            recs = book.get("Recommendations", [])
            titles = [
                r.get("Title", "") for r in recs
                if isinstance(r, dict) and r.get("Title")
            ]
            rows.append({
                "image": row["image"],
                "pipeline": row["pipeline"],
                "book_index": i,
                "Title": book.get("Title", ""),
                "Author": book.get("Author", ""),
                "ISBN13": book.get("ISBN13", ""),
                "Source": book.get("Source", ""),
                "Recommendations": " | ".join(titles),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--ext", default=".jpg")
    parser.add_argument("--catalog", default=r"..\MVP\Dataset\books_cleaned.csv")
    parser.add_argument("--chroma", default=r"..\MVP\chroma_db")
    parser.add_argument("--google-books-api-key", default=None)
    args = parser.parse_args()

    images = [x.strip() for x in args.images.split(",") if x.strip()]

    df = compare_pipelines(
        args.image_dir,
        images,
        args.ext,
        args.catalog,
        args.chroma,
        args.google_books_api_key,
    )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(build_summary(df).to_string())
