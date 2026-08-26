"""Main dependency-aware controller.

This file wires the component modules together. The individual modules do not
reach into SpinePipeline through self.
"""

import time
import chromadb
import numpy as np
import pandas as pd
from .models import rapidocr_engine, model, embedding_model
from .image_loader import load_image
from .utils import normalize_text
from .detection import detect_spines, annotate_detections, crop_spines
from .preprocessing import preprocess
from .ocr import run_ocr
from .matching import match_books
from .google_books import fetch_from_google_books, add_to_local_catalog
from .recommendations import get_recommendations, recommend_from_shelf
from .tracking import build_crop_tracking_dataframe, build_timing_dataframe
from .visualization import display_results
from .config import (
    OCR_RESULT_SCORE_THRESHOLD,
    LOCAL_MATCH_SCORE_CUTOFF,
    GOOGLE_BOOKS_MATCH_SCORE_CUTOFF,
    CATALOG_DUPLICATE_SCORE_CUTOFF,
    RECOMMENDATION_TOP_K,
    RECOMMENDATION_QUERY_SCORE_CUTOFF,
    GOOGLE_BOOKS_TIMEOUT_SECONDS,
    GOOGLE_BOOKS_MAX_RETRIES,
    GOOGLE_BOOKS_MAX_RESULTS,
    MIN_DESCRIPTION_WORDS,
    DET_UNCLIP_RATIO
)

class SpinePipeline:

    def __init__(
        self,
        catalog_path="../Dataset/Books.csv",
        chroma_path="../chroma_db",
        collection_name="books",
        google_books_api_key=None,
        score_threshold=OCR_RESULT_SCORE_THRESHOLD,
        match_score_cutoff=LOCAL_MATCH_SCORE_CUTOFF,
        google_validation_cutoff=GOOGLE_BOOKS_MATCH_SCORE_CUTOFF,
        recommendation_top_k=RECOMMENDATION_TOP_K,
        recommendation_query_score_cutoff=RECOMMENDATION_QUERY_SCORE_CUTOFF,
        det_unclip_ratio=DET_UNCLIP_RATIO,
    ):
        self.model = model
        self.rapidocr_engine = rapidocr_engine
        self.embedding_model = embedding_model
        self.ocr_result_score_threshold = score_threshold
        self.local_match_score_cutoff = match_score_cutoff
        self.google_books_match_score_cutoff = google_validation_cutoff
        self.recommendation_top_k = recommendation_top_k
        self.recommendation_query_score_cutoff = (recommendation_query_score_cutoff)
        self.det_unclip_ratio = det_unclip_ratio
        self.min_description_words = MIN_DESCRIPTION_WORDS
        self.google_books_api_key = google_books_api_key
        self.catalog_path = catalog_path
        self.catalog_df = pd.read_csv(catalog_path)
        self.catalog_df["isbn13"] = (self.catalog_df["isbn13"].astype(str))
        self.catalog_df["match_key"] = (self.catalog_df["title"].fillna("") + " " + self.catalog_df["authors"].fillna(""))
        self.catalog_df["normalized_match_key"] = (self.catalog_df["match_key"].apply(normalize_text))
        self.choices = (self.catalog_df["normalized_match_key"].tolist())
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(collection_name)

    def load_image(self, image_path):
        try:
            return load_image(image_path)
        except ValueError as exc:
            print(f"{exc}")
            return None
    
    def detect_spines(self, image):
        return detect_spines(self.model, image)

    def annotate_detections(self, image, obb_corners):
        return annotate_detections(image, obb_corners)

    def crop_spines(self, image, obb_corners):
        return crop_spines(image, obb_corners)

    def preprocess(self, crop):
        return preprocess(crop)

    def run_ocr(self, crop_records):
        return run_ocr(crop_records, self.rapidocr_engine, self.preprocess, self.ocr_result_score_threshold,)

    def match_books(self, crop_records):
        return match_books(crop_records, self.catalog_df, self.choices, self.local_match_score_cutoff, normalize_text, )

    def fetch_from_google_books(self, query, timeout=GOOGLE_BOOKS_TIMEOUT_SECONDS, max_retries=GOOGLE_BOOKS_MAX_RETRIES, max_results=GOOGLE_BOOKS_MAX_RESULTS, ):

        return fetch_from_google_books(
            query,
            self.google_books_api_key,
            self.google_books_match_score_cutoff,
            self.min_description_words,
            normalize_text,
            timeout,
            max_retries,
            max_results,
        )

    def add_to_local_catalog(self, book):
        self.catalog_df, self.choices = add_to_local_catalog(
            book,
            self.catalog_df,
            self.choices,
            self.catalog_path,
            self.collection,
            self.embedding_model,
            normalize_text,
            CATALOG_DUPLICATE_SCORE_CUTOFF,
        )

    def get_recommendations(self, isbn13, top_k=None):
        k = ( top_k if top_k is not None else self.recommendation_top_k)

        return get_recommendations( isbn13, self.collection, k, )

    def recommend_from_shelf( self, matched_books_df, query, top_k=None, ):
        k = ( top_k if top_k is not None else self.recommendation_top_k )

        return recommend_from_shelf( matched_books_df, query, self.collection, self.embedding_model, self.recommendation_query_score_cutoff, k, )

    def results(self, image, query=None):
        total_start = time.perf_counter()

        recommendation_query = query

        if isinstance(image, (str, bytes)):
            image = self.load_image(image)

            if image is None:
                return None
            
        timing = {}

        step_start = time.perf_counter()
        detections = self.detect_spines(image)
        timing["YOLO Detection"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        yolo_output_image = self.annotate_detections(image, detections, )
        timing["YOLO Annotation"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        crops = self.crop_spines(image,detections, )
        timing["Spine Cropping"] = time.perf_counter() - step_start

        crop_records = [
            {
                "crop_idx": idx,
                "bbox": np.asarray(
                    detections[idx]
                ).tolist(),
                "crop": crop,
                "ocr_status": "not_attempted",
                "ocr_text": [],
                "ocr_scores": [],
                "ocr_query": "",
                "local_match_status": "not_attempted",
                "local_match_score": None,
                "local_match": None,
                "google_status": "not_attempted",
                "google_match_score": None,
                "google_match": None,
                "match_source": None,
                "final_status": "unmatched",
                "final_book": None,
            }
            for idx, crop in enumerate(crops)
        ]

        step_start = time.perf_counter()
        crop_records = self.run_ocr(crop_records)
        timing["RapidOCR"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        crop_records = self.match_books(crop_records)
        timing["Local Catalog Matching"] = (time.perf_counter() - step_start )

        google_start = time.perf_counter()

        for record in crop_records:
            if record["local_match_status"] == "matched":
                record["final_status"] = "matched"
                record["final_book"] = record["local_match"]
                record["google_status"] = "not_needed"
                record["google_books_time"] = 0.0
                continue

            ocr_query = record.get("ocr_query", "")

            if not ocr_query:
                record["google_status"] = "not_attempted"
                record["google_books_time"] = 0.0
                continue

            if not self.google_books_api_key:
                record["google_status"] = "no_api_key"
                record["google_books_time"] = 0.0
                continue

            book_start = time.perf_counter()
            book = self.fetch_from_google_books(ocr_query)
            record["google_books_time"] = (
                time.perf_counter() - book_start
            )

            if book is None:
                record["google_status"] = "unmatched"
                continue

            record["google_status"] = "matched"
            record["google_match"] = book
            record["google_match_score"] = book.get("Match Score")
            record["match_source"] = "Google Books"
            record["final_status"] = "matched"
            record["final_book"] = book

            self.add_to_local_catalog(book)

        timing["Google Books"] = time.perf_counter() - google_start

        recommendation_start = time.perf_counter()

        for record in crop_records:
            book = record.get("final_book")

            if book:
                book["Crop Index"] = record["crop_idx"]

                book_start = time.perf_counter()

                book["Recommendations"] = (
                    self.get_recommendations(
                        book.get("ISBN13")
                    )
                )

                record["book_recommendation_time"] = (
                    time.perf_counter() - book_start
                )
            else:
                record["book_recommendation_time"] = 0.0

            crop_start = record.get("_crop_start_time")
            record["crop_total_time"] = (
                record.get("ocr_time", 0.0)
                + record.get("local_match_time", 0.0)
                + record.get("google_books_time", 0.0)
                + record.get("book_recommendation_time", 0.0)
            )

        timing["Book-to-Book Recommendations"] = ( time.perf_counter() - recommendation_start )

        tracking_df = build_crop_tracking_dataframe( crop_records )

        matched_books = [
            record["final_book"]
            for record in crop_records
            if record.get("final_book")
        ]

        book_columns = [
            "Crop Index",
            "Title",
            "Author",
            "ISBN13",
            "Source",
            "Description",
            "Thumbnail",
            "Recommendations",
        ]

        if matched_books:
            books_df = pd.DataFrame(matched_books)

            for col in book_columns:
                if col not in books_df.columns:
                    books_df[col] = ""

            books_df = books_df[book_columns]

        else:
            books_df = pd.DataFrame( columns=book_columns )

        top_query_columns = ( book_columns + ["Query Score", "Query Rank"] )

        query_start = time.perf_counter()

        if ( recommendation_query and not books_df.empty ):
            top_recommendations_df = self.recommend_from_shelf(
                books_df,
                recommendation_query,
                top_k=3,
            )
        else:
            top_recommendations_df = pd.DataFrame(
                columns=top_query_columns
            )

        timing["User Query Recommendations"] = ( time.perf_counter() - query_start )

        unmatched_records = [
            record
            for record in crop_records
            if record["final_status"] != "matched"
        ]

        unmatched_queries = [
            record["ocr_query"]
            for record in unmatched_records
            if record["ocr_query"]
        ]

        timing["Total Pipeline"] = (
            time.perf_counter() - total_start
        )

        timing_df = build_timing_dataframe(timing)

        return {
            "yolo_output_image": yolo_output_image,
            "crop_records": crop_records,
            "crop_tracking_df": tracking_df,
            "ocr_results": [
                {
                    "crop_idx": r["crop_idx"],
                    "text": r["ocr_text"],
                    "scores": r["ocr_scores"],
                }
                for r in crop_records
            ],
            "query_strings": [
                r["ocr_query"]
                for r in crop_records
                if r["ocr_query"]
            ],
            "recommendation_query": recommendation_query,
            "matched_books": books_df,
            "top_query_recommendations": top_recommendations_df,
            "unmatched_queries": unmatched_queries,
            "num_detected_spines": len(crop_records),
            "num_ocr_results": sum(
                r["ocr_status"] == "success"
                for r in crop_records
            ),
            "num_matched_books": len(matched_books),
            "num_unmatched_books": len(unmatched_records),
            "timing": timing,
            "timing_df": timing_df,
        }

    def display(self, output):
        if output is None:
            return None

        return display_results(output)