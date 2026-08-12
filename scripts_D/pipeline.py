from scripts_D.model_rapidocr import rapidocr_engine, model, embedding_model
from IPython.display import display
from PIL import Image
import cv2
import chromadb
import hashlib
import numpy as np
import pandas as pd
import requests
import re
import time
from rapidfuzz import process, fuzz


def normalize_text(text):
    """
    Cleans OCR text for both OCR output and catalog matching.

    Operations:
        • remove punctuation
        • split CamelCase
        • collapse multiple spaces
        • remove one-character tokens
        • convert to lowercase
    """

    text = str(text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)

    tokens = [
        token
        for token in text.strip().split()
        if len(token) > 1
    ]

    return " ".join(tokens).lower()


def remove_publisher_numbers(title):
    """
    Removes likely publisher/edition numbers from OCR queries.
    Numbers with >4 digits are always removed; short titles keep numbers.
    """

    if not title:
        return ""

    words = normalize_text(title).split()
    filtered = []

    for i, word in enumerate(words):
        if word.isdigit():
            if len(word) > 4:
                continue

            if len(words) < 3 or i <= 1:
                filtered.append(word)

            continue

        filtered.append(word)

    return " ".join(filtered)


class SpinePipeline:

    def __init__(
        self,
        catalog_path=r"..\MVP\Dataset\Books.csv",
        chroma_path=r"..\MVP\chroma_db",
        collection_name="books",
        google_books_api_key=None,
    ):

        self.model = model
        self.rapidocr_engine = rapidocr_engine
        self.embedding_model = embedding_model

        # Minimum RapidOCR confidence required to keep an OCR segment.
        self.ocr_score_threshold = 0.80

        # Minimum RapidFuzz score required for a local catalog match.
        self.local_match_score_cutoff = 80

        # Minimum score required to accept a Google Books result.
        self.google_match_score_cutoff = 60

        # Number of similar books returned for each matched book.
        self.recommendation_top_k = 5

        # Minimum cosine-similarity score required for a shelf recommendation.
        # Books below this score are removed even if they are in the top K.
        self.recommendation_score_cutoff = 0.35

        self.yolo_confidence_threshold = 0.50
        self.yolo_iou_threshold = 0.40

        self.det_unclip_ratio = 1.2
        self.min_description_words = 0

        self.google_books_api_key = google_books_api_key

        self.catalog_path = catalog_path

        self.catalog_df = pd.read_csv(catalog_path)

        self.catalog_df["isbn13"] = (
            self.catalog_df["isbn13"]
            .astype(str)
        )

        self.catalog_df["match_key"] = (
            self.catalog_df["title"].fillna("")
            + " "
            + self.catalog_df["authors"].fillna("")
        )

        self.catalog_df["normalized_match_key"] = (
            self.catalog_df["match_key"]
            .apply(normalize_text)
        )

        self.choices = (
            self.catalog_df["normalized_match_key"]
            .tolist()
        )

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = self.chroma_client.get_collection(
            collection_name
        )

    @staticmethod
    def _elapsed(start_time):
        """Return elapsed wall-clock time in seconds."""
        return round(time.perf_counter() - start_time, 4)

    @staticmethod
    def order_points(pts):
        """
        Returns points in the order:
        top-left, top-right, bottom-right, bottom-left
        """

        pts = np.asarray(pts, dtype=np.float32)

        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def detect_spines(self, image):

        results = self.model.predict(
            image,
            conf=self.yolo_confidence_threshold,
            iou=self.yolo_iou_threshold,
            verbose=False
        )

        return results[0].obb.xyxyxyxy.cpu().numpy()

    def annotate_detections(self, image, obb_corners):
        """Draw each YOLO OBB with its crop index."""

        annotated = image.copy()

        for crop_idx, corners in enumerate(obb_corners):

            points = np.asarray(
                corners,
                dtype=np.int32
            ).reshape((-1, 1, 2))

            cv2.polylines(
                annotated,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            centroid = points.reshape(-1, 2).mean(axis=0)

            cx = int(centroid[0])
            cy = int(centroid[1])

            label = str(crop_idx)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (tw, th), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            text_x = cx - tw // 2
            text_y = cy + th // 2

            cv2.rectangle(
                annotated,
                (
                    text_x - 4,
                    text_y - th - baseline - 2
                ),
                (
                    text_x + tw + 4,
                    text_y + baseline + 2
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return annotated

    def crop_spines(self, image, obb_corners):

        crops = []

        for crop_idx, corners in enumerate(obb_corners):

            corners = self.order_points(corners)

            width = int(max(
                np.linalg.norm(corners[2] - corners[3]),
                np.linalg.norm(corners[1] - corners[0])
            ))

            height = int(max(
                np.linalg.norm(corners[1] - corners[2]),
                np.linalg.norm(corners[0] - corners[3])
            ))

            if width < 2 or height < 2:
                crops.append(None)
                continue

            destination = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(
                corners,
                destination
            )

            crop = cv2.warpPerspective(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(
                    crop,
                    cv2.ROTATE_90_CLOCKWISE
                )

            crops.append(crop)

        return crops

    def preprocess(self, crop):

        TARGET_HEIGHT = 300
        TARGET_WIDTH = 750

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        # Mild sharpening that performed best on slightly blurred spines.
        blurred = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=1.0
        )

        gray = cv2.addWeighted(
            gray,
            1.35,
            blurred,
            -0.35,
            0
        )

        h, w = gray.shape

        scale = min(
            TARGET_WIDTH / w,
            TARGET_HEIGHT / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        interpolation = (
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )

        resized = cv2.resize(
            gray,
            (new_w, new_h),
            interpolation=interpolation
        )

        canvas = np.full(
            (TARGET_HEIGHT, TARGET_WIDTH),
            255,
            dtype=np.uint8
        )

        x = (TARGET_WIDTH - new_w) // 2
        y = (TARGET_HEIGHT - new_h) // 2

        canvas[
            y:y + new_h,
            x:x + new_w
        ] = resized

        return cv2.cvtColor(
            canvas,
            cv2.COLOR_GRAY2BGR
        )

    def run_ocr(self, crop_records):
        """Run RapidOCR while preserving crop index and timing."""

        results = []

        for record in crop_records:

            crop_start = time.perf_counter()

            crop = record["crop"]

            record["ocr_status"] = "failed"
            record["ocr_text"] = []
            record["ocr_scores"] = []
            record["ocr_query"] = ""

            if crop is None:
                record["ocr_time"] = self._elapsed(crop_start)
                results.append(record)
                continue

            image = self.preprocess(crop)

            prediction = self.rapidocr_engine(image)

            if prediction.txts:

                for text, score in zip(
                    prediction.txts,
                    prediction.scores
                ):

                    if score >= self.ocr_score_threshold:

                        cleaned = normalize_text(text)

                        if cleaned:
                            record["ocr_text"].append(cleaned)
                            record["ocr_scores"].append(
                                float(score)
                            )

            if record["ocr_text"]:

                record["ocr_status"] = "success"

                ocr_query = normalize_text(
                    " ".join(record["ocr_text"])
                )

                record["ocr_query"] = remove_publisher_numbers(
                    ocr_query
                )

            record["ocr_time"] = self._elapsed(
                crop_start
            )

            results.append(record)

        return results

    def match_books(self, crop_records):
        """Match each OCR query while preserving crop timing."""

        for record in crop_records:

            match_start = time.perf_counter()

            record["local_match_status"] = "not_attempted"
            record["local_match_score"] = None
            record["local_match"] = None
            record["match_source"] = None

            ocr_query = record.get(
                "ocr_query",
                ""
            )

            if not ocr_query:

                record["local_match_time"] = self._elapsed(
                    match_start
                )
                continue

            result = process.extractOne(
                ocr_query,
                self.choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.local_match_score_cutoff
            )

            if result is None:

                record["local_match_status"] = "unmatched"
                record["local_match_time"] = self._elapsed(
                    match_start
                )
                continue

            _, score, idx = result

            row = self.catalog_df.iloc[idx]

            book = {
                "Title": row["title"],
                "Author": row["authors"],
                "ISBN13": row["isbn13"],
                "Description": row.get(
                    "description",
                    ""
                ),
                "Thumbnail": row.get(
                    "thumbnail",
                    ""
                ),
                "Source": "Local Catalog",
                "Match Score": float(score)
            }

            record["local_match_status"] = "matched"
            record["local_match_score"] = float(score)
            record["local_match"] = book
            record["match_source"] = "Local Catalog"

            record["local_match_time"] = self._elapsed(
                match_start
            )

        return crop_records

    def fetch_from_google_books(
        self,
        query,
        timeout=5,
        max_retries=3,
        max_results=5
    ):

        url = "https://www.googleapis.com/books/v1/volumes"

        params = {
            "q": query,
            "maxResults": max_results,
            "key": self.google_books_api_key
        }

        response = None

        for attempt in range(max_retries):

            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=timeout
                )

                response.raise_for_status()

                break

            except requests.exceptions.Timeout:

                print(f"Timeout : {query}")

            except requests.exceptions.HTTPError:

                if response is not None and response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504
                ):

                    if attempt < max_retries - 1:

                        retry_after = response.headers.get(
                            "Retry-After"
                        )

                        if retry_after is not None:

                            try:
                                delay = float(retry_after)

                            except ValueError:
                                delay = 1.5 * (2 ** attempt)

                        else:
                            delay = 1.5 * (2 ** attempt)

                        if response.status_code == 429:
                            print(
                                f"429 rate limited, "
                                f"retrying in {delay:.1f}s : {query}"
                            )

                        time.sleep(delay)
                        continue

                return None

            except requests.exceptions.RequestException:

                return None

        else:
            return None

        data = response.json()

        items = data.get(
            "items",
            []
        )

        if not items:
            return None

        for item in items:

            info = item.get(
                "volumeInfo",
                {}
            )

            title = info.get("title")

            authors = info.get(
                "authors",
                []
            )

            if not title:
                continue

            candidate = normalize_text(
                title + " " + " ".join(authors)
            )

            score = fuzz.token_set_ratio(
                query,
                candidate
            )

            if score < self.google_match_score_cutoff:
                continue

            description = info.get(
                "description"
            ) or ""

            if (
                description
                and len(description.split())
                < self.min_description_words
            ):
                continue

            isbn13 = None

            for identifier in info.get(
                "industryIdentifiers",
                []
            ):

                if identifier["type"] == "ISBN_13":

                    isbn13 = identifier["identifier"]
                    break

            return {
                "Title": title,
                "Author": ", ".join(authors),
                "ISBN13": isbn13,
                "Description": description,
                "Thumbnail": info.get(
                    "imageLinks",
                    {}
                ).get(
                    "thumbnail",
                    ""
                ),
                "Source": "Google Books",

                # The score used to validate this Google Books result.
                "Match Score": float(score)
            }

        return None

    def add_to_local_catalog(self, book):

        isbn13 = book["ISBN13"]

        if not isbn13:

            source = (
                f"{book['Title']}_{book['Author']}"
            )

            isbn13 = (
                "gb_"
                + hashlib.md5(
                    source.encode()
                ).hexdigest()[:12]
            )

            book["ISBN13"] = isbn13

        if isbn13 in self.catalog_df["isbn13"].values:
            return

        normalized = normalize_text(
            book["Title"]
            + " "
            + book["Author"]
        )

        duplicate = process.extractOne(
            normalized,
            self.choices,
            scorer=fuzz.token_set_ratio,
            score_cutoff=90
        )

        if duplicate:
            return

        # Falls back to title+author when there is no description.
        text_for_embedding = (
            book["Description"]
            if book["Description"]
            else (
                f"{book['Title']} "
                f"{book['Author']}"
            )
        )

        embedding = self.embedding_model.encode(
            [text_for_embedding]
        ).tolist()[0]

        self.collection.upsert(
            ids=[isbn13],
            embeddings=[embedding],
            documents=[book["Description"]],
            metadatas=[
                {
                    "title": book["Title"],
                    "authors": book["Author"],
                    "thumbnail": book["Thumbnail"],
                    "categories": "",
                    "average_rating": 0
                }
            ]
        )

        row = {
            column: ""
            for column in self.catalog_df.columns
            if column not in (
                "match_key",
                "normalized_match_key"
            )
        }

        row.update({
            "isbn13": isbn13,
            "title": book["Title"],
            "authors": book["Author"],
            "description": book["Description"],
            "thumbnail": book["Thumbnail"]
        })

        if "title_with_subtitle" in row:
            row["title_with_subtitle"] = book["Title"]

        if "tagged_description" in row:
            row["tagged_description"] = (
                isbn13
                + " "
                + book["Description"]
            )

        self.catalog_df = pd.concat(
            [
                self.catalog_df,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

        key = normalize_text(
            book["Title"]
            + " "
            + book["Author"]
        )

        self.choices.append(key)

        columns = [
            c
            for c in self.catalog_df.columns
            if c not in (
                "match_key",
                "normalized_match_key"
            )
        ]

        pd.DataFrame(
            [row]
        )[columns].to_csv(
            self.catalog_path,
            mode="a",
            index=False,
            header=False
        )

    def get_recommendations(
        self,
        isbn13,
        top_k=None,
        score_cutoff=None
    ):
        """
        Given the ISBN13 of a matched book, finds similar books
        using the book's stored Chroma embedding.

        The returned recommendations are filtered by distance
        only indirectly through the requested top K. This function
        keeps the original book-to-book recommendation behavior.
        """

        k = (
            top_k
            if top_k is not None
            else self.recommendation_top_k
        )

        if not isbn13:
            return []

        try:

            record = self.collection.get(
                ids=[isbn13],
                include=["embeddings"]
            )

        except Exception:

            return []

        embeddings = record.get(
            "embeddings"
        )

        if (
            embeddings is None
            or len(embeddings) == 0
        ):
            return []

        embedding = embeddings[0]

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k + 1
        )

        ids = result.get(
            "ids",
            [[]]
        )[0]

        distances = result.get(
            "distances",
            [[]]
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]]
        )[0]

        recommendations = []

        for rec_id, distance, metadata in zip(
            ids,
            distances,
            metadatas
        ):

            if rec_id == isbn13:
                continue

            recommendations.append({
                "ISBN13": rec_id,
                "Title": metadata.get(
                    "title",
                    ""
                ),
                "Author": metadata.get(
                    "authors",
                    ""
                ),
                "Categories": metadata.get(
                    "categories",
                    ""
                ),
                "Average Rating": metadata.get(
                    "average_rating",
                    ""
                ),
                "Thumbnail": metadata.get(
                    "thumbnail",
                    ""
                ),
                "Distance": distance
            })

            if len(recommendations) >= k:
                break

        return recommendations

    def recommend_from_shelf(
        self,
        matched_books_df,
        query,
        top_k=None,
        score_cutoff=None
    ):
        """
        Ranks matched shelf books against a user query using
        cosine similarity.

        Unlike a simple top-K operation, a book must also meet
        recommendation_score_cutoff to be returned.

        Therefore:
            • top_k controls the maximum number of recommendations.
            • score_cutoff controls the minimum relevance.
            • fewer than top_k books may be returned.
            • zero books may be returned if none reaches the cutoff.

        The query can be a single string or a list/tuple of
        survey answers.
        """

        if (
            matched_books_df is None
            or matched_books_df.empty
        ):
            return matched_books_df

        if isinstance(query, (list, tuple)):

            query = " ".join(
                str(q)
                for q in query
                if q
            )

        query = (query or "").strip()

        if not query:
            raise ValueError(
                "recommend_from_shelf requires a non-empty query"
            )

        k = (
            top_k
            if top_k is not None
            else self.recommendation_top_k
        )

        cutoff = (
            score_cutoff
            if score_cutoff is not None
            else self.recommendation_score_cutoff
        )

        isbn_list = (
            matched_books_df["ISBN13"]
            .astype(str)
            .tolist()
        )

        try:

            records = self.collection.get(
                ids=isbn_list,
                include=["embeddings"]
            )

        except Exception:

            records = {
                "ids": [],
                "embeddings": []
            }

        embedding_lookup = {
            isbn13: embedding
            for isbn13, embedding in zip(
                records.get("ids", []),
                records.get("embeddings", [])
            )
        }

        query_embedding = np.asarray(
            self.embedding_model.encode(
                [query]
            )[0]
        )

        query_norm = np.linalg.norm(
            query_embedding
        )

        scores = []

        for isbn13, description in zip(
            isbn_list,
            matched_books_df.get(
                "Description",
                [""] * len(isbn_list)
            )
        ):

            embedding = embedding_lookup.get(
                isbn13
            )

            if (
                not isinstance(description, str)
                or not description.strip()
            ):
                description = None

            if embedding is None and description:

                embedding = self.embedding_model.encode(
                    [description]
                )[0]

            if embedding is None:

                scores.append(-1.0)
                continue

            embedding = np.asarray(
                embedding
            )

            denom = (
                query_norm
                * np.linalg.norm(embedding)
            )

            similarity = (
                float(
                    np.dot(
                        query_embedding,
                        embedding
                    ) / denom
                )
                if denom
                else -1.0
            )

            scores.append(similarity)

        ranked_df = matched_books_df.copy()

        ranked_df["Query Score"] = scores

        # First rank all candidates by query similarity.
        ranked_df = ranked_df.sort_values(
            "Query Score",
            ascending=False
        ).reset_index(drop=True)

        # Keep only genuinely relevant candidates.
        ranked_df = ranked_df[
            ranked_df["Query Score"] >= cutoff
        ].head(k).reset_index(drop=True)

        ranked_df["Query Rank"] = (
            ranked_df.index + 1
        )

        return ranked_df

    def results(self, image, query=None):

        pipeline_start = time.perf_counter()

        # Preserve the user's recommendation query separately.
        recommendation_query = query

        # ---------------------------------
        # YOLO detection
        # ---------------------------------
        step_start = time.perf_counter()

        detections = self.detect_spines(
            image
        )

        detection_time = self._elapsed(step_start)

        # ---------------------------------
        # YOLO annotation
        # ---------------------------------
        step_start = time.perf_counter()

        yolo_output_image = self.annotate_detections(
            image,
            detections
        )

        annotation_time = self._elapsed(step_start)

        # ---------------------------------
        # Spine cropping
        # ---------------------------------
        step_start = time.perf_counter()

        crops = self.crop_spines(
            image,
            detections
        )

        crop_time = self._elapsed(step_start)

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

                # Per-crop timing.
                "ocr_time": 0.0,
                "local_match_time": 0.0,
                "google_books_time": 0.0,
                "book_recommendation_time": 0.0,
                "crop_total_time": 0.0
            }
            for idx, crop in enumerate(crops)
        ]

        # ---------------------------------
        # OCR
        # ---------------------------------
        step_start = time.perf_counter()

        crop_records = self.run_ocr(
            crop_records
        )

        ocr_total_time = self._elapsed(step_start)

        # ---------------------------------
        # Local catalog matching
        # ---------------------------------
        step_start = time.perf_counter()

        crop_records = self.match_books(
            crop_records
        )

        local_match_total_time = self._elapsed(step_start)

        # ---------------------------------
        # Google Books fallback
        # ---------------------------------
        google_books_start = time.perf_counter()

        for record in crop_records:

            crop_step_start = time.perf_counter()

            if record["local_match_status"] == "matched":

                record["final_status"] = "matched"
                record["final_book"] = record["local_match"]
                record["google_status"] = "not_needed"

                record["google_books_time"] = 0.0

                continue

            # Keep OCR query separate from the user recommendation query.
            ocr_query = record.get(
                "ocr_query",
                ""
            )

            if not ocr_query:

                record["google_status"] = "not_attempted"
                record["google_books_time"] = 0.0
                continue

            if not self.google_books_api_key:

                record["google_status"] = "no_api_key"
                record["google_books_time"] = 0.0
                continue

            book = self.fetch_from_google_books(
                ocr_query
            )

            record["google_books_time"] = self._elapsed(
                crop_step_start
            )

            if book is None:

                record["google_status"] = "unmatched"
                continue

            record["google_status"] = "matched"
            record["google_match"] = book

            # Actual fuzzy score used to validate the Google result.
            record["google_match_score"] = book.get(
                "Match Score"
            )

            record["match_source"] = "Google Books"
            record["final_status"] = "matched"
            record["final_book"] = book

            self.add_to_local_catalog(
                book
            )

        google_books_total_time = self._elapsed(
            google_books_start
        )

        # ---------------------------------
        # Book-to-book recommendations
        # ---------------------------------
        recommendation_start = time.perf_counter()

        for record in crop_records:

            book = record.get(
                "final_book"
            )

            if book:

                book["Crop Index"] = (
                    record["crop_idx"]
                )

                rec_start = time.perf_counter()

                book["Recommendations"] = (
                    self.get_recommendations(
                        book.get("ISBN13")
                    )
                )

                record["book_recommendation_time"] = (
                    self._elapsed(rec_start)
                )

        recommendation_total_time = self._elapsed(
            recommendation_start
        )

        # ---------------------------------
        # Per-crop total time
        # ---------------------------------
        for record in crop_records:

            record["crop_total_time"] = round(
                record.get("ocr_time", 0.0)
                + record.get("local_match_time", 0.0)
                + record.get("google_books_time", 0.0)
                + record.get("book_recommendation_time", 0.0),
                4
            )

        # ---------------------------------
        # Tracking DataFrame
        # ---------------------------------
        tracking_columns = [
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
            "Crop Total Time (s)"
        ]

        tracking_rows = []

        for record in crop_records:

            book = (
                record.get("final_book")
                or {}
            )

            tracking_rows.append({
                "Crop Index": record["crop_idx"],
                "BBox": record["bbox"],
                "OCR Status": record["ocr_status"],
                "OCR Text": " | ".join(
                    record["ocr_text"]
                ),
                "OCR Scores": record["ocr_scores"],
                "OCR Query": record["ocr_query"],
                "Local Match Status": record[
                    "local_match_status"
                ],
                "Local Match Score": record[
                    "local_match_score"
                ],
                "Google Books Status": record[
                    "google_status"
                ],
                "Google Books Score": record[
                    "google_match_score"
                ],
                "Match Source": record[
                    "match_source"
                ],
                "Final Status": record[
                    "final_status"
                ],
                "Matched Title": book.get(
                    "Title",
                    ""
                ),
                "Matched Author": book.get(
                    "Author",
                    ""
                ),
                "ISBN13": book.get(
                    "ISBN13",
                    ""
                ),
                "OCR Time (s)": record[
                    "ocr_time"
                ],
                "Local Match Time (s)": record[
                    "local_match_time"
                ],
                "Google Books Time (s)": record[
                    "google_books_time"
                ],
                "Book Recommendation Time (s)": record[
                    "book_recommendation_time"
                ],
                "Crop Total Time (s)": record[
                    "crop_total_time"
                ]
            })

        crop_tracking_df = pd.DataFrame(
            tracking_rows,
            columns=tracking_columns
        )

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
            "Recommendations"
        ]

        if matched_books:

            books_df = pd.DataFrame(
                matched_books
            )

            for col in book_columns:

                if col not in books_df.columns:
                    books_df[col] = ""

            books_df = books_df[
                book_columns
            ]

        else:

            books_df = pd.DataFrame(
                columns=book_columns
            )

        # ---------------------------------
        # User-query shelf recommendations
        # ---------------------------------
        top_query_columns = (
            book_columns
            + [
                "Query Score",
                "Query Rank"
            ]
        )

        query_recommendation_start = time.perf_counter()

        if (
            recommendation_query
            and not books_df.empty
        ):

            top_recommendations_df = (
                self.recommend_from_shelf(
                    books_df,
                    recommendation_query,
                    top_k=self.recommendation_top_k,
                    score_cutoff=self.recommendation_score_cutoff
                )
            )

        else:

            top_recommendations_df = pd.DataFrame(
                columns=top_query_columns
            )

        query_recommendation_time = self._elapsed(
            query_recommendation_start
        )

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

        total_pipeline_time = self._elapsed(
            pipeline_start
        )

        timing_df = pd.DataFrame([
            {
                "Step": "YOLO Detection",
                "Time (s)": detection_time
            },
            {
                "Step": "YOLO Annotation",
                "Time (s)": annotation_time
            },
            {
                "Step": "Spine Cropping",
                "Time (s)": crop_time
            },
            {
                "Step": "RapidOCR",
                "Time (s)": ocr_total_time
            },
            {
                "Step": "Local Catalog Matching",
                "Time (s)": local_match_total_time
            },
            {
                "Step": "Google Books",
                "Time (s)": google_books_total_time
            },
            {
                "Step": "Book-to-Book Recommendations",
                "Time (s)": recommendation_total_time
            },
            {
                "Step": "User Query Recommendations",
                "Time (s)": query_recommendation_time
            },
            {
                "Step": "Total Pipeline",
                "Time (s)": total_pipeline_time
            }
        ])

        return {
            "yolo_output_image": yolo_output_image,
            "crop_records": crop_records,
            "crop_tracking_df": crop_tracking_df,
            "timing_df": timing_df,
            "timing": {
                "yolo_detection": detection_time,
                "yolo_annotation": annotation_time,
                "spine_cropping": crop_time,
                "rapidocr": ocr_total_time,
                "local_catalog_matching": local_match_total_time,
                "google_books": google_books_total_time,
                "book_to_book_recommendations": recommendation_total_time,
                "user_query_recommendations": query_recommendation_time,
                "total_pipeline": total_pipeline_time
            },
            "ocr_results": [
                {
                    "crop_idx": r["crop_idx"],
                    "text": r["ocr_text"],
                    "scores": r["ocr_scores"]
                }
                for r in crop_records
            ],
            "query_strings": [
                r["ocr_query"]
                for r in crop_records
                if r["ocr_query"]
            ],
            "recommendation_query": recommendation_query,
            "recommendation_score_cutoff": self.recommendation_score_cutoff,
            "recommendation_top_k": self.recommendation_top_k,
            "matched_books": books_df,
            "top_query_recommendations": (
                top_recommendations_df
            ),
            "unmatched_queries": unmatched_queries,
            "num_detected_spines": len(
                crop_records
            ),
            "num_ocr_results": sum(
                r["ocr_status"] == "success"
                for r in crop_records
            ),
            "num_matched_books": len(
                matched_books
            ),
            "num_unmatched_books": len(
                unmatched_records
            )
        }

    def display_results(self, output):

        if output.get(
            "yolo_output_image"
        ) is not None:

            print(
                "YOLO DETECTIONS (CROP NUMBERS)"
            )

            print("-" * 80)

            display(
                Image.fromarray(
                    cv2.cvtColor(
                        output[
                            "yolo_output_image"
                        ],
                        cv2.COLOR_BGR2RGB
                    )
                )
            )

        print("=" * 80)
        print("BOOK DETECTION SUMMARY")
        print("=" * 80)

        print(
            f"Detected Spines : "
            f"{output['num_detected_spines']}"
        )

        print(
            f"OCR Results     : "
            f"{output['num_ocr_results']}"
        )

        print(
            f"Matched Books   : "
            f"{output['num_matched_books']}"
        )

        print(
            f"Unmatched Books : "
            f"{output['num_unmatched_books']}"
        )

        print(
            f"Recommendation Cutoff : "
            f"{output.get('recommendation_score_cutoff', '')}"
        )

        print(
            f"Maximum Recommendations : "
            f"{output.get('recommendation_top_k', '')}"
        )

        print(
            "\nPER-CROP PIPELINE PROGRESS"
        )

        print("-" * 80)

        display(
            output["crop_tracking_df"]
        )

        print(
            "\nPIPELINE TIMING"
        )

        print("-" * 80)

        display(
            output["timing_df"]
        )

        print(
            "\nMATCHED BOOKS"
        )

        print("-" * 80)

        display(
            output["matched_books"]
        )

        if not output[
            "top_query_recommendations"
        ].empty:

            print(
                "\nTOP QUERY RECOMMENDATIONS"
            )

            print("-" * 80)

            print(
                "Books are included only when their "
                "Query Score meets the recommendation cutoff."
            )

            display(
                output[
                    "top_query_recommendations"
                ]
            )

        else:

            print(
                "\nTOP QUERY RECOMMENDATIONS"
            )

            print("-" * 80)

            print(
                "No matched shelf books reached the "
                "recommendation score cutoff."
            )

        if output[
            "unmatched_queries"
        ]:

            print(
                "\nUNMATCHED OCR QUERIES"
            )

            print("-" * 80)

            for ocr_query in output[
                "unmatched_queries"
            ]:

                print(
                    f"• {ocr_query}"
                )