from scripts_D.model_paddleocr import model, paddle_ocr, embedding_model
from IPython.display import display
import cv2
import chromadb
import hashlib
import numpy as np
import pandas as pd
import requests
import re
import time
from rapidfuzz import process, fuzz


# =====================================================================
# TEXT NORMALIZATION
# =====================================================================

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


# =====================================================================
# GOOGLE BOOKS FILTER
# =====================================================================

SPINOFF_KEYWORDS = [
    # "journal",
    # "workbook",
    # "summary",
    # "companion",
    # "study guide",
    # "notebook",
]


def is_likely_spinoff(title):

    if not title:
        return False

    title = title.lower()

    return any(keyword in title for keyword in SPINOFF_KEYWORDS)


# =====================================================================
# OCR NOISE FILTERING
# =====================================================================
# Grow this list ONLY as you actually observe an imprint name causing
# a bad match — do not add generic words. Some imprint names collide
# with real words (e.g. "Harper" is also an author surname, "House"
# appears in real titles like "Bleak House"), so being conservative
# here matters more than being comprehensive.

PUBLISHER_NOISE_WORDS = {
    "beck",
    "blanvalet",
}

MIN_BARCODE_DIGITS = 3


def is_barcode_like(text):
    """
    True if a cleaned OCR segment is purely digits and long enough
    to be a shelf-tag/barcode number rather than a title.

    NOTE: this will also drop a spine whose ONLY visible text is a
    bare number (e.g. a spine that OCR's as just "1984" with no
    other line detected). In practice queries combine multiple OCR
    lines per spine, so this is low risk, but worth knowing if a
    numeric-titled book goes unexpectedly unmatched.
    """

    return bool(text) and text.isdigit() and len(text) >= MIN_BARCODE_DIGITS


def strip_publisher_noise(text):
    """
    Removes known publisher/imprint tokens from an already-normalized
    OCR segment (token-level, since imprint names are often fused
    mid-string with the title/author rather than on their own line).
    """

    tokens = [
        token
        for token in text.split()
        if token not in PUBLISHER_NOISE_WORDS
    ]

    return " ".join(tokens)


# =====================================================================
# PIPELINE
# =====================================================================

class SpinePipeline:

    def __init__(
        self,
        catalog_path=r"..\MVP\Dataset\books_cleaned.csv",
        chroma_path=r"..\MVP\chroma_db",
        collection_name="books",
        google_books_api_key=None,
    ):

        # ---------------------------------------------------------
        # MODELS
        # ---------------------------------------------------------

        self.model = model
        self.paddle_ocr = paddle_ocr
        self.embedding_model = embedding_model

        # ---------------------------------------------------------
        # PARAMETERS
        # ---------------------------------------------------------

        self.score_threshold = 0.80
        self.match_score_cutoff = 80
        self.google_validation_cutoff = 60

        # Used by get_recommendations() when suggesting similar
        # books for an already-matched book (not for OCR matching).
        self.recommendation_top_k = 5

        self.det_unclip_ratio = 1.2
        self.min_description_words = 0

        self.google_books_api_key = google_books_api_key

        self.catalog_path = catalog_path

        # ---------------------------------------------------------
        # LOAD LOCAL CATALOG
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # CHROMA DATABASE
        # ---------------------------------------------------------

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = self.chroma_client.get_collection(
            collection_name
        )

    # ==================================================================
    # GEOMETRY
    # ==================================================================

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


    # ==================================================================
    # DETECTION
    # ==================================================================

    def detect_spines(self, image):

        results = self.model.predict(
            image,
            conf=0.50,
            iou=0.40,
            verbose=False
        )

        return results[0].obb.xyxyxyxy.cpu().numpy()


    # ==================================================================
    # CROPPING
    # ==================================================================

    def crop_spines(self, image, obb_corners):

        crops = []

        for corners in obb_corners:

            corners = self.order_points(corners)

            width = int(
                max(
                    np.linalg.norm(corners[2] - corners[3]),
                    np.linalg.norm(corners[1] - corners[0])
                )
            )

            height = int(
                max(
                    np.linalg.norm(corners[1] - corners[2]),
                    np.linalg.norm(corners[0] - corners[3])
                )
            )

            if width < 2 or height < 2:
                continue

            destination = np.array(
                [
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1],
                ],
                dtype=np.float32,
            )

            matrix = cv2.getPerspectiveTransform(
                corners,
                destination
            )

            crop = cv2.warpPerspective(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(
                    crop,
                    cv2.ROTATE_90_CLOCKWISE
                )

            crops.append(crop)

        return crops


    # ==================================================================
    # PREPROCESSING
    # ==================================================================

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


    # ==================================================================
    # OCR
    # ==================================================================

    def run_ocr(self, crops):

        if not crops:
            return []

        processed_images = [
            self.preprocess(crop)
            for crop in crops
        ]

        predictions = self.paddle_ocr.predict(
            processed_images,
            text_det_unclip_ratio=self.det_unclip_ratio
        )

        results = []

        for prediction in predictions:

            texts = []
            scores = []

            for text, score in zip(
                prediction["rec_texts"],
                prediction["rec_scores"]
            ):

                if score >= self.score_threshold:

                    cleaned = normalize_text(text)

                    if cleaned:

                        texts.append(cleaned)
                        scores.append(score)

            results.append({

                "text": texts,

                "scores": scores

            })

        return results

    # ==================================================================
    # QUERY STRINGS
    # ==================================================================

    def extract_query_strings(self, ocr_results):

        queries = []

        for spine in ocr_results:

            if not spine["text"]:
                continue

            query = normalize_text(
                " ".join(spine["text"])
            )

            if query:
                queries.append(query)

        return queries


    # ==================================================================
    # LOCAL CATALOG MATCHING (STAGE 1 — RAPIDFUZZ)
    # ==================================================================

    def match_books(self, query_strings):

        matches = []
        unmatched = []

        for query in query_strings:

            result = process.extractOne(
                query,
                self.choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.match_score_cutoff
            )

            if result is None:
                unmatched.append(query)
                continue

            _, score, idx = result

            row = self.catalog_df.iloc[idx]

            matches.append({

                "Title": row["title"],
                "Author": row["authors"],
                "ISBN13": row["isbn13"],
                "Description": row.get("description", ""),
                "Thumbnail": row.get("thumbnail", ""),
                "Source": "Local Catalog"

            })

        return matches, unmatched


    # ==================================================================
    # GOOGLE BOOKS FALLBACK
    # ==================================================================

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
                    500,
                    502,
                    503,
                    504
                ):

                    time.sleep(1.5 * (attempt + 1))
                    continue

                return None

            except requests.exceptions.RequestException:

                return None

        else:

            return None

        data = response.json()

        items = data.get("items", [])

        if not items:

            return None

        for item in items:

            info = item.get("volumeInfo", {})

            title = info.get("title")

            authors = info.get("authors", [])

            if not title:

                continue

            if is_likely_spinoff(title):

                continue

            candidate = normalize_text(
                title + " " + " ".join(authors)
            )

            score = fuzz.token_set_ratio(
                query,
                candidate
            )

            if score < self.google_validation_cutoff:

                continue

            description = info.get("description") or ""

            if description and len(description.split()) < self.min_description_words:

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

                "Source": "Google Books"

            }

        return None

    # =====================================================================
    # SAVE NEW GOOGLE BOOKS TO LOCAL DATABASE
    # =====================================================================

    def add_to_local_catalog(self, book):

        isbn13 = book["ISBN13"]

        if not isbn13:

            source = f"{book['Title']}_{book['Author']}"

            isbn13 = "gb_" + hashlib.md5(
                source.encode()
            ).hexdigest()[:12]

            book["ISBN13"] = isbn13


        # -----------------------------
        # Already exists?
        # -----------------------------

        if isbn13 in self.catalog_df["isbn13"].values:
            return


        normalized = normalize_text(

            book["Title"] + " " + book["Author"]

        )

        duplicate = process.extractOne(

            normalized,
            self.choices,
            scorer=fuzz.token_set_ratio,
            score_cutoff=90

        )

        if duplicate:

            return


        # -----------------------------
        # Generate embedding
        # -----------------------------
        # Falls back to title+author when there's no description
        # (e.g. some Google Books records lack one) — encoding an
        # empty string would otherwise produce a degenerate embedding
        # that's useless for future recommendation queries.

        text_for_embedding = (
            book["Description"]
            if book["Description"]
            else f"{book['Title']} {book['Author']}"
        )

        embedding = self.embedding_model.encode(

            [text_for_embedding]

        ).tolist()[0]


        # -----------------------------
        # Store in Chroma
        # -----------------------------

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


        # -----------------------------
        # Store in local dataframe
        # -----------------------------

        row = {

            column: ""

            for column in self.catalog_df.columns

            if column != "match_key"

            and column != "normalized_match_key"

        }


        row.update(

            {

                "isbn13": isbn13,

                "title": book["Title"],

                "authors": book["Author"],

                "description": book["Description"],

                "thumbnail": book["Thumbnail"]

            }

        )


        if "title_with_subtitle" in row:

            row["title_with_subtitle"] = book["Title"]


        if "tagged_description" in row:

            row["tagged_description"] = (

                isbn13 +

                " " +

                book["Description"]

            )


        self.catalog_df = pd.concat(

            [

                self.catalog_df,

                pd.DataFrame([row])

            ],

            ignore_index=True

        )


        # -----------------------------
        # Update lookup cache
        # -----------------------------

        key = normalize_text(

            book["Title"] +

            " " +

            book["Author"]

        )

        self.choices.append(key)


        # -----------------------------
        # Save CSV
        # -----------------------------

        columns = [

            c

            for c in self.catalog_df.columns

            if c not in (

                "match_key",

                "normalized_match_key"

            )

        ]


        pd.DataFrame([row])[columns].to_csv(

            self.catalog_path,

            mode="a",

            index=False,

            header=False

        )


    # =====================================================================
    # RECOMMENDATIONS (SIMILAR BOOKS FOR AN ALREADY-MATCHED BOOK)
    # =====================================================================

    def get_recommendations(self, isbn13, top_k=None):
        """
        Given the isbn13 of a book that has ALREADY been matched
        (from local catalog or Google Books), finds similar books by
        looking up that book's own description embedding in Chroma
        and retrieving its nearest neighbors.

        This is different from match_via_embeddings (removed): here
        we start from a known book's own embedding, not from a raw
        OCR query string, so there's no query/description domain
        mismatch — we're comparing descriptions to descriptions.

        Returns a list of dicts (empty list if isbn13 isn't in the
        Chroma collection or has no stored embedding).
        """

        k = top_k if top_k is not None else self.recommendation_top_k

        if not isbn13:
            return []

        try:

            record = self.collection.get(
                ids=[isbn13],
                include=["embeddings"]
            )

        except Exception:

            return []

        embeddings = record.get("embeddings")

        if embeddings is None or len(embeddings) == 0:
            return []

        embedding = embeddings[0]

        # +1 result requested since the book itself will typically
        # come back as its own nearest neighbor (distance ~0).
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k + 1
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        recommendations = []

        for rec_id, distance, metadata in zip(ids, distances, metadatas):

            if rec_id == isbn13:
                continue

            recommendations.append({

                "ISBN13": rec_id,
                "Title": metadata.get("title", ""),
                "Author": metadata.get("authors", ""),
                "Categories": metadata.get("categories", ""),
                "Average Rating": metadata.get("average_rating", ""),
                "Thumbnail": metadata.get("thumbnail", ""),
                "Distance": distance

            })

            if len(recommendations) >= k:
                break

        return recommendations


    # =====================================================================
    # COMPLETE PIPELINE
    # =====================================================================

    def results(self, image):

        detections = self.detect_spines(image)

        crops = self.crop_spines(

            image,

            detections

        )

        ocr = self.run_ocr(crops)

        queries = self.extract_query_strings(ocr)


        # --------------------------------
        # Stage 1 — rapidfuzz local catalog
        # --------------------------------

        matches, unmatched = self.match_books(

            queries

        )

        # --------------------------------
        # Stage 2 — Google Books fallback
        # --------------------------------
        # NOTE: previously "unmatched" was left untouched here even
        # after a query was successfully resolved via Google Books,
        # so num_unmatched_books / unmatched_queries over-reported
        # failures. still_unmatched now tracks only queries that
        # remain unresolved after every stage.

        still_unmatched = []

        if self.google_books_api_key:

            for query in unmatched:

                book = self.fetch_from_google_books(

                    query

                )

                if book is None:

                    still_unmatched.append(query)

                    continue

                matches.append(book)

                self.add_to_local_catalog(book)

        else:

            still_unmatched = unmatched

        unmatched = still_unmatched

        # --------------------------------
        # Attach recommendations to each match
        # --------------------------------

        for match in matches:

            match["Recommendations"] = self.get_recommendations(
                match.get("ISBN13")
            )

        # ----------------------------------------
        # Convert matches to DataFrame
        # ----------------------------------------

        columns = [
            "Title",
            "Author",
            "ISBN13",
            "Source",
            "Description",
            "Thumbnail",
            "Recommendations"
        ]

        if matches:
            books_df = pd.DataFrame(matches)[columns]
        else:
            books_df = pd.DataFrame(columns=columns)


        # ----------------------------------------
        # Return complete pipeline output
        # ----------------------------------------

        return {
            "ocr_results": ocr,
            "query_strings": queries,
            "matched_books": books_df,
            "unmatched_queries": unmatched,
            "num_detected_spines": len(crops),
            "num_ocr_results": len(ocr),
            "num_matched_books": len(books_df),
            "num_unmatched_books": len(unmatched)
        }


    def display_results(self, output):

            print("=" * 80)
            print("BOOK DETECTION SUMMARY")
            print("=" * 80)

            print(f"OCR Results     : {output['ocr_results']}")
            print(f"Detected Spines : {output['num_detected_spines']}")
            print(f"Matched Books   : {output['num_matched_books']}")
            print(f"Unmatched Books : {output['num_unmatched_books']}")

            print("\nMatched Books")
            print("-" * 80)

            display(output["matched_books"])

            if output["unmatched_queries"]:
                print("\nBooks Not Found")
                print("-" * 80)

                for book in output["unmatched_queries"]:
                    print(f"• {book}")