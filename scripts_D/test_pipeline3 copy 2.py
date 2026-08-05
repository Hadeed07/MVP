from scripts_D.models import model, paddle_ocr, embedding_model
import numpy as np
import cv2
import pandas as pd
import chromadb
import requests
import time
import re
import hashlib
from rapidfuzz import process, fuzz

import re

def normalize_text(text):
    """
    Normalize OCR text by:
    - Replacing punctuation with spaces
    - Splitting CamelCase words
    - Collapsing multiple spaces
    - Removing single-character tokens
    - Converting to lowercase
    """
    text = str(text)

    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Split CamelCase (e.g., PauloCoelho -> Paulo Coelho)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove single-character tokens
    tokens = [token for token in text.split() if len(token) > 1]

    return ' '.join(tokens).lower()


def split_concatenated_words(text):
    """
    Split CamelCase words.
    Kept for compatibility if used elsewhere.
    """
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

SPINOFF_KEYWORDS = ['journal', 'workbook', 'summary', 'companion', 'study guide', 'guide to', 'notebook']


def is_likely_spinoff(title):
    """Detects companion/derivative products (workbooks, journals, summaries)
    rather than the actual book."""
    if not title:
        return False
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in SPINOFF_KEYWORDS)


class SpinePipeline:
    def __init__(self, catalog_path=r"..\MVP\Dataset\books_cleaned.csv",
                 chroma_path=r"..\MVP\chroma_db",
                 collection_name="books", google_books_api_key=None):
        self.model = model  # YOLO spine-detection model
        self.paddle_ocr = paddle_ocr
        self.embedding_model = embedding_model  # loaded once in script.models, just referenced here

        self.score_threshold = 0.8
        self.match_score_cutoff = 80
        self.google_validation_cutoff = 60
        self.n_recommendations = 5
        self.google_books_api_key = google_books_api_key
        self.det_unclip_ratio = 1.2
        self.min_description_words = 25
        self.catalog_path = catalog_path

        self.catalog_df = pd.read_csv(catalog_path)
        self.catalog_df['isbn13'] = self.catalog_df['isbn13'].astype(str)
        self.catalog_df['match_key'] = (
            self.catalog_df['title'].astype(str) + " " + self.catalog_df['authors'].astype(str)
        )
        self.choices = self.catalog_df['match_key'].tolist()
        self.normalized_choices = [normalize_text(c) for c in self.choices]

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(collection_name)

    @staticmethod
    def order_points(pts):
        """
        Order points as:
        top-left, top-right, bottom-right, bottom-left
        """
        pts = np.array(pts, dtype=np.float32)

        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # Top-left
        rect[2] = pts[np.argmax(s)]      # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # Top-right
        rect[3] = pts[np.argmax(diff)]   # Bottom-left

        return rect

    def detect_spines(self, image):
        results = self.model.predict(image, conf=0.5, verbose=False, iou=0.4)
        obb_corners = results[0].obb.xyxyxyxy.cpu().numpy()
        return obb_corners

    def crop_spines(self, image, obb_corners):
        crops = []

        for corners in obb_corners:

            # Reorder corners
            corners = self.order_points(corners)

            # Compute width
            widthA = np.linalg.norm(corners[2] - corners[3])
            widthB = np.linalg.norm(corners[1] - corners[0])
            width = int(max(widthA, widthB))

            # Compute height
            heightA = np.linalg.norm(corners[1] - corners[2])
            heightB = np.linalg.norm(corners[0] - corners[3])
            height = int(max(heightA, heightB))

            if width < 2 or height < 2:
                continue

            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            # Perspective transform
            M = cv2.getPerspectiveTransform(corners, dst)

            crop = cv2.warpPerspective(
                image,
                M,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            # Make the spine horizontal
            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            crops.append(crop)

        return crops
    
    def preprocess(self, crop):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        target_height = 256
        target_width = 600

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        gray = clahe.apply(gray)

        h, w = gray.shape[:2]

        # Scale while preserving aspect ratio
        scale = min(target_width / w, target_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

        resized = cv2.resize(
            gray,
            (new_w, new_h),
            interpolation=interpolation
        )

        # Create white canvas
        canvas = np.full((target_height, target_width), 255, dtype=np.uint8)

        # Center the resized image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2

        canvas[y_offset:y_offset + new_h,
            x_offset:x_offset + new_w] = resized

        # Convert back to 3-channel BGR
        img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        return img

    def run_ocr(self, crops):
        results = []

        for crop in crops:
            pre_processed = self.preprocess(crop)
            ocr_output = self.paddle_ocr.predict(
                pre_processed,
                text_det_unclip_ratio=self.det_unclip_ratio
            )

            for res in ocr_output:
                filtered_scores = []
                filtered_texts = []

                for text, score in zip(res['rec_texts'], res['rec_scores']):
                    if score >= self.score_threshold:
                        filtered_texts.append(text)
                        filtered_scores.append(score)

                if filtered_texts:
                    results.append({'text': filtered_texts, 'scores': filtered_scores})

        return results

    def extract_query_strings(self, ocr_results):
        """
        Takes OCR results (already score-filtered by run_ocr) and returns
        a list of single joined strings, one per spine.
        """
        query_strings = []

        for spine in ocr_results:
            texts = spine['text']

            if not texts:
                continue

            joined = " ".join(texts)
            joined = split_concatenated_words(joined)
            query_strings.append(joined)

        return query_strings

    def match_books(self, query_strings):
        """
        Matches each OCR query string against the local catalog, using
        normalized (lowercased, punctuation-stripped, space-preserved) text
        on both sides for a fair comparison. Returns confident matches plus
        the raw (un-normalized) query strings that found no local match.
        """
        matches = []
        unmatched_queries = []

        for query in query_strings:
            normalized_query = normalize_text(query)

            result = process.extractOne(
                normalized_query,
                self.normalized_choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.match_score_cutoff
            )

            if result is None:
                unmatched_queries.append(query)  # keep original for Google fallback
                continue

            matched_text, score, idx = result
            row = self.catalog_df.iloc[idx]

            matches.append({
                "ocr_text": query,
                "matched_title": row['title'],
                "matched_authors": row['authors'],
                "isbn13": row['isbn13'],
                "score": score,
                "source": "local_catalog"
            })

        return matches, unmatched_queries

    def compare_scorers(self, query_strings):
        """
        DIAGNOSTIC TOOL — not used in the live pipeline. Runs each OCR query
        string against the local catalog using several different rapidfuzz
        scorers, so you can compare their behavior side-by-side.
        """
        scorers = {
            "ratio": fuzz.ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "partial_ratio": fuzz.partial_ratio,
        }

        for query in query_strings:
            normalized_query = normalize_text(query)
            print(f"Query: {query!r}")

            for scorer_name, scorer_fn in scorers.items():
                result = process.extractOne(normalized_query, self.normalized_choices, scorer=scorer_fn)
                if result is None:
                    print(f"  {scorer_name:20}: no match")
                    continue

                matched_text, score, idx = result
                row = self.catalog_df.iloc[idx]
                print(f"  {scorer_name:20}: {score:5.1f}  →  {row['title']}")

            print()

    def fetch_from_google_books(self, query, timeout=5, max_retries=3, max_results=5):
        """
        Falls back to Google Books API for a query that didn't match locally.
        Requests multiple candidates and returns the first one that passes
        both the spin-off filter and the validation score against the
        original OCR query.
        """
        base_url = "https://www.googleapis.com/books/v1/volumes"
        params = {"q": query, "maxResults": max_results, "key": self.google_books_api_key}

        response = None
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()
                break
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1} for query: {query!r}")
            except requests.exceptions.HTTPError:
                if response is not None and response.status_code in (500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
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
            volume_info = item.get("volumeInfo", {})
            title = volume_info.get("title")
            authors = volume_info.get("authors", [])

            if is_likely_spinoff(title):
                continue

            result_string = f"{title} {' '.join(authors)}" if authors else str(title)
            validation_score = fuzz.token_set_ratio(normalize_text(query), normalize_text(result_string))

            if validation_score < self.google_validation_cutoff:
                continue

            description = volume_info.get("description")
            if not description:
                continue

            if len(description.split()) <= self.min_description_words:
                continue  # too short to embed meaningfully

            return {
                "ocr_text": query,
                "matched_title": title,
                "matched_authors": ", ".join(authors) if authors else None,
                "description": description,
                "thumbnail": volume_info.get("imageLinks", {}).get("thumbnail"),
                "isbn13": next(
                    (i["identifier"] for i in volume_info.get("industryIdentifiers", [])
                     if i["type"] == "ISBN_13"),
                    None
                ),
                "score": validation_score,
                "source": "google_books"
            }

        return None

    def debug_fetch(self, query, timeout=5):
        """
        DIAGNOSTIC TOOL — shows exactly why a query passed or failed the
        Google Books fallback (HTTP error / no results / spin-off / low
        score / missing description), checking only the top result.
        """
        base_url = "https://www.googleapis.com/books/v1/volumes"
        params = {"q": query, "maxResults": 1, "key": self.google_books_api_key}

        response = requests.get(base_url, params=params, timeout=timeout)

        if response.status_code != 200:
            print(f"Query: {query!r} → HTTP ERROR {response.status_code}: {response.text[:200]}\n")
            return

        data = response.json()

        if "items" not in data or len(data["items"]) == 0:
            print(f"Query: {query!r} → NO RESULTS FROM GOOGLE (genuinely empty)\n")
            return

        volume_info = data["items"][0].get("volumeInfo", {})
        title = volume_info.get("title")
        authors = volume_info.get("authors", [])
        description = volume_info.get("description")
        result_string = f"{title} {' '.join(authors)}" if authors else str(title)

        score = fuzz.token_set_ratio(normalize_text(query), normalize_text(result_string))
        spinoff = is_likely_spinoff(title)

        print(f"Query: {query!r}")
        print(f"  Google returned: {result_string!r}")
        print(f"  Validation score: {score:.1f} (cutoff: {self.google_validation_cutoff})")
        print(f"  Is spinoff: {spinoff}")
        print(f"  Has description: {bool(description)}")
        print()

    def get_recommendations(self, isbn13=None, embedding=None):
        """
        Gets recommendations either from a stored isbn13 (local catalog books)
        or a freshly computed embedding (Google Books fallback results).
        """
        if embedding is None:
            stored = self.collection.get(ids=[isbn13], include=["embeddings", "metadatas"])
            if not stored["ids"]:
                return []
            embedding = stored["embeddings"][0]

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.n_recommendations + 1
        )

        recommendations = []
        for rec_id, rec_meta, rec_dist in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            if rec_id == isbn13:
                continue

            recommendations.append({
                "isbn13": rec_id,
                "title": rec_meta.get("title"),
                "authors": rec_meta.get("authors"),
                "categories": rec_meta.get("categories"),
                "average_rating": rec_meta.get("average_rating"),
                "thumbnail": rec_meta.get("thumbnail"),
                "distance": rec_dist
            })

        return recommendations[:self.n_recommendations]

    def add_to_local_catalog(self, book_data, embedding=None):
        """
        Persists a Google Books fallback result into both Chroma (via upsert,
        safe against duplicate ids across sessions) and the local catalog
        (for future Stage 1 fuzzy matching, avoiding a repeat API call for
        the same book). Checks for duplicates by isbn13 AND fuzzy
        title+author match, since isbn13 can be missing/inconsistent.
        """
        isbn13 = book_data.get("isbn13")

        if not isbn13:
            id_source = f"{book_data['matched_title']}_{book_data.get('matched_authors', '')}"
            isbn13 = "gb_" + hashlib.md5(id_source.encode()).hexdigest()[:12]

        # --- Duplicate check 1: exact isbn13 already present ---
        if isbn13 in self.catalog_df['isbn13'].values:
            return

        # --- Duplicate check 2: fuzzy title+author match against existing catalog ---
        candidate_key = normalize_text(f"{book_data['matched_title']} {book_data.get('matched_authors', '')}")
        dup_check = process.extractOne(
            candidate_key,
            self.normalized_choices,
            scorer=fuzz.token_set_ratio,
            score_cutoff=90  # high bar — only treat as duplicate if essentially the same book
        )
        if dup_check is not None:
            return

        if embedding is None:
            embedding = self.embedding_model.encode([book_data["description"]]).tolist()[0]

        thumbnail = book_data.get("thumbnail") or ""
        authors = book_data.get("matched_authors") or ""
        title = book_data["matched_title"]
        description = book_data.get("description") or ""

        # --- 1. Add to Chroma (upsert — safe against duplicate/repeated ids) ---
        self.collection.upsert(
            ids=[isbn13],
            embeddings=[embedding],
            documents=[description],
            metadatas=[{
                "title": title,
                "authors": authors,
                "categories": "",
                "average_rating": 0.0,
                "thumbnail": thumbnail
            }]
        )

        # --- 2. Build a full row matching the catalog's actual schema ---
        new_row = {col: "" for col in self.catalog_df.columns if col != 'match_key'}
        new_row.update({
            "isbn13": isbn13,
            "title": title,
            "authors": authors,
            "description": description,
            "thumbnail": thumbnail,
        })
        if "title_with_subtitle" in new_row:
            new_row["title_with_subtitle"] = title
        if "tagged_description" in new_row:
            new_row["tagged_description"] = f"{isbn13} {description}"

        self.catalog_df = pd.concat([self.catalog_df, pd.DataFrame([new_row])], ignore_index=True)

        new_match_key = f"{title} {authors}"
        self.choices.append(new_match_key)
        self.normalized_choices.append(normalize_text(new_match_key))

        # --- 3. Persist to CSV on disk ---
        csv_columns = [c for c in self.catalog_df.columns if c != 'match_key']
        pd.DataFrame([new_row])[csv_columns].to_csv(
            self.catalog_path, mode='a', header=False, index=False
        )

    def results(self, image):
        bboxes = self.detect_spines(image)
        crops = self.crop_spines(image, bboxes)
        ocr_results = self.run_ocr(crops)
        query_strings = self.extract_query_strings(ocr_results)

        matches, unmatched_queries = self.match_books(query_strings)

        # Attach recommendations for confident local matches
        for match in matches:
            match["recommendations"] = self.get_recommendations(isbn13=match["isbn13"])

        # Try Google Books fallback for unmatched queries
        if self.google_books_api_key:
            for query in unmatched_queries:
                fallback_result = self.fetch_from_google_books(query)

                if fallback_result is None:
                    continue  # truly unidentifiable, skip silently

                embedding = self.embedding_model.encode([fallback_result["description"]]).tolist()[0]
                fallback_result["recommendations"] = self.get_recommendations(
                    isbn13=fallback_result.get("isbn13"),
                    embedding=embedding
                )

                matches.append(fallback_result)
                self.add_to_local_catalog(fallback_result, embedding=embedding)

        return {
            "ocr_results": ocr_results,
            "query_strings": query_strings,
            "matches": matches
        }

    def display_results(self, output):
        """Pretty-prints full details for every matched book and its recommendations."""
        print(f"Detected spines: {len(output['query_strings'])}")
        print(f"Resolved matches: {len(output['matches'])}")
        print("=" * 80)

        for i, match in enumerate(output["matches"], 1):
            print(f"\n[{i}] {match['matched_title']}")
            print(f"    Source: {match['source']}")
            print(f"    OCR text: {match['ocr_text']}")
            print(f"    Authors: {match.get('matched_authors', 'N/A')}")
            print(f"    ISBN13: {match.get('isbn13', 'N/A')}")
            print(f"    Match score: {match['score']:.1f}")

            if match['source'] == 'google_books':
                print(f"    Thumbnail: {match.get('thumbnail', 'N/A')}")
                desc = match.get('description', '')
                desc_preview = (desc[:150] + '...') if desc and len(desc) > 150 else desc
                print(f"    Description: {desc_preview}")

            recs = match.get('recommendations', [])
            print(f"    Recommendations ({len(recs)}):")
            for rec in recs:
                print(f"      - {rec['title']} by {rec.get('authors', 'N/A')}  (distance: {rec['distance']:.3f})")

        print("\n" + "=" * 80)