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


def normalize_text(text):
    """Lowercase and strip punctuation, but KEEP spaces so token-based
    scoring remains robust to word-order differences."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_concatenated_words(text):
    """Insert a space wherever lowercase is immediately followed by uppercase
    (handles cases like 'PauloCoelho' -> 'Paulo Coelho')."""
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)


SPINOFF_KEYWORDS = ['journal', 'workbook', 'summary', 'companion', 'study guide', 'guide to', 'notebook']

def is_likely_spinoff(title):
    if not title:
        return False
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in SPINOFF_KEYWORDS)


class SpinePipeline:
    def __init__(self, catalog_path=r"D:\Documents\python codes\Book Recommender\MVP\Dataset\books_cleaned.csv",
                 chroma_path=r"D:\Documents\python codes\Book Recommender\MVP\chroma_db",
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
        self.catalog_path = catalog_path  # stored so add_to_local_catalog can persist to it

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
        pts = np.array(pts, dtype=np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_spines(self, image):
        results = self.model.predict(image, conf=0.5, verbose=False, iou=0.4)
        obb_corners = results[0].obb.xyxyxyxy.cpu().numpy()
        return obb_corners

    def crop_spines(self, image, obb_corners):
        crops = []
        for i, corners in enumerate(obb_corners):
            corners = self.order_points(corners)
            widthA = np.linalg.norm(corners[2] - corners[3])
            widthB = np.linalg.norm(corners[1] - corners[0])
            width = int(max(widthA, widthB))
            heightA = np.linalg.norm(corners[1] - corners[2])
            heightB = np.linalg.norm(corners[0] - corners[3])
            height = int(max(heightA, heightB))
            if width < 2 or height < 2:
                continue
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners, dst)
            crop = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            crops.append(crop)
        return crops

    def preprocess(self, crop, scale=3):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_height = 256
        max_width = 1000
        max_scale = 3
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        h, w = gray.shape[:2]
        scale = target_height / h
        if scale > max_scale:
            scale = max_scale
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
        img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
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
                unmatched_queries.append(query)
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

    def compare_scorers(self, query_strings, top_n=1):
        """DIAGNOSTIC TOOL — not used in the live pipeline."""
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
        Persists a Google Books fallback result into both Chroma and the local
        catalog, populating all matching columns. Checks for duplicates by both
        isbn13 AND fuzzy title+author match, since the same book can arrive
        with a missing/inconsistent isbn13 across different OCR reads.
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
            score_cutoff=90  # high bar — only treat as duplicate if it's essentially the same book
        )
        if dup_check is not None:
            return  # already effectively in the catalog under a different isbn/id

        if embedding is None:
            embedding = self.embedding_model.encode([book_data["description"]]).tolist()[0]

        thumbnail = book_data.get("thumbnail") or ""
        authors = book_data.get("matched_authors") or ""
        title = book_data["matched_title"]
        description = book_data.get("description") or ""

        # --- 1. Add to Chroma ---
        self.collection.add(
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
        # Fill in a couple more fields if your schema has them and Google provided equivalents
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

        for match in matches:
            match["recommendations"] = self.get_recommendations(isbn13=match["isbn13"])

        if self.google_books_api_key:
            for query in unmatched_queries:
                fallback_result = self.fetch_from_google_books(query)
                if fallback_result is None:
                    continue

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