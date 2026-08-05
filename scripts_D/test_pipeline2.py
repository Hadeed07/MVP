# Abandoning this pipeline because the result are not so great.

from script.models import model, paddle_ocr
import numpy as np
import cv2
import pandas as pd
import chromadb
from rapidfuzz import process, fuzz

class SpinePipeline:
    def __init__(self, catalog_path=r"..\Dataset\books_cleaned2.csv", chroma_path=r"..\chroma_db", collection_name="books_v3"):
        self.model = model
        self.paddle_ocr = paddle_ocr
        self.score_threshold = 0.8
        self.match_score_cutoff = 70
        self.n_recommendations = 5

        # Load catalog once at init, not per-query
        self.catalog_df = pd.read_csv(catalog_path)
        self.catalog_df['ID'] = self.catalog_df['ID'].astype(str)
        self.catalog_df['match_key'] = (
            self.catalog_df['Title'].astype(str) + " " + self.catalog_df['Authors'].astype(str)
        )
        self.choices = self.catalog_df['match_key'].tolist()

        # Connect to the persisted Chroma index once at init
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

        for i, corners in enumerate(obb_corners):

            # -------------------------------
            # Reorder the corners correctly
            # -------------------------------
            corners = self.order_points(corners)

            # -------------------------------
            # Compute width
            # -------------------------------
            widthA = np.linalg.norm(corners[2] - corners[3])
            widthB = np.linalg.norm(corners[1] - corners[0])
            width = int(max(widthA, widthB))

            # -------------------------------
            # Compute height
            # -------------------------------
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

            # -------------------------------
            # Perspective transform
            # -------------------------------
            M = cv2.getPerspectiveTransform(corners, dst)

            crop = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            crops.append(crop)

        return crops

    def preprocess(self, crop, scale=3):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_height = 256
        max_width = 1000
        max_scale = 3

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE
        gray = clahe.apply(gray)

        # Resizing
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

        # Conversion to RGB back
        img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return img

    def run_ocr(self, crops):
        results = []

        for crop in crops:
            pre_processed = self.preprocess(crop)
            ocr_output = self.paddle_ocr.predict(pre_processed)

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
            query_strings.append(joined)

        return query_strings

    def match_books(self, query_strings):
        """
        Matches each OCR query string against the catalog's title+author
        field. Returns a list of dicts, one per successfully matched spine,
        containing the original OCR text, the matched catalog title/author,
        and the ID. Unmatched spines are silently dropped.
        """
        matches = []

        for query in query_strings:
            result = process.extractOne(
                query,
                self.choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.match_score_cutoff
            )

            if result is None:
                continue  # no confident match, skip silently

            matched_text, score, idx = result
            row = self.catalog_df.iloc[idx]

            matches.append({
                "ocr_text": query,
                "matched_title": row['Title'],
                "matched_authors": row['Authors'],
                "id": row['ID'],
                "score": score
            })

        return matches

    def get_recommendations(self, book_id):
        """
        Given an ID already in the Chroma index, retrieves its stored
        embedding and queries for the most similar books (excluding itself).
        Returns an empty list if the ID isn't in the index.
        """
        stored = self.collection.get(
            ids=[book_id],
            include=["embeddings", "metadatas"]
        )

        if not stored["ids"]:
            return []  # ID not found in the Chroma index

        embedding = stored["embeddings"][0]

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.n_recommendations + 1
        )

        recommendations = []
        for rec_id, rec_meta, rec_dist in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            if rec_id == book_id:
                continue  # skip the book matching itself

            recommendations.append({
                "id": rec_id,
                "title": rec_meta.get("Title"),
                "authors": rec_meta.get("Authors"),
                "category": rec_meta.get("Category"),
                "publisher": rec_meta.get("Publisher"),
                "publish_date": rec_meta.get("Publish Date"),
                "price": rec_meta.get("Price"),
                "distance": rec_dist
            })

        return recommendations[:self.n_recommendations]

    def results(self, image):
        bboxes = self.detect_spines(image)
        crops = self.crop_spines(image, bboxes)
        ocr_results = self.run_ocr(crops)
        query_strings = self.extract_query_strings(ocr_results)
        matches = self.match_books(query_strings)

        for match in matches:
            match["recommendations"] = self.get_recommendations(match["id"])

        return matches