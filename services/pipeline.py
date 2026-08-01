from services.models import model, paddle_ocr
import numpy as np
from typing import List, Tuple
import cv2
import pandas as pd
import re
from rapidfuzz import fuzz, process

class SpinePipeline:
    def __init__(self, catalog_path='Dataset/books_cleaned.csv'):
        self.model = model
        self.paddle_ocr = paddle_ocr
        self.score_threshold = 0.4
        self.match_score_cutoff = 70
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


         # Load catalog once at init, not per-query
        self.catalog_df = pd.read_csv(catalog_path)
        self.catalog_df['isbn13'] = self.catalog_df['isbn13'].astype(str)
        self.catalog_df['match_key'] = (
            self.catalog_df['title'].astype(str) + " " + self.catalog_df['authors'].astype(str)
        )
        self.choices = self.catalog_df['match_key'].tolist()


    @staticmethod
    def normalize(corners: np.ndarray, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        return [[float(x / w), float(y / h)] for x, y in corners]


    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
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

    
    def detect_spines(self, image: np.ndarray) -> np.ndarray:
        results = self.model.predict(image, conf=0.5, verbose=False, iou=0.4)
        obb_corners = results[0].obb.xyxyxyxy.cpu().numpy()
        return obb_corners


    def crop_spines(self, image: np.ndarray, obb_corners: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        crops: List[Tuple[int, np.ndarray]] = []
        
        for idx, corners in enumerate(obb_corners):

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

            crops.append((idx, crop))

        return crops



    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        target_height = 256
        max_width = 1000
        max_scale = 3

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE
        gray = self.clahe.apply(gray)

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


    

    def run_ocr(self, crops: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, dict]]:
        results = []

        for idx, crop in crops:
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
                    results.append((idx, {'text': filtered_texts,'scores': filtered_scores}))

        return results


    def annotate(self, image: np.ndarray, obb_corners: np.ndarray) -> np.ndarray:
        annotated = image.copy()
        for corners in obb_corners:
            pts = corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return annotated


    def extract_query_strings(self, ocr_results: List[Tuple[int, dict]]) -> List[Tuple[int, str]]:
        """
        Takes OCR results (already score-filtered by run_ocr) and returns
        a list of cleaned query strings, one per spine.
        """
        query_strings = []

        for idx, spine in ocr_results:
            cleaned_texts = []

            for text in spine["text"]:
                # Replace punctuation with spaces
                text = re.sub(r"[^\w\s]", " ", text)

                # Split CamelCase
                text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

                # Collapse whitespace
                text = re.sub(r"\s+", " ", text)

                # Trim
                text = text.strip()

                # Lowercase
                text = text.lower()

                # Remove single-character tokens
                tokens = [t for t in text.split() if len(t) > 1]

                if tokens:
                    cleaned_texts.append(" ".join(tokens))

            if cleaned_texts:
                query_strings.append((idx, " ".join(cleaned_texts)))

        return query_strings


    def match_books(self, query_strings: List[Tuple[int, str]]) -> List[dict]:
        """
        Matches each OCR query string against the catalog's title+author
        field. Returns a list of dicts, one per successfully matched spine,
        containing the original OCR text, the matched catalog title/author,
        and the isbn13. Unmatched spines are silently dropped.
        """
        matches = []

        for spine_idx, query in query_strings:
            result = process.extractOne(
                query,
                self.choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.match_score_cutoff
            )

            if result is None:
                continue  # no confident match, skip silently

            matched_text, score, catalog_idx = result
            row = self.catalog_df.iloc[catalog_idx]

            matches.append({
                "spine_idx": spine_idx,
                "ocr_text": query,
                "matched_title": row['title'],
                "matched_authors": row['authors'],
                "isbn13": row['isbn13'],
                "score": score
            })

        return matches


    def results(self, image: np.ndarray) -> Tuple[List[Tuple[int, dict]], np.ndarray]:
        bboxes = self.detect_spines(image)
        crops = self.crop_spines(image, bboxes)
        ocr_results = self.run_ocr(crops)
        query_strings = self.extract_query_strings(ocr_results)
        matches = self.match_books(query_strings)

        return matches, bboxes
