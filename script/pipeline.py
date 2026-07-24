from script.models import model, paddle_ocr
import numpy as np
import cv2

class SpinePipeline:
    def __init__(self):
        self.model = model
        self.paddle_ocr = paddle_ocr
        self.score_threshold = 0.8


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
                    results.append({'text': res['rec_texts'],'scores':res['rec_scores']})

        return results


    def results(self, image):
        bboxes = self.detect_spines(image)
        crops = self.crop_spines(image, bboxes)
        results = self.run_ocr(crops)

        return results
