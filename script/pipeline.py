from script.models import model, paddle_ocr
import numpy as np
import cv2

class SpinePipeline:
    def __init__(self):
        self.model = model
        self.paddle_ocr = paddle_ocr
        self.conf_thres = 0.5
    

    def detect_spines(self, image):
        results = self.model.predict(image, conf=self.conf_thres, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy()

        return bboxes
    
    def crop_spines(self, image, bboxes):
        crops = []
        h, w = image.shape[:2]

        for x1, y1, x2, y2 in bboxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)


            crop = image[y1: y2, x1: x2]
            crops.append(crop)

        return crops



    def preprocess(self, crop, scale=3):
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Upscale
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Sharpen
        sharpen = cv2.filter2D(gray, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

        return cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)
    

    def run_ocr(self, crops):
        results = []

        for crop in crops:
            processed = self.preprocess(crop)
            ocr_output = self.paddle_ocr.predict(processed)

            for res in ocr_output:
                results.appened({'text': res['rec_texts'],'scores':res['rec_scores']})

        return results

    def process(self, image):
        bboxes = self.detect_spines(image)
        crops = self.crop_spines(image, bboxes)
        results = self.run_ocr(crops)

        return results
