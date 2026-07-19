import easyocr


class ocr():
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def extract(self, image):
        return self.reader.readtext(image)
    
    def txt(self, image):
        self.results = self.extract(image)

        for bbox, text, confidence in self.results:
            print(f"Text: {text}")
            print(f"Confidence: {confidence:.3f}")
            
