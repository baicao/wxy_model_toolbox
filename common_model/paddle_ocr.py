import os
from paddleocr import PaddleOCR
import cv2


class Paddle_OCR:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(lang="ch", use_gpu=False)

    def ocr_simple(self, image_input, **kwargs):
        try:
            if isinstance(image_input, str) and os.path.exists(image_input):
                file_content = self.read_file(image_input)
            elif "input_type" in kwargs and "opencv" == kwargs["input_type"]:
                file_content = cv2.imencode(".jpg", image_input)[1].tobytes()
            result = self.ocr.ocr(file_content, cls=False)
            if len(result) > 0:
                text = result[0][-1][1][0]
                return True, text
            else:
                return False, "empty"
        except:
            return False, "error"
