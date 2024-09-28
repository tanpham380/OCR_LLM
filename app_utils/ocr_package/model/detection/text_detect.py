import numpy as np

from app_utils.ocr_package.model.detection.rapidocr_onnxruntime import RapidOCR

class TextDect_withRapidocr():
    def __init__(self, text_score: float = 0.6, det_use_cuda: bool = False, det_model_path: str = "", rec_model_path: str = "") -> None:
        self.engine = RapidOCR(
            text_score=text_score,
            det_use_cuda=det_use_cuda,
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            rec_img_shape=[3, 48, 320],
        )

    def detect(self, img_array: np.ndarray):
        result, _ = self.engine(img_array)
        return result

