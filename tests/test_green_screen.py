import cv2
import numpy as np

from src.matting.scene_detector import SceneDetector
from src.matting.chroma_key import ChromaKey


def test_green_plate_detects_chroma_mode():
    frame = np.full((240, 320, 3), (40, 230, 145), np.uint8)  # lime screen
    cv2.rectangle(frame, (110, 80), (210, 230), (40, 40, 180), -1)
    det = SceneDetector()
    assert det.mode(frame) == "chroma"


def test_chroma_key_removes_screen_keeps_subject():
    frame = np.full((120, 160, 3), (40, 230, 145), np.uint8)
    frame[35:95, 70:100] = (40, 40, 180)
    alpha = ChromaKey()(frame)
    assert float(alpha[10, 10]) < .1
    assert float(alpha[60, 85]) > .8
