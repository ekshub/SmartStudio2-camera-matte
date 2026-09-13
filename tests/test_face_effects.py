from pathlib import Path

import cv2
import numpy as np
import pytest

from src.effects.face_effects import FaceEffectEngine


def test_face_accessory_tracks_face_and_preserves_frame_shape():
    frame = cv2.imread(str(Path(__file__).parents[1] / "assets" / "demo" / "lena.jpg"))
    if frame is None:
        pytest.skip("demo face image is unavailable")
    engine = FaceEffectEngine(max_faces=1, interval=1)
    if not engine.available:
        pytest.skip("mediapipe is not installed")
    for style in ("glasses", "sunglasses", "hat"):
        out = engine.apply(frame.copy(), None, style)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert len(engine._tracks) == 1
        assert np.any(out != frame)
    engine.close()
