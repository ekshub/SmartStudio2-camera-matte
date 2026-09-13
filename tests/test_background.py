import numpy as np

from src.compositing.background import BackgroundProvider


def test_solid_background_accepts_browser_serialized_color():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    out = BackgroundProvider("solid", "[30, 60, 100]").get(frame)
    assert out.shape == frame.shape
    assert tuple(out[0, 0]) == (30, 60, 100)
