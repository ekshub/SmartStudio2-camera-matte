import numpy as np
from src.matting.temporal_propagation import KeyframePropagator


def test_keyframe_stride_and_reset():
    p = KeyframePropagator(2)
    assert p.should_infer()
    frame = np.zeros((32, 32, 3), np.uint8)
    fg = np.full_like(frame, 120)
    a = np.ones((32, 32), np.float32)
    p.update(frame, fg, a)
    assert not p.should_infer()
    p.update(frame, fg, a)
    assert p.should_infer()
    p.reset()
    assert p.should_infer()
