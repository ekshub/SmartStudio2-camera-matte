import cv2, numpy as np


class Harmonizer:
    """Fast temporal color matching using thumbnail statistics."""

    def __init__(self, strength=.15, ema=.1, processing_scale=.25):
        self.strength = float(strength)
        self.ema = float(ema)
        self.state = None
        self.processing_scale = float(np.clip(processing_scale, .1, 1.0))

    def apply(self, fg, bg, alpha):
        h, w = fg.shape[:2]
        sw, sh = max(32, int(w * self.processing_scale)), max(32, int(h * self.processing_scale))
        fl = cv2.resize(fg, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        bl = cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        a_s = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        mask = a_s > .1
        if not np.any(mask):
            return fg
        fm, fs = fl[mask].mean(0), fl[mask].std(0) + 1e-3
        bm, bs = bl.mean((0, 1)), bl.std((0, 1)) + 1e-3
        self.state = bm if self.state is None else self.ema * bm + (1 - self.ema) * self.state
        ratio = bs / fs
        gain = (1 - self.strength) + self.strength * ratio
        bias = self.strength * (self.state - bm - fm * ratio)
        corrected = fg.astype(np.float32) * gain.reshape(1, 1, 3) + bias.reshape(1, 1, 3)
        return np.clip(corrected, 0, 255).astype(np.uint8)
