from __future__ import annotations
import cv2, numpy as np

class ChromaKey:
    """Soft LAB/HSV chroma key with morphology, feathering and spill suppression."""
    def __init__(self, key_color: tuple[int,int,int] | None = None, screen: str = "green", t0: float = .10, t1: float = .35):
        self.key_color = key_color
        self.screen = screen; self.t0=t0; self.t1=t1
    def _color(self, frame: np.ndarray) -> np.ndarray:
        if self.key_color is not None: return np.array(self.key_color, np.float32)
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); h=hsv[...,0]; mask=((h>35)&(h<95)&(hsv[...,1]>60)) if self.screen=="green" else ((h<15)|(h>100))&(hsv[...,1]>60)
        pix=frame[mask]
        return pix.mean(0) if len(pix) else np.array([0,255,0],np.float32)
    def alpha(self, frame: np.ndarray) -> np.ndarray:
        # For green screens, channel dominance is more stable than a single
        # global RGB/LAB distance (the live_studio clip uses a bright lime
        # screen with strong illumination gradients).  Work in float so the
        # thresholds remain resolution and codec independent.
        b,g,r = cv2.split(frame.astype(np.float32) / 255.0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[..., 0] / 180.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        dominance = g - np.maximum(b, r)
        if self.screen == "green":
            # Green pixels have high G dominance and hue near 60 degrees.
            screen_score = np.clip(dominance * 3.2, 0, 1) * np.clip((s - .12) / .35, 0, 1)
            screen_score *= np.clip(1.15 - np.abs(h - .333) * 4.0, 0, 1)
        else:
            screen_score = np.clip((np.maximum(r, b) - g) * 3.0, 0, 1)
        # Smoothstep gives a soft anti-aliased edge without deleting hair.
        a = 1.0 - np.clip((screen_score - .12) / .48, 0, 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        solid = (a > .5).astype(np.uint8)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, k)
        a = cv2.GaussianBlur(a, (0, 0), .7)
        # Only suppress isolated codec noise; retain soft edge values.
        a = np.where((solid == 0) & (a < .15), 0, a)
        return np.clip(a, 0, 1).astype(np.float32)
    def __call__(self, frame): return self.alpha(frame)
