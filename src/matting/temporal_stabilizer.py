"""Temporal matte lock used by live camera compositors."""
from __future__ import annotations
import cv2
import numpy as np

class TemporalAlphaStabilizer:
    def __init__(self, strength: float = .28, flow_scale: float = .25):
        self.strength = float(np.clip(strength, 0, .8))
        self.flow_scale = float(np.clip(flow_scale, .1, .5))
        self.prev_alpha = None
        self.prev_gray = None
        self._history = []

    def reset(self):
        self.prev_alpha = None
        self.prev_gray = None
        self._history.clear()

    def _warp_previous(self, gray: np.ndarray, use_flow: bool) -> np.ndarray:
        if not use_flow:
            return self.prev_alpha
        h, w = gray.shape
        sh, sw = max(48, int(h * self.flow_scale)), max(48, int(w * self.flow_scale))
        g0 = cv2.resize(self.prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        if float(np.mean(cv2.absdiff(g0, g1))) / 255.0 > .22:
            return self.prev_alpha
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, .5, 3, 19, 3, 5, 1.1, 0)
        yy, xx = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
        mapx = (xx - flow[..., 0]).astype(np.float32)
        mapy = (yy - flow[..., 1]).astype(np.float32)
        small = cv2.resize(self.prev_alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        warped = cv2.remap(small, mapx, mapy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(warped, (w, h), interpolation=cv2.INTER_LINEAR)

    def stabilize(self, alpha: np.ndarray, frame: np.ndarray,
                  use_flow: bool = True) -> np.ndarray:
        alpha = np.clip(alpha.astype(np.float32), 0, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (self.prev_alpha is None or self.prev_alpha.shape != alpha.shape
                or self.prev_gray is None):
            self.prev_alpha = alpha.copy()
            self.prev_gray = gray.copy()
            self._history[:] = [alpha.copy()]
            return alpha
        reference = self._warp_previous(gray, use_flow)
        self._history.append(alpha.copy())
        if len(self._history) > 3:
            self._history.pop(0)
        if len(self._history) == 3:
            median = np.median(np.stack(self._history, axis=0), axis=0)
            uncertain = (alpha > .12) & (alpha < .88)
            alpha = np.where(uncertain, median, alpha).astype(np.float32)
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0)
        blend = np.clip(self.strength * (.55 + .45 * edge), .18, .86)
        candidate = alpha * (1.0 - blend) + reference * blend
        hard_fg = alpha >= .88
        hard_bg = alpha <= .08
        candidate[hard_fg] = np.maximum(candidate[hard_fg], .78)
        candidate[hard_bg] = np.minimum(candidate[hard_bg], .22)
        delta = candidate - self.prev_alpha
        limit = .10 + .10 * edge
        result = self.prev_alpha + np.clip(delta, -limit, limit)
        result = np.clip(result, 0, 1).astype(np.float32)
        self.prev_alpha = result
        self.prev_gray = gray.copy()
        return result
