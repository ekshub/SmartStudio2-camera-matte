"""Key-frame propagation for real-time video matting.

RVM remains the quality model.  On intermediate frames we warp the last
key-frame foreground and alpha with a low-resolution Farneback flow.  This is
the same practical idea used by real-time video effects systems: expensive
semantic inference establishes a key frame, while motion compensation fills
the frames between key frames.
"""
from __future__ import annotations
import cv2
import numpy as np


class KeyframePropagator:
    def __init__(self, stride: int = 1, flow_scale: float = .25):
        self.stride = max(1, int(stride))
        self.flow_scale = float(np.clip(flow_scale, .1, .5))
        self.index = 0
        self.prev_frame = None
        self.prev_fg = None
        self.prev_alpha = None

    def reset(self):
        self.index = 0
        self.prev_frame = self.prev_fg = self.prev_alpha = None

    def set_stride(self, stride: int):
        stride = max(1, int(stride))
        if stride != self.stride:
            self.reset()
        self.stride = stride

    def should_infer(self):
        return self.prev_frame is None or self.index % self.stride == 0

    def propagate(self, frame):
        """Return warped (fg, alpha), or None when a key frame is required."""
        if self.prev_frame is None or self.prev_fg is None or self.prev_alpha is None:
            return None
        h, w = frame.shape[:2]
        sh, sw = max(32, int(h * self.flow_scale)), max(32, int(w * self.flow_scale))
        g0 = cv2.cvtColor(cv2.resize(self.prev_frame, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        # Avoid smearing across cuts or large exposure changes.
        if float(np.mean(cv2.absdiff(g0, g1))) / 255.0 > .18:
            return None
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, .5, 2, 15, 2, 5, 1.1, 0)
        yy, xx = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
        mapx = (xx - flow[..., 0]).astype(np.float32)
        mapy = (yy - flow[..., 1]).astype(np.float32)
        fg_s = cv2.resize(self.prev_fg, (sw, sh), interpolation=cv2.INTER_AREA)
        fg_w = cv2.remap(fg_s, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a_s = cv2.resize(self.prev_alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        a_w = cv2.remap(a_s, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(fg_w, (w, h), interpolation=cv2.INTER_LINEAR), cv2.resize(a_w, (w, h), interpolation=cv2.INTER_LINEAR)

    def update(self, frame, fg, alpha):
        self.prev_frame = frame.copy()
        self.prev_fg = fg.copy()
        self.prev_alpha = alpha.copy()
        self.index += 1
