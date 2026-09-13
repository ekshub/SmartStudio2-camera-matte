from __future__ import annotations
import cv2, numpy as np
class SceneDetector:
    def __init__(self, low=.45, high=.65): self.low,self.high=low,high; self._hybrid=False
    def confidence(self, frame: np.ndarray) -> float:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); h,s,_=cv2.split(hsv); green=((h>35)&(h<95)&(s>50));
        bw, bh = max(1, frame.shape[1] // 10), max(1, frame.shape[0] // 10)
        sides = [green[:, :bw].mean(), green[:, -bw:].mean(),
                 green[:bh, :].mean(), green[-bh:, :].mean()]
        # The live_studio plate has a green upper field but people occupy the
        # lower/side borders. Use the strongest broad border band rather than
        # concatenating all four sides (which diluted the signal).
        return float(max(sides))
    def mode(self, frame):
        s=self.confidence(frame)
        if self._hybrid:
            if s<self.low:self._hybrid=False
        elif s>self.high:self._hybrid=True
        # A stable green border is a strong signal that a chroma-key source is
        # being used.  It gets a dedicated fast path; ordinary footage keeps
        # the RVM/hybrid behavior.
        return "chroma" if self._hybrid else "rvm"
