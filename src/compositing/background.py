from __future__ import annotations
import cv2, numpy as np
from pathlib import Path
class BackgroundProvider:
    def __init__(self, kind='solid', value=(40,40,40), loop=True):
        self.kind,self.value,self.loop,self.cap=kind,value,loop,None
        self.image=None
        if self.kind=='image':
            image_path=Path(str(self.value))
            if image_path.is_file():
                self.image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if self.kind=='video': self.cap=cv2.VideoCapture(str(self.value))
    def get(self, frame, index=0):
        h,w=frame.shape[:2]
        if self.kind=='solid':
            color = self.value
            if isinstance(color, str):
                try:
                    color = tuple(int(x.strip()) for x in color.strip('[]()').split(','))
                except (ValueError, TypeError):
                    color = (40, 40, 40)
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                color = (40, 40, 40)
            return np.full((h,w,3),color,np.uint8)
        if self.kind=='blur': return cv2.GaussianBlur(frame,(0,0),15)
        if self.kind=='transparent': return np.zeros((h,w,3),np.uint8)
        if self.kind=='image':
            image=self.image
            if image is not None and image.size:
                return cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
            # Keep the live worker alive if a file is temporarily unreadable
            # (for example while it is being replaced on disk).
            return np.full((h,w,3),(40,40,40),np.uint8)
        if self.kind=='video' and self.cap is not None:
            ok,bg=self.cap.read()
            if not ok and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,bg=self.cap.read()
            if ok and bg is not None: return cv2.resize(bg,(w,h),interpolation=cv2.INTER_AREA)
        return np.full((h,w,3),(40,40,40),np.uint8)

    def close(self):
        if self.cap is not None: self.cap.release()
