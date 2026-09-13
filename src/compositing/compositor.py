import cv2, numpy as np
class Compositor:
    def compose(self,fg,bg,alpha):
        # OpenCV's native blend avoids several full-frame NumPy temporaries.
        a=np.ascontiguousarray(np.clip(alpha,0,1).astype(np.float32)); return cv2.blendLinear(fg,bg,a,1.0-a)
