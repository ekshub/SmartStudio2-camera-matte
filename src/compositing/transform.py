import cv2, numpy as np
def transform(image,scale=1.0,x=0,y=0):
    h,w=image.shape[:2]; M=np.float32([[scale,0,x],[0,scale,y]]); return cv2.warpAffine(image,M,(w,h),borderMode=cv2.BORDER_TRANSPARENT)

def transform_pair(foreground, alpha, scale=1.0, x=0, y=0):
    """Apply one canvas-preserving transform to foreground and alpha together."""
    h,w=alpha.shape[:2]; M=np.float32([[scale,0,x],[0,scale,y]])
    fg=cv2.warpAffine(foreground,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    a=cv2.warpAffine(alpha,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return fg,np.clip(a,0,1).astype(np.float32)
