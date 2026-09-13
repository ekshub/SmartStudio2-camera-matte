import cv2, numpy as np
def refine_alpha(alpha: np.ndarray) -> np.ndarray:
    out=alpha.copy(); band=(alpha>.02)&(alpha<.98)
    h,w=alpha.shape; sw,sh=max(32,w//2),max(32,h//2)
    small=cv2.resize(alpha.astype(np.float32),(sw,sh),interpolation=cv2.INTER_AREA)
    sm=cv2.bilateralFilter(small,5,0.08,3); sm=cv2.resize(sm,(w,h),interpolation=cv2.INTER_LINEAR); out[band]=sm[band]
    return np.clip(out,0,1)
