import cv2, numpy as np

def overlay_logo(frame, logo, x=20, y=20, scale=1.0, opacity=1.0):
    if logo is None or scale <= 0: return frame
    h,w=logo.shape[:2]; logo=cv2.resize(logo,(max(1,int(w*scale)),max(1,int(h*scale)))); h,w=logo.shape[:2]
    x,y=int(x),int(y); x0,y0=max(0,x),max(0,y); x1,y1=min(frame.shape[1],x+w),min(frame.shape[0],y+h)
    if x0>=x1 or y0>=y1: return frame
    roi=frame[y0:y1,x0:x1]; logo=logo[y0-y:y1-y,x0-x:x1-x]; op=float(np.clip(opacity,0,1))
    if logo.ndim==3 and logo.shape[2]>=4:
        a=(logo[...,3:4].astype(np.float32)/255.)*op; roi[:]=((1-a)*roi+a*logo[...,:3]).astype(np.uint8)
    else: roi[:]=((1-op)*roi+op*logo[...,:3]).astype(np.uint8)
    return frame
