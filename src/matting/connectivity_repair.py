"""Conservative Alpha connectivity repair for small/occluded people."""
from __future__ import annotations
import cv2, numpy as np

def repair_alpha(alpha: np.ndarray, strength: float=.35) -> np.ndarray:
    a=np.clip(alpha.astype(np.float32),0,1); s=float(np.clip(strength,0,1))
    # Close only narrow cracks in a soft foreground mask.
    core=(a>.35).astype(np.uint8); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)); closed=cv2.morphologyEx(core,cv2.MORPH_CLOSE,k)
    fill=(closed.astype(bool)&(a<.35)).astype(np.float32)
    # Restrict filling to enclosed/interior pixels, never the exterior ring.
    num,lab,stats,_=cv2.connectedComponentsWithStats(closed,8); interior=np.zeros_like(fill)
    for i in range(1,num):
        x,y,w,h,area=stats[i]
        if area<20: continue
        comp=(lab==i).astype(np.uint8); contour=np.zeros_like(comp); cv2.drawContours(contour,cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,1,1)
        interior[(comp>0)&(contour==0)]=1
    fill*=interior
    return np.clip(a+fill*(.55-a)*s,0,1).astype(np.float32)
