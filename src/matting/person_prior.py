"""Optional lightweight person-segmentation prior for multi-person videos."""
from __future__ import annotations
import cv2, numpy as np

class PersonSegmentationPrior:
    def __init__(self, weights='yolo11n-seg.pt', conf=.15, imgsz=640, every_n=2, device=None):
        self.weights,self.conf,self.imgsz,self.every_n,self.device=weights,float(conf),int(imgsz),max(1,int(every_n)),device; self.model=None; self.index=0; self.last=None
    def _load(self):
        from ultralytics import YOLO
        self.model=YOLO(self.weights)
    def reset(self): self.index=0; self.last=None
    def mask(self, frame):
        if self.model is None: self._load()
        self.index+=1
        if self.last is not None and self.index % self.every_n != 1: return self.last
        kwargs={'source':frame,'conf':self.conf,'imgsz':self.imgsz,'classes':[0],'verbose':False}
        if self.device is not None: kwargs['device']=self.device
        result=self.model.predict(**kwargs)[0]; h,w=frame.shape[:2]; out=np.zeros((h,w),np.float32)
        if result.masks is not None:
            for m,box in zip(result.masks.data.detach().cpu().numpy(),result.boxes):
                score=float(box.conf.detach().cpu().item()); mm=cv2.resize(m.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR); out=np.maximum(out,mm*score)
        self.last=np.clip(out,0,1); return self.last
