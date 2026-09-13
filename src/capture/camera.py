import cv2
from .base import FramePacket
class CameraSource:
    def __init__(self,index=0,width=1280,height=720):
        self.cap=cv2.VideoCapture(index); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); self.i=0
    def read(self):
        ok,f=self.cap.read()
        if not ok:return None
        p=FramePacket.create(self.i,f,'camera'); self.i+=1; return p
    def close(self): self.cap.release()
