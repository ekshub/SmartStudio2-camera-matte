import cv2
from .base import FramePacket
class VideoSource:
    def __init__(self,path): self.cap=cv2.VideoCapture(str(path)); self.i=0
    def read(self):
        ok,f=self.cap.read()
        if not ok: self.cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,f=self.cap.read()
        if not ok:return None
        p=FramePacket.create(self.i,f,'video'); self.i+=1; return p
    def close(self): self.cap.release()
