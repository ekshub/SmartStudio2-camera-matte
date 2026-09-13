import cv2
class ImageSource:
    def __init__(self,path): self.path=path
    def read(self): return cv2.imread(str(self.path))
