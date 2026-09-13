from dataclasses import dataclass
import time, numpy as np
@dataclass
class FramePacket:
    frame_id:int; timestamp:float; bgr:np.ndarray; width:int; height:int; source_type:str
    @classmethod
    def create(cls,i,bgr,source_type): return cls(i,time.time(),bgr,bgr.shape[1],bgr.shape[0],source_type)
