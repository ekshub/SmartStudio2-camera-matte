import numpy as np, cv2
from src.matting.temporal_stabilizer import TemporalAlphaStabilizer

def test_temporal_stabilizer_range_and_reset():
    s=TemporalAlphaStabilizer(.4); f=np.zeros((64,64,3),np.uint8); a=np.zeros((64,64),np.float32); a[20:40,20:40]=1
    assert np.array_equal(s.stabilize(a,f),a)
    b=np.roll(a,1,axis=1); out=s.stabilize(b,f); assert 0<=float(out.min())<=float(out.max())<=1
    s.reset(); assert s.prev_alpha is None
