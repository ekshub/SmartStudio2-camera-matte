import numpy as np
from src.matting.connectivity_repair import repair_alpha

def test_repair_fills_enclosed_crack_without_growing_exterior():
    a=np.zeros((40,40),np.float32); a[8:32,8:32]=1; a[18:22,18:22]=0
    out=repair_alpha(a,1.0)
    assert out[20,20]>.4
    assert out[2,2]==0
