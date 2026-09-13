import sys, numpy as np
sys.path.insert(0,'.')
from src.matting.chroma_key import ChromaKey
from src.matting.hybrid_matting import HybridMatting
from src.runtime.scheduler import RuntimeScheduler
from src.compositing.compositor import Compositor
def test_chroma_soft():
    f=np.zeros((20,20,3),np.uint8); f[:]=[0,255,0]; a=ChromaKey()(f); assert a.mean()<.2
def test_hybrid_range():
    a=HybridMatting().fuse(np.full((2,2),.5,np.float32),np.ones((2,2),np.float32),.8); assert np.all((a>=0)&(a<=1))
def test_scheduler_hysteresis():
    s=RuntimeScheduler(); old=s.ratio
    for _ in range(2): s.update(10)
    assert s.ratio==old
    s.update(10); assert s.ratio<old
def test_composite():
    out=Compositor().compose(np.full((2,2,3),255,np.uint8),np.zeros((2,2,3),np.uint8),np.ones((2,2),np.float32)); assert out.mean()==255
