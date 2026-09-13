import sys, numpy as np
sys.path.insert(0,'.')
from src.effects.effects import outline, glow, shadow
from src.effects.text import overlay_text
def test_outline_changes_boundary():
    img=np.zeros((40,40,3),np.uint8); a=np.zeros((40,40),np.float32); a[10:30,10:30]=1
    assert np.count_nonzero(outline(img,a)!=img)>0
def test_glow_and_shadow_change_pixels():
    img=np.full((40,40,3),100,np.uint8); a=np.zeros((40,40),np.float32); a[15:25,15:25]=1
    assert np.count_nonzero(glow(img,a)!=img)>0
    assert np.count_nonzero(shadow(img,a)!=img)>0
def test_text_overlay():
    img=np.zeros((80,160,3),np.uint8); assert np.count_nonzero(overlay_text(img,'TEST')!=img)>0
