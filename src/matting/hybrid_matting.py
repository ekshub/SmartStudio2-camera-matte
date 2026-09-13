import numpy as np
class HybridMatting:
    def __init__(self,max_weight=.8): self.max_weight=max_weight
    def fuse(self, alpha_r, alpha_c, screen_confidence):
        u=4*alpha_r*(1-alpha_r); g=np.exp(-((1-alpha_c)**2)/(2*.25**2)); w=np.clip(screen_confidence*u*g,0,self.max_weight)
        return np.clip((1-w)*alpha_r+w*alpha_c,0,1).astype(np.float32)
