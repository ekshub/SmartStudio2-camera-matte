import numpy as np
def suppress_spill(foreground: np.ndarray, alpha: np.ndarray, strength=.6) -> np.ndarray:
    x=foreground.astype(np.float32)/255.
    m=np.maximum(x[...,0],x[...,2])
    eg=np.maximum(0,x[...,1]-m)
    # Green contamination can push an edge pixel to alpha ~= 1.0, so testing
    # only the fractional-alpha band leaves a visible lime halo. Decontaminate
    # any sufficiently opaque pixel that still has clear green dominance.
    edge=(alpha>.02)&((alpha<.98)|(eg>.06))
    x[...,1][edge]-=float(np.clip(strength,0,1.2))*eg[edge]
    return np.clip(x*255,0,255).astype(np.uint8)
