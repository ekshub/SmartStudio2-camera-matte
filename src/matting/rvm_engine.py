from __future__ import annotations
import logging, torch, numpy as np, cv2
from pathlib import Path
log=logging.getLogger(__name__)
class RVMEngine:
    def __init__(self, checkpoint="models/rvm_mobilenetv3.pth", device=None, fp16=True, downsample_ratio=.25, inference_scale=1.0, variant="mobilenetv3"):
        self.checkpoint=Path(checkpoint); self.variant=str(variant).lower();
        # Keep old configs working when only the checkpoint path is changed.
        if "resnet50" in self.checkpoint.stem.lower(): self.variant="resnet50"
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); self.fp16=fp16 and self.device.type=="cuda"; self.ratio=downsample_ratio; self.inference_scale=float(np.clip(inference_scale,.25,1.0)); self.model=None; self.rec=[None]*4
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    def load_model(self):
        if not self.checkpoint.exists(): raise FileNotFoundError(f"RVM checkpoint not found. Please place rvm_mobilenetv3.pth in models/")
        try:
            from .rvm_model.model import MattingNetwork
            self.model=MattingNetwork(self.variant).to(self.device).eval()
            if self.device.type == "cuda": self.model.to(memory_format=torch.channels_last)
            state=torch.load(self.checkpoint,map_location=self.device,weights_only=True); self.model.load_state_dict(state,strict=True)
            log.info("Loaded checkpoint %s",self.checkpoint)
        except Exception as exc:
            log.warning("RVM model unavailable (%s); using deterministic fallback",exc); self.model=torch.nn.Identity().to(self.device).eval()
    def reset_state(self): self.rec=[None,None,None,None]
    def set_downsample_ratio(self,r): self.ratio=float(np.clip(r,.125,.4))
    def set_inference_scale(self,scale): self.inference_scale=float(np.clip(scale,.25,1.0)); self.reset_state()
    def set_precision(self,fp16): self.fp16=bool(fp16 and self.device.type=="cuda")
    def warmup(self,shape=(720,1280)):
        self.infer(np.zeros((*shape,3),np.uint8))
    def infer(self, frame: np.ndarray):
        if self.model is None: self.load_model()
        if self.model.__class__.__name__ == 'Identity':
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); a=cv2.GaussianBlur((hsv[...,1]>35).astype(np.float32),(0,0),3); return frame,a,self.rec
        orig_h,orig_w=frame.shape[:2]; work=frame if self.inference_scale>=.999 else cv2.resize(frame,(max(32,int(orig_w*self.inference_scale)),max(32,int(orig_h*self.inference_scale))),interpolation=cv2.INTER_AREA)
        rgb=cv2.cvtColor(work,cv2.COLOR_BGR2RGB); src=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
        src=src.to(self.device, dtype=torch.float16 if self.fp16 else torch.float32, non_blocking=True).div_(255)
        if self.device.type == "cuda": src=src.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.float16,enabled=self.fp16): out=self.model(src,*self.rec,downsample_ratio=self.ratio)
        fg=(out[0][0].permute(1,2,0).clamp(0,1).float().cpu().numpy()*255).astype(np.uint8); alpha=out[1][0,0].float().cpu().numpy(); self.rec=list(out[2:]); fg=cv2.cvtColor(fg,cv2.COLOR_RGB2BGR)
        if self.inference_scale<.999: fg=cv2.resize(fg,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR); alpha=cv2.resize(alpha,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
        return fg,alpha,self.rec
    def get_device_info(self): return {'device':str(self.device),'fp16':self.fp16,'vram_mb':torch.cuda.memory_allocated()/2**20 if self.device.type=='cuda' else 0}
