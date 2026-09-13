import time
from ..matting.rvm_engine import RVMEngine
from ..matting.chroma_key import ChromaKey
from ..matting.scene_detector import SceneDetector
from ..matting.hybrid_matting import HybridMatting
from ..matting.alpha_refine import refine_alpha
from ..matting.spill_suppression import suppress_spill
from ..compositing.background import BackgroundProvider
from ..compositing.compositor import Compositor
from ..compositing.harmonization import Harmonizer
from ..effects.effects import outline, glow, shadow
from ..effects.text import overlay_text
from ..effects.logo import overlay_logo
from ..effects.face_effects import FaceEffectEngine
from ..compositing.transform import transform_pair
from ..matting.temporal_stabilizer import TemporalAlphaStabilizer
from ..matting.connectivity_repair import repair_alpha
from ..matting.person_prior import PersonSegmentationPrior
from ..matting.temporal_propagation import KeyframePropagator
import cv2, numpy as np
class ProcessingPipeline:
    def __init__(self,config):
        self.cfg=config; self.processing_scale=float(max(.25,min(1.0,getattr(config.runtime,'processing_scale',1.0)))); self.rvm=RVMEngine(config.rvm.checkpoint,fp16=config.rvm.fp16,downsample_ratio=config.rvm.downsample_ratio,inference_scale=getattr(config.runtime,'inference_scale',1.0),variant=getattr(config.rvm,'variant','mobilenetv3')); self.ck=ChromaKey(); self.det=SceneDetector(config.matting.screen_low_threshold,config.matting.screen_high_threshold); self.hy=HybridMatting(config.matting.hybrid_max_weight); self.bg=BackgroundProvider(config.background.kind,config.background.value,config.background.loop); self.comp=Compositor(); self.harm=Harmonizer(config.harmonization.strength,config.harmonization.temporal_ema); self.temporal=TemporalAlphaStabilizer(getattr(config.runtime,'temporal_strength',.28)); self.propagator=KeyframePropagator(getattr(config.runtime,'inference_stride',1),.25); self.person_prior=PersonSegmentationPrior(config.runtime.person_prior_weights,config.runtime.person_prior_conf,every_n=config.runtime.person_prior_every_n) if getattr(config.runtime,'person_prior',False) else None; self.face_effects=FaceEffectEngine(getattr(config.effects,'face_effect_max_faces',2),getattr(config.effects,'face_effect_interval',2)); self._logo_cache=None; self._logo_path=''
    def reset_state(self):
        self.rvm.reset_state(); self.temporal.reset(); self.propagator.reset(); self.harm.state=None; self.face_effects.reset()
        if self.person_prior is not None: self.person_prior.reset()
    def set_processing_scale(self, scale):
        """Change the end-to-end processing scale between frames."""
        new_scale=float(max(.25,min(1.0,scale)))
        if abs(new_scale-self.processing_scale)>1e-6:
            # Recurrent tensors are resolution-dependent; avoid mixing states
            # from two quality profiles when switching during a live stream.
            self.rvm.reset_state()
            self.propagator.reset()
        self.processing_scale=new_scale
    def set_inference_stride(self, stride):
        self.propagator.set_stride(max(1, int(stride)))
        self.cfg.runtime.inference_stride=self.propagator.stride
    def process(self,frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError('Input frame is empty or invalid; check the image/video path and codec support.')
        t=time.perf_counter(); orig_h,orig_w=frame.shape[:2]; scale=self.processing_scale
        work=frame if scale >= .999 else cv2.resize(frame,(max(1,round(orig_w*scale)),max(1,round(orig_h*scale))),interpolation=cv2.INTER_AREA)
        mode=self.det.mode(work)
        # Dedicated chroma path: no neural inference is needed when a stable
        # green screen is detected. This is both sharper and substantially
        # faster than asking RVM to learn a matte from an already keyed plate.
        if mode == 'chroma' and getattr(self.cfg.matting, 'green_screen_optimized', True):
            inferred = True
            fg, ar = work, self.ck(work)
            a = ar
        else:
            inferred = self.propagator.should_infer()
            if inferred:
                fg,ar,_=self.rvm.infer(work)
            else:
                propagated=self.propagator.propagate(work)
                if propagated is None:
                    inferred = True
                    fg,ar,_=self.rvm.infer(work)
                else:
                    fg,ar=propagated
            ac=self.ck(work) if mode=='chroma' else ar
            a=self.hy.fuse(ar,ac,self.det.confidence(work)) if mode=='chroma' else ar
            a=refine_alpha(a)
        if self.person_prior is not None:
            pm=self.person_prior.mask(work)
            # When RVM misses an otherwise confidently detected person, raising
            # alpha alone is insufficient: RVM's foreground estimate is also
            # unreliable in that region. Recover color from the source frame
            # and use a soft person-mask alpha, while leaving confident RVM
            # foreground untouched.
            rescue=np.clip(pm*np.clip((.72-a)/.72,0,1),0,1)
            a=np.maximum(a,np.clip(pm*.92,0,1))
            fg=fg.astype(np.float32)*(1-rescue[...,None])+work.astype(np.float32)*rescue[...,None]
            fg=np.clip(fg,0,255).astype(np.uint8)
        # Stabilize after all matte sources have been fused.  This is
        # important when the detector is evaluated every N frames: the
        # detector mask and the RVM recurrent matte must share one temporal
        # state, otherwise rescue regions can flicker at the update cadence.
        # Key-frame propagation already performs motion compensation. Avoid a
        # second Farneback pass and expensive morphology on intermediate frames.
        if getattr(self.cfg.runtime,'connectivity_repair',True) and inferred and mode != 'chroma':
            a=repair_alpha(a,getattr(self.cfg.runtime,'connectivity_strength',.35))
        if getattr(self.cfg.runtime,'temporal_stabilization',True):
            # Apply temporal smoothing last: connectivity repair can otherwise
            # reintroduce a one-frame contour jump after the matte was stable.
            a=self.temporal.stabilize(a,work,use_flow=inferred)
        spill_strength = max(float(self.cfg.spill.strength), 1.15) if mode == 'chroma' else float(self.cfg.spill.strength)
        fg=suppress_spill(fg,a,spill_strength) if self.cfg.spill.enabled and mode=='chroma' else fg
        self.propagator.update(work,fg,a)
        e=self.cfg.effects
        # Restore source-resolution color detail for live rendering. RVM's
        # matte can be computed on a reduced canvas, but compositing an
        # upsampled low-resolution foreground makes faces, text and clothing
        # visibly soft.  The source frame is used as the color carrier while
        # the learned Alpha still controls foreground/background ownership.
        if scale < .999 and getattr(self.cfg.runtime, 'preserve_source_detail', False):
            fg=cv2.resize(frame,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
            a=cv2.resize(a,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
            work_render=frame
        else:
            work_render=work
        if e.foreground_scale != 1.0 or e.foreground_x or e.foreground_y:
            fg,a=transform_pair(fg,a,e.foreground_scale,e.foreground_x,e.foreground_y)
        bg=self.bg.get(work_render)
        # Harmonization statistics are temporally smoothed; applying them only
        # on semantic keyframes avoids repeating the same expensive color pass
        # on optical-flow frames while preserving a stable look.
        fg=self.harm.apply(fg,bg,a) if self.cfg.harmonization.enabled and inferred else fg
        if e.enabled:
            # Shadow/glow are background layers, then foreground and outline are composited above them.
            if e.shadow: bg=shadow(bg,a,e.shadow_dx,e.shadow_dy,e.shadow_blur,e.shadow_opacity,e.shadow_color)
            if e.glow: bg=glow(bg,a,color=e.glow_color,radius=e.glow_radius,opacity=e.glow_opacity)
        out=self.comp.compose(fg,bg,a)
        if getattr(e, 'face_effect_enabled', False) and getattr(e, 'face_accessory', 'none') not in {'', 'none'}:
            out=self.face_effects.apply(out, a, getattr(e, 'face_accessory', 'glasses'), getattr(e, 'face_effect_opacity', .95))
        if e.enabled:
            if e.outline: out=outline(out,a,e.outline_width,e.outline_opacity,e.outline_color)
            if e.logo:
                if self._logo_path != e.logo: self._logo_cache=cv2.imread(e.logo,cv2.IMREAD_UNCHANGED); self._logo_path=e.logo
                out=overlay_logo(out,self._logo_cache,e.logo_x,e.logo_y,e.logo_scale,e.logo_opacity)
            if e.text: out=overlay_text(out,e.text,e.text_size,e.text_x,e.text_y,e.text_opacity)
        if scale < .999 and not getattr(self.cfg.runtime, 'preserve_source_detail', False):
            out=cv2.resize(out,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
            a=cv2.resize(a,(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
        return out,a,mode,(time.perf_counter()-t)*1000
