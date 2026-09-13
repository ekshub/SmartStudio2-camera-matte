from dataclasses import dataclass
@dataclass
class RuntimeScheduler:
    target_fps: float=30; ratio: float=.25; mode: str='automatic'; margin: float=2; low_windows:int=0; high_windows:int=0
    def update(self,fps):
        if self.mode!='automatic': return self.ratio
        if fps<self.target_fps-self.margin:self.low_windows+=1; self.high_windows=0
        elif fps>self.target_fps+self.margin:self.high_windows+=1; self.low_windows=0
        else:self.low_windows=self.high_windows=0
        if self.low_windows>=3:self.ratio=max(.125,self.ratio-.025); self.low_windows=0
        if self.high_windows>=8:self.ratio=min(.4,self.ratio+.025); self.high_windows=0
        return self.ratio
