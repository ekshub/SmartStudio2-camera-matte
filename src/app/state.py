from dataclasses import dataclass
@dataclass
class RuntimeState:
    source_type:str='image'; mode:str='auto'; downsample_ratio:float=.25; target_fps:float=30.; recording:bool=False
