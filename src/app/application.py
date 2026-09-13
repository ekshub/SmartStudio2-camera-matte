import logging
from ..utils.config import load_config
from ..runtime.pipeline import ProcessingPipeline
class SmartStudioApplication:
    """Dependency-injection façade used by Qt UI; UI never touches the model directly."""
    def __init__(self,config_path='configs/default.yaml'):
        self.config=load_config(config_path); self.pipeline=ProcessingPipeline(self.config); logging.info('application initialized')
