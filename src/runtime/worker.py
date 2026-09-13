import threading, logging
from .frame_queue import LatestQueue
class ProcessingWorker(threading.Thread):
    """Capture/inference worker that always drops stale frames (queue size one)."""
    def __init__(self,pipeline,frames:LatestQueue,results:LatestQueue): super().__init__(daemon=True); self.pipeline,self.frames,self.results=pipeline,frames,results; self.stop_event=threading.Event()
    def run(self):
        while not self.stop_event.is_set():
            try:
                packet=self.frames.get(.2); image,alpha,mode,lat=self.pipeline.process(packet.bgr); self.results.put_latest((packet,image,alpha,mode,lat))
            except Exception: logging.exception('processing worker exception')
    def stop(self): self.stop_event.set()
