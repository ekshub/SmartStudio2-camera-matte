from queue import Queue, Full
class LatestQueue:
    def __init__(self): self.q=Queue(maxsize=1)
    def put_latest(self,item):
        try:self.q.put_nowait(item)
        except Full:
            try:self.q.get_nowait()
            except Exception:pass
            self.q.put_nowait(item)
    def get(self,timeout=None): return self.q.get(timeout=timeout)
