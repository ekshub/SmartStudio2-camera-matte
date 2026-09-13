from __future__ import annotations
import argparse, logging, cv2
from pathlib import Path
from src.utils.config import load_config
from src.runtime.pipeline import ProcessingPipeline
from src.output.snapshot import save
def read_image(path):
    """Read image paths containing non-ASCII characters on Windows."""
    import numpy as np
    try: return cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    except (OSError,ValueError): return None
def run_image(path, out):
    cfg=load_config(Path(__file__).parent/'configs/default.yaml'); cfg.rvm.checkpoint=str(Path(__file__).parent/cfg.rvm.checkpoint); p=ProcessingPipeline(cfg); frame=read_image(path)
    if frame is None: raise FileNotFoundError(f'Cannot read image: {path}')
    image,alpha,mode,lat=p.process(frame); save(image,out); cv2.imwrite(str(Path(out).with_name(Path(out).stem+'_alpha.png')),(alpha*255).astype('uint8')); logging.info('mode=%s latency=%.1fms',mode,lat)
def run_video(path, out):
    cfg=load_config(Path(__file__).parent/'configs/default.yaml'); cfg.rvm.checkpoint=str(Path(__file__).parent/cfg.rvm.checkpoint); p=ProcessingPipeline(cfg); cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(f'Cannot open video: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30; w,h=int(cap.get(3)),int(cap.get(4)); Path(out).parent.mkdir(parents=True,exist_ok=True); wr=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not wr.isOpened(): cap.release(); raise RuntimeError(f'Cannot create output video: {out}')
    while True:
        ok,frame=cap.read()
        if not ok: break
        image,_,_,_=p.process(frame); wr.write(image)
    cap.release(); wr.release()
def main():
    log_dir=Path(__file__).parent/'logs'; log_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,filename=log_dir/'smartstudio.log',format='%(asctime)s %(levelname)s %(message)s')
    ap=argparse.ArgumentParser(description='SmartStudio image/video processor')
    ap.add_argument('--image',metavar='PATH',help='input image path')
    ap.add_argument('--video','--VIDEO',dest='video',metavar='PATH',help='input video path (a path is required)')
    ap.add_argument('--output',default='output/result.png',help='output image or video path')
    ap.add_argument('--gui',action='store_true',help='open the Qt image preview')
    a=ap.parse_args()
    if a.image: run_image(a.image,a.output); return
    if getattr(a,'video',None): run_video(a.video,a.output); return
    if a.gui:
        from PySide6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
        app=QApplication([]); w=MainWindow(); w.show(); app.exec()
    else: ap.print_help()
if __name__=='__main__': main()
