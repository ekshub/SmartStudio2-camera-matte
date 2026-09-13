from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow,QLabel,QFileDialog,QDockWidget,QFormLayout,QWidget,QCheckBox,QSlider,QComboBox,QLineEdit,QPushButton,QMessageBox)
from PySide6.QtGui import QAction,QPixmap,QImage
from PySide6.QtCore import Qt
from ..utils.config import load_config
from ..runtime.pipeline import ProcessingPipeline

class MainWindow(QMainWindow):
    """Small but functional studio control surface backed by ProcessingPipeline."""
    def __init__(self):
        super().__init__(); self.setWindowTitle('SmartStudio Professional Studio'); self.resize(1280,800)
        self.preview=QLabel('Open an image to preview'); self.preview.setAlignment(Qt.AlignCenter); self.setCentralWidget(self.preview)
        self.cfg=load_config(Path(__file__).resolve().parents[2]/'configs/default.yaml'); self.cfg.rvm.checkpoint=str((Path(__file__).resolve().parents[2]/self.cfg.rvm.checkpoint).resolve()); self.pipe=ProcessingPipeline(self.cfg)
        act=QAction('Open image',self); act.triggered.connect(self.open_image); self.menuBar().addAction(act)
        dock=QDockWidget('Effects / Studio controls',self); panel=QWidget(); form=QFormLayout(panel)
        self.checks={}
        for name,label in [('outline','Outline'),('glow','Glow'),('shadow','Drop Shadow')]:
            c=QCheckBox(label); c.setChecked(getattr(self.cfg.effects,name)); c.toggled.connect(lambda v,n=name:setattr(self.cfg.effects,n,v)); form.addRow(c); self.checks[name]=c
        self.text=QLineEdit(''); self.text.textChanged.connect(lambda v:setattr(self.cfg.effects,'text',v)); form.addRow('Text',self.text)
        self.outline_width=QSlider(Qt.Horizontal); self.outline_width.setRange(1,12); self.outline_width.setValue(3); self.outline_width.valueChanged.connect(lambda v:setattr(self.cfg.effects,'outline_width',v)); form.addRow('Outline width',self.outline_width)
        self.glow_radius=QSlider(Qt.Horizontal); self.glow_radius.setRange(1,40); self.glow_radius.setValue(15); self.glow_radius.valueChanged.connect(lambda v:setattr(self.cfg.effects,'glow_radius',v)); form.addRow('Glow radius',self.glow_radius)
        self.effect_opacity=QSlider(Qt.Horizontal); self.effect_opacity.setRange(0,100); self.effect_opacity.setValue(40); self.effect_opacity.valueChanged.connect(self._set_opacity); form.addRow('Effect opacity',self.effect_opacity)
        self.scale=QSlider(Qt.Horizontal); self.scale.setRange(25,100); self.scale.setValue(round(self.pipe.processing_scale*100)); self.scale.valueChanged.connect(self._set_processing_scale); form.addRow('Processing %',self.scale)
        self.bg=QComboBox(); self.bg.addItems(['solid','blur','transparent','image','video']); self.bg.currentTextChanged.connect(lambda v:self._set_bg(v)); form.addRow('Background',self.bg)
        pick_bg=QPushButton('Choose background file'); pick_bg.clicked.connect(self.choose_background); form.addRow(pick_bg)
        pick_logo=QPushButton('Choose transparent logo'); pick_logo.clicked.connect(self.choose_logo); form.addRow(pick_logo)
        dock.setWidget(panel); self.addDockWidget(Qt.RightDockWidgetArea,dock)
    def _set_bg(self,kind): self.cfg.background.kind=kind; self.pipe.bg.kind=kind
    def _set_opacity(self,value):
        v=float(value)/100.; self.cfg.effects.glow_opacity=v; self.cfg.effects.shadow_opacity=v; self.cfg.effects.outline_opacity=v
    def _set_processing_scale(self,value):
        scale=float(value)/100.; self.cfg.runtime.processing_scale=scale; self.pipe.set_processing_scale(scale)
    def choose_background(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose background','','Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)')
        if path:
            self.cfg.background.value=path; self.pipe.bg.value=path; self.pipe.bg.close()
            self.pipe.bg=type(self.pipe.bg)(self.cfg.background.kind,path,self.cfg.background.loop)
    def choose_logo(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose logo','','PNG (*.png)')
        if path: self.cfg.effects.logo=path
    def open_image(self):
        path,_=QFileDialog.getOpenFileName(self,'Open image','','Images (*.png *.jpg *.jpeg)')
        if not path:return
        try:
            # cv2.imread may fail on non-ASCII Windows paths. Reading bytes
            # first and decoding them supports Chinese user/folder names.
            frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('无法读取该图片。请选择有效的 JPG、JPEG 或 PNG 文件。')
            out,_,_,_=self.pipe.process(frame)
            rgb=cv2.cvtColor(out,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
            self.preview.setPixmap(QPixmap.fromImage(QImage(rgb.data,w,h,3*w,QImage.Format_RGB888).copy()).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        except Exception as exc:
            QMessageBox.critical(self,'处理失败',str(exc))
