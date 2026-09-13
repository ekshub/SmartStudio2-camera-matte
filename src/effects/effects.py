"""Lightweight, deterministic studio effects operating on BGR/alpha layers."""
import cv2
import numpy as np

def outline(img, a, width=3, opacity=.8, color=(255,255,255)):
    width=max(1,int(width)); k=np.ones((2*width+1,2*width+1),np.uint8)
    m=np.clip(a.astype(np.float32),0,1)
    edge=np.clip(cv2.dilate(m,k)-cv2.erode(m,k),0,1)
    edge=cv2.GaussianBlur(edge,(0,0),max(.4,width*.18))*float(np.clip(opacity,0,1))
    c=np.asarray(color,np.float32).reshape(1,1,3)
    return np.clip(img.astype(np.float32)*(1-edge[...,None])+c*edge[...,None],0,255).astype(np.uint8)

def glow(img, a, color=(255,255,255), radius=15, opacity=.4):
    m=np.clip(a.astype(np.float32),0,1)
    g=cv2.GaussianBlur((m*255).astype(np.uint8),(0,0),max(.1,float(radius))).astype(np.float32)/255.
    g=np.clip(g-m,0,1)*float(np.clip(opacity,0,1)); c=np.asarray(color,np.float32).reshape(1,1,3)
    return np.clip(img.astype(np.float32)*(1-g[...,None])+c*g[...,None],0,255).astype(np.uint8)

def shadow(img, a, dx=8, dy=8, blur=12, opacity=.4, color=(0,0,0)):
    h,w=a.shape[:2]; M=np.float32([[1,0,dx],[0,1,dy]])
    m=cv2.warpAffine((np.clip(a,0,1)*255).astype(np.uint8),M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    m=cv2.GaussianBlur(m,(0,0),max(.1,float(blur))).astype(np.float32)/255.*float(np.clip(opacity,0,1))
    c=np.asarray(color,np.float32).reshape(1,1,3)
    return np.clip(img.astype(np.float32)*(1-m[...,None])+c*m[...,None],0,255).astype(np.uint8)
