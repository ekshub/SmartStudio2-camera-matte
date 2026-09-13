import cv2
def background_blur(frame,level='medium'):
    sigma={'low':5,'medium':10,'high':18}.get(level,10)
    return cv2.GaussianBlur(frame,(0,0),sigma)
