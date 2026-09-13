import cv2
def overlay_text(frame,text,size=1.0,x=20,y=40,opacity=1.0,color=(255,255,255),thickness=2):
    if not text: return frame
    layer=frame.copy(); cv2.putText(layer,str(text),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,float(size),tuple(map(int,color)),int(thickness),cv2.LINE_AA)
    return cv2.addWeighted(layer,float(max(0,min(1,opacity))),frame,1-float(max(0,min(1,opacity))),0)
