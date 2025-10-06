import cv2
import numpy as np
from preprocess import adjust_gamma

def get_webcam_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Auto-adjust brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50:
            frame = adjust_gamma(frame, gamma=0.5)
        elif brightness > 200:
            frame = adjust_gamma(frame, gamma=1.5)
        
        yield frame
    
    cap.release()
    cv2.destroyAllWindows()




