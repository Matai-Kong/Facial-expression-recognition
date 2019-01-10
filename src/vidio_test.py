import numpy as np
import cv2
import time


cap = cv2.VideoCapture(0)

'''
while (cap.isOpened()):
    start=time.time()
    ret, frame = cap.read()
    if ret == True:     
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('iframe', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    end=time.time()
    print('time = {}'.format(end-start))

'''

while True:
    start=time.time()
    ret, frame = cap.read()
    cv2.imshow('iframe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end=time.time()
    print('time = {}'.format(end-start))

cap.release()
cv2.destroyAllWindows()
