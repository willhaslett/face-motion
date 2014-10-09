import os
import numpy as np
import cv2
for input_video in os.listdir('videos-input'):
    print 'processing ' + input_video
    cap = cv2.VideoCapture('videos-input/' + input_video)
    # read the first video frame
    ret, frame1 = cap.read()
    # set up the output video file
    height, width, layers = frame1.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('videos-output/' + input_video.replace('mp4', 'avi'),
                           fourcc,
                           20.0,
                           (width,height))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        try:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        except:
            break
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        out.write(rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
    cap.release()
    out.release()
