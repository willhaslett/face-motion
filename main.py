import os
import numpy as np
import cv2
import csv

def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

output_data = []
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
    num_frames = 0
    total_mag = 0
    out_data = []
    # Only analyze the first two seconds of video
    while(num_frames < 60):
        ret, frame2 = cap.read()
        try:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        except:
            break
        num_frames += 1
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        total_mag = total_mag + np.sum(mag)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        out.write(rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
    output_data.append([input_video, str(num_frames), str(int(total_mag) / num_frames)])
    cap.release()
    out.release()
csv_writer(output_data, 'output.csv')
