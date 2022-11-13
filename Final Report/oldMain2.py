import numpy as np
import cv2 as cv
import argparse
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.image)
videoName = input("Input video: ")
videoName = videoName + ".avi"
cap = cv.VideoCapture(videoName)
# cap = cv.VideoCapture('video2bis.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

first = True

while(1):
    try:
        ret,frame = cap.read()

        #only for videos from the smartphone
        # height, width, layers = frame.shape
        # new_h = height / 2
        # new_w = width / 2
        # frame = cv.resize(frame, (new_w, new_h))
        # width = int(frame.shape[1] * 0.6)
        # height = int(frame.shape[0] * 0.6)
        # dim = (width, height)
        # frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)


        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)

#show the first frame for a longer period
        if(first):
            img = cv.add(frame, mask)
            cv.imshow('frame', img)
            cv.waitKey(0)
            first = False
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            continue

        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    except:
        cap.release()
        cv.imshow('frame', img)
        k = cv.waitKey(0)
        break
cv.destroyAllWindows()
