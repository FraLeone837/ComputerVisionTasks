import os

import cv2

vidcap = cv2.VideoCapture("originalVideos/edgeHorizontalMovement.mp4")
success, image = vidcap.read()
count = 0

while success:
    width = int(image.shape[1] * 0.4)
    height = int(image.shape[0] * 0.4)
    dim = (width, height)
    resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # cv2.imwrite("%03d.jpg" % count, resize)

    path = 'images4'
    cv2.imwrite(os.path.join(path, "%03d.jpg" % count), resize)
    success, image = vidcap.read()
    count += 1