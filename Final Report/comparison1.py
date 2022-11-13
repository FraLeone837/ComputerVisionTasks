import numpy as np
import cv2 as cv

import argparse
import sys

stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)
# image1 = cv.imread('photos/translation/4a.jpeg')
# image2 = cv.imread('photos/translation/4b.jpeg')
# image3 = cv.imread('photos/translation/4c.jpeg')
# image4 = cv.imread('photos/translation/4d.jpeg')
# imgs = (image1,image2,image3,image4)

image1 = cv.imread('photos/perspective/1a.jpeg')
image2 = cv.imread('photos/perspective/2b.jpeg')
imgs = (image1,image2)
status, pano = stitcher.stitch(imgs)

if status != cv.Stitcher_OK:
    print("Can't stitch images, error code = %d" % status)
    sys.exit(-1)

cv.imwrite('result.jpg', pano)
print("stitching completed successfully.")

print('Done')