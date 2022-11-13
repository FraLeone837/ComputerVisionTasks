import cv2
import numpy as np
import math

from interpolation import interpolate


def vectorLength(vector):
  return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def pointsDistance(point1, point2):
  return vectorLength((point1[0] - point2[0],point1[1] - point2[1]))

 #if the point is outside the image, reposition it inside
def clamp(value, minimum, maximum):
  return max(min(value,maximum),minimum)

def warp(image, points):

  height, width, ch = image.shape
  warpedImage = np.zeros((height, width, 3), np.uint8)
  warpedImage2 = np.zeros((height, width, 3), np.uint8)


  for y in range(0, height):
    for x in range(0, width):

      offset = [0,0]

      for point in points:
        pointPosition = (point[0] + point[2],point[1] + point[3])
        shift_vector = (point[2],point[3])

        #Inverse distance weighting  -> need to be tested to find possible improvement
        #helper = 1.0 / (3 * (pointsDistance((x,y),pointPosition) / vectorLength(shift_vector)) ** 4 + 1)
        helper = 1.0 / ((pointsDistance((x, y), pointPosition) / vectorLength(shift_vector)) ** 4 + 1)

        offset[0] -= helper * shift_vector[0]
        offset[1] -= helper * shift_vector[1]

      offset[0] = round(offset[0], 2)
      offset[1] = round(offset[1], 2)

      coords2 = (clamp(x + int(offset[0]),0, width - 1),clamp(y + int(offset[1]),0,height - 1))
      warpedImage2[y, x] = image[coords2[1], coords2[0]]

      coords = (clamp(x + offset[0], 0, width - 1), clamp(y + offset[1], 0, height - 1))
      warpedImage[y, x] = interpolate(image, coords)


      #add interpolation





# highlight the warped points in the original image and in the warped image
  cv2.imshow('warpedImage', warpedImage)
  cv2.imshow('warpedImage2', warpedImage2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  for point in points:
      x = point[0]
      y = point[1]
      for i in range(y - 2, y + 2):
          for j in range(x - 2, x + 2):
              image[i, j] = (0, 0, 255)

      pointPosition = (point[0] + point[2], point[1] + point[3])
      x = pointPosition[0]
      y = pointPosition[1]
      for i in range(y-2, y+2):
          for j in range(x-2, x+2):
              warpedImage[i, j] = (0, 0, 255)

  return warpedImage

img = cv2.imread('imageTest.jpg')
image = warp(img,[(50,50,100,100), (600,600,-100,0), (100,500,0,-100)]) #,(1000,700,-500,-300)])

cv2.imshow('originalImage',img)
cv2.imshow('warpedImage',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
