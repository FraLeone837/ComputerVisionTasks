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

def distance(point, vector):
  return math.sqrt((point[0]-vector[0]) ** 2 + (point[1]-vector[1]) ** 2)

def interpolate(image, point):
    height, width, ch = image.shape

    x = point[0]
    y = point[1]
    modX = round(x, 1) - int(x)
    modY = round(y, 1) - int(y)

    intX = int(x)
    intY = int(y)
    nextX = 0
    nextY = 0


    if modX == 0 and modY == 0:
        return image[intY,intX]

    if modX > 0.49 :
        nextX = min(intX + 1, width-1)
    else:
        nextX = max(intX - 1, 0)

    if modY > 0.49 :
        nextY = min(intY + 1, height-1)
    else:
        nextY = max(intY - 1, 0)

    a = np.array([intX, intY])
    b = np.array([intX, nextY])
    c = np.array([nextX, nextY])
    d = np.array([nextX, intY])

    distA = distance(point, a)
    distB = distance(point, b)
    distC = distance(point, c)
    distD = distance(point, d)

    coeff = distA + distB + distC + distD
    pixel = np.array([0,0,0])
    # setP = np.array([a,b,c,d])

    for i in (a,b,c,d):
        # tempPixel = np.array(image[i[1], i[0]])
        x = i[0]
        y = i[1]

        # tempPixel = np.array(image[y, x])
        # tempPixel = (tempPixel*distance(point, i))/coeff
        # pixel = pixel + tempPixel

        b, g, r = image[y, x]
        b = (b*distance(point, i))/coeff
        g = (g * distance(point, i)) / coeff
        r = (r * distance(point, i)) / coeff
        pixel = pixel + (int(b),int(g),int(r))

    return pixel

def warp(image, points):

  height, width, ch = image.shape
  warpedImage = np.zeros((height, width, 3), np.uint8)
  # warpedImage2 = np.zeros((height, width, 3), np.uint8)


  for y in range(0, height):
    for x in range(0, width):

      offset = [0,0]

      for point in points:
        pointPosition = (point[0] + point[2],point[1] + point[3])
        shift_vector = (point[2],point[3])

        #Inverse distance weighting  -> need to be tested to find possible improvement
        #helper = 1.0 / (3 * (pointsDistance((x,y),pointPosition) / vectorLength(shift_vector)) ** 4 + 1)
        helper = 1.0 / (3*(pointsDistance((x, y), pointPosition) / vectorLength(shift_vector)) ** 4 + 1)

        offset[0] -= helper * shift_vector[0]
        offset[1] -= helper * shift_vector[1]

      offset[0] = round(offset[0], 1)
      offset[1] = round(offset[1], 1)

      # coords2 = (clamp(x + int(offset[0]),0, width - 1),clamp(y + int(offset[1]),0,height - 1))
      # warpedImage2[y, x] = image[coords2[1], coords2[0]]

      coords = (clamp(x + offset[0], 0, width - 1), clamp(y + offset[1], 0, height - 1))
      warpedImage[y, x] = interpolate(image, coords)


      #add interpolation





# highlight the warped points in the original image and in the warped image
#   cv2.imshow('warpedImage', warpedImage)
#   cv2.imshow('warpedImage2', warpedImage2)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
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

image = cv2.imread('imageTest.jpg')
# image = cv2.resize(image,(800, 400))

width = int(image.shape[1] * 0.6)
height = int(image.shape[0] * 0.6)
dim = (width, height)
image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
warpedImage = warp(image,[(50,50,30,30), (400,200,-50,0), (100,200,0,-40)]) #,(1000,700,-500,-300)])

cv2.imshow('originalImage',image)
cv2.imshow('warpedImage',warpedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()