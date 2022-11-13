import cv2
import numpy as np
import math

def distance(point, vector):
  return math.sqrt((point[0]-vector[0]) ** 2 + (point[1]-vector[1]) ** 2)

def interpolate(image, point):

    x = point[0]
    y = point[1]
    modX = round(x, 2) - int(x)
    modY = round(y, 2) - int(y)

    intX = int(x)
    intY = int(y)
    nextX = 0
    nextY = 0


    if modX == 0 and modY == 0:
        return image[intY,intX]

    if modX > 0.49 :
        nextX = intX - 1
    else:
        nextX = intX + 1

    if modY > 0.49 :
        nextY = intY - 1
    else:
        nextY = intY + 1

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
        pixel = pixel + (round(b),round(g),round(r))

    return pixel


# img = np.zeros((200,300, 3), np.uint8)
# img[0:99, 0:149]= (255,255,255)
# img[0:99, 150:299]= (255,0,0)
# img[100:199, 0:149]= (0,255,0)
# img[100:199, 150:299]= (0,0,255)
# (x, y, z) = interpolate(img, (50.5, 100.5))
# print(x, y, z)