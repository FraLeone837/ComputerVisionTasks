import cv2
import numpy as np
import math

def vectorLength(vector):
  return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def pointsDistance(point1, point2):
  return vectorLength((point1[0] - point2[0],point1[1] - point2[1]))



image = cv2.imread('imageTest.jpg')
width = int(image.shape[1] * 0.6)
height = int(image.shape[0] * 0.6)
dim = (width, height)
image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)

point = [0.4,0.3]

a = np.array([0, 0, 0])
b = np.array([1, 1, 0])
c = np.array([1, 0, 0])
d = np.array([0, 1, 0])

print(a)
print(b)
print(c)
print(d)

distA = pointsDistance(point, a)
distB = pointsDistance(point, b)
distC = pointsDistance(point, c)
distD = pointsDistance(point, d)
coeff = distA + distB + distC + distD

# coeffA = (distB + distC + distD)/(3*coeff)
# coeffB = (distA + distC + distD) / (3 * coeff)
# coeffC = (distA + distB + distD) / (3 * coeff)
# coeffD = (distA + distB + distC) / (3 * coeff)

print(distA, distB, distC, distD, coeff)


a= np.array([a[0], a[1], (distB + distC + distD)/(3*coeff)])
b= np.array([b[0], b[1],(distA + distC + distD) / (3 * coeff)])
c= np.array([c[0], c[1],(distA + distB + distD) / (3 * coeff)])
d= np.array([d[0], d[1],(distA + distB + distC) / (3 * coeff)])
pixel = np.array([0,0,0])

print(a)
print(b)
print(c)
print(d)


for i in (a, b, c, d):
    x = int(i[0])
    y = int(i[1])
    coeff = i[2]
    print(image[y, x])
    b, g, r = image[y, x]
    b = b * coeff
    g = g * coeff
    r = r * coeff
    # b = (b * pointsDistance(point, i)) / coeff
    # g = (g * pointsDistance(point, i)) / coeff
    # r = (r * pointsDistance(point, i)) / coeff
    pixel = pixel + (int(b), int(g), int(r))

print(pixel)