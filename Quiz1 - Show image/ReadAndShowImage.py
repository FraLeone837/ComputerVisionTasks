import cv2

imageTest = cv2.imread('imageTest.jpg', 1)
# Resize the image
#resizedImage = cv2.resize(image, (1000, 500))

cv2.imshow('imageTest', imageTest)
cv2.waitKey(0)