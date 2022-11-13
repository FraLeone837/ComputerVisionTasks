import math
import cv2
import numpy as np

def pointsDistance(point1, point2):
    vector = (point1[0] - point2[0],point1[1] - point2[1])
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def resize(image, fact):
    width = int(image.shape[1] * fact)
    height = int(image.shape[0] * fact)
    dim = (width, height)
    imageRes = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return imageRes

def highlightRandom(image, points):
    for point in points:
        color = np.random.randint(0, 255, 3)
        color = (int(color[0]), int(color[1]), int(color[2]))
        x = int(point[0])
        y = int(point[1])
        image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
    return image

def highlightRed(image, points):
    for point in points:
        x = int(point[0])
        y = int(point[1])
        image = cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    return image

def centrate(image1, image2, a, b, factx, facty):
    width = int(max(image1.shape[1], image2.shape[1]) * factx)
    height = int(max(image1.shape[0], image2.shape[0]) * facty)
    dim = (width, height)
    x = int((width-width/factx))/2
    y = int((height-height/facty))/2
    M3 = np.float32([[1, 0, x], [0, 1, y]])
    image1 = cv2.warpAffine(image1, M3, dim)
    image2 = cv2.warpAffine(image2, M3, dim)
    for i in range(4):
        a[i] = (a[i][0] + x, a[i][1] + y)
        b[i] = (b[i][0] + x, b[i][1] + y)
    return image1, image2, a, b


def bestFeaturesFirst(matches, kp1, kp2):
    a1 = (int(kp1[matches[0].queryIdx].pt[0]), int(kp1[matches[0].queryIdx].pt[1]))
    a2 = (int(kp1[matches[1].queryIdx].pt[0]), int(kp1[matches[1].queryIdx].pt[1]))
    a3 = (int(kp1[matches[2].queryIdx].pt[0]), int(kp1[matches[2].queryIdx].pt[1]))
    a4 = (int(kp1[matches[3].queryIdx].pt[0]), int(kp1[matches[3].queryIdx].pt[1]))
    b1 = (int(kp2[matches[0].trainIdx].pt[0]), int(kp2[matches[0].trainIdx].pt[1]))
    b2 = (int(kp2[matches[1].trainIdx].pt[0]), int(kp2[matches[1].trainIdx].pt[1]))
    b3 = (int(kp2[matches[2].trainIdx].pt[0]), int(kp2[matches[2].trainIdx].pt[1]))
    b4 = (int(kp2[matches[3].trainIdx].pt[0]), int(kp2[matches[3].trainIdx].pt[1]))
    a = np.int16([a1, a2, a3, a4])
    b = np.int16([b1, b2, b3, b4])
    return a, b

def bestFeaturesQuadrant(dim, matches, kp1, kp2):
    a = [0,0,0,0]
    b = [0,0,0,0]
    q1 = (0, int(dim[0]/2), 0, int(dim[1]/2))
    q2 = (0, int(dim[0]/2), int(dim[1]/2), int(dim[1]))
    q3 = (int(dim[0]/2), int(dim[0]), 0, int(dim[1]/2))
    q4 = (int(dim[0]/2), int(dim[0]), int(dim[1]/2), int(dim[1]))
    q = [q1, q2, q3, q4]
    i = 0
    j = 0
    while(i<4):
        x1 = (int(kp1[matches[j].queryIdx].pt[0]), int(kp1[matches[j].queryIdx].pt[1]))
        y1 = (int(kp2[matches[j].trainIdx].pt[0]), int(kp2[matches[j].trainIdx].pt[1]))
        if((x1[1] in range(q[i][0], q[i][1])) and (x1[0] in range(q[i][2], q[i][3]))):
            a[i] = x1
            b[i] = y1
            j = 0
            i = i+1
        else:
            j = j+1
    return a, b

def bestFeaturesDistance(distance, matches, kp1, kp2):
    a = []
    b = []
    i = 0
    j = 1
    while(i<4):
        if i==0:
            x = [(int(kp1[matches[0].queryIdx].pt[0]), int(kp1[matches[0].queryIdx].pt[1]))]
            y = [(int(kp2[matches[0].trainIdx].pt[0]), int(kp2[matches[0].trainIdx].pt[1]))]
            a = a + x
            b = b + y
            i = i+1
        else:
            while(True):
                ok = True
                x1 = (int(kp1[matches[j].queryIdx].pt[0]), int(kp1[matches[j].queryIdx].pt[1]))
                y1 = (int(kp2[matches[j].trainIdx].pt[0]), int(kp2[matches[j].trainIdx].pt[1]))
                for k in range(len(a)):
                    if(pointsDistance(x1, a[k])<distance):
                        ok = False
                        break
                if(ok):
                    a = a + [x1]
                    b = b + [y1]
                    j = j+1
                    i = i+1
                    break
                else:
                    j = j+1
    return a, b

def findFeaturesORB(image1, image2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    img1 = image1
    img2 = image2
    # a, b = bestFeaturesFirst(matches, kp1, kp2)
    a, b = bestFeaturesDistance(60, matches, kp1, kp2)
    # width = int(min(img1.shape[1], img2.shape[1]))
    # height = int(min(img1.shape[0], img2.shape[0]))
    # dim = (width, height)
    # a, b = bestFeaturesQuadrant(dim, matches, kp1, kp2)
    # img1 = highlightRandom(img1, a)
    # img2 = highlightRandom(img2, b)
    # img1 = highlightRed(img1, a)
    # img2 = highlightRed(img2, b)
    return img1, img2, a, b

def mergePixelsAvg(pixel1, pixel2):
    x1 = pixel1[0]
    y1 = pixel1[1]
    z1 = pixel1[2]
    x2 = pixel2[0]
    y2 = pixel2[1]
    z2 = pixel2[2]
    if(x1==0 and y1==0 and z1==0):
        pixel = (x2, y2, z2)
    elif(x2==0 and y2==0 and z2==0):
        pixel = (x1, y1, z1)
    else:
        x = (int(x1) + int(x2)) / 2
        y = (int(y1) + int(y2)) / 2
        z = (int(z1) + int(z2)) / 2
        pixel = (x,y,z)
    return pixel

def mergePixelsPrior(pixel1, pixel2):
    x1 = pixel1[0]
    y1 = pixel1[1]
    z1 = pixel1[2]
    x2 = pixel2[0]
    y2 = pixel2[1]
    z2 = pixel2[2]
    if(x1==0 and y1==0 and z1==0):
        pixel = (x2, y2, z2)
    else:
        pixel = (x1, y1, z1)
    return pixel

def mergeImages (image1, image2):
    width = int(image1.shape[1])
    height = int(image1.shape[0])
    mergedImage = np.zeros((height, width, 3), np.uint8)
    for x in range(width):
        for y in range(height):
            # mergedImage[y, x] = mergePixelsAvg(image1[y, x], image2[y, x])
            mergedImage[y, x] = mergePixelsPrior(image2[y, x], image1[y, x])
    return mergedImage

def translation(image1, image2):
    img1, img2, a, b = findFeaturesORB(image1, image2)
    a1 = a[0]
    b1 = b[0]
    pts1 = a1[0] - b1[0]
    pts2 = a1[1] - b1[1]
    width = int((max(img1.shape[1], img2.shape[1]) + abs(pts1)) *1)
    height = int((max(img1.shape[0], img2.shape[0]) + abs(pts2)) *1)
    dim = (width, height)
    imga = img2
    imgb = img1

#the translations occur so that the images are not cropped
    if pts1 < 0:
        pts1 = -pts1
        M1 = np.float32([[1, 0, pts1], [0, 1, 0]])
        imgb = cv2.warpAffine(img1, M1, dim)
    else:
        M1 = np.float32([[1, 0, pts1], [0, 1, 0]])
        imga = cv2.warpAffine(img2, M1, dim)
    imgc = imga
    imgd = imgb
    if pts2 < 0:
        pts2 = -pts2
        M2 = np.float32([[1, 0, 0], [0, 1, pts2]])
        imgd = cv2.warpAffine(imgb, M2, dim)
    else:
        M2 = np.float32([[1, 0, 0], [0, 1, pts2]])
        imgc = cv2.warpAffine(imga, M2, dim)
    M3 = np.float32([[1, 0, 0], [0, 1, 0]])
    imgc = cv2.warpAffine(imgc, M3, dim)
    imgd = cv2.warpAffine(imgd, M3, dim)
    return imgc, imgd

def perspective(image1, image2):
    img1, img2, a, b = findFeaturesORB(image1, image2)
    img1 ,img2, a, b = centrate(img1, img2, a, b, factx=1.8, facty=1.25)
    c0 = ((a[0][0] + b[0][0]) / 2, (a[0][1] + b[0][1]) / 2)
    c1 = ((a[1][0] + b[1][0]) / 2, (a[1][1] + b[1][1]) / 2)
    c2 = ((a[2][0] + b[2][0]) / 2, (a[2][1] + b[2][1]) / 2)
    c3 = ((a[3][0] + b[3][0]) / 2, (a[3][1] + b[3][1]) / 2)
    c = (c0, c1, c2, c3)
    a = np.float32(a)
    b = np.float32(b)
    c = np.float32(c)
    width = int(max(img1.shape[1], img2.shape[1]))
    height = int(max(img1.shape[0], img2.shape[0]))
    dim = (width, height)
    imga = img1
    imgb = img2

# # these can be used to warp only one image
#     # M1 = cv2.getPerspectiveTransform(a, b)
#     # imga = cv2.warpPerspective(imga, M1, dim)
# #this second one is better
#     M2 = cv2.getPerspectiveTransform(b, a)
#     imgb = cv2.warpPerspective(imgb, M2, dim)

    M1 = cv2.getPerspectiveTransform(a, c)
    imga = cv2.warpPerspective(imga, M1, dim)
    M2 = cv2.getPerspectiveTransform(b, c)
    imgb = cv2.warpPerspective(imgb, M2, dim)
    return imga, imgb


#perspective, 2 images
def testPerspective(name1, name2):
# name1 and name2 are the names of the two images to stitch
# here the names are turned into paths
    name1 = 'photos/perspective/'+name1+'.jpeg'
    name2 = 'photos/perspective/' + name2 + '.jpeg'
    image1 = cv2.imread(name1)
    image2 = cv2.imread(name2)
    image1 = resize(image1, 0.25)
    image2 = resize(image2, 0.25)

    img1 = image1
    img2 = image2
    # imga, imgb = translation(img1, img2)
    imga, imgb = perspective(img1, img2)
    fimg = mergeImages(imga, imgb)

#uncomment to see all steps
    # cv2.imshow('i1', img1)
    # cv2.imshow('i2', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('wi1', imga)
    # cv2.imshow('wi2', imgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imshow('fi', fimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testTranslation():
    #translation, 4 images
    image1 = cv2.imread('photos/translation/4a.jpeg')
    image2 = cv2.imread('photos/translation/4b.jpeg')
    image3 = cv2.imread('photos/translation/4c.jpeg')
    image4 = cv2.imread('photos/translation/4d.jpeg')
    image1 = resize(image1, 0.25)
    image2 = resize(image2, 0.25)
    image3 = resize(image3, 0.25)
    image4 = resize(image4, 0.25)

    imga, imgb = translation(image1, image2)
    fimg1 = mergeImages(imga, imgb)
# uncomment to see all steps
    # cv2.imshow('i1', imga)
    # cv2.imshow('i2', imgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('fi1', fimg1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imgc, imgd = translation(image3, image4)
    fimg2 = mergeImages(imgc, imgd)
# uncomment to see all steps
    # cv2.imshow('i3', imgc)
    # cv2.imshow('i4', imgd)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('fi2', fimg2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imge, imgf = translation(fimg1, fimg2)
    fimg = mergeImages(imge, imgf)
# uncomment to see all steps
#     cv2.imshow('wi1', imge)
#     cv2.imshow('wi2', imgf)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    cv2.imshow('fi', fimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test = int(input("Want to test perspective(1) or translation(2)?"))
if(test==1):
    image1, image2 = input("what couple of images must be used?").split()
    testPerspective(image1, image2)
elif(test==2):
    testTranslation()
else:
    print("command not valid")