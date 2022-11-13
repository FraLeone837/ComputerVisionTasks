import cv2
import transformation

#perspective, 2 images
def testPerspective(name1, name2):
# name1 and name2 are the names of the two images to stitch
# here the names are turned into paths
    name1 = 'photos/perspective/'+name1+'.jpeg'
    name2 = 'photos/perspective/' + name2 + '.jpeg'
    image1 = cv2.imread(name1)
    image2 = cv2.imread(name2)
    image1 = transformation.resize(image1, 0.25)
    image2 = transformation.resize(image2, 0.25)

    img1 = image1
    img2 = image2
    # imga, imgb = transformation.translation(img1, img2)
    imga, imgb = transformation.perspective(img1, img2)
    fimg = transformation.mergeImages(imga, imgb)

#uncomment to see all steps
    cv2.imshow('i1', img1)
    cv2.imshow('i2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    image1 = transformation.resize(image1, 0.25)
    image2 = transformation.resize(image2, 0.25)
    image3 = transformation.resize(image3, 0.25)
    image4 = transformation.resize(image4, 0.25)

    imga, imgb = transformation.translation(image1, image2)
    fimg1 = transformation.mergeImages(imga, imgb)
# uncomment to see all steps
    # cv2.imshow('i1', imga)
    # cv2.imshow('i2', imgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('fi1', fimg1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imgc, imgd = transformation.translation(image3, image4)
    fimg2 = transformation.mergeImages(imgc, imgd)
# uncomment to see all steps
    # cv2.imshow('i3', imgc)
    # cv2.imshow('i4', imgd)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('fi2', fimg2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imge, imgf = transformation.translation(fimg1, fimg2)
    fimg = transformation.mergeImages(imge, imgf)
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