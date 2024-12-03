import cv2 as cv
cam = cv.VideoCapture(-1) # default is 480, 640, 3
# cam.set(cv.CAP_PROP_FRAME_WIDTH,1920)
# cam.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
cam.set(cv.CAP_PROP_EXPOSURE,2)
ret,img = cam.read()
while True:
    ret,img = cam.read()

    cv.imshow("capture",img)
    cv.waitKey(1)
print(img.shape)
