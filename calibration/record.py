import cv2 as cv 
cap = cv.VideoCapture(-1)
i = 0


cap.set(cv.CAP_PROP_SETTINGS, 0)

# cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT,800)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE,3)
# cap.set(cv.CAP_PROP_AUTO_EXPOSURE,0.25)

# cap.set(cv.CAP_PROP_EXPOSURE,100)
# cap.set(cv.CAP_PROP_FPS,30)
# cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
# cap.set(cv.CAP_PROP_SETTINGS, 0)
while True:
    result,image = cap.read()
    cv.imshow('frame',image)

    # Press 'q' to exit
    if cv.waitKey(1) == ord('c'):
        cv.imwrite(str(i)+'.jpg',image)
        i+=1
        print("recording!")
    