import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
import torch
intrinsics = np.loadtxt('intrinsics0')
distortion = np.loadtxt('distortion0')

cam = cv.VideoCapture(-1)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
ret,img = cam.read()
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(intrinsics, distortion, (w,h), 1, (w,h))
x, y, w, h = roi
undistort = lambda image:(cv.undistort(image, intrinsics, distortion, None, newcameramtx)[y:y+h, x:x+w])
last_img = undistort(img)

key = cv.waitKey(0)

detections = model(last_img)
bbox = detections.xyxy[0].numpy().astype(np.int32)[0,:4]
print(bbox)
print(last_img.shape)
dx = 80
dy=30
x0,y0,x1,y1 = bbox[0]+dy,bbox[1]+dy,bbox[2]-dx,bbox[3]-dy

last_img = last_img[y0:y1,x0:x1,:]




translations = []
rotations = []
### the anchor is the "origin". it gets updated once there aren't enough matches.
anchor_rotation = np.eye(3)
anchor_translation = np.zeros(3)

rotation = np.eye(3)
translation = np.zeros(3)

def find_matches(bfm,des1,des2):
    matches = bfm.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append([m])
    return good
sift = cv.SIFT_create()
fast = cv.FastFeatureDetector_create()
# find the keypoints and descriptors with SIFT

# BFMatcher with default params
bf = cv.BFMatcher()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
 
flann = cv.FlannBasedMatcher(index_params,search_params)
kp2,des2 = sift.detectAndCompute(last_img,None)


while True:
    ret,img = cam.read()
    if(ret):
        curr_img = undistort(img)
        # cv.waitKey(0)
        kp1,des1 = sift.detectAndCompute(curr_img,None)

        good = find_matches(flann,des1,des2)
        num_match = len(good)
        pts1 = [kp1[good[i][0].queryIdx] for i in range(num_match)]
        a = cv.drawKeypoints(curr_img,pts1,0)
        cv.imshow("input",a)
        pts2 = [kp2[good[i][0].trainIdx] for i in range(num_match)]

        b = cv.drawKeypoints(last_img,pts2,0)
        cv.imshow("anchor",b)

        pts1 = [point.pt for point in pts1]
        pts2 = [point.pt for point in pts2]

        if(num_match<6):
            continue

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
         
        E,mask = cv.findEssentialMat(pts1, pts2, intrinsics, cv.RANSAC, 0.999, 1.0)

        if(E is not None):
            #if(E.shape!=(3,3)):

            _, R, t, mask = cv.recoverPose(E=E, points1=pts1, points2=pts2,cameraMatrix=intrinsics)
            rotation = anchor_rotation @ np.array(R)
            # translation = anchor_rotation @ np.array(t).reshape(-1) + anchor_translation
            # translations.append(np.array(t))
            rotations.append(np.array(R))
            extrinsic = np.hstack((np.array(R),np.array(t).reshape((3,1))))
            #print(intrinsics.dtype)
            # point_cloud = cv.triangulatePoints(np.hstack((intrinsics,np.zeros((3,1)))),intrinsics @ extrinsic,pts1.T.astype(np.float64),pts2.T.astype(np.float64))
            # point_cloud = point_cloud[:3]/point_cloud[2:3]
            # print(point_cloud.shape)
        euler = Rotation.from_matrix(np.array(rotation)).as_euler('zxy')
        print(euler)

        # #F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

        #E = intrinsics.T @ np.array(F) @ intrinsics # essential matrix
        #print(E)
        # if(num_match<50):
        #     print("switching anchor!")
        #     last_img = curr_img.copy()
        #     # kp2 = kp1.copy()
        #     # des2 = des1.copy()
        #     kp2,des2 = sift.detectAndCompute(curr_img,None)
        #     anchor_rotation = np.copy(rotation)
        #     #anchor_translation = np.copy(translation)
        key = cv.waitKey(1)
        if(key==ord('q')):
            np.save('rotations',rotations)
            np.save('translations',translations)
            print(extrinsic)
            #point_cloud = cv.convertPointsToHomogeneous(point_cloud,0)
            np.save('point cloud',point_cloud)
            exit(0)