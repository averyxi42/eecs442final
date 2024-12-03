import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
intrinsics = np.loadtxt('intrinsics0')
#intrinsics = np.zeros(5)
distortion = np.loadtxt('distortion0')

cam = cv.VideoCapture(-1) # default is 480, 640, 3
cam.set(cv.CAP_PROP_AUTO_EXPOSURE,3)
# cam.set(cv.CAP_PROP_FRAME_WIDTH,1920)
# cam.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
# cam.set(cv.CAP_PROP_EXPOSURE, -1)
# cam.set(cv.CAP_PROP_AUTO_EXPOSURE,2)

# intrinsics[0]*=3
# intrinsics[1]*=2.25
#distortion/=2

#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
ret,img = cam.read()
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(intrinsics, distortion, (w,h), 1, (w,h))
x, y, w, h = roi
undistort = lambda image:(cv.undistort(image, intrinsics, distortion, None, newcameramtx)[y:y+h, x:x+w])
last_img = undistort(img)

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
        if m.distance < 0.6*n.distance: #0.5 for sift 0.8 for orb>?
            good.append([m])
    return good
sift = cv.SIFT_create()
orb = cv.ORB_create()
fast = cv.FastFeatureDetector_create()
detector = sift
# find the keypoints and descriptors with SIFT

# BFMatcher with default params
bf = cv.BFMatcher()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
 
flann = cv.FlannBasedMatcher(index_params,search_params)
kp2,des2 = detector.detectAndCompute(last_img,None)
# cv.namedWindow('input', cv.WINDOW_KEEPRATIO)
# cv.namedWindow('anchor', cv.WINDOW_KEEPRATIO)

while True:
    ret,img = cam.read()
    if(ret):
        curr_img = undistort(img)
        # cv.waitKey(0)
        kp1,des1 = detector.detectAndCompute(curr_img,None)
        good = find_matches(bf,des1,des2)
        num_match = len(good)
        pts1 = [kp1[good[i][0].queryIdx] for i in range(num_match)]
        out = cv.drawKeypoints(curr_img,pts1,0)
        #cv.imshow("input",out)
        pts2 = [kp2[good[i][0].trainIdx] for i in range(num_match)]

        out = cv.drawKeypoints(last_img,pts2,0)
        #cv.imshow("anchor",out)
        cv.imshow("input",curr_img)
        img3 = cv.drawMatchesKnn(curr_img,kp1,last_img,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("matching",img3)
        pts1 = [point.pt for point in pts1]
        pts2 = [point.pt for point in pts2]



        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        if(num_match<5): continue
        E,mask = cv.findEssentialMat(pts1, pts2, intrinsics, cv.RANSAC, 0.9999, 1.0)
        if(mask is None or E is None): continue
        if(len(E)!=3 or E.shape[1]!=3): continue

        inlier_ratio = np.sum(mask)/np.size(mask)
        if(inlier_ratio>0.5):
            _, R, t, mask = cv.recoverPose(E=E, points1=pts1, points2=pts2,cameraMatrix=intrinsics)
            rotvec = Rotation.from_matrix(np.array(R)).as_rotvec()
            angle = np.linalg.norm(rotvec) #angle of rotation from anchor
            rotation = anchor_rotation @ np.array(R)
            # translation = anchor_rotation @ np.array(t).reshape(-1) + anchor_translation
            # translations.append(np.array(t))
            rotations.append(np.array(R))
            extrinsic = np.hstack((np.array(R),np.array(t).reshape((3,1))))
            #print(intrinsics.dtype)
            # point_cloud = cv.triangulatePoints(np.hstack((intrinsics,np.zeros((3,1)))),intrinsics @ extrinsic,pts1.T.astype(np.float64),pts2.T.astype(np.float64))
            # point_cloud = point_cloud[:3]/point_cloud[2:3]
            # print(point_cloud.shape)
            euler = Rotation.from_matrix(rotation).as_euler('zxy')
            euler_delta = Rotation.from_matrix(np.array(R)).as_euler('zxy')
            #print(euler_delta)
            print(euler)
        #print(euler_delta,end=' ')

        # #F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

        #E = intrinsics.T @ np.array(F) @ intrinsics # essential matrix
        #print(E)
            if(num_match<150 and inlier_ratio>0.7 and angle<np.pi/3 and angle>np.pi/10): #160 is good for sift,
                # print("switching anchor!")
                last_img = curr_img
                # kp2 = kp1.copy()
                # des2 = des1.copy()
                kp2,des2 = detector.detectAndCompute(curr_img,None)
                anchor_rotation = np.copy(rotation)
            #anchor_translation = np.copy(translation)
        key = cv.waitKey(1)
        if(key==ord('q')):
            np.save('rotations',rotations)
            np.save('translations',translations)
            print(extrinsic)
            #point_cloud = cv.convertPointsToHomogeneous(point_cloud,0)
            np.save('point cloud',point_cloud)
            exit(0)