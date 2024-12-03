## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': lambda x: f'{x:4}'})
import cv2 as cv
from scipy.spatial.transform import Rotation
from time import time
from filterpy.kalman import UnscentedKalmanFilter,MerweScaledSigmaPoints

print(cv.__version__)

# create the stereo stream
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
depth_sensor = device.query_sensors()[0]

if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)

device_product_line = str(device.get_info(rs.camera_info.product_line))

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 1280,800, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared,2, 1280,800, rs.format.y8, 30)
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.infrared) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)

# create the imu stream
imu_pipeline = rs.pipeline()
imu_config = rs.config()
imu_config.enable_stream(rs.stream.accel)
imu_config.enable_stream(rs.stream.gyro)
imu_pipeline.start(imu_config)
# config.enable_stream(rs.stream.infrared, 2, 1280,720, rs.format.y8, 6)
# Start streaming

def find_knn_matches(bfm,des1,des2):
    matches = bfm.knnMatch(des1,des2,k=2)
    # matches = bfm.match(des1,des2)
    # return matches
    # Apply ratio test
    good = []
    # matches = sorted(matches, key = lambda x:x.distance)
    try:
        for m,n in matches:
            if m.distance < 0.5*n.distance: #0.5 for sift 0.8 for orb>?
                good.append([m])
    except:
        return []
    return good

def find_matches(bfm,des1,des2):
    matches = bfm.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    good = []
    # matches = sorted(matches, key = lambda x:x.distance)
    try:
        for m,n in matches:
            if m.distance < 0.5*n.distance: #0.5 for sift 0.8 for orb>?
                good.append([m])
    except:
        return []
    return good

sift = cv.SIFT_create()
orb = cv.ORB_create()
fast = cv.FastFeatureDetector_create()
agast = cv.AgastFeatureDetector_create()
freak = cv.xfeatures2d.FREAK_create()
detector = orb


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
 
bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
flann = cv.FlannBasedMatcher(index_params,search_params)


intrinsics = np.array([[640.778,0,643.648],
                       [0,640.778,400.925],
                       [0,0,1]])

pm_l = np.hstack((intrinsics,np.zeros((3,1)))) #projection matrix of left camera is just intrinsics
pm_r = pm_l @ np.array([[1,0,0,-5],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]]).astype(np.float64) #left camera is 5cm to the left or right camera
#p[643.648 400.925]  f[640.778 640.778]
anchor = {}
#kp1: list of cv:Keypoint
#match: list of cv.DMatch
class StereoFrame:
    def __init__(self,l,r):
        self.left = l
        self.right = r
    def create_keypoints(self,detector):
        self.kpl,self.desl = detector.detectAndCompute(self.left,None)
        self.kpr,self.desr = detector.detectAndCompute(self.right,None)
    def match_keypoints(self,bf):
        matches =  bf.match(self.desl,self.desr)
        good = sorted(matches, key = lambda x:x.distance)
        self.matches = good[:len(good)//3*2]
        self.pt1,self.pt2 = get_matched_points(self.kpl,self.kpr,self.matches)
        self.pc = triangulate(self.pt1,self.pt2)
        self.pc = self.pc[:3,:]/self.pc[3:,:]
        # print((self.pc[2]<500).shape)
        mask = np.where(self.pc[2]<800) # np.ones_like(self.pc[2]).astype(np.bool_)
        self.pc = self.pc[:,mask].squeeze()
        # self.pc = self.pc[:,]
        # print(self.desl.shape)
        self.des = np.array([self.desr[self.matches[i].trainIdx] for i in range(len(self.matches))],dtype=self.desl.dtype) #descriptor per point cloud element!
        # print(self.des.shape)
        self.des = self.des[mask]
        print(self.pc.shape)
        # print(self.des.shape)

def triangulate(ptsl,ptsr):
    point_cloud = cv.triangulatePoints(pm_l,pm_r,ptsl.T.astype(np.float64),ptsr.T.astype(np.float64))
    return point_cloud


def get_data(frames):
    ir_frame1 = frames.get_infrared_frame(1)
    ir_frame2 = frames.get_infrared_frame(2)
    left = np.asanyarray(ir_frame1.get_data())
    right = np.asanyarray(ir_frame2.get_data())
    fr = StereoFrame(left,right)
    fr.create_keypoints(detector)
    return fr

def get_matched_points(kp1,kp2,match):
    pts_1 = [kp1[match[i].queryIdx].pt for i in range(len(match))] #matched keypoint in left image
    pts_2 = [kp2[match[i].trainIdx].pt for i in range(len(match))] #use .pt to get tuple of coordinates
    pts_1 = np.int32(pts_1)
    pts_2 = np.int32(pts_2)
    return pts_1,pts_2

def get_pose(pts1,pts2,intrinsics=intrinsics):
    E,mask = cv.findEssentialMat(pts1, pts2, intrinsics, cv.RANSAC, 0.9999, 1.0)
    if(mask is None or E is None): return None
    if(len(E)!=3 or E.shape[1]!=3): return None
    inlier_ratio = np.sum(mask)/np.size(mask)
    if(inlier_ratio>0.5):
        _, R, t, mask = cv.recoverPose(E=E, points1=pts_l, points2=pts_r,cameraMatrix=intrinsics)
        return R,t
    
def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv.Rodrigues(R)
    points = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(np.int32)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 3)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 3)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    return img

    
    
# anchor_rotation = np.eye(3)
# anchor_translation = np.zeros(3)
anchor_pose = np.eye(4)
frames = pipeline.wait_for_frames()
last_t = time() 

anchor = get_data(frames)
anchor.create_keypoints(detector)
anchor.match_keypoints(bf)



def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

accel = np.zeros(3)
gyro = np.zeros(3)

#measurement function
def ori_g_func(x:np.ndarray,dt:float):
    X = Rotation.from_rotvec(x[:3])
    dX = Rotation.from_rotvec(gyro*dt)
    return (X*dX).as_rotvec()

def ori_h_func(x:np.ndarray):
    g_world = np.array([0,-9.8,0])
    X = Rotation.from_rotvec(x[:3])
    return X.inv().apply(g_world)

points = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=-1)

kf_ori = UnscentedKalmanFilter(3,3,None,ori_h_func,ori_g_func,points,)
kf_ori.x = np.zeros(3) #initial state
kf_ori.P*=0.2  #initial covariance
z_std = 20
kf_ori.R = np.diag([z_std**2,z_std**2,z_std**2])
x_std = 0.01
kf_ori.Q = np.diag([x_std**2,x_std**2,x_std**2])

points2 = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=-1)

kf_position = UnscentedKalmanFilter(3,3,None,lambda x:x,lambda x,dt:x,points2)
kf_position.x = np.zeros(3)
kf_position.P*=0.2
kf_position.R = np.diag(np.ones(3)*4)
kf_position.Q = np.diag(np.ones(3)*8)

# lastRot = np.eye(3)
positions = []
try:
    while True:
        f = pipeline.wait_for_frames()
        frame = get_data(f)
        f = imu_pipeline.wait_for_frames()
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        
        curr_t = time()
        dt = curr_t-last_t
        last_t = curr_t
        # print(dt)

        kf_ori.predict(dt)
        kf_ori.update(accel)

        frame.create_keypoints(detector)
        frame.match_keypoints(bf)
        
        Rot = Rotation.from_rotvec(kf_ori.x).as_matrix()

        # matches = find_matches(bf,frame.des,anchor.des)
        matches = bf.match(frame.des,anchor.des)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:60]#[:len(matches)//3*2]
        query_id = [matches[i].queryIdx for i in range(len(matches))]
        train_id = [matches[i].trainIdx for i in range(len(matches))]
        src_pts = frame.pc.T[query_id]
        dst_pts = anchor.pc.T[train_id]
        # print(src_pts.shape)
        t = -np.median(Rot @ src_pts.T - anchor_pose[:3,:3] @ dst_pts.T,axis=1).reshape((3,1))
        print(t)

        # if(np.linalg.norm(t)>10000):
        #     t*=0
        gt = t+anchor_pose[:3,3:]
        kf_position.predict(0.1)

        kf_position.update(gt.reshape(3))

        dR = Rotation.from_matrix(Rot.T@anchor_pose[:3,:3]).as_rotvec()
        R = Rot
        print(len(matches))
        # print(np.linalg.norm(t))
        # print(t)
        # # Estimate the rigid transformation
        # _,mat,mask = cv.estimateAffine3D(src_pts, dst_pts)
        # accuracy = np.sum(mask)/np.size(mask)
        # # print(mat.shape)
        # dR = np.linalg.norm(Rotation.from_matrix(mat[:,:3]).as_rotvec()[:2])

        # mat =   anchor_pose @ np.vstack((mat,np.array([[0,0,0,1]])))
        # R = mat[:3,:3]
        # t = mat[:3,3:]
        # print(R.T@ np.array([[0,0,100]]).T-t)
        img = cv.cvtColor(frame.left,cv.COLOR_GRAY2RGB)
        img = draw_axis(img,R.T,R.T@ (np.array([[0,0,50]]).T-kf_position.x.reshape((3,1))),intrinsics)


        # print(t[2,0])
        #print(dR)
        # if(accuracy>0.5 and dR>0.01):
        #     anchor = frame
        #     anchor_pose = mat
        #     print("switching anchor!")
        #img = draw_axis(frame.left,R.T,np.array([[0,0.0,100]]),intrinsics)
        if(np.linalg.norm(t)>15 or np.linalg.norm(dR)>0.3):
            anchor = frame
            anchor_pose[:3,:3] = Rot
            anchor_pose[:3,3] = kf_position.x  #gt.squeeze() #
            print("updating anchor")
        # R = Rotation.from_matrix().as_rotvec()

        # depth = np.average(frame.pc[2])
        # print(depth)
        #depth_frame = frames.get_depth_frame()
        

        # matches =  bf.match(frame.desl,frame.desr)
        # good = sorted(matches, key = lambda x:x.distance)
        # good = good[:len(good)//3*2]
        # num_match = len(good)

        # # pts_l = [kpl[good[i][0].queryIdx] for i in range(num_match)] #matched keypoint in left image
        # # pts_r = [kpr[good[i][0].trainIdx] for i in range(num_match)]
        # pts_l = [frame.kpl[good[i].queryIdx].pt for i in range(num_match)] #matched keypoint in left image
        # pts_r = [frame.kpr[good[i].trainIdx].pt for i in range(num_match)]
        # pts_l = np.int32(pts_l)
        # pts_r = np.int32(pts_r)

        # E,mask = cv.findEssentialMat(pts_l, pts_r, intrinsics, cv.RANSAC, 0.9999, 1.0)
        # if(mask is None or E is None): continue
        # if(len(E)!=3 or E.shape[1]!=3): continue

        # inlier_ratio = np.sum(mask)/np.size(mask)
        # if(inlier_ratio>0.5):
        #     _, R, t, mask = cv.recoverPose(E=E, points1=pts_l, points2=pts_r,cameraMatrix=intrinsics)
        #     print(t)


        # out = cv.drawKeypoints(curr_img,pts1,0)
        # #cv.imshow("input",out)

        # out = cv.drawKeypoints(last_img,pts2,0)
        # #cv.imshow("anchor",out)
        # cv.imshow("input",curr_img)
        if(len(frame.matches)>10):
          #both = cv.drawMatchesKnn(left,kpl,right,kpr,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
          both = cv.drawMatches(frame.left,frame.kpl,frame.right,frame.kpr,frame.matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) #orb

        else:
            both = np.concatenate((frame.left,frame.right),1)
        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    #           # Show images
        cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        cv.imshow('RealSense', img)
        cv.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    imu_pipeline.stop()

# find the keypoints and descriptors with SIFT

# BFMatcher with default params
bf = cv.BFMatcher()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
 
flann = cv.FlannBasedMatcher(index_params,search_params)
kpr,desr = detector.detectAndCompute(last_img,None)
# cv.namedWindow('input', cv.WINDOW_KEEPRATIO)
# cv.namedWindow('anchor', cv.WINDOW_KEEPRATIO)

while True:
    ret,img = cam.read()
    if(ret):
        curr_img = undistort(img)
        # cv.waitKey(0)
        kpl,desl = detector.detectAndCompute(curr_img,None)
        good = find_knn_matches(bf,desl,desr)
        num_match = len(good)
        pts1 = [kpl[good[i][0].queryIdx] for i in range(num_match)]
        out = cv.drawKeypoints(curr_img,pts1,0)
        #cv.imshow("input",out)
        pts2 = [kpr[good[i][0].trainIdx] for i in range(num_match)]

        out = cv.drawKeypoints(last_img,pts2,0)
        #cv.imshow("anchor",out)
        cv.imshow("input",curr_img)
        img3 = cv.drawMatchesKnn(curr_img,kpl,last_img,kpr,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("matching",img3)
        pts1 = [point.pt for point in pts1]
        pts2 = [point.pt for point in pts2]
