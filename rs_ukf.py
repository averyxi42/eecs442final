import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation
from filterpy.kalman import UnscentedKalmanFilter,MerweScaledSigmaPoints
from time import time
import cv2 as cv
imu_pipeline = rs.pipeline()
imu_config = rs.config()
imu_config.enable_stream(rs.stream.accel)
imu_config.enable_stream(rs.stream.gyro)
imu_pipeline.start(imu_config)


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

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

accel = np.zeros(3)
gyro = np.zeros(3)

#measurement function
def g_func(x:np.ndarray,dt:float):
    X = Rotation.from_rotvec(x[:3])
    dX = Rotation.from_rotvec(gyro*dt)
    return (X*dX).as_rotvec()

def h_func(x:np.ndarray):
    g_world = np.array([0,-9.8,0])
    X = Rotation.from_rotvec(x[:3])
    return X.inv().apply(g_world)

points = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=-1)
kf = UnscentedKalmanFilter(3,3,None,h_func,g_func,points,)
kf.x = np.zeros(3) #initial state
kf.P*=0.2  #initial covariance
z_std = 20
kf.R = np.diag([z_std**2,z_std**2,z_std**2])
x_std = 0.01
kf.Q = np.diag([x_std**2,x_std**2,x_std**2])

last_t = time()


K = np.array([[500,0,250],
              [0,500,250],
              [0,0,1.0]])
try:
    while True:
        f = imu_pipeline.wait_for_frames()
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        curr_t = time()
        dt = curr_t-last_t
        last_t = curr_t

        kf.predict(dt)
        kf.update(accel)
        # print(kf.x)   
        # print(accel-h_func(kf.x))

        img = np.zeros((500,500,3))
        draw_axis(img,Rotation.from_rotvec(kf.x).as_matrix(),np.array([[0,0.0,20]]),K)
        cv.imshow("gyro",img)
        cv.waitKey(1)
        print(accel)

finally:
    imu_pipeline.stop()