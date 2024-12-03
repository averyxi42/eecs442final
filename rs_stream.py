## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 1280,800, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared,2, 1280,800, rs.format.y8, 30)

# config.enable_stream(rs.stream.infrared, 2, 1280,720, rs.format.y8, 6)
# Start streaming
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.infrared) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        ir_frame1 = frames.get_infrared_frame(1)
        ir_frame2 = frames.get_infrared_frame(2)

        # color_frame = frames.get_color_frame()
        # if not depth_frame:
        #     continue

        # # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())
        both = np.concatenate((ir_image1,ir_image2),1)
        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    #           # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', both)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()