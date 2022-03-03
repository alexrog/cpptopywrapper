import nanodet_openvino as nanodet # the cpp to py wrapper for our nanodet inference
from BoundingBoxes import BoundingBoxes
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# DONT USE THIS, NOT FINISHED YET
def video_inference(video_pth):
    width = 512
    height = 288
    cap = cv2.VideoCapture(video_pth)
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (cap.get(3), cap.get(4)))
    if cap.isOpened() == False:
        print("Error opening video file")
    while cap.isOpened() == True:
        ret, frame = cap.read()
        if ret:
            # img = cv2.resize(frame, (height, width))
            bboxes = nanodet.inference(frame)
            bboxes = BoundingBoxes(bboxes)
            detects, scores, classes = bboxes.bytetrack_input()
        else:
            break
    cap.release()

# function which sets up the intel realsense camera
def setup_camera():
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline

# function which gets live feed from camera and performs inference
def live_inference():
    pipeline = setup_camera()

    while True:
        start = time.time()

        # get feed from camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # perform inference and store bounding boxes
        bboxes = nanodet.inference(color_image)
        bboxes = BoundingBoxes(bboxes) # this class is in BoundingBoxes.py

        # get the cropped images from bounding box (can be used for other algorithms)
        crop_imgs = bboxes.crop(color_image)
        
        # get the information to pass to bytetrack, use color_image
        detects, scores, classes = bboxes.bytetrack_input()

        print(bboxes)

        # cv2.imshow('RealSense', color_image)
        if(cv2.waitKey(1) >= 0):
            break
        print(time.time()-start)
    pipeline.stop()

# function which will show all cropped images after calling BoundingBoxes crop function
def show_cropped_imgs(crop_imgs):
    for crop_img in crop_imgs:
        cv2.imshow("image", crop_img)
        cv2.waitKey(0)

if __name__ == "__main__":
    if nanodet.isModelInit() == False:
        print('Initializing model')
        nanodet.initModel("nanodet_model/nanodet.xml", "MYRIAD")
        print('Model has been initialized')
    # live_inference()
    video_inference('rc1.m4v')