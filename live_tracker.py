import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import object_tracker
import os
from datetime import datetime

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

frame_rate = 15
num_frames = 3000
image_shape = [480, 680]
num_particles = 200
num_bins = 12
runs_per_frame = 3
current_time = datetime.now()
results_folder = current_time.strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(results_folder)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frame_rate)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, frame_rate)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()

Track = False
frame_count = 0

for i in list(range(frame_rate * 5)):
    frames = pipeline.wait_for_frames()
print('Began Capturing Images...')

while frame_count < num_frames:
    frames = pipeline.wait_for_frames()

    # Aligning depth and rgb frames
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not aligned_color_frame:
        continue

    # Getting camera intrinsics
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrinsics = aligned_color_frame.profile.as_video_stream_profile().intrinsics

    # Timing
    currentDT = datetime.now()
    currentDT = str(currentDT).split(' ')[1].split(':')[2]
    current_seconds = int(currentDT.split('.')[0])
    current_millis = int(currentDT.split('.')[1]) // 1000
    if frame_count == 0:
        previous_seconds = current_seconds
        previous_millis = current_millis
    if previous_seconds == current_seconds:
        time_step = current_millis - previous_millis
    else:
        time_step = current_millis + (1000 - previous_millis)
    previous_seconds = current_seconds
    previous_millis = current_millis
    dt = time_step

    # Getting IMU readings
    accel = accel_data(frames[2].as_motion_frame().get_motion_data())
    gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())

    # Getiing color and depth images
    color_image = np.asanyarray(aligned_color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    #Cropping pixels with depth ~ 2m
    depth_image = depth_image * (depth_image < 2000)
    if float(np.max(depth_image)) == 0:
        continue

    # Converting depth values to 8-bit integers
    depth_scaled = ((depth_image / float(np.max(depth_image))) * 255).astype('uint8')

    # Constructing an RGBD image from the frames
    rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
    rgbd_image[:, :, 0:3] = color_image
    rgbd_image[:, :, 3] = depth_scaled
    # image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


    # Particle Filtering
    if Track == True:
        if frame_count == 0:
            x, y, Hx, Hy = object_tracker.define_initial_target_region(color_image)
            q = object_tracker.get_color_distribution([x, y, Hx, Hy], color_image, num_bins)
            s_t_1, pi_t_1 = object_tracker.get_initial_state(x, y, Hx, Hy, q, num_particles, num_bins, color_image)

        for i in list(range(runs_per_frame)):
            s_t, pi_t, s_t_mean, q = object_tracker.particle_filter(s_t_1, pi_t_1, q, num_bins, color_image)
            s_t_1 = s_t
            pi_t_1 = pi_t
        color_image = object_tracker.plot_state(s_t, s_t_mean, color_image, frame_count, results_folder, save = False, display_all = True)
        frame_count = frame_count + 1

    # Visualization
    cv2.namedWindow('Live Tracking', cv2.WINDOW_AUTOSIZE)
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Live Tracking', image)


    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        Track = True
        frame_count = 0