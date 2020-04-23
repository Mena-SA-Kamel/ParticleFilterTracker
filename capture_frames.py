import pyrealsense2 as rs
import numpy as np
import os
import cv2
from PIL import Image
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import datetime
from pyntcloud import PyntCloud


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def butterworth_filter(type, data, cutoff, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    b, a = butter(order, normal_cutoff, btype=type, analog=False)
    y = lfilter(b, a, data)
    return y

def capture_frames(num_frames, frame_rate = 15, dataset_name = 'Tracking Dataset Accelerometer Gyro'):
    ## Setting up work directories
    if not os.path.exists(dataset_name):
        dataset_path = os.path.join("Datasets", dataset_name)
        os.mkdir(dataset_path)
        folders = ['rgb', 'depth']
        for folder in folders:
            folder_path = os.path.join(dataset_path, folder)
            os.mkdir(folder_path)

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
    frame_count = 0
    accelerometer_data = np.zeros((num_frames, 3))
    gyroscope_data = np.zeros((num_frames, 3))
    time_data = np.zeros(num_frames)

    for i in list(range(frame_rate*5)):
        frames = pipeline.wait_for_frames()
    print('Began Capturing Images...')

    try:
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
            # print('Depth intrinsics: ', depth_intrinsics)
            # print('Color intrinsics: ', color_intrinsics)

            # Computing 3D points:
            depth_pixel_1 = [508, 272]
            depth_pixel_2 = [142, 262]
            pixel_distance_in_meters_1 = aligned_depth_frame.get_distance(depth_pixel_1[0], depth_pixel_1[1])
            pixel_distance_in_meters_2 = aligned_depth_frame.get_distance(depth_pixel_2[0], depth_pixel_2[1])
            depth_point_1 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel_1, depth_scale * pixel_distance_in_meters_1)
            depth_point_2 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel_2, depth_scale * pixel_distance_in_meters_2)
            #
            # print("pt 1: ", np.array(depth_point_1)/ depth_scale)
            # print("pt 2: ", np.array(depth_point_2) / depth_scale)
            # print ("##########################")
            #
            # import code;
            # code.interact(local=dict(globals(), **locals()))

            # Timing
            currentDT = datetime.datetime.now()
            currentDT = str(currentDT).split(' ')[1].split(':')[2]
            current_seconds = int(currentDT.split('.')[0])
            current_millis = int(currentDT.split('.')[1])//1000
            if frame_count ==0:
                previous_seconds = current_seconds
                previous_millis = current_millis
            if previous_seconds == current_seconds:
                time_step = current_millis - previous_millis
            else:
                time_step = current_millis + (1000 - previous_millis)
            previous_seconds = current_seconds
            previous_millis = current_millis
            time_data[frame_count] = time_step

            # Getting IMU readings
            accel = accel_data(frames[2].as_motion_frame().get_motion_data())
            gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())
            accelerometer_data[frame_count, :] = accel
            gyroscope_data[frame_count, :] = gyro


            color_image = np.asanyarray(aligned_color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # Cropping pixels with depth ~ 2m
            depth_image = depth_image * (depth_image < 2000)
            if float(np.max(depth_image)) == 0:
                continue

            # Converting depth values to 8-bit integers
            depth_scaled = ((depth_image / float(np.max(depth_image))) * 255).astype('uint8')

            # Constructing an RGBD image from the frames
            rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
            rgbd_image[:, :, 0:3] = color_image
            rgbd_image[:, :, 3] = depth_scaled

            # Saving the images
            image_name = frame_count
            color_image_path = os.path.join(dataset_path, 'rgb', str(image_name) + '.png')
            depth_image_path = os.path.join(dataset_path, 'depth', str(image_name) + '.png')

            Image.fromarray(color_image).save(color_image_path)
            Image.fromarray(depth_scaled).save(depth_image_path)

            # Visualization
            depth_3_channel = cv2.cvtColor(depth_scaled,cv2.COLOR_GRAY2RGB)
            color_image_new = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            images = np.hstack((color_image_new, depth_3_channel))
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)

            frame_count = frame_count + 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                plt.imshow(color_image)
                plt.show()

    finally:
        pipeline.stop()
        return accelerometer_data, gyroscope_data, time_data, depth_intrinsics, depth_scale
#
#
num_frames = 150
frame_rate = 15
noise_level = 0.025
accelerometer_data, gyroscope_data, time_data, depth_intrinsics, depth_scale = capture_frames(num_frames,frame_rate, 'Tracking Dataset 7')
# # import code;
# #
# # code.interact(local=dict(globals(), **locals()))
#
#
# accelerometer_filtered = np.zeros(accelerometer_data.shape)
# for i in list(range(3)):
#     accelerometer_data[:, i] = accelerometer_data[:,i] - np.mean(accelerometer_data[:,i])
#     noise = np.logical_or((accelerometer_data[:, i] < -1 * noise_level), (accelerometer_data[:, i] > noise_level))
#     accelerometer_data[:, i] = accelerometer_data[:, i] * noise
#     max_unfiltered = np.max(accelerometer_data[:, i])
#     if np.sum(accelerometer_data[:, i]) == 0:
#         accelerometer_filtered[:, i] = accelerometer_data[:, i]
#         continue
#     # accelerometer_filtered[:,i] = butterworth_filter('low',accelerometer_data[:,i],1.5,15)
#     # accelerometer_filtered[:, i] = butterworth_filter('high',accelerometer_filtered[:,i],0.01,15)
#     accelerometer_filtered[:, i] = accelerometer_data[:, i]
#     max_filtered = np.max(accelerometer_filtered[:, i])
#     scale = max_unfiltered / max_filtered
#     accelerometer_filtered[:, i] = accelerometer_filtered[:, i] * scale
#
#
# # t = 1/frame_rate
# t = time_data/1000
# dx = np.zeros(num_frames)
# dy = np.zeros(num_frames)
# dz = np.zeros(num_frames)
# vx = 0; vy = 0; vz = 0
# i = 0
# for a in accelerometer_filtered:
#     dx[i] = (vx*t[i]) + 0.5*a[0]*t[i]**2
#     dy[i] = (vy*t[i]) + 0.5*a[1]*t[i]**2
#     dz[i] = (vz*t[i]) + 0.5*a[2]*t[i]**2
#     vx = vx + a[0] * t[i]
#     vy = vy + a[1] * t[i]
#     vz = vz + a[2] * t[i]
#     i = i + 1
#
# time = list(range(num_frames))
# x_acceleration = accelerometer_data[:,0]
# y_acceleration = accelerometer_data[:,1]
# z_acceleration = accelerometer_data[:,2]
# fig4,(ax1, ax2, ax3) = plt.subplots(3,1, sharex='col', sharey='row')
# ax1.plot(time, x_acceleration); ax1.set_title('x'); ax1.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax2.plot(time, y_acceleration);ax2.set_title('y'); ax2.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax3.plot(time, z_acceleration);ax3.set_title('z'); ax3.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax1.set_title('A unfiltered')
# fig4.show()
#
# from scipy.fftpack import fft
# N = num_frames
# T = 1/frame_rate
# x = np.linspace(0.0, N*T, N)
# x_acceleration_fft = fft(x_acceleration)
# x_acceleration_filtered_fft = fft(accelerometer_filtered[:,0])
# xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
#
# fig5,(ax1) = plt.subplots(1,1)
# ax1.plot(xf, 2.0/N * np.abs(x_acceleration_fft[0:int(N/2)]))
# ax1.grid()
# fig5.show()
#
# fig6,(ax1) = plt.subplots(1,1)
# ax1.plot(xf, 2.0/N * np.abs(x_acceleration_filtered_fft[0:int(N/2)]))
# ax1.grid()
# fig6.show()
#
# time = list(range(num_frames))
# accelerometer_data = accelerometer_filtered
# x_acceleration = accelerometer_data[:,0]
# y_acceleration = accelerometer_data[:,1]
# z_acceleration = accelerometer_data[:,2]
# fig,(ax1, ax2, ax3) = plt.subplots(3,1)
# # import code; code.interact(local=dict(globals(), **locals()))
#
# ax1.plot(time, x_acceleration); ax1.set_title('x'); ax1.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax2.plot(time, y_acceleration);ax2.set_title('y'); ax2.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax3.plot(time, z_acceleration);ax3.set_title('z'); ax3.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax1.set_title('A filtered')
# fig.show()
#
# x_gyro = gyroscope_data[:,0]
# y_gyro = gyroscope_data[:,1]
# z_gyro = gyroscope_data[:,2]
# fig2,(ax1, ax2, ax3) = plt.subplots(3,1)
# ax1.plot(time, x_gyro); ax1.set_title('x'); ax1.set_ylim(np.min(gyroscope_data)-0.05, np.max(gyroscope_data)+0.05)
# ax2.plot(time, y_gyro);ax2.set_title('y'); ax2.set_ylim(np.min(gyroscope_data)-0.05, np.max(gyroscope_data)+0.05)
# ax3.plot(time, z_gyro);ax3.set_title('z'); ax3.set_ylim(np.min(gyroscope_data)-0.05, np.max(gyroscope_data)+0.05)
# ax1.set_title('Gyroscope unfiltered')
# fig2.show()
#
# fig4,(ax1, ax2, ax3) = plt.subplots(3,1)
# d = np.array([dx, dy, dz])
# ax1.plot(time, dx); ax1.set_title('x'); ax1.set_ylim(np.min(d)-0.05, np.max(d)+0.05)
# ax2.plot(time, dy);ax2.set_title('y'); ax2.set_ylim(np.min(d)-0.05, np.max(d)+0.05)
# ax3.plot(time, dz);ax3.set_title('z'); ax3.set_ylim(np.min(d)-0.05, np.max(d)+0.05)
# ax1.set_title('distance')
# fig4.show()
#
# plt.show()