import pyrealsense2 as rs
import numpy as np
import os
import cv2
from PIL import Image

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def capture_frames(num_frames, dataset_name = 'Tracking Dataset Accelerometer Gyro'):
    ## Setting up work directories

    dataset_name = dataset_name
    if not os.path.exists(dataset_name):
        os.mkdir(dataset_name)
        folders = ['rgb', 'depth']
        for folder in folders:
            folder_path = os.path.join(dataset_name, folder)
            os.mkdir(folder_path)

    pipeline = rs.pipeline()
    config = rs.config()
    frame_rate = 15
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

    for i in list(range(frame_rate*5)):
        frames = pipeline.wait_for_frames()
    print('Began Capturing Images...')
    try:
        while frame_count < num_frames:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            # import code;
            # code.interact(local=dict(globals(), **locals()))
            accel = accel_data(frames[2].as_motion_frame().get_motion_data())
            gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())
            accelerometer_data[frame_count, :] = accel
            gyroscope_data[frame_count, :] = gyro
            print("accelerometer: ", accel)
            print("gyro: ", gyro)
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image = depth_image * (depth_image < 2000)
            if float(np.max( depth_image )) == 0:
                continue
            depth_scaled = ((depth_image / float(np.max( depth_image ))) * 255).astype('uint8')
            rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
            rgbd_image[:, :, 0:3] = color_image
            rgbd_image[:, :, 3] = depth_scaled
            depth_3_channel = cv2.cvtColor(depth_scaled,cv2.COLOR_GRAY2RGB)
            color_image_new = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            images = np.hstack((color_image_new, depth_3_channel))
            image_name = frame_count
            color_image_path = os.path.join(dataset_name, 'rgb', str(image_name)+'.png')
            depth_image_path = os.path.join(dataset_name, 'depth', str(image_name)+'.png')

            color_image = Image.fromarray(color_image)
            depth_scaled = Image.fromarray(depth_scaled)
            # color_image.save(color_image_path)
            # depth_scaled.save(depth_image_path)

            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)
            frame_count = frame_count + 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
        return accelerometer_data, gyroscope_data

# import matplotlib.pyplot as plt
# accelerometer_data, gyroscope_data = capture_frames(250)
# time = list(range(250))
# x_acceleration = accelerometer_data[:,0]
# y_acceleration = accelerometer_data[:,1]
# z_acceleration = accelerometer_data[:,2]
# fig,(ax1, ax2, ax3) = plt.subplots(3,1)
# ax1.plot(time, x_acceleration); ax1.set_title('x')
# ax2.plot(time, y_acceleration);ax2.set_title('y')
# ax3.plot(time, z_acceleration);ax3.set_title('z')
# fig.show()
#
# x_gyro = gyroscope_data[:,0]
# y_gyro = gyroscope_data[:,1]
# z_gyro = gyroscope_data[:,2]
# fig2,(ax1, ax2, ax3) = plt.subplots(3,1)
# ax1.plot(time, x_gyro); ax1.set_title('x')
# ax2.plot(time, y_gyro);ax2.set_title('y')
# ax3.plot(time, z_gyro);ax3.set_title('z')
# fig2.show()
#
# plt.show()
# import code;
#
# code.interact(local=dict(globals(), **locals()))
#
# import code;
#
# code.interact(local=dict(globals(), **locals()))