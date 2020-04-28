########################################################################################################################
# ECE 9516 - Topics in Autonomous Robotics - Final Project
# Mena SA Kamel
# Student Number: 251064703
# MESc Candidate, Robotics and Control
# Electrical and Computer Engineering, Western University
########################################################################################################################

import draw_target_region
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io
import math
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from skimage.feature import hog
import skimage.io
import numpy as np
from skimage import exposure
from PIL import Image

def validate_box(x, y, hx, hy, image_shape):

    # Validates if a box is within the image
    if x <= 0:
        x = 10
    if x >= image_shape[1]:
        x = image_shape[1] - 10
    if y <= 0:
        y = 10
    if y >= image_shape[0]:
        y = image_shape[0] - 10
    if hx <= 2:
        hx = 3
    if hy <= 2:
        hy = 3
    if ((x + 0.5*hx) > image_shape[1]):
        diff = int((x + 0.5*hx) - image_shape[1])
        hx = hx - 2*(abs(diff)) - 1
    elif ((x - 0.5*hx) < 0):
        diff = int(x - 0.5*hx)
        hx = hx - 2*(abs(diff)) - 1
    if ((y + 0.5*hy) > image_shape[0]):
        diff = int((y + 0.5*hy) - image_shape[0])
        hy = hy - 2*(abs(diff)) - 1
    elif ((y - 0.5*hy) < 0):
        diff = int(y - 0.5*hy)
        hy = hy - 2*(abs(diff)) - 1
    return x, y, hx, hy

def get_hog_distributions(image, num_bins):

    # Work in progress
    pixels_per_cell = 8
    fd, hog_image = skimage.feature.hog(image, orientations=num_bins, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    x_dimension = int(image.shape[1] / pixels_per_cell)
    y_dimension = int(image.shape[0] / pixels_per_cell)
    fd_reshaped = fd.reshape(x_dimension*y_dimension, num_bins)
    gradients = np.max(fd_reshaped, axis = 1)
    orientations = np.argmax(fd_reshaped, axis = 1)
    orientations = orientations.reshape(y_dimension, x_dimension)
    gradients = gradients.reshape(y_dimension, x_dimension)
    return [orientations, gradients]

def get_color_distribution(coordinates, image, bins):
    x, y, hx, hy = coordinates
    x0 = x - int(0.5 * hx)
    x1 = x + int(0.5 * hx)
    y0 = y - int(0.5 * hy)
    y1 = y + int(0.5 * hy)

    target_region = image[y0:y1, x0:x1, :]
    target_region = np.array(target_region)

    # Creating a weighting chart, w
    hx = target_region.shape[1]
    hy = target_region.shape[0]
    x_coords = np.arange(hx) - int((hx) / 2)
    y_coords = np.arange(hy) - int((hy) / 2)
    X, Y = np.meshgrid(x_coords, y_coords, sparse=True)
    a = math.sqrt(hx ** 2 + hy ** 2)
    d = np.sqrt(X ** 2 + Y ** 2) / a
    weights = (d < 1) * (1 - d ** 2)
    weights_flattened = weights.flatten()

    ####################################################################################################################
    # Visualizing weights
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(weights, cmap='gray')
    # axs[0].set_title('k(r)')
    # axs[1].imshow(target_region[:,:, 0:3].astype('uint8'))
    # axs[1].set_title('RoI')
    # plt.show()
    ####################################################################################################################

    # Calculating the histogram
    channel_bins = bins
    num_channels = target_region.shape[2]
    histogram = []
    for channel in list(range(num_channels)):
        channel_pixels = target_region[:,:,channel].flatten()
        if channel == 3:
            channel_bins = int(0.5*bins)
        # Calculating the weighted histogram
        channel_histogram = np.histogram(channel_pixels, bins=channel_bins, range=(0, 255), weights=weights_flattened)[0]
        histogram.extend(channel_histogram/ np.sum(channel_histogram)*0.25)

    histogram = np.array(histogram).flatten()
    histogram = histogram
    histogram = histogram.reshape((1, len(histogram)))

    ####################################################################################################################
    # Visualizing color distribution
    # fig, ax1 = plt.subplots(1)
    # bar_chart = ax1.bar(list(range(histogram.shape[1])), histogram[0])
    # colours = ['r', 'g', 'b', 'k']
    # for i in list(range(histogram.shape[1])):
    #     index = int(i/bins)
    #     if index > 3:
    #         index = 3
    #     bar_chart[i].set_color(colours[index])
    # ax1.set_title('Color distribution p(y)')
    # ax1.set_xlabel('Bins')
    # ax1.set_ylabel('p(y)')
    # plt.show()
    ####################################################################################################################
    return histogram

def plot_state(state, image, frame_number = 0, mean_state = '', save_dir = '', save = True, display_mean = False, display_all_states = False, img_name = '', title = '', history = []):

    # Plotting function, can specify if you want to plot all states, save or not, plotting motion history
    num_particles = state.shape[0]
    fig2, ax2 = plt.subplots(1)
    color_image = image[:,:,0:3].astype('uint8')
    ax2.imshow(color_image)
    ax2.set_title(title)
    if display_all_states:
        for i in list(range(num_particles)):
            Hx = int(state[i][4])
            Hy = int(state[i][5])
            x_coord = int(state[i][0]) - int(Hx / 2)
            y_coord = int(state[i][2]) - int(Hy / 2)
            rect = patches.Rectangle((x_coord, y_coord), Hx, Hy, linewidth=1, edgecolor='g', facecolor='none')
            ax2.add_patch(rect)
    if display_mean:
        i = 0
        if not len(history) ==0:
            for prev_state in history:
                if i%5 == 0:
                    history_new = np.array(history)
                    history_x = history_new[:, 0]
                    history_y = history_new[:, 2]
                    Hx = int(prev_state[4])
                    Hy = int(prev_state[5])
                    x_coord = int(prev_state[0]) - int(Hx / 2)
                    y_coord = int(prev_state[2]) - int(Hy / 2)
                    plt.plot(history_x, history_y, linewidth=1, color='c')
                    history_rect = patches.Rectangle((x_coord, y_coord), Hx, Hy, linewidth=0.5, edgecolor='r',
                                                      facecolor='none')
                    ax2.add_patch(history_rect)
                i = i + 1
        Hx = int(mean_state[4])
        Hy = int(mean_state[5])
        x_coord = int(mean_state[0]) - int(Hx / 2)
        y_coord = int(mean_state[2]) - int(Hy / 2)
        mean_rect = patches.Rectangle((x_coord, y_coord), Hx, Hy, linewidth=2.5, edgecolor='b', facecolor='none')
        ax2.add_patch(mean_rect)
    if save:
        fig_name = os.path.join(save_dir, str(frame_number) + img_name +'.png')
        fig2.savefig(fig_name)
    fig2.canvas.draw()
    image_from_plot = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image_from_plot

def get_bhattacharyya_coef(p, q):

    # Calculating the Bhattacharya coefficient
    m = p.shape[1]
    coef = 0
    for bin in list(range(m)):
        coef = coef + math.sqrt(p[0, bin] * q[0, bin])
        if coef > 1:
            coef = 1
    return coef

def get_bhattacharyya_distance(p, q):

    # Calculating the Bhattacharya distance from the Bhattacharya coefficient
    bhattacharyya_distance = math.sqrt(1 - get_bhattacharyya_coef(p,q))
    return bhattacharyya_distance

def get_initial_state(x, y, Hx, Hy, q, num_particles, num_bins, image):

    # Defines the initial state of the particle filter. All samples are placed around the initial target position with
    # a spread of 0.01
    initial_spread_ratio = 0.01
    sigma = 0.05 #  decrease to make it narrower / more selective

    q = get_color_distribution([x, y, Hx, Hy], image, num_bins)

    x_init = np.random.normal(x, math.sqrt(initial_spread_ratio * Hx), size=(num_particles - 1)).astype('int16')
    x_init = np.append(x_init, x)  # Adding the target particle x coordinate

    y_init = np.random.normal(y, math.sqrt(initial_spread_ratio * Hy), size=(num_particles - 1)).astype('int16')
    y_init = np.append(y_init, y)  # Adding the target particle y coordinate

    Hx_init = np.random.normal(Hx, math.sqrt(initial_spread_ratio * Hx), size=(num_particles - 1)).astype('int16')
    Hx_init = np.append(Hx_init, Hx)  # Adding the target particle x coordinate

    Hy_init = np.random.normal(Hy, math.sqrt(initial_spread_ratio * Hy), size=(num_particles - 1)).astype('int16')
    Hy_init = np.append(Hy_init, Hy)  # Adding the target particle y coordinate

    x_dot_init = np.zeros(num_particles)
    y_dot_init = np.zeros(num_particles)

    s_t_1 = np.zeros((num_particles, 6))
    pi_t_1 = np.zeros(num_particles)

    image_shape = image.shape
    for i in list(range(num_particles)):
        x, y, Hx, Hy = [x_init[i], y_init[i], Hx_init[i], Hy_init[i]]
        x, y, Hx, Hy = validate_box(x, y, Hx, Hy, image_shape)
        s_t_1[i, 0] = x
        s_t_1[i, 1] = x_dot_init[i]
        s_t_1[i, 2] = y
        s_t_1[i, 3] = y_dot_init[i]
        s_t_1[i, 4] = Hx
        s_t_1[i, 5] = Hy
        p = get_color_distribution([x, y, Hx, Hy], image, num_bins)
        d = get_bhattacharyya_distance(p, q)
        pi_t_1[i] = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d ** 2) / (2 * sigma ** 2))
    pi_t_1 = pi_t_1 / np.max(pi_t_1)
    return s_t_1, pi_t_1

def create_cummulative_probability_distribution(pi_t_1):
    # Creating a normalized cumulative probability distribution
    num_particles = len(pi_t_1)
    c_t_1 = np.zeros(num_particles)
    c_t_1[0] = pi_t_1[0]
    for i in list(range(1, num_particles)):
        c_t_1[i] = c_t_1[i - 1] + pi_t_1[i]
    c_t_1 = c_t_1 / c_t_1[-1]
    return c_t_1

def sample_from_cumulative_distribution(c_t_1, frame_number, results_folder):

    # Samples particles using stochastic universsal sampling given the past state cumulative distribution function
    num_samples = len(c_t_1)
    new_samples = []
    u = np.zeros(num_samples + 1)
    u[0] = np.random.uniform(0, 1/num_samples)
    i = 0
    for j in list(range(num_samples)):
        while (u[j] > c_t_1[i]):
            i = i + 1
        new_samples.append(i)
        u[j+1] = u[j] + 1/num_samples

    ####################################################################################################################
    # Visualizing resampling
    # fig3, ax3 = plt.subplots(1)
    # ax3.plot(list(range(num_samples)), c_t_1)
    # ax3.scatter(new_samples, c_t_1[new_samples], color = 'r')
    # ax3.set_title("Particle cumulative distribution function - Frame: " + str(frame_number))
    # ax3.set_xlabel('Particle index, i')
    # ax3.set_ylabel('Cumulative probability, c')
    # fig_name = os.path.join(results_folder, str(frame_number) + '_sus')
    # fig3.savefig(fig_name)
    # plt.close()
    ####################################################################################################################
    return new_samples

def get_coords(particle):
    # Extracts the RoI coordinates in the format [x, y, Hx, Hy] = [x(center), y(center), width, height]
    particle = particle.reshape(1, len(particle))
    x = int(particle[0][0])
    y = int(particle[0][2])
    Hx = int(particle[0][4])
    Hy = int(particle[0][5])
    return [x, y, Hx, Hy]

def get_mean_state(s_t, pi_t, sigma, q, image, num_bins):

    # Calculates the mean state using the provided sample set
    pi_observed_normalized = pi_t / np.sum(pi_t)
    s_t_mean = np.zeros(6)
    num_particles = len(pi_t)
    image_shape = image.shape
    for i in list(range(num_particles)):
        s_t_mean = s_t_mean + (s_t[i] * pi_observed_normalized[i])
    s_t_mean = s_t_mean.astype('int16')
    x, y, Hx, Hy = get_coords(s_t_mean)
    s_t_mean[0], s_t_mean[2], s_t_mean[4], s_t_mean[5] = validate_box(x, y, Hx, Hy, image_shape)
    x, y, Hx, Hy = get_coords(s_t_mean)
    p_estimated = get_color_distribution([x, y, Hx, Hy], image, num_bins)
    d_mean_state = get_bhattacharyya_distance(p_estimated, q)
    pi_t_mean = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d_mean_state ** 2) / (2 * sigma ** 2))
    return [s_t_mean, pi_t_mean]

def particle_filter(s_t_1, pi_t_1, q, num_bins, image, probabilities, target_updates, results_folder = '',
                    frame_number = 0, start_num = 0, pi_thresh = 1):
    # Inputs:
    # s_t_1 - Previous state particle states
    # pi_t_1 - Previous state particle weights
    # q - target model
    # image - image of current frame
    # probabilities - past mean state weights
    # target_updates - tracks whether the target model was updated or not
    # pi_thresh - Threshold for target model updates

    num_particles = len(pi_t_1)
    image_shape = image.shape
    sigma = 0.05 # Specifies the sensitivity of the weights
    alpha = 0.25 # Balances between current and previous state when updating the target model

    c_t_1 = create_cummulative_probability_distribution(pi_t_1) # Creating a cumulative probability distribution

    # Selecting N samples based on weights
    resampled_particles_indices = sample_from_cumulative_distribution(c_t_1, frame_number, results_folder)
    s_t_1_resampled = s_t_1[resampled_particles_indices]


    s_t = np.zeros(s_t_1_resampled.shape)

    # Degining state transition parameters
    dt = 1
    A = np.array([[1, dt, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, dt, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1], ])

    # Propagating the particles using the state transition model
    for i in list(range(num_particles)):
        particle = s_t_1_resampled[i, :]
        t = 1

        # Accelerations in the x and y direction - To be replaced by acceleration from accelerometer + gyro
        x2 = int(np.random.normal(0, math.sqrt(300)))
        y2 = int(np.random.normal(0, math.sqrt(300)))

        # Changes in RoI width and height
        hx_noise = int(np.random.normal(0, math.sqrt(10)))
        hy_noise = int(np.random.normal(0, math.sqrt(10)))

        # System noise
        w_t_1 = np.array([[0.5 * x2 * t ** 2],
                          [t * x2],
                          [0.5 * y2 * t ** 2],
                          [t * y2],
                          [hx_noise],
                          [hy_noise]])


        particle = particle.reshape(1, len(particle))
        particle_new = A.dot(np.transpose(particle)) + w_t_1
        x, y, Hx, Hy = get_coords(particle_new)

        # Validating the bounding boxes are within the image
        particle_new[0], particle_new[2], particle_new[4], particle_new[5] = validate_box(x, y, Hx, Hy, image_shape)
        s_t[i, :] = np.transpose(particle_new)

    # Measurement update
    pi_t = np.zeros(num_particles)
    distance_data = []
    weights_data = []
    for i in list(range(num_particles)):
        particle = s_t[i, :]
        x, y, Hx, Hy = get_coords(particle)
        particle[0], particle[2], particle[4], particle[5] = validate_box(x, y, Hx, Hy, image_shape)
        x, y, Hx, Hy = get_coords(particle)
        # Calculating the probability distribution (histogram) for each particle
        p = get_color_distribution([x, y, Hx, Hy], image, num_bins)

        # Calculating the Bhattacharya distance between the target region and the current particle
        d = get_bhattacharyya_distance(p, q)

        # Computing the weight for each particle
        weight = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d ** 2) / (2 * sigma ** 2))

        pi_t[i] = weight
        distance_data.append([x, y, d])
        weights_data.append([x, y, weight])

    ####################################################################################################################
    # Visualizing Bhattacharya Distance and Weights
    # fig = plt.figure()
    # data = np.array(distance_data)
    # ax2 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax2.scatter(data[:, 0], data[:, 1], data[:, 2])
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('d')
    # ax2.set_title('Bhattacharya distance, d')
    #
    # data = np.array(weights_data)
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax2.scatter(data[:, 0], data[:, 1], data[:, 2])
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('π')
    # ax2.set_title('Weights, π, σ = 0.05')
    # plt.show()
    ####################################################################################################################

    # Calculating the past and present mean states
    s_t_1_mean, pi_t_1_mean = get_mean_state(s_t_1, pi_t_1, sigma, q, image, num_bins)
    s_t_mean, pi_t_mean = get_mean_state(s_t, pi_t, sigma, q, image, num_bins)
    x, y, Hx, Hy = get_coords(s_t_mean)
    p_mean = get_color_distribution([x, y, Hx, Hy], image, num_bins)

    probabilities.append(pi_t_mean)
    update = 0
    if frame_number - start_num == 0:
        # Initializing pi_thresh if this is the first frame
        pi_thresh = 0.05*pi_t_mean

    # Target model update
    if abs(pi_t_mean - pi_t_1_mean) < pi_thresh:
        q = (1 - alpha)*q + (alpha *p_mean)
        update = pi_thresh*2

    target_updates.append(update)
    s_t_1 = s_t
    pi_t_1 = pi_t

    return s_t_1, pi_t_1, s_t_mean, pi_t_mean, q, probabilities, target_updates, pi_thresh


def run():
    # This is the main function. It creates a new directory to save the logs, reads all image frames and calls particle
    # filter algorithm on each frame

    dataset_path = 'Datasets/Tracking Dataset 6' # Dataset path
    num_images = len(os.listdir(os.path.join(dataset_path, 'rgb')))
    start_image_number = 5 # Start tracking from this frame
    frame_numbers = list(list(range(start_image_number, num_images + start_image_number - 1)))
    frame_numbers = frame_numbers[0:150]

    num_particles = 500 # specifying the number of particles
    num_bins = 8 # Number of bins for the RGB color channels
    pi_thresh = 1

    # Creating the direcctoy under the Logs folder to store the results
    current_time = datetime.now()
    results_folder = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    results_folder = dataset_path.split('/')[1] + '-' + results_folder
    results_folder = os.path.join('Logs',results_folder)
    os.mkdir(results_folder)

    probabilities = []
    target_updates = []
    counter = 0
    state_history = []

    for frame_number in frame_numbers:
        frame_name = str(frame_number) + '.png'

        # Reading the color and depth images
        color_image = skimage.io.imread(os.path.join(dataset_path, 'rgb', frame_name))
        depth_image = skimage.io.imread(os.path.join(dataset_path, 'depth', frame_name))

        # Scaling depth image to 8 bits to match the RGB channels
        depth_scaled = ((depth_image / float(np.max(depth_image))) * 255).astype('uint8')
        rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
        rgbd_image[:, :, 0:3] = color_image
        rgbd_image[:, :, 3] = depth_scaled

        if counter == 0:
            # Initializing the particle filter states, and target model, q
            x, y, Hx, Hy = draw_target_region.draw_region(image, image_path)
            q = get_color_distribution([x, y, Hx, Hy], rgbd_image, num_bins)
            s_t_1, pi_t_1 = get_initial_state(x, y, Hx, Hy, q, num_particles, num_bins, rgbd_image)

        # Calling the particle filter algorithm and updating the current state
        s_t, pi_t, s_t_mean, pi_t_mean, q, probabilities, target_updates, pi_thresh = particle_filter(s_t_1, pi_t_1, q, num_bins,
                                                                                           rgbd_image, probabilities,
                                                                                           target_updates, results_folder,
                                                                                           frame_number, start_image_number,
                                                                                           pi_thresh)
        s_t_1 = s_t
        pi_t_1 = pi_t
        state_history.append(s_t_mean)

        # Plotting the results to the results folder
        plot_state(s_t, rgbd_image, frame_number, mean_state= s_t_mean, save_dir=results_folder, save=True, display_mean=True,
                   display_all_states=False, img_name='', title='Frame #: '+str(frame_number - start_image_number), history = [])
        counter = counter + 1

    ####################################################################################################################
    # Visualizing target model updates
    # fig3, ax3 = plt.subplots(1)
    # ax3.plot(frame_numbers, probabilities, label='Mean state weight,' + r'$\pi_{E[s]}$')
    # ax3.plot(frame_numbers, target_updates, label='Target model updates')
    # ax3.set_xlabel('Frame numbers')
    # ax3.set_ylabel('Mean state weight')
    # ax3.set_title('Target model updates')
    # fig3.show()
    # fig_name = os.path.join(results_folder, 'Mean State Weight')
    # fig3.savefig(fig_name)
    # ax3.legend()
    # plt.show()
    ####################################################################################################################


# Running Instructions
# 1. When the first image gets displayed on the screen, begin selecting the RoI by double clicking on the image
# 2. Specify the points necessary to describe the RoI by left clicking
# 3. Double Right click when done, a yellow box should highlight the RoI
# 4. Press Esc. to exit the region selector, the tracker will start and output the results in the Logs folder

run()