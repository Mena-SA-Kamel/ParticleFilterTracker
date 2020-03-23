import capture_frames
import draw_target_region
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import skimage.io
import math
from datetime import datetime
import matplotlib.patches as patches
import operator

image_shape = [480, 640]
dataset_path = 'Tracking Dataset/rgb'
num_images = len(os.listdir(dataset_path))
first_image_in_dataset = 23
start_image_number = 53
frames = list(list(range(start_image_number, num_images + first_image_in_dataset)))
# import code; code.interact(local=dict(globals(), **locals()))
frame_name = str(frames[0]) + '.png'
image_path = os.path.join(dataset_path, frame_name)
img = skimage.io.imread(image_path)


def get_ordered_frame_list(start_frame, end_frame):
    image_list_ordered = []
    image_numbers = list(range(start_frame, end_frame + 1))
    for image_number in image_numbers:
        image_list_ordered.append(str(image_number) + '.png')
    return image_list_ordered

def k(r):
    if r < 1:
        return (1 - r**2)
    else:
        return 0

def kronecker_delta(num):
    if num == 0:
        return 1
    else:
        return 0

def get_bhattacharyya_coef(p, q):
    m = p.shape[1]
    num_channels = p.shape[0]
    coef = np.zeros(num_channels)
    for channel in list(range(num_channels)):
        for bin in list(range(m)):
            coef[channel] = coef[channel] + math.sqrt(p[channel,bin] * q[channel,bin])
            if coef[channel] > 1:
                coef[channel] = 1
    return coef

def get_bhattacharyya_distance(p, q):
    num_channels = p.shape[0]
    distances = np.zeros(num_channels)
    for channel in list(range(num_channels)):
        distances[channel] = math.sqrt(1 - get_bhattacharyya_coef(p,q)[channel])

    return distances

def get_color_distribution(coordinates, image, bins):
    x, y, hx, hy = coordinates
    if (hx == 0):
        x0 = x - 1
        x1 = x + 1
    else:
        x0 = x - int(0.5 * hx)
        x1 = x + int(0.5 * hx)
    if (hy == 0):
        y0 = y - 1
        y1 = y + 1
    else:
        y0 = y - int(0.5 * hy)
        y1 = y + int(0.5 * hy)
    if x0 <0 or x1 < 0 or y0 < 0 or y1< 0 :
        print ('errorrrr')
        import code;

        code.interact(local=dict(globals(), **locals()))

    target_region = image[y0:y1, x0:x1, :]
    target_region = np.array(target_region)
    a = math.sqrt(hx ** 2 + hy ** 2)

    hx = target_region.shape[1]
    hy = target_region.shape[0]
    x_coords = np.arange(hx) - int((hx) / 2)
    y_coords = np.arange(hy) - int((hy) / 2)
    X, Y = np.meshgrid(x_coords, y_coords, sparse=True)
    d = np.sqrt(X ** 2 + Y ** 2) / a
    weights = (d < 1) * (1 - d ** 2)
    weights_flattened = weights.flatten()

    num_channels = target_region.shape[2]
    histogram = []
    for channel in list(range(num_channels)):
        channel_pixels = target_region[:,:,channel].flatten()
        channel_histogram = np.histogram(channel_pixels, bins=bins, range=(0, 255), weights=weights_flattened)[0]
        histogram.append(channel_histogram/ np.sum(channel_histogram))
    histogram = np.array(histogram).flatten()
    histogram = histogram / np.sum(histogram)
    histogram = histogram.reshape((1, len(histogram)))
    for hist in histogram[0]:
        if np.isnan(hist):
            print('nan found')
    return histogram

def validate_box(x, y, hx, hy):
    if x <= 0:
        x = 10
    if x >= image_shape[1]:
        x = image_shape[1] - 10
    if y <= 0:
        y = 10
    if y >= image_shape[0]:
        y = image_shape[0] - 10
    if hx <= 0:
        hx = 2
    if hy <= 0:
        hy = 2
    if ((x + 0.5*hx) > image_shape[1]):
        diff = int((x + 0.5*hx) - image_shape[1])
        hx = hx - 2*(abs(diff)) - 1
    if ((x - 0.5*hx) < 0):
        diff = int(x - 0.5*hx)
        hx = hx - 2*(abs(diff)) - 1
    if ((y + 0.5*hy) > image_shape[0]):
        diff = int((y + 0.5*hy) - image_shape[0])
        hy = hy - 2*(abs(diff)) - 1
    if ((y - 0.5*hy) < 0):
        diff = int(y - 0.5*hy)
        hy = hy - 2*(abs(diff)) - 1
    if x <= 0 or y <= 0 or hx <= 0 or hy <= 0:
        print('error validate box')
        import code;

        code.interact(local=dict(globals(), **locals()))
    return x, y, hx, hy

def sample_from_cumulative_distribution(c):
    num_samples = len(c)
    new_samples = []
    u = np.zeros(num_samples + 1)
    u[0] = np.random.uniform(0, 1/num_samples)
    i = 0
    for j in list(range(num_samples)):
        while (u[j] > c[i]):
            i = i + 1
        new_samples.append(i)
        u[j+1] = u[j] + 1/num_samples
    return new_samples

def get_coords(particle):
    particle = particle.reshape(1, len(particle))
    x = int(particle[0][0])
    y = int(particle[0][2])
    Hx = int(particle[0][4])
    Hy = int(particle[0][5])
    return [x, y, Hx, Hy]

########################################## Testing ##########################################
# x,y,Hx,Hy = draw_target_region.draw_region(image_path)
# sample_region = img[y0:y1, x0:x1, :]
# sample_region = Image.fromarray(sample_region)
# target_region_image.save('target_region.png')
# target_region_image = skimage.io.imread('target_region.png')
# target_region_image = np.array(target_region_image)
# plt.imshow(target_region_image)
# plt.show()
# binned_histogram = get_histogram(target_region_image, 8)
# p = get_color_distribution([x0,x1,y0,y1], img, 8)
# p = get_color_distribution([200,300,100,100], img, 8)
# q = get_color_distribution([x, y, Hx, Hy], img, 8)
# q = get_color_distribution([358, 299, 112, 210], img, 8)
# print('Bhattacharyya Coef: ', get_bhattacharyya_coef(p,q))
# print ('Bhattacharyya Dist: ', get_bhattacharyya_distance(p,q))

########################################## Defining Dynamic Model ##########################################
# x, y, Hx, Hy = [358, 299, 112, 210]
x, y, Hx, Hy = draw_target_region.draw_region(image_path)
t = 1

A = np.array([[1, t, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, t, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],])


########################################## Initialization ##########################################
num_particles = 200
num_particles = num_particles - 1
frame = 0
sigma = 0.001
num_bins = 8
num_channels = 3
pi_thresh = 0.75
alpha = 0.4

results_folder = 'Result 2 - sigma_ ' + str(sigma) +  '_particles_' + str(num_particles) + '_alpha_ ' + str(alpha)
if not os.path.exists(results_folder):
    os.mkdir(results_folder)


# Initializing the particle x and y coordinates
# originally 10, 10
x_init = np.random.normal(x, math.sqrt(10*Hx), size=(num_particles)).astype('int16')
y_init = np.random.normal(y, math.sqrt(10*Hy), size=(num_particles)).astype('int16')
#
# x_init = np.random.uniform(10, 630, size=(num_particles)).astype('int16')
# y_init = np.random.uniform(10, 470, size=(num_particles)).astype('int16')

x_init = np.append(x_init, x) # Adding the target particle x coordinate
y_init = np.append(y_init, y) # Adding the target particle x coordinate

# Initializing the particle Hx and Hy states
Hx_init = np.random.normal(Hx, math.sqrt(Hx), size=(num_particles)).astype('int16')
Hy_init = np.random.normal(Hy, math.sqrt(Hy), size=(num_particles)).astype('int16')
Hx_init = np.append(Hx_init, Hx) # Adding the target particle x coordinate
Hy_init = np.append(Hy_init, Hy) # Adding the target particle x coordinate

# Initializing the particle x_dot and y_dot
# originally spread 10. 10
x_dot_init = np.random.normal(0, math.sqrt(10), size=(num_particles + 1)).astype('int16')
y_dot_init = np.random.normal(0, math.sqrt(10), size=(num_particles + 1)).astype('int16')

num_states = 6
# x_init, x_dot_init, y_init, y_dot_init, Hx_init, Hy_init
s_t_1 = np.zeros((num_particles + 1, 6))
for i in list(range(num_particles + 1)):
    x_init[i], y_init[i], Hx_init[i], Hy_init[i] = validate_box(x_init[i], y_init[i], Hx_init[i], Hy_init[i])
    s_t_1[i, 0] = x_init[i]
    s_t_1[i, 1] = x_dot_init[i]
    s_t_1[i, 2] = y_init[i]
    s_t_1[i, 3] = y_dot_init[i]
    s_t_1[i, 4] = Hx_init[i]
    s_t_1[i, 5] = Hy_init[i]

# Initializing the particle weights, pi
q = get_color_distribution([x, y, Hx, Hy], img, num_bins) # to be adapted eventually

pi_t_1 = np.zeros(num_particles + 1)
for i in list(range(num_particles + 1)):
    x, y, Hx, Hy = get_coords(s_t_1[i])
    p = get_color_distribution([x, y, Hx, Hy], img, num_bins)
    d = get_bhattacharyya_distance(p, q)
    pi_t_1[i] = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d ** 2) / (2 * sigma ** 2))


# print('end (efficient) : ', datetime.now())

# # Plotting the Box Centres
# plt.imshow(img,zorder=1)
# plt.scatter(x_init,y_init, c='r',zorder=2)
# plt.show()
# #
# # # Plotting the Boxes
# fig,ax = plt.subplots(1)
# ax.imshow(img)
#
# for i in list(range(num_particles + 1)):
#     hx = Hx_init[i]
#     hy = Hy_init[i]
#     x_coord = x_init[i] - int(hx/2)
#     y_coord = y_init[i] - int(hy/2)
#     rect = patches.Rectangle((x_coord,y_coord),hx,hy,linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
# fig.show()

########################################## Particle Filter ##########################################

runs_per_frame = 1
probabilities = []
updates = []
for frame in frames:
    for i in list(range(runs_per_frame)):
        frame_name = str(frame) + '.png'
        image_path = os.path.join(dataset_path, frame_name)
        img = skimage.io.imread(image_path)
        # Creating a normalized cumulative probability distribution
        c_t_1 = np.zeros(num_particles + 1)
        c_t_1[0] = pi_t_1[0]
        for i in list(range(1, num_particles + 1)):
            c_t_1[i] = c_t_1[i-1] + pi_t_1[i]
        c_t_1 = c_t_1 / c_t_1[-1]

        # Selecting Samples
        resampled_particles_indices = sample_from_cumulative_distribution(c_t_1)
        # candidate_regions = np.unique(resampled_particles_indices)
        s_t_1_resampled = s_t_1[resampled_particles_indices]

        st = np.zeros(s_t_1_resampled.shape)

        # Propagating
        for i in list(range(num_particles + 1)):
            t = 1
            x2 = int(np.random.normal(0, math.sqrt(10)))
            y2 = int(np.random.normal(0, math.sqrt(10)))
            # x2 = 0
            # y2 = 0
            # originally 3, 3
            hx_noise = int(np.random.normal(0, math.sqrt(3)))
            hy_noise = int(np.random.normal(0, math.sqrt(3)))

            w_t_1 = np.array([[0.5 * x2 * t ** 2],
                               [t * x2],
                               [0.5 * y2 * t ** 2],
                               [t * y2],
                               [hx_noise],
                               [hy_noise]])

            particle = s_t_1_resampled[i, :]
            particle = particle.reshape(1, len(particle))
            particle_new = A.dot(np.transpose(particle)) + w_t_1
            x, y, Hx, Hy = get_coords(particle_new)
            particle_new[0], particle_new[2], particle_new[4], particle_new[5] = validate_box(x, y, Hx, Hy)
            st[i, :] = np.transpose(particle_new)


        # Observing
        # x_init, x_dot_init, y_init, y_dot_init, Hx_init, Hy_init
        pi_t = np.zeros(num_particles + 1)
        for i in list(range(num_particles + 1)):
            particle = st[i, :]
            x, y, Hx, Hy = get_coords(particle)
            particle[0], particle[2], particle[4], particle[5] = validate_box(x, y, Hx, Hy)
            x, y, Hx, Hy = get_coords(particle)
            p = get_color_distribution([x, y, Hx, Hy], img, num_bins)
            d = get_bhattacharyya_distance(p, q)
            weight = (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-(d**2)/(2*sigma**2))
            if np.isnan(weight):
                import code;

                code.interact(local=dict(globals(), **locals()))
                print('nan found')
                weight = 2*np.max(pi_t)
            pi_t[i] = weight
        # import code;
        #
        # code.interact(local=dict(globals(), **locals()))

        # Estimating mean state
        pi_observed_normalized = pi_t / np.sum(pi_t)
        s_t_mean = np.zeros(len(particle))
        for i in list(range(num_particles + 1)):
            s_t_mean = s_t_mean + (st[i] * pi_observed_normalized[i])
        s_t_mean = s_t_mean.astype('int16')
        x, y, Hx, Hy = get_coords(s_t_mean)
        s_t_mean[0], s_t_mean[2], s_t_mean[4], s_t_mean[5] = validate_box(x, y, Hx, Hy)
        x, y, Hx, Hy = get_coords(s_t_mean)
        p_estimated = get_color_distribution([x, y, Hx, Hy], img, num_bins)

        d_mean_state = get_bhattacharyya_distance(p_estimated, q)
        pi_t_mean = (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-(d_mean_state**2)/(2*sigma**2))
        if pi_t_mean >= pi_thresh:
            q = ((1 - alpha)* q) + (alpha * (p_estimated))
            updates.append(pi_thresh)
        else:
            updates.append(0)

        # print(pi_t_mean, frame)
        probabilities.append(pi_t_mean)

        # Updating to next time step
        s_t_1 = st
        pi_t_1 = pi_t

    if True:
        # # Plotting the Boxes
        fig2, ax2 = plt.subplots(1)
        ax2.imshow(img)
        for i in list(range(num_particles + 1)):
            Hx = int(st[i][4])
            Hy = int(st[i][5])
            x_coord = int(st[i][0]) - int(Hx / 2)
            y_coord = int(st[i][2]) - int(Hy / 2)
            rect = patches.Rectangle((x_coord,y_coord),Hx,Hy,linewidth=1,edgecolor='r',facecolor='none')
            ax2.add_patch(rect)
            st[i] = s_t_mean
            Hx = int(st[i][4])
            Hy = int(st[i][5])
            x_coord = int(st[i][0]) - int(Hx / 2)
            y_coord = int(st[i][2]) - int(Hy / 2)
            mean_rect = patches.Rectangle((x_coord,y_coord),Hx,Hy,linewidth=1,edgecolor='b',facecolor='none')
            ax2.add_patch(mean_rect)

        # fig2.show()
        fig_name = os.path.join(results_folder, str(frame))
        fig2.savefig(fig_name)
        plt.close()
    frame = frame + 1



    #
    # for i in candidate_regions:
    #     hx = Hx_init[i]
    #     hy = Hy_init[i]
    #     x_coord = x_init[i] - int(hx/2)
    #     y_coord = y_init[i] - int(hy/2)
    #     rect = patches.Rectangle((x_coord,y_coord),hx,hy,linewidth=1,edgecolor='r',facecolor='none')
    #     ax2.add_patch(rect)
#
# plt.show()
# import code;
#
# code.interact(local=dict(globals(), **locals()))
fig3, ax3 = plt.subplots(1)
ax3.plot(frames, probabilities)
ax3.plot(frames, updates)
fig3.show()
fig_name = os.path.join(results_folder, 'Mean State Weight')
fig3.savefig(fig_name)
plt.show()











