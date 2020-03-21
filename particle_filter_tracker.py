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
image_path = 'Tracking Dataset/rgb/23.png'
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
    return coef

def get_bhattacharyya_distance(p, q):
    num_channels = p.shape[0]
    distances = np.zeros(num_channels)
    for channel in list(range(num_channels)):
        distances[channel] = math.sqrt(1 - get_bhattacharyya_coef(p,q)[channel])
    return distances

def get_color_distribution(coordinates, image, bins):
    x, y, hx, hy = coordinates
    x0 = x - int(0.5 * hx)
    x1 = x + int(0.5 * hx)
    y0 = y - int(0.5 * hy)
    y1 = y + int(0.5 * hy)

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
    # plt.bar(list(range(bins*num_channels)), histogram[0])
    # plt.show()
    return histogram

def validate_box(x, y, hx, hy):
    if ((x + 0.5*hx) > image_shape[1]):
        diff = int((x + 0.5*hx) - image_shape[1])
        hx = hx - 2*diff
    if ((x - 0.5*hx) < 0):
        diff = int(x - 0.5*hx)
        hx = hx - 2*(abs(diff))
    if ((y + 0.5*hy) > image_shape[0]):
        diff = int((y + 0.5*hy) - image_shape[0])
        hy = hy - 2*(diff)
    if ((y - 0.5*hy) < 0):
        diff = int(y - 0.5*hy)
        hy = hy - 2*(diff)
    return hx, hy

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
x, y, Hx, Hy = [363, 291, 110, 215]
t = 1

A = np.array([[1, t, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, t, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],])


########################################## Initialization ##########################################
num_particles = 50
num_particles = num_particles - 1

# Initializing the particle x and y coordinates
x_init = np.random.normal(x, math.sqrt(10*Hx), size=(num_particles)).astype('int16')
y_init = np.random.normal(y, math.sqrt(10*Hy), size=(num_particles)).astype('int16')
x_init = np.append(x_init, x) # Adding the target particle x coordinate
y_init = np.append(y_init, y) # Adding the target particle x coordinate

# Initializing the particle Hx and Hy states
Hx_init = np.random.normal(Hx, math.sqrt(Hx), size=(num_particles)).astype('int16')
Hy_init = np.random.normal(Hy, math.sqrt(Hy), size=(num_particles)).astype('int16')
Hx_init = np.append(Hx_init, Hx) # Adding the target particle x coordinate
Hy_init = np.append(Hy_init, Hy) # Adding the target particle x coordinate

# Initializing the particle x_dot and y_dot
x_dot_init = np.random.normal(0, math.sqrt(10), size=(num_particles + 1)).astype('int16')
y_dot_init = np.random.normal(0, math.sqrt(10), size=(num_particles + 1)).astype('int16')

num_states = 6
# x_init, x_dot_init, y_init, y_dot_init, Hx_init, Hy_init
st_init = np.zeros((num_particles + 1, 6))
for i in list(range(num_particles + 1)):
    Hx_init[i], Hy_init[i] = validate_box(x_init[i], y_init[i], Hx_init[i], Hy_init[i])
    st_init[i, 0] = x_init[i]
    st_init[i, 1] = x_dot_init[i]
    st_init[i, 2] = y_init[i]
    st_init[i, 3] = y_dot_init[i]
    st_init[i, 4] = Hx_init[i]
    st_init[i, 5] = Hy_init[i]

# Initializing the particle weights, pi
pi_init = np.zeros(num_particles + 1)
sigma = 0.03
num_bins = 8
num_channels = 3
q = get_color_distribution([363, 291, 110, 215], img, num_bins)
print('start (efficient) : ', datetime.now())
for i in list(range(num_particles + 1)):
    p = get_color_distribution([x_init[i], y_init[i], Hx_init[i], Hy_init[i]], img, num_bins)
    d = get_bhattacharyya_distance(p, q)
    pi_init[i] = (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-(d**2)/(2*sigma**2))
    # print (pi_init[i], d)
    # print('Bhattacharyya Dist: ', d)

print('end (efficient) : ', datetime.now())


# Plotting the Box Centres
plt.imshow(img,zorder=1)
plt.scatter(x_init,y_init, c='r',zorder=2)
plt.show()
#
# # Plotting the Boxes
fig,ax = plt.subplots(1)
ax.imshow(img)

for i in list(range(num_particles + 1)):
    hx = Hx_init[i]
    hy = Hy_init[i]
    x_coord = x_init[i] - int(hx/2)
    y_coord = y_init[i] - int(hy/2)
    rect = patches.Rectangle((x_coord,y_coord),hx,hy,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
fig.show()

########################################## Particle Filter ##########################################
q = get_color_distribution([358, 299, 112, 210], img, num_bins)

# Creating a normalized cumulative probability distribution
c_init = np.zeros(num_particles + 1)
c_init[0] = pi_init[0]
for i in list(range(1, num_particles + 1)):
    c_init[i] = c_init[i-1] + pi_init[i]
c_init = c_init / c_init[-1]

# Selecting Samples
resampled_particles_indices = sample_from_cumulative_distribution(c_init)
candidate_regions = np.unique(resampled_particles_indices)

st_resampled = st_init[resampled_particles_indices]

st_propagated = np.zeros(st_resampled.shape)

# Propagating
for i in list(range(num_particles + 1)):
    t = 1
    x2 = int(np.random.normal(0, math.sqrt(3)))
    y2 = int(np.random.normal(0, math.sqrt(3)))
    hx_noise = int(np.random.normal(0, math.sqrt(5)))
    hy_noise = int(np.random.normal(0, math.sqrt(5)))

    w_init = np.array([[0.5 * x2 * t ** 2],
                       [t * x2],
                       [0.5 * y2 * t ** 2],
                       [t * y2],
                       [hx_noise],
                       [hy_noise]])

    particle = st_resampled[i, :]
    particle = particle.reshape(1, len(particle))
    particle_new = A.dot(np.transpose(particle)) + w_init
    st_propagated[i, :] = np.transpose(particle_new)

# Observing
# x_init, x_dot_init, y_init, y_dot_init, Hx_init, Hy_init
pi_observed = np.zeros(num_particles + 1)
for i in list(range(num_particles + 1)):
    particle = st_propagated[i, :]
    x = int(particle[0])
    y = int(particle[2])
    Hx = int(particle[4])
    Hy = int(particle[5])
    p = get_color_distribution([x, y, Hx, Hy], img, num_bins)
    d = get_bhattacharyya_distance(p, q)
    pi_observed[i] = (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-(d**2)/(2*sigma**2))
    # print (pi_init[i], d)
    # print('Bhattacharyya Dist: ', d)

# Estimating
pi_observed_normalized = pi_observed / np.sum(pi_observed)
st_estimate = np.zeros(len(particle))
for i in list(range(num_particles + 1)):
    st_estimate = st_estimate + (st_propagated[i] * pi_observed_normalized[i])
print (st_estimate)
# import code;
# code.interact(local=dict(globals(), **locals()))
#


# # Plotting the Boxes
fig2,ax2 = plt.subplots(1)
ax2.imshow(img)
Hx = int(st_estimate[4])
Hy = int(st_estimate[5])
x_coord = int(st_estimate[0]) - int(hx/2)
y_coord = int(st_estimate[2]) - int(hy/2)

rect = patches.Rectangle((x_coord,y_coord),Hx,Hy,linewidth=1,edgecolor='r',facecolor='none')
ax2.add_patch(rect)

#
# for i in candidate_regions:
#     hx = Hx_init[i]
#     hy = Hy_init[i]
#     x_coord = x_init[i] - int(hx/2)
#     y_coord = y_init[i] - int(hy/2)
#     rect = patches.Rectangle((x_coord,y_coord),hx,hy,linewidth=1,edgecolor='r',facecolor='none')
#     ax2.add_patch(rect)
fig2.show()
plt.show()










