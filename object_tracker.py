import draw_target_region
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io
import math
import matplotlib.patches as patches
from datetime import datetime
from PIL import Image

def validate_box(x, y, hx, hy, image_shape):
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

def get_color_distribution(coordinates, image, bins):
    x, y, hx, hy = coordinates
    x0 = x - int(0.5 * hx)
    x1 = x + int(0.5 * hx)
    y0 = y - int(0.5 * hy)
    y1 = y + int(0.5 * hy)

    if x0 <0 or x1 < 0 or y0 < 0 or y1< 0 :
        print ('ERROR: - get_color_distribution - Value less than zero\n')
        import code;
        code.interact(local=dict(globals(), **locals()))

    if x1<=x0 or y1<=y0:
        print('ERROR: - get_color_distribution - Cant define target region\n')
        import code;
        code.interact(local=dict(globals(), **locals()))

    target_region = image[y0:y1, x0:x1, :]
    target_region = np.array(target_region)

    # Creating a weighting chart
    hx = target_region.shape[1]
    hy = target_region.shape[0]
    x_coords = np.arange(hx) - int((hx) / 2)
    y_coords = np.arange(hy) - int((hy) / 2)
    X, Y = np.meshgrid(x_coords, y_coords, sparse=True)
    a = math.sqrt(hx ** 2 + hy ** 2)
    d = np.sqrt(X ** 2 + Y ** 2) / a
    weights = (d < 1) * (1 - d ** 2)
    weights_flattened = weights.flatten()

    # Calculating the histogram
    num_channels = target_region.shape[2]
    histogram = []
    for channel in list(range(num_channels)):
        channel_pixels = target_region[:,:,channel].flatten()
        channel_histogram = np.histogram(channel_pixels, bins=bins, range=(0, 255), weights=weights_flattened)[0]
        histogram.append(channel_histogram/ np.sum(channel_histogram))
    histogram = np.array(histogram).flatten()
    histogram = histogram / np.sum(histogram)
    histogram = histogram.reshape((1, len(histogram)))

    # plt.bar(list(range(len(histogram[0]))), histogram[0])
    # plt.show()
    return histogram

def plot_state(state, image, frame_number = 0, mean_state = '', save_dir = '', save = True, display_mean = False, display_all_states = False, img_name = ''):
    num_particles = state.shape[0]
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(image)
    if display_all_states:
        for i in list(range(num_particles)):
            Hx = int(state[i][4])
            Hy = int(state[i][5])
            x_coord = int(state[i][0]) - int(Hx / 2)
            y_coord = int(state[i][2]) - int(Hy / 2)
            rect = patches.Rectangle((x_coord, y_coord), Hx, Hy, linewidth=1, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
    if display_mean:
        Hx = int(mean_state[4])
        Hy = int(mean_state[5])
        x_coord = int(mean_state[0]) - int(Hx / 2)
        y_coord = int(mean_state[2]) - int(Hy / 2)
        mean_rect = patches.Rectangle((x_coord, y_coord), Hx, Hy, linewidth=1, edgecolor='b', facecolor='none')
        ax2.add_patch(mean_rect)
    if save:
        fig_name = os.path.join(save_dir, str(frame_number) + img_name +'.png')
        fig2.savefig(fig_name)
    # plt.show()
    fig2.canvas.draw()
    image_from_plot = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image_from_plot

def define_initial_target_region(image, image_path = ''):
    x, y, Hx, Hy = draw_target_region.draw_region(image, image_path)
    return [x, y, Hx, Hy]

def get_bhattacharyya_coef(p, q):
    m = p.shape[1]
    coef = 0
    for bin in list(range(m)):
        coef = coef + math.sqrt(p[0, bin] * q[0, bin])
        if coef > 1:
            coef = 1
    return coef

def get_bhattacharyya_distance(p, q):
    bhattacharyya_distance = math.sqrt(1 - get_bhattacharyya_coef(p,q))
    return bhattacharyya_distance

def get_initial_state(x, y, Hx, Hy, q, num_particles, num_bins, image):
    initial_spread_ratio = 0.01
    sigma = 0.05 # or 0.2 - decrease to make it narrower
    # # Deciding on Sigma value
    # sigmas = np.linspace(0, 0.5, 20)
    # for sigma in sigmas:
    #     print(sigma)
    #     plt.figure()  # Create a new figure window
    #     xlist = np.linspace(0, 1, 1000)  # Create 1-D arrays for x,y dimensions
    #     F =  (1 / (sigma * np.sqrt(2 * math.pi))) * np.exp(-(xlist ** 2) / (2 * sigma ** 2))
    #     plt.plot(xlist, F)
    #     plt.show()

    q = get_color_distribution([x, y, Hx, Hy], image, num_bins)

    # x_init = np.random.uniform(10, 630, size=(num_particles)).astype('int16')
    # y_init = np.random.uniform(10, 470, size=(num_particles)).astype('int16')
    x_init = np.random.normal(x, math.sqrt(initial_spread_ratio * Hx), size=(num_particles - 1)).astype('int16')
    x_init = np.append(x_init, x)  # Adding the target particle x coordinate

    y_init = np.random.normal(y, math.sqrt(initial_spread_ratio * Hy), size=(num_particles - 1)).astype('int16')
    y_init = np.append(y_init, y)  # Adding the target particle y coordinate

    Hx_init = np.random.normal(Hx, math.sqrt(initial_spread_ratio * Hx), size=(num_particles - 1)).astype('int16')
    Hx_init = np.append(Hx_init, Hx)  # Adding the target particle x coordinate

    Hy_init = np.random.normal(Hy, math.sqrt(initial_spread_ratio * Hy), size=(num_particles - 1)).astype('int16')
    Hy_init = np.append(Hy_init, Hy)  # Adding the target particle y coordinate

    # x_dot_init = np.random.normal(0, math.sqrt(15), size=(num_particles + 1)).astype('int16')
    # y_dot_init = np.random.normal(0, math.sqrt(15), size=(num_particles + 1)).astype('int16')
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
    # print(pi_t_1)
    return s_t_1, pi_t_1

def create_cummulative_probability_distribution(pi_t_1):
    # Creating a normalized cumulative probability distribution
    num_particles = len(pi_t_1)
    c_t_1 = np.zeros(num_particles)
    c_t_1[0] = pi_t_1[0]
    for i in list(range(1, num_particles)):
        c_t_1[i] = c_t_1[i - 1] + pi_t_1[i]
    c_t_1 = c_t_1 / c_t_1[-1]
    # plt.plot(list(range(num_particles)), c_t_1)
    # plt.show()
    return c_t_1

def sample_from_cumulative_distribution(c_t_1):
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
    return new_samples

def get_coords(particle):
    particle = particle.reshape(1, len(particle))
    x = int(particle[0][0])
    y = int(particle[0][2])
    Hx = int(particle[0][4])
    Hy = int(particle[0][5])
    return [x, y, Hx, Hy]

# def get_mean_state(s_t, pi_t, sigma, q, image, num_bins):
#     # Estimating mean state
#     pi_observed_normalized = pi_t / np.sum(pi_t)
#     s_t_mean = np.zeros(6)
#     num_particles = len(pi_t)
#     image_shape = image.shape
#     for i in list(range(num_particles)):
#         s_t_mean = s_t_mean + (s_t[i] * pi_observed_normalized[i])
#     s_t_mean = s_t_mean.astype('int16')
#     x, y, Hx, Hy = get_coords(s_t_mean)
#     s_t_mean[0], s_t_mean[2], s_t_mean[4], s_t_mean[5] = validate_box(x, y, Hx, Hy, image_shape)
#     x, y, Hx, Hy = get_coords(s_t_mean)
#     p_estimated = get_color_distribution([x, y, Hx, Hy], image, num_bins)
#     d_mean_state = get_bhattacharyya_distance(p_estimated, q)
#     pi_t_mean = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d_mean_state ** 2) / (2 * sigma ** 2))
#     return [s_t_mean, pi_t_mean]

def particle_filter(s_t_1, pi_t_1, q, num_bins, image, results_folder = '', frame_number = 0):
    num_particles = len(pi_t_1)
    image_shape = image.shape
    sigma = 0.05

    c_t_1 = create_cummulative_probability_distribution(pi_t_1)

    # Selecting N samples based on weights
    resampled_particles_indices = sample_from_cumulative_distribution(c_t_1)
    s_t_1_resampled = s_t_1[resampled_particles_indices]
    if not results_folder == '':
        plot_state(s_t_1_resampled, image, frame_number, save_dir = results_folder, save = True, display_all_states = True, img_name= '_resampled')

    s_t = np.zeros(s_t_1_resampled.shape)

    # Motion model.
    dt = 1
    A = np.array([[1, dt, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, dt, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1], ])

    # Propagating
    for i in list(range(num_particles)):
        t = 1
        x2 = int(np.random.normal(0, math.sqrt(60))) # To be replaced by acceleration from accelerometer + gyro
        y2 = int(np.random.normal(0, math.sqrt(60))) # To be replaced by acceleration from accelerometer + gyro
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
        particle_new[0], particle_new[2], particle_new[4], particle_new[5] = validate_box(x, y, Hx, Hy, image_shape)
        s_t[i, :] = np.transpose(particle_new)

    # Observing
    # x_init, x_dot_init, y_init, y_dot_init, Hx_init, Hy_init
    pi_t = np.zeros(num_particles)
    for i in list(range(num_particles)):
        particle = s_t[i, :]
        x, y, Hx, Hy = get_coords(particle)
        particle[0], particle[2], particle[4], particle[5] = validate_box(x, y, Hx, Hy, image_shape)
        x, y, Hx, Hy = get_coords(particle)
        p = get_color_distribution([x, y, Hx, Hy], image, num_bins)
        d = get_bhattacharyya_distance(p, q)
        weight = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d ** 2) / (2 * sigma ** 2))
        if np.isnan(weight):
            print('ERROR: - particle_filter - Infinite weight\n')
            import code;
            code.interact(local=dict(globals(), **locals()))
            weight = 2 * np.max(pi_t)
        pi_t[i] = weight

    # Estimating mean state
    pi_observed_normalized = pi_t / np.sum(pi_t)
    s_t_mean = np.zeros(6)
    for i in list(range(num_particles)):
        s_t_mean = s_t_mean + (s_t[i] * pi_observed_normalized[i])
    s_t_mean = s_t_mean.astype('int16')
    x, y, Hx, Hy = get_coords(s_t_mean)
    s_t_mean[0], s_t_mean[2], s_t_mean[4], s_t_mean[5] = validate_box(x, y, Hx, Hy, image_shape)
    x, y, Hx, Hy = get_coords(s_t_mean)
    p_estimated = get_color_distribution([x, y, Hx, Hy], image, num_bins)

    d_mean_state = get_bhattacharyya_distance(p_estimated, q)
    pi_t_mean = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(d_mean_state ** 2) / (2 * sigma ** 2))
    # import code;
    # code.interact(local=dict(globals(), **locals()))
    # s_t_mean, pi_t_mean = get_mean_state(s_t, pi_t_1, sigma, q, image, num_bins)
    s_t_1 = s_t
    pi_t_1 = pi_t
    return s_t_1, pi_t_1, s_t_mean, q

#
# # #### Main Function
# image_shape = [480, 680]
# dataset_path = 'Tracking Dataset 4/rgb'
# num_images = len(os.listdir(dataset_path))
# start_image_number = 10
# frame_numbers = list(list(range(start_image_number, num_images + start_image_number)))
#
# num_particles = 500
# num_bins = 12
# current_time = datetime.now()
# results_folder = current_time.strftime("%Y-%m-%d-%H-%M-%S")
# os.mkdir(results_folder)
#
# for frame_number in frame_numbers:
#     frame_name = str(frame_number) + '.png'
#     image_path = os.path.join(dataset_path, frame_name)
#     image = skimage.io.imread(image_path)
#
#     if frame_number - start_image_number == 0:
#         x, y, Hx, Hy = define_initial_target_region(image)
#         q = get_color_distribution([x, y, Hx, Hy], image, num_bins)
#         s_t_1, pi_t_1 = get_initial_state(x, y, Hx, Hy, q, num_particles, num_bins, image)
#     s_t, pi_t, s_t_mean, q = particle_filter(s_t_1, pi_t_1, q, num_bins, image, results_folder, frame_number)
#     s_t_1 = s_t
#     pi_t_1 = pi_t
#     plot_state(s_t, image, frame_number, mean_state= s_t_mean, save_dir=results_folder, save=True, display_mean=True,
#                display_all_states=True, img_name='')
#
