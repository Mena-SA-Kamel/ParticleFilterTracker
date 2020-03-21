def get_color_distribution(regions_bounds, image, bins):
    x0, x1, y0, y1 = regions_bounds
    target_region = image[y0:y1, x0:x1, :]
    m = bins # number of bins
    Hx = target_region.shape[1] # width of rectangular bounding box
    Hy = target_region.shape[0] # height of rectangular bounding box
    num_channels = target_region.shape[2]
    y_center = (int(Hy / 2),int(Hx / 2)) # location of rectangle center
    a = math.sqrt(Hx**2 + Hy**2) # adapts the size of the rehion
    I = Hx * Hy # number of pixels in the region
    py = np.zeros((3, bins))
    bin_width = 255 / bins
    sample_image = np.zeros(np.shape(target_region))
    current_time = datetime.now()
    print('start : ', current_time)

    for c in list(range(num_channels)):
        for bin in list(range(bins)):
            for i in list(range(Hx)):
                for j in list(range(Hy)):
                    pixel_location = [j, i]
                    pixel_value = target_region[j, i, c]
                    r = math.sqrt((pixel_location[0] - y_center[0])**2 + (pixel_location[1] - y_center[1])**2)
                    weighted_magnitude = k(r / a)
                    sample_image[j, i, c] = weighted_magnitude
                    h_xi = int(pixel_value / bin_width)
                    py[c, bin] = py[c, bin] + (weighted_magnitude * kronecker_delta(h_xi - bin))
        py[c, :] = py[c, :] / np.sum(py[c, :])
    current_time = datetime.now()
    print('stop : ', current_time)
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.bar(list(range(bins)), py[0, :], align='center', color='r')
    ax1.set_title('Red')
    ax2.bar(list(range(bins)), py[1, :], align='center', color='g')
    ax2.set_title('Green')
    ax3.bar(list(range(bins)), py[2, :], align='center', color='b')
    ax3.set_title('Blue')
    fig2.show()

    fig3, (ax1) = plt.subplots(1, 1)
    ax1.imshow(sample_image)
    fig3.show()

A = np.array([[1, t, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, t, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, t],
              [0, 0, 0, 0, 0, 0, 0, 1]])


def get_color_distribution(coordinates, image, bins):
    x, y, hx, hy = coordinates
    x0 = x - int(0.5 * hx)
    x1 = x + int(0.5 * hx)
    y0 = y - int(0.5 * hy)
    y1 = y + int(0.5 * hy)
    target_region = image[y0:y1, x0:x1, :]
    target_region = np.array(target_region)
    m = bins # number of bins
    num_channels = target_region.shape[2]
    y_center = (y,x) # location of rectangle center
    Hx = x1 - x0
    Hy = y1 - y0
    a = math.sqrt(Hx**2 + Hy**2) # adapts the size of the region
    I = Hx * Hy # number of pixels in the region
    py = np.zeros((3, m))
    bin_width = 255 / m
    # print('start (efficient) : ', datetime.now())

    weighting_chart = np.zeros((Hy, Hx))
    for i in list(range(Hx)):
        for j in list(range(Hy)):
            pixel_location = [j, i]
            r = math.sqrt((pixel_location[0] - y_center[0]) ** 2 + (pixel_location[1] - y_center[1]) ** 2)
            weighted_magnitude = k(r / a)
            weighting_chart[j, i] = weighted_magnitude

    flattened_image = np.zeros((3, I))
    weights_flattened = weighting_chart.flatten()
    # sorted_indices = np.zeros((3, I))
    for channel in list(range(num_channels)):
        flattened_image[channel, :] = target_region[:,:,channel].flatten()
        # sorted_indices[channel, :] = np.argsort(flattened_image[channel, :])

    # sorted_flattened_image = operator.itemgetter(*sorted_indices[0,:])(flattened_image)
    for channel in list(range(num_channels)):
        for i in list(range(len(weights_flattened))):
            r = weights_flattened[i]
            pixel_value = flattened_image[channel, i]
            weighted_magnitude = k(r / a)
            h_xi = int(pixel_value / bin_width)
            if pixel_value == 255:
                h_xi = h_xi - 1
            py[channel, h_xi] = py[channel, h_xi] + weighted_magnitude
        # py[channel, :] = py[channel, :] / np.sum(py[channel, :])
    py= py / np.sum(py)
    # print('end (efficient) : ', datetime.now())
    #
    # fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # ax1.bar(list(range(m)), py[0, :], align='center', color='r')
    # ax1.set_title('Red')
    # ax2.bar(list(range(m)), py[1, :], align='center', color='g')
    # ax2.set_title('Green')
    # ax3.bar(list(range(m)), py[2, :], align='center', color='b')
    # ax3.set_title('Blue')
    # fig2.show()
    #
    # fig3, (ax1) = plt.subplots(1, 1)
    # ax1.imshow(weighting_chart)
    # fig3.show()
    # return py
    py = py.flatten()
    py = py.reshape((1, len(py)))
    return py

def get_histogram(image, bins):
    num_rows = np.shape(image)[0]
    num_cols = np.shape(image)[1]
    num_channels = np.shape(image)[2]
    bin_width = 255/bins
    binned_histogram = np.zeros((3, bins)).astype('uint8')

    for i in list(range(num_rows)):
        for j in list(range(num_cols)):
            for k in list(range(num_channels)):
                pixel_value = image[i,j,k]
                if pixel_value == 255:
                    pixel_value = pixel_value - 1
                bin_location = int(pixel_value / bin_width)
                binned_histogram[k, bin_location] = binned_histogram[k, bin_location]  + 1
    print('done')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.bar(list(range(bins)), binned_histogram[0,:], align='center', color = 'r')
    ax1.set_title('Red')
    ax2.bar(list(range(bins)), binned_histogram[1,:], align='center', color = 'g')
    ax2.set_title('Green')
    ax3.bar(list(range(bins)), binned_histogram[2,:], align='center', color = 'b')
    ax3.set_title('Blue')
    fig.show()
    return binned_histogram