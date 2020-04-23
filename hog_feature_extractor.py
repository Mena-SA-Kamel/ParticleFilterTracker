import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import skimage.io
import numpy as np

def get_hog_histogram(image, num_bins):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    fd_reshaped = fd.reshape(40*30, 8)
    degrees = np.array(list(range(8))) * (180/8)
    gradients = []
    orientations = []

    for image_segment in fd_reshaped:
        gradient = np.max(image_segment)
        orientation = np.argmax(image_segment)
        gradients.append(gradient)
        orientations.append(degrees[orientation])

    orientations_histogram = np.histogram(orientations, bins=num_bins)[0]
    gradients_histogram = np.histogram(gradients, bins=num_bins)[0]
    return orientations_histogram, gradients_histogram

# angles = np.array(orientations).reshape(30,40)
# gradients_reshaped = np.array(gradients).reshape(30,40)
# x = gradients_reshaped * np.cos(angles)
# y = gradients_reshaped * np.sin(angles)
#
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()
#
# import code;
#
# code.interact(local=dict(globals(), **locals()))

# get_hog_distributions(target_region[:,:,0:3].astype('uint8'), bins)