# Load the mandrill image
from matplotlib import cm
from matplotlib.pyplot import figure, imshow, subplot

mandrill = datasets.mandrill()

# Show mandrill
figure(1)
imshow(mandrill, cmap=cm.gray, clim=(0,1))

import dtcwt
transform = dtcwt.Transform2d()

# Compute two levels of dtcwt with the defaul wavelet family
mandrill_t = transform.forward(mandrill, nlevels=2)

# Show the absolute images for each direction in level 2.
# Note that the 2nd level has index 1 since the 1st has index 0.
figure(2)
for slice_idx in range(mandrill_t.highpasses[1].shape[2]):
    subplot(2, 3, slice_idx)
    imshow(numpy.abs(mandrill_t.highpasses[1][:,:,slice_idx]), cmap=cm.spectral, clim=(0, 1))

# Show the phase images for each direction in level 2.
figure(3)
for slice_idx in range(mandrill_t.highpasses[1].shape[2]):
    subplot(2, 3, slice_idx)
    imshow(numpy.angle(mandrill_t.highpasses[1][:,:,slice_idx]), cmap=cm.hsv, clim=(-np.pi, np.pi))e_idx]), cmap=cm.hsv, clim=(-np.pi, np.pi))