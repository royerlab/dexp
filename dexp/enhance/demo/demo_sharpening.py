from napari import gui_qt, Viewer
from tifffile import imread

from dexp.enhance.sharpen import median_sharpening, gaussian_sharpening, hybrid_sharpening, sharpen
from dexp.utils.timeit import timeit

image = imread('../../data/zfish/zfish2.tif')
print(image.shape)

size = 25

with timeit('gaussian_sharpening'):
    gaussian_sharpened = sharpen(image, mode='gaussian', size=size, min=0, max=1024)

with timeit('median_sharpening'):
    median_sharpened = sharpen(image, mode='median', size=size, min=0, max=1024)

with timeit('hybrid_sharpening'):
    hybrid_sharpened = sharpen(image, mode='hybrid', size=size, min=0, max=1024)

with gui_qt():
    viewer = Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(gaussian_sharpened, name='gaussian_sharpened')
    viewer.add_image(median_sharpened, name='median_sharpened')
    viewer.add_image(hybrid_sharpened, name='hybrid_sharpened')




