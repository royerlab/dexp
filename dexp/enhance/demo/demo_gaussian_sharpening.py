from napari import gui_qt, Viewer
from tifffile import imread

from dexp.enhance.sharpen import median_sharpening, gaussian_sharpening, hybrid_sharpening
from dexp.utils.timeit import timeit

image = imread('../../data/zfish/zfish2.tif')
print(image.shape)

size = 20


with timeit('gaussian_sharpening'):
    gaussian_sharpened, gaussian_filtered = gaussian_sharpening(image, size=size)


with gui_qt():
    viewer = Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(gaussian_sharpened, name='gaussian_sharpened')
    viewer.add_image(gaussian_filtered, name='gaussian_filtered')



