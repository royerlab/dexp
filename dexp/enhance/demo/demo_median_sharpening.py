from napari import gui_qt, Viewer
from tifffile import imread

from dexp.enhance.sharpen import median_sharpening
from dexp.utils.timeit import timeit

image = imread('../../data/zfish/zfish2.tif')
print(image.shape)

with timeit('median_sharpening'):
    median_sharpened, median_filtered = median_sharpening(image, size=32)

with gui_qt():
    viewer = Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(median_filtered, name='median_filtered')
    viewer.add_image(median_sharpened, name='median_sharpened')

