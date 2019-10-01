from napari import gui_qt, Viewer
from tifffile import imread

from dexp.enhance.sharpen import median_sharpening, gaussian_sharpening, hybrid_sharpening
from dexp.utils.timeit import timeit

image = imread('../../data/zfish/zfish2.tif')
print(image.shape)

with timeit('hybrid_sharpening'):
    hybrid_sharpened, hybrid_filtered = hybrid_sharpening(image, size=20)

with gui_qt():
    viewer = Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(hybrid_filtered, name='median_filtered')
    viewer.add_image(hybrid_sharpened, name='median_sharpened')

