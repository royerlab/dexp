import numpy
from dexp.enhance import sharpen
from napari import gui_qt, Viewer
from scipy.ndimage import zoom
from tifffile import imread, imwrite

from dexp.processing.isonet import IsoNet
from dexp.utils.timeit import timeit

# image = imread('data/retina/cropped_farred_RFP_GFP_2109175_2color_sub_10.20.tif')
image = imread('../data/zfish/zfish2.tif')

image = zoom(image, zoom=(1, 0.5, 0.5), order=0)

image = sharpen(image, mode='hybrid', min=0, max=1024, margin_pad=False)

print(f"shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
image_crop = image  # [36-16:36+16,750-128:750+128,750-128:750+128]
print(image_crop.shape)

dxy = 0.4 * 2
dz = 6.349
subsampling = dz / dxy
print(f'subsampling={subsampling}')

# psf = nikon16x08na(dxy=dxy,dz=dz, size=16)[:,(15-1)//2,:]
psf = numpy.ones((3, 3)) / 9

isonet = IsoNet()

# with timeit("Preparation:"):
#     isonet.prepare(image_crop, subsampling=subsampling, psf=psf, threshold=0.97)
#
# with timeit("Training:"):
#     isonet.train(max_epochs=40)

with timeit("Evaluation:"):
    restored = isonet.apply_pair(image_crop, subsampling=subsampling, batch_size=1)

print(restored.shape)

imwrite('../data/isotropic.tif', restored, imagej=True, resolution=(1 / dxy, 1 / dxy), metadata={'spacing': dz, 'unit': 'um'})

image = zoom(image, zoom=(subsampling, 1, 1), order=0)

with gui_qt():
    viewer = Viewer()

    viewer.add_image(image, name='image (all)')
    viewer.add_image(image_crop, name='image selection')
    viewer.add_image(restored, name='restored')
    viewer.add_image(psf, name='psf')
