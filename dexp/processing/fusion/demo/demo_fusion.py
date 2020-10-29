
# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff
import time

import numpy
#from aydin.it.llr_deconv import ImageTranslatorLearnedLRDeconv
from napari import gui_qt, Viewer
from tifffile import imread

from dexp.processing.fusion import SimpleFusion


filepath = '/home/royer/Desktop/test_data/embryo_4views.tif'

fusion = SimpleFusion(backend='cupy')
start = time.time()

print(f"Loading data...")
array = imread(filepath)
print(f"Done loading.")

C0L0 = array[0]
C0L1 = array[1]
C1L0 = array[2]
C1L1 = array[3]

C1L0 = numpy.flip(C1L0, -1)
C1L1 = numpy.flip(C1L1, -1)

CxLx, shifts = fusion.fuse_2I2D(C0L0, C0L1, C1L0, C1L1, as_numpy=True)
CxLx = CxLx.astype(numpy.float32)
print(f"Shifts = {shifts}")

#lr = ImageTranslatorLearnedLRDeconv(psf_kernel=psf_kernel, max_num_iterations=40)

stop = time.time()
print(f"Elapsed fusion time:  {stop-start} (includes loading)")

with gui_qt():
    viewer = Viewer()
    viewer.add_image(C0L0, name='C0L0')
    viewer.add_image(C0L1, name='C0L1')
    viewer.add_image(C1L0, name='C1L0')
    viewer.add_image(C1L1, name='C1L1')
    viewer.add_image(CxLx, name='CxLx')


