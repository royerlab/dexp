


# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff
import time

import numpy
from napari import gui_qt, Viewer
from tifffile import imread

from dexp.processing.fusion import SimpleFusion


filepath = '/home/royer/Desktop/test_data/embryo_4views.tif'

print(f"Loading data...")
array = imread(filepath)
print(f"Done loading.")

C0L0 = array[0]
C0L1 = array[1]
C1L0 = array[2]
C1L1 = array[3]

C1L0 = numpy.flip(C1L0, -1)
C1L1 = numpy.flip(C1L1, -1)

ff = SimpleFusion(backend='cupy')
start = time.time()
print(f"equalise_intensity...")
C0L0, C0L1 = ff.equalise_intensity(C0L0, C0L1, as_numpy=False)
print(f"fuse_lightsheets...")
C0lx = ff.fuse_lightsheets(C0L0, C0L1, as_numpy=False)
C0L0 = ff._cn(C0L0)
C0L1 = ff._cn(C0L1)

print(f"equalise_intensity...")
C1L0, C1L1 = ff.equalise_intensity(C1L0, C1L1,  as_numpy=False)
print(f"fuse_lightsheets...")
C1lx = ff.fuse_lightsheets(C1L0, C1L1, as_numpy=False)
C1L0 = ff._cn(C1L0)
C1L1 = ff._cn(C1L1)

print(f"equalise_intensity...")
C0lx, C1lx = ff.equalise_intensity(C0lx, C1lx, as_numpy=False)
print(f"register_stacks...")
C0lx, C1lx, shifts = ff.register_stacks(C0lx, C1lx, as_numpy=False)
print(f"shifts = {shifts}")
print(f"fuse_cameras...")
CxLx = ff.fuse_cameras(C0lx, C1lx, as_numpy=False)
stop = time.time()
print(f"Elapsed fusion time:  {stop-start} for numexpr ")

with gui_qt():
    viewer = Viewer()
    viewer.add_image(ff._cn(C0L0), name='C0L0')
    viewer.add_image(ff._cn(C0L1), name='C0L1')
    viewer.add_image(ff._cn(C1L0), name='C1L0')
    viewer.add_image(ff._cn(C1L1), name='C1L1')
    viewer.add_image(ff._cn(C0lx), name='C0lx')
    viewer.add_image(ff._cn(C1lx), name='C1lx')
    viewer.add_image(ff._cn(CxLx), name='CxLx')


