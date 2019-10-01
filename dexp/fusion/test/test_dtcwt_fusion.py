import dtcwt
import numpy
from napari import gui_qt, Viewer

GRID_SIZE = 64
SPHERE_RAD = int(0.45 * GRID_SIZE) + 0.5

grid = numpy.arange(-(GRID_SIZE>>1), GRID_SIZE>>1)
X, Y, Z = numpy.meshgrid(grid, grid, grid)
r = numpy.sqrt(X*X + Y*Y + Z*Z)

sphere = 0.5 + 0.5 * numpy.clip(SPHERE_RAD-r, -1, 1)

trans = dtcwt.Transform3d()
sphere_t = trans.forward(sphere, nlevels=2)


with gui_qt():
    viewer = Viewer()

    viewer.add_image(sphere, name='sphere')

    viewer.add_image(sphere_t.lowpass, name=f'sphere lowpass')

    counter=0

    for level in sphere_t.highpasses:
        viewer.add_image(numpy.absolute(level), name=f'sphere highpass level {counter}')
        counter +=1
