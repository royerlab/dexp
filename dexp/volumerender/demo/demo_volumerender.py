from napari import Viewer, gui_qt
from numpy import dot, linspace

from dexp.volumerender.transform_matrices import (
    mat4_perspective,
    mat4_rotation,
    mat4_scale,
    mat4_translate,
)
from dexp.volumerender.volumerender import VolumeRenderer

rend = VolumeRenderer((400, 400))

Nx, Ny, Nz = 200, 150, 50
d = linspace(0, 10000, Nx * Ny * Nz).reshape([Nz, Ny, Nx])

rend.set_data(d)
rend.set_units([1.0, 1.0, 0.1])
rend.set_projection(mat4_perspective(60, 1.0, 1, 10))
rend.set_modelView(dot(mat4_translate(0, 0, -1), dot(mat4_rotation(1), mat4_scale(0.7, 0.7, 0.7))))

rend.render()

print(rend.output)

with gui_qt():
    viewer = Viewer()
    viewer.add_image(rend.output, name="output")
    viewer.add_image(rend.output_alpha, name="output_alpha")
    viewer.add_image(rend.output_depth, name="output_depth")
