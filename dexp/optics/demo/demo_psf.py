from napari import gui_qt, Viewer

from dexp.optics.psf import nikon16x08na

psf_xyz = nikon16x08na()

print(f'psf shape: {psf_xyz.shape}')

with gui_qt():
    viewer = Viewer()
    viewer.add_image(psf_xyz, name='psf')