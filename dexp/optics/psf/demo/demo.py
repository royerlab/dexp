#!/usr/bin/env python

from dexp.optics.psf.standard_psfs import nikon16x08na, olympus20x10na


def demo_standard_psfs():
    """
    Particle scan, focus offset.
    """
    nikon16x08na_psf = nikon16x08na()
    olympus20x10na_psf = olympus20x10na()

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(nikon16x08na_psf, name="nikon16x08na_psf")
        viewer.add_image(olympus20x10na_psf, name="olympus20x10na_psf")


demo_standard_psfs()
