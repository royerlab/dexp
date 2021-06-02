import os
import random
import tempfile
from os.path import join

from arbol import aprint, asection
from dask.array.image import imread

from dexp.datasets.operations.copy import dataset_copy
from dexp.datasets.operations.deconv import dataset_deconv
from dexp.datasets.operations.deskew import dataset_deskew
from dexp.datasets.operations.projrender import dataset_projection_rendering
from dexp.datasets.zarr_dataset import ZDataset
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data


def demo_deskew_numpy():
    with NumpyBackend():
        _demo_deskew()


def demo_deskew_cupy():
    try:
        with CupyBackend():
            _demo_deskew(length_xy=128, zoom=4)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_deskew(length_xy=96,
                 zoom=1,
                 n=8,
                 display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # generate nuclei image:
    _, _, image = generate_nuclei_background_data(add_noise=False,
                                                  length_xy=length_xy,
                                                  length_z_factor=1,
                                                  independent_haze=True,
                                                  sphere=True,
                                                  zoom=zoom,
                                                  dtype=xp.float32)

    with asection("prepare simulated timelapse:"):
        # move to backend:
        image = Backend.to_backend(image)

        # Add blur:
        psf = nikon16x08na()
        psf = psf.astype(dtype=image.dtype, copy=False)
        image = fft_convolve(image, psf)
        image = image.astype(dtype=xp.uint16)

        # generate reference 'ground truth' timelapse
        alpha = 1
        matrix = xp.asarray([[1, 0, alpha], [0, 1, 0], [0, 0, 0]])
        skewed_image = sp.ndimage.affine_transform(image, matrix)

        # Add noise:
        skewed_image += random.uniform(0, 20)
        skewed_image = skewed_image.astype(dtype=xp.uint16)


    with tempfile.TemporaryDirectory() as tmpdir:
        aprint('created temporary directory', tmpdir)

        with asection("Prepare dataset..."):
            input_path = join(tmpdir, 'dataset.zarr')
            dataset = ZDataset(path=input_path,
                               mode='w',
                               store='dir')

            dataset.add_channel(name='channel',
                                shape=skewed_image.shape,
                                chunks=(1, 64, 64, 64),
                                dtype=skewed_image.dtype)

            dataset.write_array(channel='channel',
                                array=Backend.to_numpy(skewed_image))

            source_array = dataset.get_array('channel')

        with asection("Deskew..."):
            # output_folder:
            output_path = join(tmpdir, 'deskewed.zarr')

            # deskew:
            dataset_deskew(dataset=dataset,
                         dest_path=output_path,
                         channels=('channel',),
                         slicing=(slice(2, 3), ...))

            deconv_dataset = ZDataset(path=output_path, mode='a')
            deconv_array = deconv_dataset.get_array('channel')

            assert deconv_array.shape[0] == 1
            assert deconv_array.shape[1:] == source_array.shape[1:]

        if display:

            def _c(array):
                return Backend.to_numpy(array)

            import napari
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(_c(source_array), name='source_array')
            viewer.add_image(_c(deconv_array), name='deconv_array')
            viewer.grid.enabled = True
            napari.run()


if __name__ == "__main__":
    if not demo_deskew_cupy():
        demo_deskew_numpy()
