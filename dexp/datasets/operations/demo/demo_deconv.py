import random
import tempfile
from os.path import join

from arbol import aprint, asection

from dexp.datasets.operations.deconv import dataset_deconv
from dexp.datasets.zarr_dataset import ZDataset
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data


def demo_deconv_numpy():
    with NumpyBackend():
        _demo_deconv()


def demo_deconv_cupy():
    try:
        with CupyBackend():
            _demo_deconv(length_xy=128, zoom=4)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_deconv(length_xy=96,
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

        # generate reference 'ground truth' timelapse
        images = (image.copy() for _ in range(n))

        # modify each image:
        psf = nikon16x08na()
        psf = psf.astype(dtype=image.dtype, copy=False)
        images = (fft_convolve(image, psf) for image in images)
        images = (image + random.uniform(-10, 10) for image in images)

        # turn into array:
        images = xp.stack(images)

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint('created temporary directory', tmpdir)

        with asection("Prepare dataset..."):
            input_path = join(tmpdir, 'dataset.zarr')
            dataset = ZDataset(path=input_path,
                               mode='w',
                               store='dir')

            dataset.add_channel(name='channel',
                                shape=images.shape,
                                chunks=(1, 64, 64, 64),
                                dtype=images.dtype)

            dataset.write_array(channel='channel',
                                array=Backend.to_numpy(images))

            source_array = dataset.get_array('channel')

        with asection("Deconvolve..."):
            # output_folder:
            output_path = join(tmpdir, 'deconv.zarr')

            # Do deconvolution:
            dataset_deconv(dataset=dataset,
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
    if not demo_deconv_cupy():
        demo_deconv_numpy()
