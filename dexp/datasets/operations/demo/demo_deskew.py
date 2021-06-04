import math
import tempfile
from os.path import join

from arbol import aprint, asection

from dexp.datasets.operations.deskew import dataset_deskew
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
            _demo_deskew(length=128, zoom=4)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_deskew(length=96,
                 zoom=1,
                 shift=1,
                 display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("prepare simulated timelapse:"):
        # generate nuclei image:
        _, _, image = generate_nuclei_background_data(add_noise=False,
                                                      length_xy=length,
                                                      length_z_factor=1,
                                                      independent_haze=True,
                                                      sphere=True,
                                                      zoom=zoom,
                                                      add_offset=False,
                                                      background_stength=0.07,
                                                      dtype=xp.float32)

        # Pad:
        pad_width = ((int(shift * zoom * length // 2), int(shift * zoom * length // 2)), (0, 0), (0, 0),)
        image = xp.pad(image, pad_width=pad_width)

        # Add blur:
        psf = nikon16x08na()
        psf = psf.astype(dtype=image.dtype, copy=False)
        image = fft_convolve(image, psf)

        # apply skew:
        angle = 45
        matrix = xp.asarray([[math.cos(angle * math.pi / 180), math.sin(angle * math.pi / 180), 0], [0, 1, 0], [0, 0, 1]])
        offset = 0 * xp.asarray([image.shape[0] // 2, 0, 0])
        # matrix = xp.linalg.inv(matrix)
        skewed = sp.ndimage.affine_transform(image, matrix, offset=offset)

        # Add noise and clip
        # skewed += xp.random.uniform(-1, 1)
        skewed = xp.clip(skewed, a_min=0, a_max=None)

        # cast to uint16:
        skewed = skewed.astype(dtype=xp.uint16)

        # Single timepoint:
        skewed = skewed[xp.newaxis, ...]


    with tempfile.TemporaryDirectory() as tmpdir:
        aprint('created temporary directory', tmpdir)

        with asection("Prepare dataset..."):
            input_path = join(tmpdir, 'dataset.zarr')
            dataset = ZDataset(path=input_path,
                               mode='w',
                               store='dir')

            dataset.add_channel(name='channel',
                                shape=skewed.shape,
                                chunks=(1, 64, 64, 64),
                                dtype=skewed.dtype)

            dataset.write_array(channel='channel',
                                array=Backend.to_numpy(skewed))

            source_array = dataset.get_array('channel')

        with asection("Deskew..."):
            # output_folder:
            output_path = join(tmpdir, 'deskewed.zarr')

            # deskew:
            dataset_deskew(dataset=dataset,
                           dest_path=output_path,
                           channels=('channel',),
                           slicing=(slice(0, 1),),
                           dx=1,
                           dz=1,
                           angle=angle)

            deskew_dataset = ZDataset(path=output_path, mode='a')
            deskew_array = deskew_dataset.get_array('channel')

            assert deskew_array.shape[0] == 1


        if display:

            def _c(array):
                return Backend.to_numpy(array)

            import napari
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(_c(source_array), name='source_array')
            viewer.add_image(_c(deskew_array), name='deconv_array')
            viewer.grid.enabled = True
            napari.run()


if __name__ == "__main__":
    if not demo_deskew_cupy():
        demo_deskew_numpy()
