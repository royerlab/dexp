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

# Input dataset
# Please provide path to dataset fro demoing here:
input_path = '/mnt/raid0/pisces_datasets/test.zarr'


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


    with asection("Deskew..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            aprint('created temporary directory', tmpdir)

            # Open input dataset:
            dataset = ZDataset(path=input_path, mode='r')

            #Input array:
            input_array = dataset.get_array('v0c0')

            with asection("Deskew yang..."):
                # output path:
                output_path = join(tmpdir, 'deskewed_yang.zarr')

                # deskew:
                dataset_deskew(dataset=dataset,
                               dest_path=output_path,
                               channels=('v0c0',),
                               flips=(True,),
                               slicing=(slice(0, 1),),
                               mode='yang')

                # read result
                deskewed_yang_dataset = ZDataset(path=output_path, mode='r')
                deskewed_yang_array = deskewed_yang_dataset.get_array('v0c0')

                assert deskewed_yang_array.shape[0] == 1

            with asection("Deskew classic..."):
                # output path:
                output_path = join(tmpdir, 'deskewed_classic.zarr')

                # deskew:
                dataset_deskew(dataset=dataset,
                               dest_path=output_path,
                               channels=('v0c0',),
                               flips=(True,),
                               slicing=(slice(0, 1),),
                               mode='classic',
                               padding=True)

                # read result
                deskewed_classic_dataset = ZDataset(path=output_path, mode='r')
                deskewed_classic_array = deskewed_classic_dataset.get_array('v0c0')

                assert deskewed_classic_array.shape[0] == 1

                deskewed_classic_array = xp.rot90(Backend.to_backend(deskewed_classic_array[0]), 1, axes=(0, 1))
                deskewed_classic_array = xp.rot90(Backend.to_backend(deskewed_classic_array), 1, axes=(1, 2))

            if display:

                def _c(array):
                    return Backend.to_numpy(array)

                import napari
                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(_c(input_array), name='input_array')
                viewer.add_image(_c(deskewed_yang_array), name='deskewed_yang_array')
                viewer.add_image(_c(deskewed_classic_array), name='deskewed_classic_array')
                viewer.grid.enabled = True
                napari.run()


if __name__ == "__main__":
    if not demo_deskew_cupy():
        demo_deskew_numpy()
