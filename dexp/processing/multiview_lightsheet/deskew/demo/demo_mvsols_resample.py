# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff

from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.utils.timeit import timeit

dataset_path = '/mnt/raid0/pisces_datasets/data2_fish_TL100_range1300um_step0.31_6um_20ms_dualv_300tp_2_first10tp.zarr'


def demo_mvsols_resample_numpy():
    with NumpyBackend():
        _mvsols_resample()


def demo_mvsols_resample_cupy():
    try:
        with CupyBackend():
            _mvsols_resample()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _mvsols_resample():
    xp = Backend.get_xp_module()


    with timeit(f"loading"):
        zdataset = ZDataset(path=dataset_path, mode='r')

        print(zdataset.channels())

        view1 = zdataset.get_stack('v0c0', 0)
        view2 = zdataset.get_stack('v1c0', 0)

        print(f"view1 shape={view1.shape}, dtype={view1.dtype}")
        print(f"view2 shape={view2.shape}, dtype={view2.dtype}")


    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)
        viewer = Viewer()
        viewer.add_image(_c(view1), name='view1', colormap='bop blue', blending='additive')
        viewer.add_image(_c(view2), name='view2', colormap='bop orange', blending='additive')



if __name__ == "__main__":
    # demo_simview_fuse_numpy()
    demo_mvsols_resample_cupy()
