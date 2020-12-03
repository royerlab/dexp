# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff

from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.utils.timeit import timeit

dataset_path = '/mnt/raid0/pisces_datasets/data2_fish_TL100_range1300um_step0.31_6um_20ms_dualv_300tp_2_first10tp.zarr'


def demo_mvsols_resample_numpy():
    with NumpyBackend():
        _mvsols_resample()


def demo_mvsols_resample_cupy():
    try:
        with CupyBackend(1):
            _mvsols_resample()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _mvsols_resample():
    xp = Backend.get_xp_module()

    with timeit(f"Load"):
        zdataset = ZDataset(path=dataset_path, mode='r')

        print(zdataset.channels())

        C0L0 = zdataset.get_stack('v0c0', 0)[...,700:-700]
        C0L1 = zdataset.get_stack('v1c0', 0)[...,700:-700]

        print(f"C0L0 shape={C0L0.shape}, dtype={C0L0.dtype}")
        print(f"C0L1 shape={C0L1.shape}, dtype={C0L1.dtype}")

        metadata = zdataset.get_metadata()
        print(metadata)

    with timeit(f"Fuse"):
        angle = metadata['angle']
        channel = metadata['channel']
        dz = metadata['dz']
        res = metadata['res']

        C0Lx, _ = msols_fuse_1C2L(C0L0, C0L1,
                                  zero_level=0,
                                  angle=angle,
                                  dx=res,
                                  dz=dz)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
        viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')
        viewer.add_image(_c(C0Lx), name='C0Lx', colormap='bop blue', blending='additive')


if __name__ == "__main__":
    # demo_simview_fuse_numpy()
    demo_mvsols_resample_cupy()
