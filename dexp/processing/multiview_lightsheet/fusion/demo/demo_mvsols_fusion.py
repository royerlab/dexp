# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff
from arbol import aprint, asection
from napari import Viewer

from dexp.datasets import ZDataset
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend

dataset_path = "/mnt/raid0/dexp_datasets/tail/raw.zarr"


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
    with asection("Load"):
        zdataset = ZDataset(path=dataset_path, mode="r")

        aprint(zdataset.channels())

        C0L0 = zdataset.get_stack("v0c0", 0)
        C0L1 = zdataset.get_stack("v1c0", 0)

        aprint(f"C0L0 shape={C0L0.shape}, dtype={C0L0.dtype}")
        aprint(f"C0L1 shape={C0L1.shape}, dtype={C0L1.dtype}")

        metadata = zdataset.get_metadata()
        aprint(metadata)

    with asection("Fuse"):
        angle = metadata["angle"]
        dz = metadata["dz"]
        res = metadata["res"]

        C0Lx, model, ratios = msols_fuse_1C2L(
            C0L0,
            C0L1,
            zero_level=0,
            angle=angle,
            dx=res,
            dz=dz,
            equalise=True,
            registration_confidence_threshold=0.6,
            registration_max_residual_shift=64,
            z_pad=8,
            z_apodise=96,
        )
        aprint(f"model = {model}, ratios={ratios}")

    from napari import gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        # viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
        # viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')
        viewer.add_image(_c(C0Lx), name="C0Lx", colormap="bop blue", blending="additive")


if __name__ == "__main__":
    # demo_simview_fuse_numpy()
    demo_mvsols_resample_cupy()
