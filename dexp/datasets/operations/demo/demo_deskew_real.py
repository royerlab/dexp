import tempfile
from pathlib import Path

from arbol import aprint, asection
from toolz import curry

from dexp.datasets import ZDataset
from dexp.datasets.operations.deskew import dataset_deskew
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend

# Input dataset
# Please provide path to dataset fro demoing here:
input_path = "/mnt/raid0/pisces_datasets/test.zarr"


def demo_deskew_numpy():
    with NumpyBackend():
        _demo_deskew()


def demo_deskew_cupy():
    try:
        with CupyBackend():
            _demo_deskew()
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_deskew(display=True):
    xp = Backend.get_xp_module()

    with asection("Deskew..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            aprint("created temporary directory", tmpdir)
            tmpdir = Path(tmpdir)

            # Open input dataset:
            dataset = ZDataset(path=input_path, mode="r")
            dataset.set_slicing((slice(0, 1),))

            # Input array:
            input_array = dataset.get_array("v0c0")
            angle = 45

            deskew_func = curry(
                dataset_deskew,
                input_dataset=dataset,
                channels=("v0c0",),
                dx=1,
                dz=1,
                angle=angle,
                flips=(True,),
                camera_orientation=0,
                depth_axis=0,
                lateral_axis=1,
                padding=True,
                devices=[
                    0,
                ],
            )

            with asection("Deskew yang..."):
                # output path:
                output_path = tmpdir / "deskewed_yang.zarr"
                output_dataset = ZDataset(path=input_path, mode="w", parent=dataset)

                # deskew:
                deskew_func(
                    output_dataset=output_dataset,
                    mode="yang",
                )

                # read result
                deskewed_yang_dataset = ZDataset(path=output_path, mode="r")
                deskewed_yang_array = deskewed_yang_dataset.get_array("v0c0")

                assert deskewed_yang_array.shape[0] == 1

            with asection("Deskew classic..."):
                # output path:
                output_path = tmpdir / "deskewed_classic.zarr"
                output_dataset = ZDataset(path=input_path, mode="w", parent=dataset)

                # deskew:
                deskew_func(
                    output_dataset=output_dataset,
                    mode="classic",
                )

                # read result
                deskewed_classic_dataset = ZDataset(path=output_path, mode="r")
                deskewed_classic_array = deskewed_classic_dataset.get_array("v0c0")

                assert deskewed_classic_array.shape[0] == 1

                deskewed_classic_array = xp.rot90(Backend.to_backend(deskewed_classic_array[0]), 1, axes=(0, 1))
                deskewed_classic_array = xp.rot90(Backend.to_backend(deskewed_classic_array), 1, axes=(1, 2))

            if display:

                def _c(array):
                    return Backend.to_numpy(array)

                import napari

                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(_c(input_array), name="input_array")
                viewer.add_image(_c(deskewed_yang_array), name="deskewed_yang_array")
                viewer.add_image(_c(deskewed_classic_array), name="deskewed_classic_array")
                viewer.grid.enabled = True
                napari.run()


if __name__ == "__main__":
    if not demo_deskew_cupy():
        demo_deskew_numpy()
