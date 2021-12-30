import tempfile
from os.path import join

from skimage.data import binary_blobs
from skimage.filters import gaussian

from dexp.datasets import ZDataset


def demo():
    import napari

    with tempfile.TemporaryDirectory() as tmpdir:
        print("created temporary directory", tmpdir)

        zdataset = ZDataset(path=join(tmpdir, "test.zarr"), mode="w", store="dir")

        array1 = zdataset.add_channel(
            name="first", shape=(10, 100, 100, 100), chunks=(1, 50, 50, 50), dtype="f4", codec="zstd", clevel=3
        )

        array2 = zdataset.add_channel(
            name="second", shape=(17, 10, 20, 30), chunks=(1, 5, 1, 6), dtype="f4", codec="zstd", clevel=3
        )

        for i in range(0, 10):
            blobs = binary_blobs(length=100, n_dim=3, blob_size_fraction=0.1).astype("f4")
            blobs = gaussian(blobs, sigma=1)
            array1[i] = blobs

        for i in range(0, 17):
            blobs = binary_blobs(length=30, n_dim=3, blob_size_fraction=0.03).astype("f4")
            blobs = gaussian(blobs, sigma=1)
            array2[i] = blobs[0:10, 0:20, 0:30]

        print(array1.info)
        print(array2.info)

        viewer = napari.Viewer()
        viewer.add_image(array1, name="array_first")
        viewer.add_image(array2, name="array_second")

        napari.run()

        zdataset.close()
        del zdataset
        del array1
        del array2

        zdataset_read = ZDataset(path=join(tmpdir, "test.zarr"), mode="r")

        array1 = zdataset_read.get_array("first")
        array2 = zdataset_read.get_array("second")

        print(array1.info)
        print(array2.info)

        viewer = napari.Viewer()
        viewer.add_image(array1, name="array_first")
        viewer.add_image(array2, name="array_second")

        napari.run()


if __name__ == "__main__":
    demo()
