import tempfile
from os.path import join

from skimage.data import binary_blobs
from skimage.filters import gaussian

from dexp.datasets.zarr_dataset import ZDataset


def test_zarr_dataset_livecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        # self, path:str, mode:str ='r', store:str ='dir'
        zdataset = ZDataset(path=join(tmpdir, 'test.zarr'),
                            mode='w',
                            store='dir')

        array1 = zdataset.add_channel(name='first',
                                      shape=(10, 100, 100, 100),
                                      chunks=(1, 50, 50, 50),
                                      dtype='f4',
                                      codec='zstd',
                                      clevel=3)

        array2 = zdataset.add_channel(name='second',
                                      shape=(17, 10, 20, 30),
                                      chunks=(1, 5, 1, 6),
                                      dtype='f4',
                                      codec='zstd',
                                      clevel=3)

        assert len(zdataset.channels()) == 2
        assert 'first' in zdataset.channels()
        assert 'second' in zdataset.channels()

        assert zdataset.shape('first') == (10, 100, 100, 100)
        assert zdataset.chunks('first') == (1, 50, 50, 50)
        assert zdataset.dtype('first') == 'f4'

        assert zdataset.shape('second') == (17, 10, 20, 30)
        assert zdataset.chunks('second') == (1, 5, 1, 6)
        assert zdataset.dtype('second') == 'f4'

        assert not zdataset.check_integrity()

        for i in range(0, 10):
            blobs = binary_blobs(length=100, n_dim=3, blob_size_fraction=0.1).astype('f4')
            blobs = gaussian(blobs, sigma=1)
            array1[i] = blobs

        for i in range(0, 17):
            blobs = binary_blobs(length=30, n_dim=3, blob_size_fraction=0.03).astype('f4')
            blobs = gaussian(blobs, sigma=1)
            array2[i] = blobs[0:10, 0:20, 0:30]

        assert zdataset.check_integrity()

        # with napari.gui_qt():
        #     viewer = Viewer()
        #     viewer.add_image(array1, name='image')


def test_zarr_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        # self, path:str, mode:str ='r', store:str ='dir'
        zdataset = ZDataset(path=join(tmpdir, 'test.zarr'),
                            mode='w',
                            store='dir')

        array = zdataset.add_channel(name='first',
                                     shape=(10, 100, 100, 100),
                                     chunks=(1, 50, 50, 50),
                                     dtype='f4',
                                     codec='zstd',
                                     clevel=3)

        # we initialise almost everything:
        array[0:9] = 0

        # the dataset integrity must be False!
        assert not zdataset.check_integrity()

        # we initialise everything:
        array[...] = 1
        # the dataset integrity must be True!
        assert zdataset.check_integrity()
