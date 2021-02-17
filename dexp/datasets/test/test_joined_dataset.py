import tempfile
from os.path import join

from skimage.data import binary_blobs
from skimage.filters import gaussian

from dexp.datasets.joined_dataset import JoinedDataset
from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.numpy_backend import NumpyBackend


def test_joined_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        size = 12
        chunks = 6

        dataset_list = []

        for i in range(3):
            zdataset = ZDataset(path=join(tmpdir, f'test{i}.zarr'),
                                mode='w',
                                store='dir')
            dataset_list.append(zdataset)

            zdataset.add_channel(name='first',
                                 shape=(10 + 5 * i, size, size, size),
                                 chunks=(1, chunks, chunks, chunks),
                                 dtype='f4',
                                 codec='zstd',
                                 clevel=2)

            zdataset.add_channel(name='second',
                                 shape=(17 + i // 2, size, size, size),
                                 chunks=None,
                                 dtype='f4',
                                 codec='zstd',
                                 clevel=3)

            with NumpyBackend() as backend:
                xp = backend.get_xp_module()

                blobs = binary_blobs(length=size, n_dim=3, blob_size_fraction=0.1).astype('f4')

                for i in range(0, zdataset.nb_timepoints('first')):
                    blobs = gaussian(blobs, sigma=1 + 0.1 * i)
                    zdataset.write_stack('first', i, blobs)

                for i in range(0, zdataset.nb_timepoints('second')):
                    blobs = gaussian(blobs, sigma=1 + 0.5 * i)
                    zdataset.write_stack('second', i, blobs)

            assert zdataset.check_integrity()

        joined_dataset = JoinedDataset(dataset_list)

        for channel in joined_dataset.channels():
            for dataset in dataset_list:
                assert joined_dataset.nb_timepoints(channel) == sum(dataset.nb_timepoints(channel) for dataset in dataset_list)
                assert joined_dataset.shape(channel)[1:] == dataset.shape(channel)[1:]
                assert joined_dataset.dtype(channel) == dataset.dtype(channel)
