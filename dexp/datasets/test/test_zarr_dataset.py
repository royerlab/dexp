import os
import tempfile
from os.path import join

import numpy
from skimage.data import binary_blobs
from skimage.filters import gaussian

from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.numpy_backend import NumpyBackend
from ome_zarr.utils import info


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

        assert array1 is not None
        assert array2 is not None

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

        with NumpyBackend() as backend:
            xp = backend.get_xp_module()
            blobs = binary_blobs(length=100, n_dim=3, blob_size_fraction=0.1).astype('f4')
            for i in range(0, 10):

                blobs = gaussian(blobs, sigma=1 + 0.1 * i)
                zdataset.write_stack('first', i, blobs)

                # import napari
                # with napari.gui_qt():
                #     from napari import Viewer
                #     viewer = Viewer()
                #     viewer.add_image(blobs, name='blobs')
                #     viewer.add_image(zdataset.get_stack('first', i), name='(zdataset.get_stack(first, i)')

                assert xp.all((zdataset.get_stack('first', i) == blobs))

                for axis in range(3):
                    assert xp.all((zdataset.get_projection_array('first', axis=axis)[i] == xp.max(blobs, axis=axis)))

            blobs = binary_blobs(length=30, n_dim=3, blob_size_fraction=0.03).astype('f4')
            for i in range(0, 17):
                blobs = gaussian(blobs, sigma=1 + 0.1 * i)
                blobs = blobs[0:10, 0:20, 0:30]
                zdataset.write_stack('second', i, blobs)

                assert xp.all((zdataset.get_stack('second', i) == blobs))

                for axis in range(3):
                    assert xp.all((zdataset.get_projection_array('second', axis=axis)[i] == xp.max(blobs, axis=axis)))

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


def test_add_channels_to():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        # self, path:str, mode:str ='r', store:str ='dir'
        dataset1_path = join(tmpdir, 'test1.zarr')
        zdataset1 = ZDataset(path=dataset1_path,
                             mode='w',
                             store='dir')

        array1 = zdataset1.add_channel(name='first',
                                       shape=(10, 11, 12, 13),
                                       chunks=None,
                                       dtype='f4',
                                       codec='zstd',
                                       clevel=3)

        array1[...] = 1

        dataset2_path = join(tmpdir, 'test2.zarr')
        zdataset2 = ZDataset(path=dataset2_path,
                             mode='w',
                             store='dir')

        array2 = zdataset2.add_channel(name='second',
                                       shape=(17, 10, 20, 30),
                                       chunks=None,
                                       dtype='f4',
                                       codec='zstd',
                                       clevel=3)

        array2[...] = 1

        zdataset2.add_channels_to(dataset1_path,
                                  channels=('second',),
                                  rename=('second-',),
                                  )

        zdataset1_reloaded = ZDataset(path=dataset1_path,
                                      mode='r',
                                      store='dir')

        assert len(zdataset1_reloaded.channels()) == 2
        assert 'first' in zdataset1_reloaded.channels()
        assert 'second-' in zdataset1_reloaded.channels()

        assert zdataset1_reloaded.shape('first') == zdataset1.shape('first')
        assert zdataset1_reloaded.shape('second-') == zdataset2.shape('second')
        assert zdataset1_reloaded.dtype('first') == zdataset1.dtype('first')
        assert zdataset1_reloaded.dtype('second-') == zdataset2.dtype('second')

        a = zdataset1_reloaded.get_array('second-', wrap_with_dask=True).compute()
        b = zdataset2.get_array('second', wrap_with_dask=True).compute()
        assert numpy.all(a == b)

        zdataset1.close()
        zdataset2.close()
        zdataset1_reloaded.close()


def test_zarr_cli_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        # self, path:str, mode:str ='r', store:str ='dir'
        source_path = join(tmpdir, 'test.zarr')
        zdataset = ZDataset(path=source_path,
                            mode='w',
                            store='dir')

        zdataset.add_channel(name='first',
                             shape=(5, 20, 20, 20),
                             chunks=(1, 10, 10, 10),
                             dtype='f4',
                             codec='zstd',
                             clevel=3)

        first_copy = join(tmpdir, 'first.zarr.zip')
        first_command = f'dexp copy {source_path} -o {first_copy}'
        os.system(first_command)

        second_copy = join(tmpdir, 'second.zarr.zip')
        second_command = f'dexp copy {first_copy} -o {second_copy}'
        os.system(second_command)

        first_ds = ZDataset(first_copy)
        second_ds = ZDataset(second_copy)

        assert first_command in first_ds.get_metadata()['cli_history']
        assert first_command in second_ds.get_metadata()['cli_history']
        assert second_command in second_ds.get_metadata()['cli_history']


def test_zarr_parent_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        # self, path:str, mode:str ='r', store:str ='dir'
        source_path = join(tmpdir, 'test.zarr')
        dataset = ZDataset(path=source_path, mode='w', store='dir')

        dataset.append_metadata({
            'dz': 4.0,
            'Channel1': {'dt': 10.0}
        })
        parent_metadata = dataset.get_metadata()
        parent_metadata.pop('cli_history')

        child_dataset = ZDataset(path=join(tmpdir, 'child.zarr'), mode='w', parent=dataset)
        child_metadata = child_dataset.get_metadata()
        child_metadata.pop('cli_history')
        assert child_metadata == parent_metadata


def test_ome_zarr_convertion():
    with tempfile.TemporaryDirectory() as tmpdir:
        print('created temporary directory', tmpdir)

        # self, path:str, mode:str ='r', store:str ='dir'
        source_path = join(tmpdir, 'test_ome.zarr')
        zdataset = ZDataset(path=source_path,
                            mode='w',
                            store='dir')

        zdataset.add_channel(name='first',
                             shape=(5, 20, 20, 20),
                             chunks=(1, 10, 10, 10),
                             dtype='f4',
                             codec='zstd',
                             clevel=3)
        
        ome_zarr_path = join(tmpdir, 'test_ome.ome.zarr')
        zdataset.to_ome_zarr(ome_zarr_path)
        info(ome_zarr_path)
