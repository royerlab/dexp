from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2C2L
from dexp.processing.registration.model.model_io import model_list_from_file, model_list_to_file


def dataset_fuse(dataset,
                 output_path,
                 channels,
                 slicing,
                 store,
                 compression,
                 compression_level,
                 overwrite,
                 microscope,
                 equalise,
                 zero_level,
                 clip_too_high,
                 fusion,
                 fusion_bias_strength,
                 dehaze_size,
                 dark_denoise_threshold,
                 loadreg,
                 workers,
                 workersbackend,
                 devices,
                 check):
    if microscope == 'simview':
        if channels is None:
            channels = ('C0L0', 'C0L1', 'C1L0', 'C1L1')
    elif microscope == 'mvsols':
        if channels is None:
            channels = ('C0L0', 'C0L1')

    views = tuple(dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in channels)

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        views = tuple(view[slicing] for view in views)

    # shape and dtype of views to fuse:
    shape = views[0].shape
    dtype = views[0].dtype

    # We allocate last minute once we know the shape...
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(output_path, mode, store)

    with NumpyBackend():
        model_list_filename = "registration_models.txt"
        if loadreg:
            aprint(f"Loading registration shifts from existing file! ({model_list_filename})")
            models = model_list_from_file(model_list_filename)
        else:
            models = [None, ] * shape[0]

    def process(tp, device):
        try:

            with asection(f"Loading channels {channels} for time point {tp}"):
                views_tp = tuple(view[tp].compute() for view in views)

            with CupyBackend(device):

                model = models[tp]

                if microscope == 'simview':
                    array, model = simview_fuse_2C2L(*views_tp,
                                                     registration_model=model,
                                                     equalise=equalise,
                                                     zero_level=zero_level,
                                                     clip_too_high=clip_too_high,
                                                     fusion=fusion,
                                                     fusion_bias_exponent=2 if fusion_bias_strength > 0 else 1,
                                                     fusion_bias_strength=fusion_bias_strength,
                                                     dehaze_size=dehaze_size,
                                                     dark_denoise_threshold=dark_denoise_threshold)
                elif microscope == 'mvsols':
                    metadata = dataset.get_metadata()
                    angle = metadata['angle']
                    channel = metadata['channel']
                    dz = metadata['dz']
                    res = metadata['res']

                    array, model = msols_fuse_1C2L(*views_tp,
                                                   equalise=equalise,
                                                   zero_level=0,
                                                   angle=angle,
                                                   dx=res,
                                                   dz=dz)

                array = Backend.to_numpy(array, dtype=dtype, force_copy=False)

                if not loadreg:
                    models[tp] = model

            if 'fused' not in dest_dataset.channels():
                dest_dataset.add_channel('fused',
                                         shape=(shape[0],) + array.shape,
                                         dtype=dtype,
                                         codec=compression,
                                         clevel=compression_level)

            with asection(f"Saving fused stack for time point {tp}, shape:{array.shape}, dtype:{array.dtype}"):
                dest_dataset.get_array('fused')[tp] = array

            aprint(f"Done processing time point: {tp} .")

        except Exception as error:
            aprint(error)
            aprint(f"Error occurred while processing time point {tp} !")
            import traceback
            traceback.print_exc()

    if workers == -1:
        workers = len(devices)
    aprint(f"Number of workers: {workers}")

    if workers > 1:
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
    else:
        for tp in range(0, shape[0]):
            process(tp, devices[0])

    if not loadreg:
        model_list_to_file(model_list_filename, models)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()
