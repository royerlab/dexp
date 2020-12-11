from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2C2L
from dexp.processing.registration.model.model_factory import from_json


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
                 fusion,
                 fusion_bias_strength,
                 dehaze_size,
                 dark_denoise_threshold,
                 load_shifts,
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

    registration_models_file = open("registration_models.txt", "r" if load_shifts else 'w')
    if load_shifts:
        aprint(f"Loading registration shifts from existing file! ({registration_models_file.name})")

    def process(tp, device):
        with CupyBackend(device):

            with asection(f"Loading channels {channels} for time point {tp}"):
                views_tp = tuple(view[tp].compute() for view in views)

            model = None
            if load_shifts:
                try:
                    line = registration_models_file.readline().strip()
                    model = from_json(line)
                    aprint(f"loaded model: {line} ")
                except ValueError:
                    aprint(f"Cannot read model from line: {line}, most likely we have reached the end of the shifts file, have the channels a different number of time points?")

            if microscope == 'simview':
                array, model = simview_fuse_2C2L(*views_tp,
                                                 registration_model=model,
                                                 equalise=equalise,
                                                 zero_level=zero_level,
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

            if not load_shifts:
                json_text = model.to_json()
                registration_models_file.write(json_text + '\n')

        aprint(f'Writing array of dtype: {array.dtype}')
        if 'fused' not in dest_dataset.channels():
            dest_dataset.add_channel('fused',
                                     shape=(shape[0],) + array.shape,
                                     dtype=dtype,
                                     codec=compression,
                                     clevel=compression_level)

        with asection(f"Saving fused image"):
            dest_dataset.get_array('fused')[tp] = array

    if workers == -1:
        workers = len(devices)

    aprint(f"workers={workers}")

    if workers > 1:
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
    else:
        for tp in range(0, shape[0]):
            process(tp, devices[0])

    registration_models_file.close()

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()
