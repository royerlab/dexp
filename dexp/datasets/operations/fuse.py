import numpy
from joblib import Parallel

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2I2D
from dexp.processing.registration.model.model_factory import from_json


def dataset_fuse(dataset,
                 path,
                 slicing,
                 store,
                 compression,
                 compression_level,
                 overwrite,
                 workers,
                 zero_level,
                 fusion,
                 fusion_bias_strength,
                 dehaze_size,
                 dark_denoise_threshold,
                 load_shifts,
                 devices,
                 check):
    print(f"getting Dask arrays for all channels to fuse...")
    array_C0L0 = dataset.get_array('C0L0', per_z_slice=False, wrap_with_dask=True)
    array_C0L1 = dataset.get_array('C0L1', per_z_slice=False, wrap_with_dask=True)
    array_C1L0 = dataset.get_array('C1L0', per_z_slice=False, wrap_with_dask=True)
    array_C1L1 = dataset.get_array('C1L1', per_z_slice=False, wrap_with_dask=True)

    if slicing is not None:
        print(f"Slicing with: {slicing}")
        array_C0L0 = array_C0L0[slicing]
        array_C0L1 = array_C0L1[slicing]
        array_C1L0 = array_C1L0[slicing]
        array_C1L1 = array_C1L1[slicing]

    # shape and dtype of views to fuse:
    shape = array_C0L0.shape
    dtype = array_C0L0.dtype

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    dest_array = dest_dataset.add_channel('fused',
                                          shape=shape,
                                          dtype=dtype,
                                          chunks=dataset._default_chunks,
                                          codec=compression,
                                          clevel=compression_level)

    registration_models_file = open("registration_models.txt", "r" if load_shifts else 'w')
    if load_shifts:
        print(f"Loading registration shifts from existing file! ({registration_models_file.name})")

    def process(tp, device):
        print(f"Writing time point: {tp} ")

        C0L0 = array_C0L0[tp].compute()
        C0L1 = array_C0L1[tp].compute()
        C1L0 = array_C1L0[tp].compute()
        C1L1 = array_C1L1[tp].compute()

        C1L0 = numpy.flip(C1L0, -1)
        C1L1 = numpy.flip(C1L1, -1)

        model = None
        if load_shifts:
            try:
                line = registration_models_file.readline().strip()
                model = from_json(line)
                print(f"loaded model: {line} ")
            except ValueError:
                print(f"Cannot read model from line: {line}, most likely we have reached the end of the shifts file, have the channels a different number of time points?")

        print(f'Fusing...')

        with CupyBackend(device):
            array, model = simview_fuse_2I2D(C0L0, C0L1, C1L0, C1L1,
                                             registration_model=model,
                                             zero_level=zero_level,
                                             fusion=fusion,
                                             fusion_bias_exponent=2 if fusion_bias_strength > 0 else 1,
                                             fusion_bias_strength=fusion_bias_strength,
                                             dehaze_size=dehaze_size,
                                             dark_denoise_threshold=dark_denoise_threshold)

            array = Backend.to_numpy(array, dtype=dest_array.dtype, force_copy=False)

        if not load_shifts:
            json_text = model.to_json()
            registration_models_file.write(json_text + '\n')

        print(f'Writing array of dtype: {array.dtype}')
        dest_array[tp] = array


    if workers > 1:
        Parallel(n_jobs=workers)(process(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
    else:
        for tp in range(0, shape[0]):
            process(tp)

    registration_models_file.close()

    print(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()
