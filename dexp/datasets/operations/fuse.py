import numpy


def dataset_fuse(dataset,
         path,
         slicing,
         store,
         compression,
         compression_level,
         overwrite,
         workers,
         zero_level,
         load_shifts,
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

    from dexp.processing.fusion import SimpleFusion
    fusion = SimpleFusion()

    shifts_file = open("registration_shifts.txt", "r" if load_shifts else 'w')
    if load_shifts:
        print(f"Loading registration shifts from existing file! ({shifts_file.name})")

    def process(tp):
        print(f"Writing time point: {tp} ")

        C0L0 = array_C0L0[tp].compute()
        C0L1 = array_C0L1[tp].compute()
        C1L0 = array_C1L0[tp].compute()
        C1L1 = array_C1L1[tp].compute()

        C1L0 = numpy.flip(C1L0, -1)
        C1L1 = numpy.flip(C1L1, -1)

        if load_shifts:
            try:
                line = shifts_file.readline().strip()
                shifts = tuple(float(shift) for shift in line.split('\t'))
                print(f"loaded shifts: {shifts} ")
            except ValueError:
                print(f"Cannot read shift from line: {line}, most likely we have reached the end of the shifts file, have the channels a different number of time points?")

        else:
            shifts = None

        print(f'Fusing...')
        array, shifts = fusion.simview_fuse_2I2D(C0L0, C0L1, C1L0, C1L1, shifts=shifts, zero_level=zero_level, as_numpy=True)

        if not load_shifts:
            for shift in shifts:
                shifts_file.write(f"{shift}\t")
            shifts_file.write(f"\n")

        array = array.astype(dest_array.dtype, copy=False)

        print(f'Writing array of dtype: {array.dtype}')
        dest_array[tp] = array

    # TODO: we are not yet distributing computation over GPUs, that would require a proper use of DASK for that.
    # See: https://medium.com/rapids-ai/parallelizing-custom-cupy-kernels-with-dask-4d2ccd3b0732

    for tp in range(0, shape[0]):
        process(tp)

    shifts_file.close()

    print(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()