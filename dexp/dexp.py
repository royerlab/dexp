import glob

import dask.array as da
import tifffile as tiff
from dask import delayed


def get_dask_array(filename_pattern):
    # filename_pattern ='/Volumes/GoogleDrive/My\ Drive/Datasets/AutoPilot/Dme_E1_His2AvRFP_01_TL_20131204_140355.corrected/TimeFused/SPC0_TM*_CM0_CM1_CHN00_CHN01.fusedStack.tif'

    imread = delayed(tiff.imread, pure=True)  # Lazy version of imread

    filenames = sorted(glob.glob(filename_pattern))

    for filename in filenames:
        print(filename)

    print("Number of files found: %d" % len(filenames))

    lazy_images = [imread(path) for path in filenames]  # Lazily evaluate imread on each path

    # print(lazy_images)

    sample = lazy_images[0].compute()  # load the first image (assume rest are same shape/dtype)

    arrays = [da.from_delayed(lazy_image,  # Construct a small Dask array
                              dtype=sample.dtype,  # for every lazy value
                              shape=sample.shape)
              for lazy_image in lazy_images]

    stack = da.stack(arrays, axis=0)  # Stack all small Dask arrays into one

    print(stack.shape)
    print(stack.dtype)

    return stack
