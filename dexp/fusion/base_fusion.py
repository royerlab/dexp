import os
from abc import ABC, abstractmethod

import numpy
import zarr
from numcodecs.blosc import Blosc
from tifffile import memmap, TiffWriter
from zarr import open_group

from dexp.enhance.sharpen import sharpen


class BaseFusion(ABC):

    def __init__(self):
        """

        """

    @abstractmethod
    def fuse(self, C0L0, C0L1, C1L0, C1L1):
        pass



