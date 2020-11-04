from abc import ABC, abstractmethod


class BaseFusion(ABC):

    def __init__(self):
        """

        """

    @abstractmethod
    def equalise_intensity(self, image1, image2, zero_level=90, percentile=0.999):
        pass

    @abstractmethod
    def fuse_lightsheets(self, CxL0, CxL1, asnumpy=True):
        pass

    @abstractmethod
    def register_stacks(self, C0Lx, C1Lx):
        pass

    @abstractmethod
    def fuse_cameras(self, C0Lx, C1Lx, asnumpy=True):
        pass
