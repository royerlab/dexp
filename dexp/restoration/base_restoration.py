from abc import ABC, abstractmethod


class BaseRestoration(ABC):

    def __init__(self):
        """

        """

    @abstractmethod
    def calibrate(self, images):
        pass

    @abstractmethod
    def restore(self, image, asnumpy=True):
        pass










