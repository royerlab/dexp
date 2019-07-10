from abc import ABC, abstractmethod


class DatasetBase(ABC):

    def __init__(self):
        """

        """

    @abstractmethod
    def channels(self):
        pass

    @abstractmethod
    def shape(self, channel):
        pass

    @abstractmethod
    def nb_timepoints(self, channel):
        pass

    @abstractmethod
    def get_stacks(self, channel):
        pass

    @abstractmethod
    def get_stack(self, channel, time_point):
        pass
