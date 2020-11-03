from abc import abstractmethod
from typing import Any, Union

import numpy
import torch
from torch.types import Device

from dexp.processing.backends.backend import Backend


class PytorchBackend(Backend):

    def __init__(self, device: Device = 'cpu'):
        """ Instanciates a Pytorch-based Image Processing backend

        """
        self.device = device

    def close(self):
        #Nothing to do
        pass

    def to_numpy(self, array, dtype=None, copy: bool = False) -> numpy.ndarray:
        if torch.is_tensor(array):
            array = array.cpu().detach().numpy()
            if dtype:
                return array.astype(dtype, copy=copy)
            elif copy:
                return array.copy()
            else:
                return array
        else:
            return array.astype(dtype, copy=False)

    def to_backend(self, array, dtype=None, copy: bool = False) -> Any:
        #TODO: handle better dtype conversion...
        if torch.is_tensor(array):
            if copy:
                return array.detach().clone()
            else:
                return array
        else:
            if dtype:
                array = array.astype(dtype, copy=copy)
            return torch.tensor(array, requires_grad=False, device=self.device)

    def get_xp_module(self, array=None) -> Any:
        raise NotImplementedError("numpy-like module not defined for Pytorch backend")

    def get_sp_module(self, array=None) -> Any:
        raise NotImplementedError("numpy-like module not defined for Pytorch backend")



