from typing import Any

import numpy
import torch
from torch.types import Device

from dexp.processing.backends.backend import Backend


class PytorchBackend(Backend):

    def __init__(self, device: Device = 'cpu'):
        """ Instanciates a Pytorch-based Image Processing backend

        """
        self.device = device

    def __str__(self):
        return "PytorchBackend"

    def close(self):
        # Nothing to do
        pass

    def to_numpy(self, array, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        if torch.is_tensor(array):
            array = array.cpu().detach().numpy()
            if dtype:
                return array.astype(dtype, copy=force_copy)
            elif force_copy:
                return array.copy()
            else:
                return array
        else:
            return array.astype(dtype, copy=False)

    def to_backend(self, array, dtype=None, force_copy: bool = False) -> Any:

        array = self.to_numpy(array)

        # TODO: handle better dtype conversion...
        if torch.is_tensor(array):
            if force_copy:
                return array.detach().clone()
            else:
                return array
        else:
            if dtype:
                array = array.astype(dtype, copy=force_copy)
            return torch.tensor(array, requires_grad=False, device=self.device)

    def get_xp_module(self, array=None) -> Any:
        raise NotImplementedError("numpy-like module not defined for Pytorch backend")

    def get_sp_module(self, array=None) -> Any:
        raise NotImplementedError("numpy-like module not defined for Pytorch backend")
