from typing import Callable, Tuple

import numpy as np

from dexp.processing.utils.element_wise_affine import element_wise_affine
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


class Normalise:
    def __init__(
        self,
        image: xpArray,
        low: float = 0.0,
        high: float = 1.0,
        minmax: Tuple[float, float] = None,
        quantile: float = 0,
        clip: bool = False,
        do_normalise: bool = True,
        in_place: bool = True,
        dtype=None,
    ) -> Tuple[Callable, Callable]:
        """Returns a pair of functions: the first normalises the given image to the range [low, high],
        the second denormalises back to the original range (and dtype).
        Usefull when determining the normalisation parameters and doing the actual normalisation must be decoupled,
        for example when splitting an image into chunks and sensing computation to the GPU.


        Parameters
        ----------
        image : image to normalise
        low, high : normalisation range
        minmax : min and max values of the image if already known.
        quantile : if quantile>0 then quantile normalisation is used to find the min and max values.
            Value must be within [0,1]
        clip : clip after normalisation/denormalisation
        do_normalise : If False, the returned functions are pass-through identity functions
            -- usefull to turn on and off normalisation while still using the functions themselves.
        in_place : In-place computation is allowed and inputs may be modified
        dtype : dtype to normalise to.

        Returns
        -------

        """
        xp = Backend.get_xp_module()

        if isinstance(Backend.current(), NumpyBackend):
            dtype = np.float32

        elif dtype is None:
            if image.itemsize <= 2:
                dtype = xp.float16
            elif image.itemsize == 4 or image.itemsize == 8:
                dtype = xp.float32
            else:
                raise ValueError(f"Failed to converted type {image.dtype}")

        self.dtype = dtype
        self.original_dtype = image.dtype
        self.clip = clip
        self.low = low
        self.high = high
        self.min_value = None
        self.max_value = None
        self.in_place = in_place
        self.do_normalise = do_normalise
        self.norm_alpha = None
        self.norm_beta = None
        self.denorm_alpha = None
        self.denorm_beta = None

        if not self.do_normalise:
            return

        if minmax is None:
            min_value, max_value = 0, 0
            if quantile > 0:
                min_value = xp.quantile(image, q=quantile)
                max_value = xp.quantile(image, q=1 - quantile)

            # if we did not use quantiles or we get some weird result, let's fallback to standard min max:
            if min_value >= max_value:
                min_value = xp.min(image)
                max_value = xp.max(image)

        else:
            min_value, max_value = minmax

        self.min_value = min_value
        self.max_value = max_value

        # Normalise:
        norm_denom = max_value - min_value
        if norm_denom == 0:
            norm_denom = 1

        norm_alpha = (high - low) / norm_denom
        norm_beta = low - norm_alpha * min_value

        # Ensure correct type:
        self.norm_alpha = xp.asarray(norm_alpha, dtype=dtype)
        self.norm_beta = xp.asarray(norm_beta, dtype=dtype)

        # Denormalise:
        denorm_alpha = (max_value - min_value) / (high - low)
        denorm_beta = min_value - denorm_alpha * low

        # Ensure correct type:
        self.denorm_alpha = xp.asarray(denorm_alpha, dtype=dtype)
        self.denorm_beta = xp.asarray(denorm_beta, dtype=dtype)

    def _transform(
        self, array: xpArray, alpha: float, beta: float, low: float, high: float, dtype: np.dtype
    ) -> xpArray:

        if self.do_normalise:
            array = Backend.to_backend(array, dtype=self.dtype)
            out = array if self.in_place else None
            array = element_wise_affine(array, alpha, beta, out=out)
            if self.clip:
                array = array.clip(low, high, out=array)
        return array.astype(dtype, copy=False)

    def forward(self, array: xpArray) -> xpArray:
        return self._transform(array, self.norm_alpha, self.norm_beta, self.low, self.high, self.dtype)

    def backward(self, array: xpArray) -> xpArray:
        return self._transform(
            array, self.denorm_alpha, self.denorm_beta, self.min_value, self.max_value, self.original_dtype
        )
