from typing import Any, List, Optional

import numpy as np

from dexp.processing.registration.model import RegistrationModel
from dexp.utils import xpArray


class BaseFusion:
    def __init__(
        self,
        registration_model: Optional[RegistrationModel],
        equalise: bool,
        equalisation_ratios: List[Optional[float]],
        zero_level: float,
        clip_too_high: int,
        fusion: str,
        dehaze_before_fusion: bool,
        dehaze_size: int,
        dehaze_correct_max_level: bool,
        dark_denoise_threshold: int,
        dark_denoise_size: int,
        butterworth_filter_cutoff: float,
        internal_dtype: np.dtype,
    ):

        self._registration_model = registration_model
        self._equalise = equalise
        self._equalisation_ratios = equalisation_ratios.copy()
        self._zero_level = zero_level
        self._clip_too_high = clip_too_high
        self._fusion = fusion
        self._dehaze_before_fusion = dehaze_before_fusion
        self._dehaze_size = dehaze_size
        self._dehaze_correct_max_level = dehaze_correct_max_level
        self._dark_denoise_threshold = dark_denoise_threshold
        self._dark_denoise_size = dark_denoise_size
        self._butterworth_filter_cutoff = butterworth_filter_cutoff
        self._internal_dtype = internal_dtype

    @staticmethod
    def _match_input(a: xpArray, b: xpArray) -> None:
        if a.shape != b.shape:
            raise ValueError("The views must have the same shape")
        if a.dtype != b.dtype:
            raise ValueError("The views must have the same dtype")

    def preprocess(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def postprocess(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def fuse(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def compute_registration(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        pass

    @property
    def registration_model(self) -> RegistrationModel:
        return self._registration_model
