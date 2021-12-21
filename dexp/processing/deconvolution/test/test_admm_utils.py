import pytest
import numpy as np
from numpy.testing import assert_allclose

from dexp.processing.deconvolution.admm_utils import *
import scipy.ndimage as ndi


@pytest.mark.parametrize(
    'derivative_kernels, derivative_func', [
        (first_derivative_kernels, first_derivative_func),
        (second_derivative_kernels, second_derivative_func),
    ]
)
def test_derivatives(derivative_kernels, derivative_func):
    image = np.random.randint(256, size=(64, 64, 64))

    # ignoring the border, the line kernel is exact
    # the diagonal kernel ignores the borders
    centering = tuple((slice(1, 63) for _ in range(image.ndim)))

    Ks = derivative_kernels(image.ndim)
    Daxes = derivative_axes(image.ndim)

    for K, axes in zip(Ks, Daxes):
        convolved = ndi.convolve(image, K, mode='nearest')[centering]
        finite_dif = derivative_func(image, axes, transpose=True)[centering]

        assert_allclose(convolved, finite_dif, err_msg=f'Axes {axes}', verbose=True)

        correlated = ndi.correlate(image, K, mode='nearest')[centering]
        finite_dif = derivative_func(image, axes, transpose=False)[centering]

        assert_allclose(correlated, finite_dif, err_msg=f'Axes {axes}', verbose=True)
