import pytest

from dexp.datasets.synthetic_datasets import (
    binary_blobs,
    generate_fusion_test_data,
    generate_nuclei_background_data,
)


@pytest.fixture
def dexp_binary_blobs(request):
    return binary_blobs(request.param)


@pytest.fixture
def dexp_nuclei_background_data(request):
    return generate_nuclei_background_data(**request.param)


@pytest.fixture
def dexp_fusion_test_data(request):
    return generate_fusion_test_data(**request.param)
