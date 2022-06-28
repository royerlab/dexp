import pytest

from dexp.datasets.synthetic_datasets import (
    binary_blobs,
    generate_dexp_zarr_dataset,
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


@pytest.fixture(scope="session")
def dexp_zarr_path(request, tmp_path_factory) -> str:
    ds_path = tmp_path_factory.mktemp("data") / "input.zarr"
    generate_dexp_zarr_dataset(ds_path, **request.param)
    return str(ds_path)


def pytest_addoption(parser):
    """pytest command line parser"""
    parser.addoption("--display", action="store", type=bool, default=False)


@pytest.fixture()
def display_test(pytestconfig):
    """display test fixture from pytest parser"""
    return pytestconfig.getoption("display")
