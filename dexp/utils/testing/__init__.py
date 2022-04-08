import pytest

from dexp.utils.testing.testing import cupy_only, execute_both_backends


def test_as_demo(file: str) -> None:
    """Executes this tests as demos.

    Should be used as `test_as_demo(__file__)`
    """
    pytest.main([file, "-s", "-rs", "--display", "True"])
