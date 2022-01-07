import pytest
from arbol import aprint, asection

from dexp.processing.utils.linear_solver import linsolve
from dexp.utils.backends.backend import Backend
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
@pytest.mark.parametrize(
    "w, h, noise, note",
    [(37, 41, 1e-2, "small")],
)
def test_linear_solver(w: int, h: int, noise: float, note: str) -> None:
    with asection(f"Running l2 solver on {note} matrix: "):
        error = _run_solver(w=w, h=h, noise=noise)
        aprint(f"Error = {error}")


def _run_solver(w=37, h=41, noise=1e-2, **kwargs):
    xp = Backend.get_xp_module()

    xp.random.seed(42)

    a = xp.random.rand(w, h)
    a *= xp.random.rand(w, h) > 0.9
    x_gt = xp.random.rand(h)
    x_gt *= xp.random.rand(h) > 0.5
    y_obs = a @ x_gt + noise * (xp.random.rand(w) - 0.5)
    x = linsolve(a, y_obs, **kwargs)
    mean_abs_error = xp.mean(xp.absolute(x - x_gt)).item()

    aprint(f"X_gt  : {x_gt} ")
    aprint(f"u     : {x} ")
    aprint(f"error : {xp.absolute(x - x_gt)} ")

    return mean_abs_error
