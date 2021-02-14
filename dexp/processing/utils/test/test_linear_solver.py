import numpy
from arbol import aprint, asection
from numpy import random

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.linear_solver import linsolve


def test_linear_solver_numpy():
    with NumpyBackend():
        _test_linear_solver_small()
        # _test_linear_solver_large()
        # test_linear_solver_compare()


def test_linear_solver_cupy():
    try:
        with CupyBackend():
            _test_linear_solver_small()
            # _test_linear_solver_large()
            # test_linear_solver_compare()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_linear_solver_small():
    w = 37
    h = 41
    noise = 1e-2

    a = random.rand(w, h)
    a *= random.rand(w, h) > 0.9
    x_gt = random.rand(h)
    x_gt *= random.rand(h) > 0.5
    y_obs = a @ x_gt + noise * (random.rand(w) - 0.5)

    with asection("Running l2 solver on small matrix: "):
        aprint(f"error = {numpy.mean(numpy.absolute(linsolve(a, y_obs, order_error=2, alpha_reg=0) - x_gt))}")

    # with asection("Running l1 solver on small matrix: "):
    #     aprint(f"error = {numpy.mean(numpy.absolute(linsolve(a, y_obs, order_error=1, alpha_reg=0) - x_gt))}")
    #
    # with asection("Running l0.5 solver on small matrix: "):
    #     aprint(f"error = {numpy.mean(numpy.absolute(linsolve(a, y_obs, order_error=0.5, alpha_reg=0) - x_gt))}")
    #
    # with asection("Running l2l1 solver on small matrix: "):
    #     aprint(f"error = {numpy.mean(numpy.absolute(linsolve(a, y_obs, order_error=2, order_reg=1, alpha_reg=0.1) - x_gt))}")
    #
    # with asection("Running l1l1 solver on small matrix: "):
    #     aprint(f"error = {numpy.mean(numpy.absolute(linsolve(a, y_obs, order_error=1, order_reg=1, alpha_reg=0.1) - x_gt))}")
    #
    # with asection("Running l0.5l1 solver on small matrix: "):
    #     aprint(f"error = {numpy.mean(numpy.absolute(linsolve(a, y_obs, order_error=0.5, order_reg=1, alpha_reg=0.1) - x_gt))}")


def _run_solver(w=37, h=41, noise=1e-2, display=False, **kwargs):
    a = random.rand(w, h)
    a *= random.rand(w, h) > 0.9
    x_gt = random.rand(h)
    x_gt *= random.rand(h) > 0.5
    y_obs = a @ x_gt + noise * (random.rand(w) - 0.5)
    x = linsolve(a, y_obs, **kwargs)
    mean_abs_error = numpy.mean(numpy.absolute(x - x_gt))

    if display:
        aprint(f"x_gt  : {x_gt} ")
        aprint(f"x     : {x} ")
        aprint(f"error : {numpy.absolute(x - x_gt)} ")

    return mean_abs_error


def _test_linear_solver_large():
    with asection("Running solver on large matrix: "):
        aprint(_run_solver(w=1280, h=1370, order_error=1, order_reg=1, alpha_reg=0.1, display=True, limited=True))

# def test_linear_solver_compare():
#     n = 16
#
#     mean_error = numpy.mean(list(_run_solver(order_error=2, alpha_reg=0) for _ in range(n)))
#     aprint(f"L2 mean_error={mean_error}")
#
#     mean_error = numpy.mean(list(_run_solver(order_error=1, alpha_reg=0) for _ in range(n)))
#     aprint(f"L1 mean_error={mean_error}")
#
#     mean_error = numpy.mean(list(_run_solver(order_error=0.5, alpha_reg=0) for _ in range(n)))
#     aprint(f"L0.5 mean_error={mean_error}")
#
#     mean_error = numpy.mean(list(_run_solver(order_error=2, order_reg=2, alpha_reg=0.1) for _ in range(n)))
#     aprint(f"L2L2 mean_error={mean_error}")
#
#     mean_error = numpy.mean(list(_run_solver(order_error=2, order_reg=1, alpha_reg=0.1) for _ in range(n)))
#     aprint(f"L2L1 mean_error={mean_error}")
#
#     mean_error = numpy.mean(list(_run_solver(order_error=1, order_reg=1, alpha_reg=0.1) for _ in range(n)))
#     aprint(f"L1L1 mean_error={mean_error}")
#
#     mean_error = numpy.mean(list(_run_solver(order_error=1, order_reg=0.5, alpha_reg=0.1) for _ in range(n)))
#     aprint(f"L1L0.5 mean_error={mean_error}")
