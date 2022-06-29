import warnings
from typing import Optional, Sequence, Tuple

import numpy
from arbol import aprint
from scipy.optimize import minimize

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def linsolve(
    a: xpArray,
    y: xpArray,
    x0: Optional[xpArray] = None,
    maxiter: int = 1e12,
    maxfun: int = 1e12,
    tolerance: float = 1e-6,
    order_error: float = 1,
    order_reg: float = 1,
    alpha_reg: float = 1e-1,
    l2_init: bool = False,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    limited: bool = True,
    verbose: bool = False,
) -> xpArray:
    xp = Backend.get_xp_module()

    a = Backend.to_backend(a)
    y = Backend.to_backend(y)

    if x0 is None:
        if l2_init:
            x0 = linsolve(a, y, x0=x0, maxiter=maxiter, tolerance=tolerance, order_error=2, alpha_reg=0, l2_init=False)
        else:
            x0 = numpy.zeros(a.shape[1])

    beta = (1.0 / y.shape[0]) ** (1.0 / order_error)
    alpha = (1.0 / x0.shape[0]) ** (1.0 / order_reg)

    def fun(x):
        x = Backend.to_backend(x)
        if alpha_reg == 0:
            objective = beta * float(xp.linalg.norm(a @ x - y, ord=order_error))
            # aprint(f"objective={objective}, regterm=N/A ")
        else:
            objective = beta * float(xp.linalg.norm(a @ x - y, ord=order_error))
            regularisation_term = (alpha_reg * alpha) * float(xp.linalg.norm(x, ord=order_reg))
            # aprint(f"objective={objective}, regterm={regularisation_term} ")
            objective += regularisation_term
        return objective

    result = minimize(
        fun,
        x0,
        method="L-BFGS-B" if limited else "BFGS",
        tol=tolerance,
        bounds=bounds if limited else None,
        options={
            "disp": verbose,
            "maxiter": maxiter,
            "maxfun": maxfun,
            "gtol": tolerance,
            "eps": 1e-5,  # the minimization would not converge sometimes without this param.
        },
    )

    if result.nit == 0:
        aprint(f"Warning: optimisation finished after {result.nit} iterations!")

    if not result.success:
        warnings.warn(
            f"Convergence failed: '{result.message}' after {result.nit} "
            + "iterations and {result.nfev} function evaluations."
        )
        return Backend.to_backend(x0)

    return Backend.to_backend(result.x)
