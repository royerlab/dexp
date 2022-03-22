import itertools
from typing import Callable, Dict

import numpy
from numpy.typing import ArrayLike

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def j_invariant_grid_search(
    image: xpArray,
    function: Callable,
    loss_fun: Callable[[xpArray, xpArray], float],
    grid: Dict[str, ArrayLike],
    proportion_mask: float = 0.01,
    median_window: int = 3,
    display: bool = False,
) -> Dict[str, float]:

    backend = Backend.current()
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image = backend.to_backend(image)

    median = sp.ndimage.median_filter(image, size=median_window)
    mask = xp.random.uniform(size=image.shape) < proportion_mask

    # overwrite the values of the mask with median
    masked = image.copy()
    masked[mask] = median[mask]

    if display:
        import napari

        viewer = napari.view_image(backend.to_numpy(image), name="original")
        viewer.add_image(backend.to_numpy(median), name="median")
        viewer.add_image(backend.to_numpy(masked), name="mask")
        viewer.add_image(backend.to_numpy(mask), name="masked")

    history = []
    # compute grid search over possible values
    for params in itertools.product(*grid.values()):
        params = {k: p for k, p in zip(grid.keys(), params)}
        # predict results
        estimation = function(masked, **params)
        # compute loss function
        loss = loss_fun(estimation[mask], image[mask])
        loss = numpy.nan_to_num(loss, nan=1e12)

        name = f"{params}: loss={loss:0.3f}"
        if display:
            viewer.add_image(backend.to_numpy(estimation), name=name)
        print(name)
        history.append((loss, params))

    if display:
        napari.run()

    # return optimal (minimum)
    return min(history, key=lambda x: x[0])[1]
