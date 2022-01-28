from typing import Tuple


def get_3d_image_graph(shape: Tuple[int, ...]):
    assert len(shape) == 3
    import higra as hg

    neighbors = []
    for i in range(3):
        for j in (1, -1):
            shift = [0] * 3
            shift[i] = j
            neighbors.append(shift)

    return hg.get_nd_regular_graph(shape, neighbors)
