import os

from arbol import aprint


def compute_num_workers(n_workers: int, n_time_points: int) -> int:
    if n_workers < 0:
        n_workers = int(os.cpu_count() / -n_workers)
    n_workers = min(max(1, n_workers), n_time_points)
    aprint(f"Number of workers: {n_workers}")
    return n_workers
