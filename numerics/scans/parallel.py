"""Independent tile-decomposition parallel scan utilities."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def make_tiles(
    n_rows: int,
    n_cols: int,
    n_tiles: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """
    Split an n_rows x n_cols grid into roughly n_tiles rectangular tiles.

    Returns
    -------
    tiles : list of (row_start, row_end, col_start, col_end, local_i, local_j)
        Index ranges in the global grid and the local spiral center for each tile.
    """
    if n_tiles < 1:
        raise ValueError("n_tiles must be >= 1")

    n_tile_rows = max(1, int(round(math.sqrt(n_tiles * n_rows / max(n_cols, 1)))))
    n_tile_cols = max(1, math.ceil(n_tiles / n_tile_rows))

    n_tile_rows = min(n_tile_rows, n_rows)
    n_tile_cols = min(n_tile_cols, n_cols)

    row_edges = np.linspace(0, n_rows, n_tile_rows + 1, dtype=int)
    col_edges = np.linspace(0, n_cols, n_tile_cols + 1, dtype=int)

    tiles = []
    for ri in range(n_tile_rows):
        for ci in range(n_tile_cols):
            row_start, row_end = int(row_edges[ri]), int(row_edges[ri + 1])
            col_start, col_end = int(col_edges[ci]), int(col_edges[ci + 1])
            if row_end <= row_start or col_end <= col_start:
                continue
            center_i = (row_start + row_end - 1) // 2
            center_j = (col_start + col_end - 1) // 2
            local_i = center_i - row_start
            local_j = center_j - col_start
            tiles.append((row_start, row_end, col_start, col_end, local_i, local_j))

    return tiles


class TileScanRunner:
    """
    Run a scan function independently over a set of rectangular tiles using a
    process pool.

    Parameters
    ----------
    worker_func : callable
        Function called in each worker as ``worker_func(tile_args)``.
    n_workers : int
        Number of worker processes.
    initializer : callable, optional
        ``ProcessPoolExecutor`` initializer; typically sets per-worker globals.
    initargs : tuple
        Arguments passed to ``initializer``.
    verbose : bool
        Print tile-completion progress.
    """

    def __init__(
        self,
        worker_func: Callable,
        n_workers: int = 8,
        initializer: Callable | None = None,
        initargs: tuple = (),
        verbose: bool = True,
    ) -> None:
        self.worker_func = worker_func
        self.n_workers = n_workers
        self.initializer = initializer
        self.initargs = initargs
        self.verbose = verbose

    def run(
        self,
        tiles: Sequence[tuple[int, int, int, int, int, int]],
        tile_args: Sequence,
        merge_func: Callable[[tuple, any], None],
    ) -> None:
        """
        Dispatch tiles to workers and merge each result as it completes.

        Parameters
        ----------
        tiles : sequence of tile tuples
        tile_args : sequence
            One argument object per tile, passed to ``worker_func``.
        merge_func : callable
            ``merge_func(tile, result)`` is called in the parent process for
            each completed tile.
        """
        n_total = len(tiles)
        if n_total == 0:
            return

        if self.verbose:
            print(f"  Dispatching {n_total} tiles to {self.n_workers} workers...")

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=self.initializer,
            initargs=self.initargs,
        ) as executor:
            futures = {
                executor.submit(self.worker_func, args): tile
                for tile, args in zip(tiles, tile_args)
            }
            completed = 0
            for fut in as_completed(futures):
                tile = futures[fut]
                result = fut.result()
                completed += 1
                if self.verbose and (
                    completed % max(1, n_total // 10) == 0 or completed == n_total
                ):
                    print(f"  Parallel scan: {completed}/{n_total} tiles completed", flush=True)
                merge_func(tile, result)
