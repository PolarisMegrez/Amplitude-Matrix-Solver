"""Grid traversal orders for continuation-style parameter scans."""

from __future__ import annotations


def spiral_order(n_rows: int, n_cols: int, center_i: int | None = None, center_j: int | None = None):
    """
    Yield grid indices in a Chebyshev-ring (square-spiral) order.

    Only the perimeter of each ring is emitted so the total work is O(N^2)
    instead of O(N^3).
    """
    if center_i is None:
        center_i = n_rows // 2
    if center_j is None:
        center_j = n_cols // 2

    order: list[tuple[int, int]] = []
    max_radius = max(center_i, n_rows - 1 - center_i, center_j, n_cols - 1 - center_j)
    for r in range(max_radius + 1):
        top, bottom = center_i - r, center_i + r
        left, right = center_j - r, center_j + r

        if 0 <= top < n_rows:
            for j in range(max(0, left), min(n_cols, right + 1)):
                order.append((top, j))
        if 0 <= bottom < n_rows and bottom != top:
            for j in range(max(0, left), min(n_cols, right + 1)):
                order.append((bottom, j))
        if r > 0 and 0 <= left < n_cols:
            for i in range(max(0, top + 1), min(n_rows, bottom)):
                order.append((i, left))
        if r > 0 and 0 <= right < n_cols and right != left:
            for i in range(max(0, top + 1), min(n_rows, bottom)):
                order.append((i, right))
    return order
