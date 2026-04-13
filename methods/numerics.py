from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq


def find_first_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Find the first x-value where y crosses a given level.

    Uses linear interpolation between the two samples bracketing the crossing.

    Args:
        x: Independent variable array.
        y: Dependent variable array (same length as ``x``).
        level: The y-value to find.

    Returns:
        Interpolated x at the first crossing, or ``np.nan`` if no crossing exists.
    """
    shifted = y - level
    sign_changes = np.where(np.diff(np.sign(shifted)) != 0)[0]

    if len(sign_changes) == 0:
        return np.nan

    i = sign_changes[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = shifted[i], shifted[i + 1]

    if y1 == y0:
        return x0

    return x0 - y0 * (x1 - x0) / (y1 - y0)


def fit_rising_line(
    t: np.ndarray,
    y: np.ndarray,
    start_index: int,
    npoints: int,
) -> tuple[float, float]:
    """Fit a line ``y = m·t + b`` to a short rising segment of a signal.

    Args:
        t: Time vector.
        y: Signal vector (same length as ``t``).
        start_index: Index of the first sample included in the fit.
        npoints: Number of consecutive samples to include.

    Returns:
        A tuple ``(slope, intercept)`` of the fitted line.
    """
    stop_index = start_index + npoints
    design = np.column_stack((np.ones_like(t), t))
    b, m = lstsq(design[start_index:stop_index], y[start_index:stop_index], rcond=None)[0]
    return m, b
