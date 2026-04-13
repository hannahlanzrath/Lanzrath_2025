from __future__ import annotations

from typing import Any

import numpy as np

from .numerics import fit_rising_line


def compute_intercept(
    t: np.ndarray,
    y: np.ndarray,
    line_start_index: int,
    npoints: int,
) -> dict[str, Any]:
    """Estimate a reference time from the x-intercept of a local linear fit.

    Fits a line to the rising segment of ``y`` starting at ``line_start_index``
    and uses its x-intercept as the reference time.

    Args:
        t: Time vector.
        y: Signal vector (same length as ``t``).
        line_start_index: Index of the first sample included in the fit.
        npoints: Number of consecutive samples to include in the fit.

    Returns:
        Dict with keys:
            - ``slope``: Fitted slope.
            - ``intercept``: Fitted y-intercept.
            - ``t_ref``: Reference time (x-intercept of the fitted line).
    """
    m, b = fit_rising_line(t, y, line_start_index, npoints)

    return {
        "slope": m,
        "intercept": b,
        "t_ref": -b / m,
    }
