from __future__ import annotations

from typing import Any

import numpy as np

from .numerics import fit_rising_line


def compute_halfmax_intercept(
    t: np.ndarray,
    y: np.ndarray,
    line_start_index: int,
    npoints: int,
) -> dict[str, Any]:
    """Estimate a reference time from the half-maximum of a local linear fit.

    Fits a line to the rising segment of ``y`` and returns the time at which
    the fitted line reaches 0.5 (half-maximum of a normalised signal).

    Args:
        t: Time vector.
        y: Signal vector (same length as ``t``), expected to be normalised to [0, 1].
        line_start_index: Index of the first sample included in the fit.
        npoints: Number of consecutive samples to include in the fit.

    Returns:
        Dict with keys:
            - ``slope``: Fitted slope.
            - ``intercept``: Fitted y-intercept.
            - ``halfmax``: The half-maximum level (always 0.5).
            - ``t_ref``: Reference time at which the fitted line equals ``halfmax``.
    """
    m, b = fit_rising_line(t, y, line_start_index, npoints)
    halfmax = 0.5

    return {
        "slope": m,
        "intercept": b,
        "halfmax": halfmax,
        "t_ref": (halfmax - b) / m,
    }
