from __future__ import annotations

from typing import Any

import numpy as np
from scipy import interpolate

from .numerics import find_first_crossing


def compute_halfmax_spline(
    t: np.ndarray,
    y: np.ndarray,
    spline_points: int,
    spline_end: float,
) -> dict[str, Any]:
    """Estimate a reference time from the first half-maximum crossing of a cubic spline.

    Fits a natural cubic spline to ``(t, y)`` and evaluates it on a dense grid up to
    ``spline_end``. The reference time is the first grid point where the spline reaches
    half of its maximum value.

    Args:
        t: Time vector (used as spline knots).
        y: Signal vector (same length as ``t``).
        spline_points: Number of points in the dense evaluation grid.
        spline_end: End time of the dense evaluation grid [min].

    Returns:
        Dict with keys:
            - ``spline``: Fitted :class:`scipy.interpolate.CubicSpline` object.
            - ``t_dense``: Dense time grid used for evaluation.
            - ``y_dense``: Spline values on ``t_dense``.
            - ``ymax``: Maximum value of the spline on the dense grid.
            - ``halfmax``: Half of ``ymax``.
            - ``t_ref``: Reference time (first crossing of ``halfmax``).
    """
    t_dense = np.linspace(0.0, spline_end, spline_points)
    spline = interpolate.CubicSpline(t.ravel(), y.ravel(), bc_type="natural")
    y_dense = spline(t_dense)

    ymax = np.max(y_dense)
    halfmax = ymax / 2.0
    t_ref = find_first_crossing(t_dense, y_dense, halfmax)

    return {
        "spline": spline,
        "t_dense": t_dense,
        "y_dense": y_dense,
        "ymax": ymax,
        "halfmax": halfmax,
        "t_ref": t_ref,
    }
