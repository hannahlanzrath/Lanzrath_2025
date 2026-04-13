from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import fsolve


def compute_mckay(
    t: np.ndarray,
    y: np.ndarray,
    mckay_dt: float,
    mckay_end: float,
    mckay_points: int,
) -> dict[str, Any]:
    """Estimate a reference time using the McKay binding model.

    The signal is interpolated with a cubic spline, then split into bound (``B``)
    and unbound (``G``) fractions by solving for the binding rate ``sv`` that makes
    the bound fraction equal to the measured signal at the final time point.
    The centroid of the unbound fraction is returned as the reference time.

    Args:
        t: Measured time vector [min].
        y: Measured signal vector (same length as ``t``), normalised to [0, 1].
        mckay_dt: Integration step size [min].
        mckay_end: End time of the evaluation grid [min].
        mckay_points: Number of points in the evaluation grid.

    Returns:
        Dict with keys:
            - ``sv``: Solved binding rate.
            - ``T``: Evaluation time grid.
            - ``M``: Spline-interpolated signal on ``T``.
            - ``G``: Unbound fraction on ``T``.
            - ``B``: Bound fraction on ``T``.
            - ``t_ref``: Reference time (centroid of unbound fraction).
    """
    spline = interpolate.CubicSpline(t.ravel(), y.ravel(), bc_type="natural")
    T = np.linspace(0.0, mckay_end, mckay_points)
    M = spline(T).ravel()
    dt = mckay_dt

    def terminal_residual(sv_arr: np.ndarray, signal: np.ndarray) -> float:
        """Return residual between final bound fraction and final measured signal."""
        sv = float(sv_arr[0])
        B = np.zeros_like(signal)
        for j in range(1, len(signal)):
            B[j] = B[j - 1] + sv * (signal[j - 1] - B[j - 1]) * dt
        return B[-1] - signal[-1]

    def integrate_binding(sv_values: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Integrate bound fraction for each binding rate in ``sv_values``."""
        B = np.zeros((len(sv_values), len(signal)))
        for j, sv in enumerate(sv_values):
            for i in range(1, len(signal)):
                B[j, i] = B[j, i - 1] + sv * (signal[i - 1] - B[j, i - 1]) * dt
        return B

    sv_init = 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sv = fsolve(terminal_residual, sv_init, args=(M,))

    B = integrate_binding(sv, M).ravel()
    G = M - B

    t_ref = integrate.simpson(T * G, x=T) / integrate.simpson(G, x=T)

    return {
        "sv": float(sv[0]),
        "T": T,
        "M": M,
        "G": G,
        "B": B,
        "t_ref": t_ref,
    }
