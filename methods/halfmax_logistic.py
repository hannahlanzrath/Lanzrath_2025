from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize


def logistic4(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Evaluate a four-parameter logistic rise."""
    return a / (1.0 + np.exp(-c * (x - d))) + b

def compute_logistic(
    t: np.ndarray,
    y: np.ndarray,
    initial_guesses: tuple[float, float, float, float],
    bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
) -> dict[str, Any]:
    """Reference time from midpoint of a logistic fit."""
    peak_idx = int(np.argmax(y))
    fit_t = t[: peak_idx + 1]
    fit_y = y[: peak_idx + 1]

    try:
        popt, _ = optimize.curve_fit(
            logistic4,
            fit_t,
            fit_y,
            p0=initial_guesses,
            bounds=bounds,
            maxfev=5000,
        )
    except RuntimeError as exc:
        return {"fit_error": str(exc)}

    a, b, c, d = popt
    y_fit = logistic4(t, a, b, c, d)
    halfmax = b + a / 2.0

    return {
        "params": {"a": a, "b": b, "c": c, "d": d},
        "halfmax": halfmax,
        "t_ref": d,
        "y_fit": y_fit,
    }