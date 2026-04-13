from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityFitResult:
    """Result of a distance-time linear fit.

    Attributes:
        slope: Fitted slope (transport velocity) in mm/min.
        intercept: Fitted y-intercept in mm.
        x_intercept: x-intercept of the fitted line (-intercept / slope).
        r2: Coefficient of determination (R²).
        fit: Raw ``np.polyfit`` output ``[slope, intercept]``.
        covariance: 2×2 covariance matrix of the fit coefficients.
            Filled with NaN when fewer than two unique time points are available.
        slope_std: Standard deviation of the slope estimate.
        mbinter_v: Array of shape (1, 3) containing ``[slope, intercept, x_intercept]``.
    """

    slope: float
    intercept: float
    x_intercept: float
    r2: float
    fit: np.ndarray
    covariance: np.ndarray
    slope_std: float
    mbinter_v: np.ndarray


def fit_distance_time_line(times: np.ndarray, positions: np.ndarray) -> VelocityFitResult:
    """Fit a line through distance-time reference points and compute velocity.

    Args:
        times: Reference times [min] for each ROI.
        positions: Spatial positions [mm] for each ROI.

    Returns:
        A :class:`VelocityFitResult` with slope, intercept, R², covariance, and
        the composite ``mbinter_v`` array.
    """
    can_compute_cov = len(np.unique(times)) >= 2

    if can_compute_cov:
        fit, covariance = np.polyfit(times, positions, 1, cov=True)
        slope_std = float(np.sqrt(covariance[0][0]))
        r2 = float(np.corrcoef(times, positions)[0, 1] ** 2)
    else:
        fit = np.polyfit(times, positions, 1)
        covariance = np.full((2, 2), np.nan)
        slope_std = np.nan
        r2 = np.nan

    slope, intercept = float(fit[0]), float(fit[1])
    x_intercept = -intercept / slope
    mbinter_v = np.array([[slope, intercept, x_intercept]])

    return VelocityFitResult(
        slope=slope,
        intercept=intercept,
        x_intercept=x_intercept,
        r2=r2,
        fit=fit,
        covariance=covariance,
        slope_std=slope_std,
        mbinter_v=mbinter_v,
    )
