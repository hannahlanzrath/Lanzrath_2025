from __future__ import annotations

import numpy as np
from skimage.measure import LineModelND, ransac


def get_t_from_RANSAC(
    t: np.ndarray,
    y: np.ndarray,
    start_idx: int,
    end_idx: int,
    threshold: float = 0.01,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Estimate a reference time via RANSAC linear regression on a signal segment.

    Fits a line to the rising portion of ``y`` (from ``start_idx`` to ``end_idx``)
    using RANSAC and extrapolates the x-intercept as the reference time.

    Args:
        t: Time vector.
        y: Signal vector (same length as ``t``).
        start_idx: First index of the segment used for fitting.
        end_idx: One-past-last index of the segment used for fitting.
        threshold: RANSAC inlier residual threshold.

    Returns:
        A tuple of:
            - regression_line: Predicted y-values over the full time vector.
            - intercept: Estimated x-intercept (reference time).
            - inliers: List of [t_inliers, y_inliers] arrays.
    """
    t = t.ravel()
    y = y.ravel()

    data = np.vstack((t[start_idx:end_idx], y[start_idx:end_idx])).T
    model_robust, inlier_mask = ransac(
        data, LineModelND, min_samples=2,
        residual_threshold=threshold, max_trials=1000,
    )

    regression_line = model_robust.predict_y(t)
    # x-intercept: solve predict_y(t) = 0 analytically using origin + direction
    intercept = -(model_robust.predict(np.array([0]))[0][1] / model_robust.direction[1])

    return regression_line, intercept, [data[inlier_mask, 0], data[inlier_mask, 1]]
