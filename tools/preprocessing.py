from __future__ import annotations

import numpy as np


DEFAULT_DECAY_CONSTANT = 0.000567 * 60


def decay_correct_and_normalize(
    data: np.ndarray,
    t: np.ndarray,
    decay_constant: float = DEFAULT_DECAY_CONSTANT,
) -> np.ndarray:
    """Apply decay correction and normalize to the maximum.

    Works for 1-D signals and 2-D matrices (time × ROI).
    """
    corrected = (data.T / np.exp(-decay_constant * t)).T
    return corrected / np.max(corrected)


def extract_rois_from_matrix(
    data: np.ndarray,
    roi_indices: tuple[int, ...],
) -> list[np.ndarray]:
    """Extract column slices from a 2-D data matrix."""
    return [data[:, i] for i in roi_indices]


def prepare_signal(
    t: np.ndarray,
    data: np.ndarray,
    roi_index: int,
    decay_constant: float,
) -> np.ndarray:
    """Extract, correct, and normalize one ROI signal."""
    return decay_correct_and_normalize(
        data[:, roi_index], t, decay_constant=decay_constant,
    )
