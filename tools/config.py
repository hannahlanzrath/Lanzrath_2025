from __future__ import annotations

from dataclasses import dataclass

from Lanzrath_2025.tools.preprocessing import DEFAULT_DECAY_CONSTANT


@dataclass(frozen=True)
class ArtificialDataConfig:
    decay_constant: float = DEFAULT_DECAY_CONSTANT
    noise_std: float = 0.007
    roi_indices: tuple[int, ...] = (0, 1, 2)

    ransac_min: int = 5
    ransac_max: int = 11
    ransac_thresholds: tuple[float, ...] = (0.04, 0.05, 0.04)

    spline_points: int = 500
    spline_eval_end: float = 140.0
    spline_plot_end: float = 100.0

    mckay_dt: float = 1.0
    mckay_end: float = 145.0
    mckay_points: int = 30

    buehler_model_name: str = "M02"
    buehler_model_file: str = "model.h5"
    buehler_gauss_type: str = "stretched"
    buehler_fit_dt: float = 0.000567 * 60


@dataclass(frozen=True)
class PlantDataConfig:
    decay_constant: float = DEFAULT_DECAY_CONSTANT

    ransac_min: int | tuple[int, ...] = 0
    ransac_max: int | tuple[int, ...] = 120
    ransac_thresholds: tuple[float, float, float] = (0.15, 0.15, 0.15)

    spline_points: int = 400
    spline_eval_end: float = 150.0
    spline_plot_end: float = 150.0

    mckay_dt: float | None = None   # None → derived from t: t[1] - t[0]
    mckay_end: float | None = None  # None → derived from t: t[-1]
    mckay_points: int | None = None # None → derived from t: len(t)

    buehler_fit_dt: float = 0.5
    buehler_gauss_type: str = "stretched"
