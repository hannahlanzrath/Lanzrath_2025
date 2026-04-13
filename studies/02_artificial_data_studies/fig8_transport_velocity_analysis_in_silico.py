from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from Lanzrath_2025.tools.preprocessing import decay_correct_and_normalize, extract_rois_from_matrix
from Lanzrath_2025.tools.analysis import (
    analyze_ransac_reference_times,
    analyze_halfmax_spline_reference_times,
    analyze_mckay_reference_times,
    analyze_buehler_transport_velocity,
)
from Lanzrath_2025.tools.config import ArtificialDataConfig
from Lanzrath_2025.tools.simulation import simulate_transport_case
from Lanzrath_2025.tools.utils import add_gaussian_noise, save_figure

np.random.seed(seed=123)


def run_transport_velocity_analysis(
    xi: list[float] | None = None,
    steps: int = 30,
    t_start: float = 0.0,
    t_end: float = 145.0,
    model_name: str = "M02",
    model_file: str = "model.h5",
    params: list[float] | None = None,
    config: ArtificialDataConfig | None = None,
    show: bool = True,
    save: bool = False,
    save_folder: str = "figures/figure_8",
    file_prefix: str = "fig8_transport_velocity_analysis_in_silico",
    add_noise: bool = False,
    optimize: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run transport-velocity analysis for several reference-time methods."""
    config = config or ArtificialDataConfig()
    xi = xi or [0, 20, 40]
    t = np.linspace(t_start, t_end, steps)

    suffix = "noise" if add_noise else "clean"
    file_prefix = f"{file_prefix}_{suffix}"

    positions, t, data = simulate_transport_case(
        model_name=model_name,
        model_file=model_file,
        xi=xi,
        t=t,
        params=params or [5.0, 100, 40, 10, 2, 0.0],
    )

    selected_data = data[:, list(config.roi_indices)]

    if add_noise:
        data_used = add_gaussian_noise(selected_data, config.noise_std)
    else:
        data_used = selected_data

    data_processed = decay_correct_and_normalize(
        data_used, t, decay_constant=config.decay_constant,
    )
    rois_decay_corrected = extract_rois_from_matrix(
        data_processed, roi_indices=tuple(range(len(config.roi_indices))),
    )

    global_max = np.max(data_used)
    if global_max == 0:
        global_max = 1.0
    data_raw_normalized = data_used / global_max
    rois_raw = extract_rois_from_matrix(
        data_raw_normalized, roi_indices=tuple(range(len(config.roi_indices))),
    )

    selected_positions = positions[list(config.roi_indices)]

    return {
        "ransac_intercept": analyze_ransac_reference_times(
            t, rois_decay_corrected, selected_positions, config,
            show=show, save=save, save_folder=save_folder, file_prefix=file_prefix,
            save_fn=save_figure,
        ),
        "halfmax_spline": analyze_halfmax_spline_reference_times(
            t, rois_decay_corrected, selected_positions, config,
            show=show, save=save, save_folder=save_folder, file_prefix=file_prefix,
            save_fn=save_figure,
        ),
        "mckay": analyze_mckay_reference_times(
            t, rois_decay_corrected, selected_positions, config,
            show=show, save=save, save_folder=save_folder, file_prefix=file_prefix,
            save_fn=save_figure,
        ),
        "buehler_model": analyze_buehler_transport_velocity(
            t, rois_raw, selected_positions, config,
            model_name=config.buehler_model_name,
            model_file=config.buehler_model_file,
            p0=[5.00838376e+00, 9.99959064e+01, 3.99955292e+01,
                9.94804318e+00, 1.99461275e+00, 1.02952116e-04],
            show=show, save=save, save_folder=save_folder, file_prefix=file_prefix,
            save_fn=save_figure,
            optimize=optimize,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Run least-squares fitting before extracting velocity")
    args = parser.parse_args()
    optimize = args.optimize

    run_transport_velocity_analysis(show=True, save=True, add_noise=True, optimize=optimize)
    run_transport_velocity_analysis(show=True, save=True, add_noise=False, optimize=optimize)


if __name__ == "__main__":
    main()
