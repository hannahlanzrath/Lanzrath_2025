from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

from Lanzrath_2025.methods.intercept_ransac import get_t_from_RANSAC
from Lanzrath_2025.tools.preprocessing import decay_correct_and_normalize, extract_rois_from_matrix
from Lanzrath_2025.tools.config import ArtificialDataConfig
from Lanzrath_2025.tools.fit import VelocityFitResult, fit_distance_time_line
from Lanzrath_2025.tools.simulation import simulate_transport_case
from Lanzrath_2025.tools.utils import add_gaussian_noise, save_figure



@dataclass(frozen=True)
class RansacWindowConfig:
    """Manual fit-window definition for the three ROIs."""

    starts: tuple[int, int, int]
    lengths: tuple[int, int, int]
    line_style: str = "--"
    line_color: str = "tab:grey"
    line_width: float = 1.0
    point_color: str = "r"
    label_suffix: str = ""


@dataclass(frozen=True)
class RansacTransportFigureConfig:
    """Shared parameters for RANSAC-based transport-velocity plots."""

    roi_markers: tuple[str, str, str] = ("o", "s", "^")
    roi_colors: tuple[str, str, str] = ("tab:blue", "tab:orange", "tab:green")

    ransac_min: int = 5
    ransac_max: int = 11
    ransac_thresholds: tuple[float, float, float] = (0.04, 0.05, 0.04)

    figure_size: tuple[float, float] = (5.5, 5)
    xlim_activity: tuple[float, float] = (-5, 150)
    ylim_activity: tuple[float, float] = (-0.1, 1.1)
    xlim_activity_clean: tuple[float, float] = (0, 150)

    save_folder: str = "figures/figure_9"
    file_prefix: str = "transport_velocity_ransac"


def _setup_activity_plot(config: RansacTransportFigureConfig) -> None:
    plt.figure(figsize=config.figure_size)
    plt.xlabel("Time [min]", fontsize=11)
    plt.ylabel("Activity [a.u.]", fontsize=11)


def _setup_distance_plot(config: RansacTransportFigureConfig) -> None:
    plt.figure(figsize=config.figure_size)
    plt.xlabel("Time [min]")
    plt.ylabel("X [mm]")


def _finalize_plot(
    *,
    save: bool,
    show: bool,
    save_folder: str,
    filename: str,
) -> None:
    if save:
        save_figure(save_folder, filename)

    if show and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    plt.close()


def _compute_ransac_lines(
    t: np.ndarray,
    rois: list[np.ndarray],
    config: RansacTransportFigureConfig,
) -> tuple[list[np.ndarray], np.ndarray, list[Any]]:
    """Compute RANSAC regression lines, intercepts, and inliers."""
    regression_lines: list[np.ndarray] = []
    intercepts: list[float] = []
    inliers: list[Any] = []

    for roi, threshold in zip(rois, config.ransac_thresholds):
        regression_line, intercept, inlier = get_t_from_RANSAC(
            t,
            roi,
            config.ransac_min,
            config.ransac_max,
            threshold=threshold,
        )
        regression_lines.append(regression_line)
        intercepts.append(intercept)
        inliers.append(inlier)

    return regression_lines, np.array(intercepts, dtype=float), inliers


def _compute_manual_window_intercepts(
    t: np.ndarray,
    rois: list[np.ndarray],
    window_config: RansacWindowConfig,
) -> np.ndarray:
    """Compute x-axis intercepts from manually selected linear windows."""
    I = np.ones((len(t), 1))
    M = np.array(np.append(I, t.reshape(len(t), 1), axis=1))

    mbinter = np.array([]).reshape(0, 3)

    for roi, start, length in zip(rois, window_config.starts, window_config.lengths):
        b, m = lstsq(M[start : start + length], roi[start : start + length], rcond=None)[0]
        mbinter = np.vstack((mbinter, (m, b, -b / m)))

    return mbinter


def _plot_rois(
    t: np.ndarray,
    rois: list[np.ndarray],
    config: RansacTransportFigureConfig,
) -> None:
    """Plot the three ROI series with fixed markers and colors."""
    for roi, marker, color in zip(rois, config.roi_markers, config.roi_colors):
        plt.plot(
            t,
            roi,
            marker=marker,
            linestyle="none",
            color=color,
            markerfacecolor="none",
        )


def _plot_selected_window_points(
    t: np.ndarray,
    rois: list[np.ndarray],
    window_config: RansacWindowConfig,
) -> None:
    """Highlight the manually selected points used for the local linear fit."""
    for roi, start, length in zip(rois, window_config.starts, window_config.lengths):
        plt.plot(
            t[start : start + length],
            roi[start : start + length],
            marker="o",
            linestyle="none",
            color=window_config.point_color,
        )


def _plot_manual_window_lines(
    t: np.ndarray,
    mbinter: np.ndarray,
    window_config: RansacWindowConfig,
) -> None:
    """Plot the local linear fits used to derive intercepts."""
    for m, b, _ in mbinter:
        plt.plot(
            t,
            m * t + b,
            color=window_config.line_color,
            linewidth=window_config.line_width,
            linestyle=window_config.line_style,
        )


def _plot_ransac_activity_figure(
    *,
    t: np.ndarray,
    rois: list[np.ndarray],
    regression_lines: list[np.ndarray],
    inliers: list[Any],
    config: RansacTransportFigureConfig,
    show: bool,
    save: bool,
) -> None:
    """Recreate the original RANSAC activity-time figure."""
    _setup_activity_plot(config)
    plt.hlines(0, -9, 209, linestyle=(0, (1, 1)))

    for regression_line in regression_lines:
        plt.plot(t, regression_line, color="k", linewidth=1)

    _plot_rois(t, rois, config)

    for inlier in inliers:
        plt.plot(inlier[0], inlier[1], "or")

    plt.plot(t[5 : 5 + 5], rois[0][5 : 5 + 5], "or")
    plt.plot(t[5 : 5 + 6], rois[1][5 : 5 + 6], "or")
    plt.plot(t[6 : 6 + 5], rois[2][6 : 6 + 5], "or")

    plt.xlim(config.xlim_activity_clean)
    plt.ylim(config.ylim_activity)

    _finalize_plot(
        save=save,
        show=show,
        save_folder=config.save_folder,
        filename=f"{config.file_prefix}_ransac_activity",
    )


def _plot_window_activity_figure(
    *,
    t: np.ndarray,
    rois: list[np.ndarray],
    regression_lines: list[np.ndarray],
    mbinter: np.ndarray,
    window_config: RansacWindowConfig,
    config: RansacTransportFigureConfig,
    show: bool,
    save: bool,
) -> None:
    """Plot one activity-time figure with manual offset windows."""
    _setup_activity_plot(config)
    plt.hlines(0, -9, 209, linestyle=(0, (1, 1)))

    _plot_manual_window_lines(t, mbinter, window_config)

    for regression_line in regression_lines:
        plt.plot(t, regression_line, color="k", linewidth=1)

    _plot_rois(t, rois, config)
    _plot_selected_window_points(t, rois, window_config)

    plt.xlim(config.xlim_activity)
    plt.ylim(config.ylim_activity)
    plt.ylabel("")

    plt.tick_params(
        axis="y",
        which="both",
        labelleft=False,
    )

    _finalize_plot(
        save=save,
        show=show,
        save_folder=config.save_folder,
        filename=f"{config.file_prefix}_{window_config.label_suffix}_activity",
    )


def _plot_distance_time_fit(
    intercepts: np.ndarray,
    positions: np.ndarray,
    fit: VelocityFitResult,
    *,
    config: RansacTransportFigureConfig,
    filename: str,
    show: bool,
    save: bool,
) -> None:
    """Plot the distance-time fit for one set of intercepts."""
    _setup_distance_plot(config)

    for i in range(len(fit.mbinter_v)):
        plt.plot(
            intercepts,
            fit.mbinter_v[i][0] * np.array(intercepts) + fit.mbinter_v[i][1],
            label=np.round(fit.mbinter_v[i][0], 3),
            color="k",
            linewidth=1,
        )

    plt.plot(intercepts, positions, marker="o", linestyle="None", markerfacecolor="none")
    plt.title(f"$v:$ {np.around(fit.mbinter_v[0][0], 2)}, $R^2$: {np.around(fit.r2, 4)}")

    print(fit.fit, fit.slope_std)

    _finalize_plot(
        save=save,
        show=show,
        save_folder=config.save_folder,
        filename=filename,
    )


def analyze_ransac_transport_velocity_variants(
    *,
    t: np.ndarray,
    roi1_c: np.ndarray,
    roi2_c: np.ndarray,
    roi3_c: np.ndarray,
    xi: np.ndarray | list[float],
    show: bool = True,
    save: bool = False,
    config: RansacTransportFigureConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Recreate and save the 6 RANSAC-related figures:
    1. RANSAC activity-time
    2. RANSAC distance-time
    3. Offset variant A activity-time
    4. Offset variant A distance-time
    5. Offset variant B activity-time
    6. Offset variant B distance-time
    """
    config = config or RansacTransportFigureConfig()
    rois = [roi1_c, roi2_c, roi3_c]
    positions = np.array(xi[0:3], dtype=float)

    regression_lines, ransac_intercepts, inliers = _compute_ransac_lines(t, rois, config)
    ransac_fit = fit_distance_time_line(ransac_intercepts, positions)

    _plot_ransac_activity_figure(
        t=t,
        rois=rois,
        regression_lines=regression_lines,
        inliers=inliers,
        config=config,
        show=show,
        save=save,
    )

    _plot_distance_time_fit(
        ransac_intercepts,
        positions,
        ransac_fit,
        config=config,
        filename=f"{config.file_prefix}_ransac_distance_time",
        show=show,
        save=save,
    )

    window_variants = [
        RansacWindowConfig(
            starts=(4, 4, 5),
            lengths=(5, 6, 5),
            label_suffix="offset_minus_1",
        ),
        RansacWindowConfig(
            starts=(6, 6, 7),
            lengths=(5, 6, 5),
            label_suffix="offset_plus_1",
        ),
    ]

    results: dict[str, dict[str, Any]] = {
        "ransac": {
            "intercepts": ransac_intercepts,
            "fit": ransac_fit,
            "regression_lines": regression_lines,
            "inliers": inliers,
        }
    }

    for window_config in window_variants:
        mbinter = _compute_manual_window_intercepts(t, rois, window_config)
        intercepts = np.array([x[2] for x in mbinter], dtype=float)
        fit = fit_distance_time_line(intercepts, positions)

        _plot_window_activity_figure(
            t=t,
            rois=rois,
            regression_lines=regression_lines,
            mbinter=mbinter,
            window_config=window_config,
            config=config,
            show=show,
            save=save,
        )

        _plot_distance_time_fit(
            intercepts,
            positions,
            fit,
            config=config,
            filename=f"{config.file_prefix}_{window_config.label_suffix}_distance_time",
            show=show,
            save=save,
        )

        results[window_config.label_suffix] = {
            "mbinter": mbinter,
            "intercepts": intercepts,
            "fit": fit,
        }

    return results

def main() -> None:
    """
    Run RANSAC transport velocity figures using the same data
    as the M01 noisy case from the main analysis file.
    """

    config = ArtificialDataConfig()

    positions, t, data = simulate_transport_case(
        model_name="M01",                    
        model_file="model.h5",
        xi=[0, 20, 40],
        t=np.linspace(0, 145, 30),
        params=[5.0, 100, 40, 10, 2, 0.0],
    )

    # select ROIs (same as other pipeline)
    selected_data = data[:, list(config.roi_indices)]

    # add noise (this is the "noise case")
    noisy_data = add_gaussian_noise(selected_data, config.noise_std)

    # decay correction + normalization
    data_processed = decay_correct_and_normalize(
        noisy_data, t, decay_constant=config.decay_constant,
    )

    # extract ROIs
    rois = extract_rois_from_matrix(
        data_processed,
        roi_indices=tuple(range(len(config.roi_indices))),
    )

    roi1_c, roi2_c, roi3_c = rois
    selected_positions = positions[list(config.roi_indices)]

    # --- run your new figure generator ---
    analyze_ransac_transport_velocity_variants(
        t=t,
        roi1_c=roi1_c,
        roi2_c=roi2_c,
        roi3_c=roi3_c,
        xi=selected_positions,
        show=True,
        save=True,
    )


if __name__ == "__main__":
    main()