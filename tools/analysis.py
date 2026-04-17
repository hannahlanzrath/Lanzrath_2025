"""Shared analysis pipeline: reference-time methods and plot helpers.

Used by both in-silico (fig8) and plant-data (fig10-12) studies.
Callers supply a ``save_fn(folder, filename)`` callable so each study can
write figures to its own preferred location.
"""
from __future__ import annotations

from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Lanzrath_2025.methods.buehler_model import fit_buehler_model
from Lanzrath_2025.methods.halfmax_spline import compute_halfmax_spline
from Lanzrath_2025.methods.intercept_ransac import get_t_from_RANSAC
from Lanzrath_2025.methods.mckay import compute_mckay
from Lanzrath_2025.tools.fit import VelocityFitResult, fit_distance_time_line


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _setup_activity_plot() -> None:
    """Create a new figure with standard activity-plot axis labels."""
    plt.figure(figsize=(5.5, 5))
    plt.xlabel("Time [min]", fontsize=11)
    plt.ylabel("Activity [a.u.]", fontsize=11)


def _setup_distance_time_plot() -> None:
    """Create a new figure with standard distance-time axis labels."""
    plt.figure(figsize=(5.5, 5))
    plt.xlabel("Time [min]")
    plt.ylabel("X [mm]")


def _plot_roi_points(t: np.ndarray, rois: list[np.ndarray]) -> None:
    """Plot ROI data points onto the current axes with fixed marker styles.

    Args:
        t: Time vector.
        rois: List of ROI signal arrays, one per position.
    """
    markers = ["o", "s", "^"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for roi, marker, color in zip(rois, markers, colors):
        plt.plot(
            t, roi,
            marker=marker,
            linestyle="none",
            color=color,
            markerfacecolor="none",
        )


def _plot_distance_time_fit(
    times: np.ndarray,
    positions: np.ndarray,
    fit: VelocityFitResult,
) -> None:
    """Overlay the fitted distance-time line and data points on the current axes.

    Args:
        times: Reference times [min] for each ROI.
        positions: Spatial positions [mm] for each ROI.
        fit: Velocity fit result containing slope/intercept and R².
    """
    for i in range(len(fit.mbinter_v)):
        slope, intercept = fit.mbinter_v[i][0], fit.mbinter_v[i][1]
        plt.plot(
            times,
            slope * np.array(times) + intercept,
            label=np.round(slope, 3),
            color="k",
            linewidth=1,
        )
    plt.plot(times, positions, marker="o", linestyle="None", markerfacecolor="none")
    plt.title(f"$v:$ {np.around(fit.mbinter_v[0][0], 2)}, $R^2$: {np.around(fit.r2, 4)}")


def _finalize_plot(
    save: bool,
    save_folder: str,
    filename: str,
    show: bool,
    save_fn: Callable[[str, str], None] | None = None,
) -> None:
    """Save and/or display the current figure, then close it.

    Args:
        save: Whether to save the figure.
        save_folder: Destination folder passed to ``save_fn``.
        filename: File stem passed to ``save_fn``.
        show: Whether to display the figure interactively.
            Silently skipped when the backend is non-interactive (e.g. Agg).
        save_fn: Callable ``(folder, filename) -> None`` that performs the save.
            If ``None``, saving is skipped even when ``save=True``.
    """
    if save and save_fn is not None:
        save_fn(save_folder, filename)
    if show and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    plt.close()


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_ransac_reference_times(
    t: np.ndarray,
    rois: list[np.ndarray],
    positions: np.ndarray,
    config: Any,
    show: bool = True,
    save: bool = False,
    save_folder: str = "figures",
    file_prefix: str = "analysis",
    save_fn: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Estimate reference times by RANSAC intercept and fit a velocity line.

    For each ROI, a line is fitted to the rising portion of the signal using RANSAC.
    The x-intercept of each line gives a reference time. A distance-time line is then
    fitted across all reference times to extract transport velocity.

    Args:
        t: Time vector.
        rois: List of ROI signal arrays (decay-corrected and normalised).
        positions: Spatial positions [mm] of each ROI.
        config: Configuration object with ``ransac_min``, ``ransac_max``, and
            ``ransac_thresholds`` attributes.
        show: Display figures interactively.
        save: Save figures to disk.
        save_folder: Output folder.
        file_prefix: Filename prefix for saved figures.
        save_fn: Callable ``(folder, filename) -> None`` used for saving.

    Returns:
        Dict with keys ``method``, ``times``, ``positions``, ``regression_lines``,
        ``inliers``, and ``fit``.
    """
    regression_lines: list[np.ndarray] = []
    intercepts: list[float] = []
    inliers: list[Any] = []

    ransac_mins = config.ransac_min if isinstance(config.ransac_min, tuple) else (config.ransac_min,) * len(rois)
    ransac_maxs = config.ransac_max if isinstance(config.ransac_max, tuple) else (config.ransac_max,) * len(rois)

    for roi, threshold, r_min, r_max in zip(rois, config.ransac_thresholds, ransac_mins, ransac_maxs):
        regression_line, intercept, inlier = get_t_from_RANSAC(
            t, roi, r_min, r_max, threshold=threshold,
        )
        regression_lines.append(regression_line)
        intercepts.append(intercept)
        inliers.append(inlier)

    times = np.array(intercepts, dtype=float)
    fit = fit_distance_time_line(times, positions)

    _setup_activity_plot()
    plt.hlines(0, -9, 209, linestyle=(0, (1, 1)))
    for regression_line in regression_lines:
        plt.plot(t, regression_line, color="k", linewidth=1)
    _plot_roi_points(t, rois)
    plt.xlim([-5, 150])
    plt.ylim([-0.1, 1.1])
    _finalize_plot(save, save_folder, f"{file_prefix}_ransac_activity", show, save_fn)

    _setup_distance_time_plot()
    _plot_distance_time_fit(times, positions, fit)
    _finalize_plot(save, save_folder, f"{file_prefix}_ransac_distance_time", show, save_fn)

    return {
        "method": "ransac_intercept",
        "times": times,
        "positions": positions,
        "regression_lines": regression_lines,
        "inliers": inliers,
        "fit": fit,
    }


def analyze_halfmax_spline_reference_times(
    t: np.ndarray,
    rois: list[np.ndarray],
    positions: np.ndarray,
    config: Any,
    show: bool = True,
    save: bool = False,
    save_folder: str = "figures",
    file_prefix: str = "analysis",
    save_fn: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Estimate reference times via spline half-maximum and fit a velocity line.

    A cubic spline is fitted to each ROI and the time at which the spline first
    reaches half of its maximum value is used as the reference time.

    Args:
        t: Time vector.
        rois: List of ROI signal arrays (decay-corrected and normalised).
        positions: Spatial positions [mm] of each ROI.
        config: Configuration object with ``spline_points``, ``spline_eval_end``,
            and ``spline_plot_end`` attributes.
        show: Display figures interactively.
        save: Save figures to disk.
        save_folder: Output folder.
        file_prefix: Filename prefix for saved figures.
        save_fn: Callable ``(folder, filename) -> None`` used for saving.

    Returns:
        Dict with keys ``method``, ``times``, ``positions``, ``results``, and ``fit``.
    """
    results = [
        compute_halfmax_spline(
            t, roi,
            spline_points=config.spline_points,
            spline_end=config.spline_eval_end,
        )
        for roi in rois
    ]
    times = np.array([r["t_ref"] for r in results], dtype=float)
    fit = fit_distance_time_line(times, positions)

    _setup_activity_plot()
    for r in results:
        plt.hlines(r["ymax"], -9, 209, linestyle=(0, (1, 1)))
        plt.hlines(r["halfmax"], -9, 209, colors="C2", linestyle="--")
    for r in results:
        mask = r["t_dense"] <= config.spline_plot_end
        plt.plot(r["t_dense"][mask], r["y_dense"][mask], "k", linewidth=1)
    _plot_roi_points(t, rois)
    plt.ylim([-0.001, 1.1])
    plt.xlim([-9, 150])
    _finalize_plot(save, save_folder, f"{file_prefix}_halfmax_spline_activity", show, save_fn)

    _setup_distance_time_plot()
    _plot_distance_time_fit(times, positions, fit)
    _finalize_plot(save, save_folder, f"{file_prefix}_halfmax_spline_distance_time", show, save_fn)

    return {
        "method": "halfmax_spline",
        "times": times,
        "positions": positions,
        "results": results,
        "fit": fit,
    }


def analyze_mckay_reference_times(
    t: np.ndarray,
    rois: list[np.ndarray],
    positions: np.ndarray,
    config: Any,
    show: bool = True,
    save: bool = False,
    save_folder: str = "figures",
    file_prefix: str = "analysis",
    save_fn: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Estimate reference times via the McKay binding model and fit a velocity line.

    The McKay model separates each ROI signal into bound and unbound fractions.
    The centroid of the unbound fraction gives the reference time.

    Args:
        t: Time vector.
        rois: List of ROI signal arrays (decay-corrected and normalised).
        positions: Spatial positions [mm] of each ROI.
        config: Configuration object with optional ``mckay_dt``, ``mckay_end``,
            and ``mckay_points`` attributes. Any ``None`` value is derived from ``t``.
        show: Display figures interactively.
        save: Save figures to disk.
        save_folder: Output folder.
        file_prefix: Filename prefix for saved figures.
        save_fn: Callable ``(folder, filename) -> None`` used for saving.

    Returns:
        Dict with keys ``method``, ``times``, ``positions``, ``results``, and ``fit``.
    """
    mckay_dt = config.mckay_dt if config.mckay_dt is not None else float(t[1] - t[0])
    mckay_end = config.mckay_end if config.mckay_end is not None else float(t[-1])
    mckay_points = config.mckay_points if config.mckay_points is not None else len(t)

    results = [
        compute_mckay(
            t, roi,
            mckay_dt=mckay_dt,
            mckay_end=mckay_end,
            mckay_points=mckay_points,
        )
        for roi in rois
    ]

    valid = [(pos, r) for pos, r in zip(positions, results) if "t_ref" in r]
    valid_positions = np.array([pos for pos, _ in valid], dtype=float)
    valid_results = [r for _, r in valid]
    times = np.array([r["t_ref"] for r in valid_results], dtype=float)
    fit = fit_distance_time_line(times, valid_positions)

    if valid_results:
        first = valid_results[0]
        _setup_activity_plot()
        plt.plot(
            first["T"], first["M"],
            label="Data", color="darkorange",
            marker="o", markerfacecolor="none", linestyle="none",
        )
        plt.plot(first["T"], first["G"], label="Unbound activity", linestyle=(0, (1, 1)))
        plt.plot(first["T"], first["B"], label="Bound activity", color="C2", linestyle="dashed")
        plt.vlines(first["t_ref"], 0, np.max(first["G"]), "k")
        plt.legend()
        _finalize_plot(save, save_folder, f"{file_prefix}_mckay_activity", show, save_fn)

        _setup_distance_time_plot()
        _plot_distance_time_fit(times, valid_positions, fit)
        _finalize_plot(save, save_folder, f"{file_prefix}_mckay_distance_time", show, save_fn)

    return {
        "method": "mckay",
        "times": times,
        "positions": valid_positions,
        "results": results,
        "fit": fit,
    }


def analyze_buehler_transport_velocity(
    t: np.ndarray,
    rois: list[np.ndarray],
    positions: np.ndarray,
    config: Any,
    model_name: str,
    model_file: str,
    p0: list[float],
    show: bool = True,
    save: bool = False,
    save_folder: str = "figures",
    file_prefix: str = "analysis",
    save_fn: Callable[[str, str], None] | None = None,
    optimize: bool = False,
) -> dict[str, Any]:
    """Simulate the Bühler model and extract transport velocity from parameters.

    The velocity is the first element of the parameter vector. No distance-time
    fit is needed — the model gives velocity directly.

    Args:
        t: Time vector.
        rois: List of raw (not decay-corrected), globally normalised ROI arrays.
        positions: Spatial positions [mm] of each ROI.
        config: Configuration object with ``buehler_decay`` and
            ``buehler_gauss_type`` attributes.
        model_name: CADET model identifier (e.g. ``"M02"``).
        model_file: Path to the CADET model ``.h5`` file.
        p0: Initial parameter vector for the Bühler model.
        show: Display figures interactively.
        save: Save figures to disk.
        save_folder: Output folder.
        file_prefix: Filename prefix for saved figures.
        save_fn: Callable ``(folder, filename) -> None`` used for saving.
        optimize: If ``True``, run least-squares fitting before extracting velocity.

    Returns:
        Dict with keys ``method``, ``velocity``, ``params_opt``, ``simulated``,
        ``measured``, and ``residual_sum_squares``.
    """
    result = fit_buehler_model(
        t, rois, positions,
        model_name=model_name,
        model_file=model_file,
        p0=p0,
        decay=config.buehler_decay,
        gauss_type=config.buehler_gauss_type,
        normalize_measured=False,
        optimize=optimize,
    )

    velocity = float(result.params_opt[0])

    _setup_activity_plot()
    for i in range(result.simulated_hires.shape[1]):
        label = "Bühler model" if i == 0 else None
        plt.plot(result.t_hires, result.simulated_hires[:, i], color="k", linewidth=1, label=label)
    _plot_roi_points(t, rois)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlim([-5, 150])
    plt.ylim([-0.1, 1.1])
    plt.title(f"Bühler model, $v$ = {velocity:.3f} mm/min")
    _finalize_plot(save, save_folder, f"{file_prefix}_buehler_activity", show, save_fn)

    return {
        "method": "buehler_model",
        "velocity": velocity,
        "params_opt": result.params_opt,
        "simulated": result.simulated,
        "measured": result.measured,
        "residual_sum_squares": result.residual_sum_squares,
    }
