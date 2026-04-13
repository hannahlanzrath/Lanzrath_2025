from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Lanzrath_2025.tools.simulation import CadetContainer
from Lanzrath_2025.tools.utils import save_figure

from Lanzrath_2025.methods.halfmax_intercept import compute_halfmax_intercept
from Lanzrath_2025.methods.halfmax_spline import compute_halfmax_spline
from Lanzrath_2025.methods.intercept import compute_intercept
from Lanzrath_2025.methods.halfmax_logistic import compute_logistic
from Lanzrath_2025.methods.mckay import compute_mckay
from Lanzrath_2025.tools.preprocessing import DEFAULT_DECAY_CONSTANT, prepare_signal


DEFAULT_XLIM = (-9.0, 150.0)
DEFAULT_HLINE_X = (-9.0, 209.0)


@dataclass(frozen=True)
class AnalysisConfig:
    """Shared parameters for reference-time analysis."""

    roi_index: int = 0
    decay_constant: float = DEFAULT_DECAY_CONSTANT
    line_start_index: int = 3
    npoints: int = 3
    spline_points: int = 500
    spline_end: float = 200.0
    logistic_initial_guesses: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)
    logistic_bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (0.0, 0.0, -np.inf, -np.inf),
        (np.inf, np.inf, np.inf, np.inf),
    )
    mckay_dt: float = 5.0
    mckay_end: float = 145.0
    mckay_points: int = 30


@dataclass(frozen=True)
class SimulationCase:
    """Single model-method combination."""

    model_key: str
    model_name: str
    params: list[float]
    method: str


def _setup_reference_plot(
    t: np.ndarray,
    y: np.ndarray,
    xlabel: str = "Time [min]",
    ylabel: str = "Activity [a.u.]",
) -> None:
    """Create the standard reference-time plot."""
    plt.figure(figsize=(5.5, 5))
    plt.plot(
        t,
        y,
        marker="o",
        linestyle="none",
        color="darkorange",
        markerfacecolor="none",
        label="Decay corrected data",
    )
    plt.xlim(DEFAULT_XLIM)
    plt.ylim([-0.1, 1.1])
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)


def _plot_horizontal_levels(*levels: tuple[float, dict[str, Any]]) -> None:
    """Plot horizontal guide lines."""
    for y, kwargs in levels:
        plt.hlines(y, *DEFAULT_HLINE_X, **kwargs)


def _plot_fitted_line(t: np.ndarray, slope: float, intercept: float) -> None:
    """Plot a fitted straight line."""
    plt.plot(t, slope * t + intercept, "k")


def _plot_method_result(method: str, t: np.ndarray, y: np.ndarray, result: dict[str, Any]) -> None:
    """Plot method-specific overlays."""
    if method == "intercept":
        _plot_horizontal_levels(
            (0.0, {"linestyle": (0, (1, 1))}),
            (0.045, {"colors": "C2", "linestyle": "--"}),
        )
        _plot_fitted_line(t, result["slope"], result["intercept"])

    elif method == "halfmax_intercept":
        _plot_horizontal_levels(
            (1.0, {"linestyle": (0, (1, 1))}),
            (0.5, {"colors": "C2", "linestyle": "--"}),
        )
        _plot_fitted_line(t, result["slope"], result["intercept"])

    elif method == "halfmax_spline":
        plt.plot(result["t_dense"], result["y_dense"], "k")
        _plot_horizontal_levels(
            (result["ymax"], {"linestyle": (0, (1, 1))}),
            (result["halfmax"], {"colors": "C2", "linestyle": "--"}),
        )

    elif method in {"logistic", "logistic_M01", "logistic_M02"}:
        if "y_fit" in result:
            plt.plot(t, result["y_fit"], "k", label="Fitted curve")
            plt.hlines(
                result["halfmax"],
                t.min(),
                t.max(),
                colors="C2",
                linestyle="--",
            )

    elif method == "mckay":
        plt.plot(
            result["T"],
            result["M"],
            label="Data",
            color="darkorange",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
        )
        plt.plot(result["T"], result["G"], label="Unbound activity", linestyle=(0, (1, 1)))
        plt.plot(result["T"], result["B"], label="Bound activity", color="C2", linestyle="dashed")
        plt.vlines(result["t_ref"], 0, np.max(result["G"]), "k")
        plt.legend()


def _compute_intercept_method(
    t: np.ndarray,
    y: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute intercept-based reference time."""
    return compute_intercept(
        t,
        y,
        line_start_index=config.line_start_index,
        npoints=config.npoints,
    )


def _compute_halfmax_intercept_method(
    t: np.ndarray,
    y: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute half-maximum intercept reference time."""
    return compute_halfmax_intercept(
        t,
        y,
        line_start_index=config.line_start_index,
        npoints=config.npoints,
    )


def _compute_halfmax_spline_method(
    t: np.ndarray,
    y: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute spline half-maximum reference time."""
    return compute_halfmax_spline(
        t,
        y,
        spline_points=config.spline_points,
        spline_end=config.spline_end,
    )


def _compute_logistic_method(
    t: np.ndarray,
    y: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute logistic-fit reference time."""
    return compute_logistic(
        t,
        y,
        initial_guesses=config.logistic_initial_guesses,
        bounds=config.logistic_bounds,
    )


def _compute_mckay_method(
    t: np.ndarray,
    y: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute McKay reference time."""
    return compute_mckay(
        t,
        y,
        mckay_dt=config.mckay_dt,
        mckay_end=config.mckay_end,
        mckay_points=config.mckay_points,
    )


METHODS: dict[str, Callable[[np.ndarray, np.ndarray, AnalysisConfig], dict[str, Any]]] = {
    "intercept": _compute_intercept_method,
    "halfmax_intercept": _compute_halfmax_intercept_method,
    "halfmax_spline": _compute_halfmax_spline_method,
    "logistic": _compute_logistic_method,
    "logistic_M01": _compute_logistic_method,
    "logistic_M02": _compute_logistic_method,
    "mckay": _compute_mckay_method,
}


def reference_time_analysis(
    t: np.ndarray,
    data: np.ndarray,
    roi_index: int = 0,
    method: str = "intercept",
    decay_constant: float = DEFAULT_DECAY_CONSTANT,
    line_start_index: int = 3,
    npoints: int = 3,
    spline_points: int = 500,
    spline_end: float = 200.0,
    logistic_initial_guesses: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0),
    logistic_bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (0.0, 0.0, -np.inf, -np.inf),
        (np.inf, np.inf, np.inf, np.inf),
    ),
    mckay_dt: float = 5.0,
    mckay_end: float = 145.0,
    mckay_points: int = 30,
    show: bool = True,
) -> dict[str, Any]:
    """Run one reference-time method on one signal.

    Args:
        roi_index: Column index of the ROI in `data`.
        method: Name of estimator.
        show: Display plot.

    Returns:
        Result dictionary with method-specific outputs.
    """
    config = AnalysisConfig(
        roi_index=roi_index,
        decay_constant=decay_constant,
        line_start_index=line_start_index,
        npoints=npoints,
        spline_points=spline_points,
        spline_end=spline_end,
        logistic_initial_guesses=logistic_initial_guesses,
        logistic_bounds=logistic_bounds,
        mckay_dt=mckay_dt,
        mckay_end=mckay_end,
        mckay_points=mckay_points,
    )
    y = prepare_signal(t, data, config.roi_index, config.decay_constant)

    try:
        method_fn = METHODS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown method: {method}") from exc

    result = {"method": method, "t": t, "y": y}
    result.update(method_fn(t, y, config))

    if show:
        _setup_reference_plot(t, y)
        _plot_method_result(method, t, y, result)
        if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
            plt.show()
        plt.close()

    return result


def get_default_cases() -> list[SimulationCase]:
    """Return the default simulation cases."""
    model_configs = {
        "M01": ("M01", [5.0, 20, 30, 8, 3]),
        "M02": ("M02", [5.0, 20, 30, 5, 2, 0.06]),
        "M13": ("M13", [5.0, 20, 30, 5, 2, 0.5, 0.2, 0.01]),
    }

    analysis_plan = [
        ("M01", "intercept"),
        ("M01", "halfmax_intercept"),
        ("M01", "halfmax_spline"),
        ("M01", "mckay"),
        ("M01", "logistic_M01"),
        ("M02", "logistic_M02"),
    ]

    return [
        SimulationCase(
            model_key=model_key,
            model_name=model_configs[model_key][0],
            params=model_configs[model_key][1],
            method=method,
        )
        for model_key, method in analysis_plan
    ]


def simulate_case(
    case: SimulationCase,
    t: np.ndarray,
    xi: list[float],
    model_file: str,
) -> np.ndarray:
    """Simulate one model case."""
    sim = CadetContainer(
        case.model_name,
        model_file,
        xi=xi,
        t=t,
        gauss_type="stretched",
    )
    return sim.simulate(case.params)


def run_reference_time_examples(
    steps: int = 30,
    xi: list[float] | None = None,
    t_start: float = 0.0,
    t_end: float = 145.0,
    model_file: str = "model.h5",
    save: bool = False,
    save_folder: str = "figures/figure_3",
) -> dict[str, dict[str, Any]]:
    """Run the default reference-time analyses.

    Args:
        steps: Number of simulation time points.
        xi: Spatial positions used in simulation.
        save: Save generated figures.

    Returns:
        Nested results grouped by model and method.
    """
    xi = xi or [0, 20, 40, 60, 80]
    t = np.linspace(t_start, t_end, steps)

    results: dict[str, dict[str, Any]] = {}

    for case in get_default_cases():
        data = simulate_case(case, t=t, xi=xi, model_file=model_file)
        result = reference_time_analysis(t, data, method=case.method, show=False)

        _setup_reference_plot(result["t"], result["y"])
        _plot_method_result(case.method, result["t"], result["y"], result)
        plt.title(f"{case.model_key}: {case.method}")

        if save:
            save_figure(save_folder, f"fig3_data_driven_methods_{case.model_key}_{case.method}")

        if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
            plt.show()
        plt.close()

        results.setdefault(case.model_key, {})
        results[case.model_key][case.method] = result

    return results


def main() -> None:
    """Run the reference-time analysis examples."""
    run_reference_time_examples(save=True)


if __name__ == "__main__":
    main()