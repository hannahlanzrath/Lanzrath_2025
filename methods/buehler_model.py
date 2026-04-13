from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Lanzrath_2025.tools.simulation import CadetContainer


@dataclass
class BuehlerModelResult:
    params_opt: np.ndarray
    simulated: np.ndarray
    measured: np.ndarray
    reference_times: np.ndarray
    residual_sum_squares: float
    fit_dataframe: pd.DataFrame


def _as_time_major_matrix(rois: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Convert ROI input to shape (n_time, n_rois)."""
    if isinstance(rois, list):
        matrix = np.column_stack([np.asarray(r, dtype=float) for r in rois])
    else:
        matrix = np.asarray(rois, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("ROI data must be a 2D array.")

    return matrix


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize each ROI column to its own maximum."""
    matrix = np.asarray(matrix, dtype=float)
    maxima = np.max(matrix, axis=0)
    maxima[maxima == 0.0] = 1.0
    return matrix / maxima


def _ensure_time_major(simulated: np.ndarray, n_time: int, n_rois: int) -> np.ndarray:
    """Ensure simulation output has shape (n_time, n_rois)."""
    simulated = np.asarray(simulated, dtype=float)

    if simulated.shape == (n_time, n_rois):
        return simulated

    if simulated.shape == (n_rois, n_time):
        return simulated.T

    raise ValueError(
        f"Unexpected simulation shape {simulated.shape}. "
        f"Expected ({n_time}, {n_rois}) or ({n_rois}, {n_time})."
    )


def _build_fit_dataframe(
    t: np.ndarray,
    measured: np.ndarray,
    positions: np.ndarray,
) -> pd.DataFrame:
    """
    Recreate the notebook input format:
    first column = time
    following columns = ROI signals named by position strings.
    """
    data = {"Unnamed: 0": t}
    for i, pos in enumerate(positions):
        data[str(pos)] = measured[:, i]
    return pd.DataFrame(data)


def fit_buehler_model(
    t: np.ndarray,
    rois: list[np.ndarray] | np.ndarray,
    positions: np.ndarray,
    *,
    model_name: str = "M02",
    model_file: str = "model.h5",
    p0: list[float] | np.ndarray | None = None,
    fit_dt: float = 0.000567 * 60,
    gauss_type: str = "stretched",
    normalize_measured: bool = False,
    optimize: bool = False,
) -> BuehlerModelResult:
    """Simulate the Bühler / CADET model with p0 and extract peak reference times.

    By default the model is only simulated with the provided ``p0`` parameters.
    Set ``optimize=True`` to run the least-squares optimisation first and use
    the fitted parameters instead.
    """
    t = np.asarray(t, dtype=float)
    positions = np.asarray(positions, dtype=float)
    measured = _as_time_major_matrix(rois)

    if measured.shape[0] != len(t):
        raise ValueError(
            f"Time length {len(t)} does not match ROI matrix shape {measured.shape}."
        )

    if measured.shape[1] != len(positions):
        raise ValueError(
            f"Number of ROI columns {measured.shape[1]} does not match "
            f"number of positions {len(positions)}."
        )

    if normalize_measured:
        measured = _normalize_columns(measured)

    fit_df = _build_fit_dataframe(t, measured, positions)

    p0 = np.asarray(p0, dtype=float)

    simulation = CadetContainer(
        model_name,
        model_file,
        fit_df,
        fit_dt,
        gauss_type=gauss_type,
    )

    if optimize:
        _, params_opt, _ = simulation.fit_p_LS(p0, live_plot=True, live_plot_every=1)
        params_opt = np.asarray(params_opt, dtype=float)
    else:
        params_opt = p0

    simulated = np.asarray(simulation.simulate(params_opt), dtype=float)
    simulated = _ensure_time_major(simulated, n_time=len(t), n_rois=len(positions))

    reference_times = t[np.argmax(simulated, axis=0)].astype(float)
    rss = float(np.sum((measured - simulated) ** 2))

    return BuehlerModelResult(
        params_opt=params_opt,
        simulated=simulated,
        measured=measured,
        reference_times=reference_times,
        residual_sum_squares=rss,
        fit_dataframe=fit_df,
    )
