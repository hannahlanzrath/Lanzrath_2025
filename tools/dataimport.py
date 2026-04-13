from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Lanzrath_2025.tools.preprocessing import decay_correct_and_normalize
from Lanzrath_2025.tools.utils import infer_xi_from_dataframe


@dataclass
class ExperimentalPlantCase:
    name: str
    csv_path: str
    delimiter: str
    model_name: str
    model_file: str

    # xi position indices taken from the parsed xi vector
    xi_indices: tuple[int, int, int]

    # actual data columns in the CSV numeric matrix
    data_columns: tuple[int, int, int]

    # Bühler model start parameters
    buehler_p0: list[float]

    # optional override if automatic xi parsing is not correct
    positions_override: tuple[float, float, float] | None = None

    time_column: int = 0


def load_experimental_case(
    case: ExperimentalPlantCase,
    decay_constant: float,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one experimental plant case from CSV.

    Args:
        case: Experiment specification (CSV path, columns, model, initial parameters).
        decay_constant: Exponential decay constant [1/min].
        base_dir: Directory that contains the CSV file. Defaults to the current
            working directory.

    Returns:
        A tuple of ``(positions, t, data_raw_norm, data_decay_corr_norm)`` where:
            - ``positions``: ROI positions in mm, shape (n_rois,).
            - ``t``: Time vector in minutes, shape (n_time,).
            - ``data_raw_norm``: Globally normalised raw ROI matrix, shape (n_time, n_rois).
            - ``data_decay_corr_norm``: Decay-corrected and normalised ROI matrix,
              shape (n_time, n_rois).

    Raises:
        ValueError: If xi positions cannot be inferred from column names and
            ``positions_override`` is not provided.
    """
    base_dir = base_dir or Path.cwd()
    df = pd.read_csv(base_dir / case.csv_path, delimiter=case.delimiter)
    all_data = df.to_numpy()

    t = all_data[:, case.time_column].astype(float)
    selected_raw = all_data[:, list(case.data_columns)].astype(float)

    raw_max = np.nanmax(selected_raw)
    if not np.isfinite(raw_max) or raw_max == 0:
        raw_max = 1.0
    data_raw_norm = selected_raw / raw_max

    data_decay_corr_norm = decay_correct_and_normalize(
        selected_raw, t, decay_constant=decay_constant,
    )

    if case.positions_override is not None:
        positions = np.array(case.positions_override, dtype=float)
    else:
        xi = infer_xi_from_dataframe(df)
        if len(xi) == 0:
            raise ValueError(
                f"Could not infer xi positions from column names in {case.csv_path}. "
                "Please provide positions_override."
            )
        positions = xi[list(case.xi_indices)]

    return positions, t, data_raw_norm, data_decay_corr_norm
