"""Reporting utilities for plant-data velocity studies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Lanzrath_2025.tools.analysis import (
    analyze_buehler_transport_velocity,
    analyze_halfmax_spline_reference_times,
    analyze_mckay_reference_times,
    analyze_ransac_reference_times,
)
from Lanzrath_2025.tools.config import PlantDataConfig
from Lanzrath_2025.tools.dataimport import ExperimentalPlantCase, load_experimental_case
from Lanzrath_2025.tools.preprocessing import extract_rois_from_matrix
from Lanzrath_2025.tools.utils import sanitize_name, save_figure_local


def run_plant_velocity_analysis(
    case: ExperimentalPlantCase,
    config: PlantDataConfig,
    base_dir: Path,
    show: bool = True,
    save: bool = True,
    save_folder: str = "figures",
    optimize: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the full velocity analysis pipeline for one plant experiment.

    Loads the experimental data, applies decay correction, and runs all four
    reference-time methods (RANSAC, half-maximum spline, McKay, Bühler model).

    Args:
        case: Experiment specification (CSV path, columns, model, initial parameters).
        config: Analysis configuration (RANSAC bounds, spline settings, etc.).
        base_dir: Directory containing the experiment's CSV file.
        show: Display figures interactively.
        save: Save figures to disk.
        save_folder: Output directory for saved figures.
        optimize: If ``True``, run least-squares fitting before extracting the
            Bühler model velocity.

    Returns:
        Dict mapping method name to its result dict.
        Keys: ``"ransac_intercept"``, ``"halfmax_spline"``, ``"mckay"``,
        ``"buehler_model"``.
    """
    positions, t, data_raw_norm, data_decay_corr_norm = load_experimental_case(
        case, config.decay_constant, base_dir=base_dir
    )

    rois_raw = extract_rois_from_matrix(data_raw_norm, tuple(range(data_raw_norm.shape[1])))
    rois_decay_corrected = extract_rois_from_matrix(
        data_decay_corr_norm, tuple(range(data_decay_corr_norm.shape[1]))
    )

    prefix = sanitize_name(case.name)

    return {
        "ransac_intercept": analyze_ransac_reference_times(
            t, rois_decay_corrected, positions, config,
            show=show, save=save, save_folder=save_folder, file_prefix=prefix,
            save_fn=save_figure_local,
        ),
        "halfmax_spline": analyze_halfmax_spline_reference_times(
            t, rois_decay_corrected, positions, config,
            show=show, save=save, save_folder=save_folder, file_prefix=prefix,
            save_fn=save_figure_local,
        ),
        "mckay": analyze_mckay_reference_times(
            t, rois_decay_corrected, positions, config,
            show=show, save=save, save_folder=save_folder, file_prefix=prefix,
            save_fn=save_figure_local,
        ),
        "buehler_model": analyze_buehler_transport_velocity(
            t, rois_raw, positions, config,
            model_name=case.model_name,
            model_file=case.model_file,
            p0=case.buehler_p0,
            show=show, save=save, save_folder=save_folder, file_prefix=prefix,
            save_fn=save_figure_local,
            optimize=optimize,
        ),
    }


def summarize_velocity_results(
    plant_name: str,
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a summary DataFrame from velocity analysis results.

    Args:
        plant_name: Name of the plant experiment (used as the ``plant`` column).
        results: Output of :func:`run_plant_velocity_analysis`.

    Returns:
        DataFrame with one row per method and columns:
        ``plant``, ``method``, ``velocity_mm_per_min``, ``intercept_mm``,
        ``r2``, ``slope_std``.
    """
    rows = []
    for method, result in results.items():
        if "fit" in result:
            fit = result["fit"]
            rows.append({
                "plant": plant_name,
                "method": method,
                "velocity_mm_per_min": fit.mbinter_v[0][0],
                "intercept_mm": fit.mbinter_v[0][1],
                "r2": fit.r2,
                "slope_std": fit.slope_std,
            })
        else:
            rows.append({
                "plant": plant_name,
                "method": method,
                "velocity_mm_per_min": result["velocity"],
                "intercept_mm": float("nan"),
                "r2": float("nan"),
                "slope_std": float("nan"),
            })
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a velocity summary DataFrame to stdout.

    Args:
        df: Summary DataFrame as returned by :func:`summarize_velocity_results`.
    """
    print()
    print(df.to_string(index=False))
