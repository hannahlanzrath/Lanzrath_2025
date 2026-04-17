from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Lanzrath_2025.tools.config import PlantDataConfig
from Lanzrath_2025.tools.dataimport import ExperimentalPlantCase
from Lanzrath_2025.tools.reporting import run_plant_velocity_analysis, summarize_velocity_results, print_summary_table
from Lanzrath_2025.tools.utils import ensure_dir

BASE_DIR = Path(__file__).resolve().parent

np.random.seed(123)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true",
                        help="Run least-squares fitting before extracting velocity")
    args = parser.parse_args()

    config = PlantDataConfig(
        decay_constant=0.000567 * 60.0,
        ransac_min=3,
        ransac_max=12,
        ransac_thresholds=(0.15, 0.15, 0.15),
        spline_points=400,
        spline_eval_end=150.0,
        spline_plot_end=150.0,
        buehler_decay=0.5,
        buehler_gauss_type="stretched",
    )

    case = ExperimentalPlantCase(
        name="Barley",
        csv_path="Barley.csv",
        delimiter=";",
        model_name="M02",
        model_file="barley.h5",
        xi_indices=(5, 6, 7),
        data_columns=(6, 7, 8),
        buehler_p0=[
            34.18303274,
            82.84921912,
            27.60535732,
            3.48562844,
            2.08851686,
            0.29770133,
        ],
    )

    save_folder = "figures/figure_11"
    print(f"\nRunning analysis for {case.name} ...")
    results = run_plant_velocity_analysis(
        case=case,
        config=config,
        base_dir=BASE_DIR,
        show=True,
        save=True,
        save_folder=save_folder,
        optimize=args.optimize,
    )

    params_opt = results["buehler_model"]["params_opt"]
    print(f"\nBühler model optimized parameters: {np.array2string(params_opt, precision=6, separator=', ')}")

    summary = summarize_velocity_results(case.name, results)
    print_summary_table(summary)
    ensure_dir(save_folder)
    summary.to_csv(f"{save_folder}/velocity_summary.csv", index=False)
    print(f"\nSaved summary to {save_folder}/velocity_summary.csv")


if __name__ == "__main__":
    main()
