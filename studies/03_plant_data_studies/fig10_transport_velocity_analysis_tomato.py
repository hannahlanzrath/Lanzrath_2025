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
        ransac_min=(5, 5, 7),
        ransac_max=(12, 12, 11),
        ransac_thresholds=(0.04, 0.04, 0.06),
        spline_points=400,
        spline_eval_end=150.0,
        spline_plot_end=150.0,
        buehler_fit_dt=0.5,
        buehler_gauss_type="stretched",
    )

    case = ExperimentalPlantCase(
        name="Tomato",
        csv_path="Tomato.csv",
        delimiter="\t",
        model_name="M02",
        model_file="tomato.h5",
        xi_indices=(0, 1, 2),
        data_columns=(1, 2, 3),
        buehler_p0=[
            6.23380315e+00,
            7.32754103e+01,
            4.14619520e+01,
            8.42817545e-02,
            1.27408756e+00,
            1.18790019e-02,
        ],
    )

    save_folder = "figures/figure_10"
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
