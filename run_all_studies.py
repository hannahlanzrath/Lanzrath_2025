"""Run all study scripts in sequence."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no windows, no blocking

STUDIES_ROOT = Path(__file__).resolve().parent / "studies"

SCRIPTS = [
    # (group label, relative path inside studies/)
    ("01 Illustrative examples",   "01_illustrative_examples/fig2_decay_correction.py"),
    ("01 Illustrative examples",   "01_illustrative_examples/fig5_spatial_temporal_distribution.py"),
    ("01 Illustrative examples",   "01_illustrative_examples/fig7_transport_types.py"),
    ("02 Artificial data studies", "02_artificial_data_studies/fig3_data_driven_methods.py"),
    ("02 Artificial data studies", "02_artificial_data_studies/fig8_transport_velocity_analysis_in_silico.py"),
    ("02 Artificial data studies", "02_artificial_data_studies/fig9_intercept_offset_study.py"),
    ("03 Plant data studies",      "03_plant_data_studies/fig10_transport_velocity_analysis_tomato.py"),
    ("03 Plant data studies",      "03_plant_data_studies/fig11_transport_velocity_analysis_barley.py"),
    ("03 Plant data studies",      "03_plant_data_studies/fig12_transport_velocity_analysis_phaseolus.py"),
]


def _load_and_run(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    current_group = None
    for group, rel_path in SCRIPTS:
        if group != current_group:
            print(f"\n=== {group} ===")
            current_group = group
        script = STUDIES_ROOT / rel_path
        print(f"  Running {script.name} ...")
        _load_and_run(script)


if __name__ == "__main__":
    main()
