# Lanzrath 2025 - Tracer Transport Velocity Analysis

[![GitHub](https://img.shields.io/badge/GitHub-hannahlanzrath%2FLanzrath__2025-blue?logo=github)](https://github.com/hannahlanzrath/Lanzrath_2025)

Repositry with companion code for:

> **Lanzrath et al. (2025)** - *Analyzing time activity curves from spatio-temporal tracer data to determine tracer transport velocity in plants*
> Mathematical Biosciences, 2025.
> https://doi.org/10.1016/j.mbs.2025.109430

Implements and benchmarks four methods (RANSAC intercept, half-maximum spline, McKay binding model, Bühler/MCT mechanistic model based on CADET) for estimating phloem transport velocity from PET imaging data. Reproduces figures 2,3,5,7-12 in the paper.

---

## Notes on Implementation

The Bühler model, referred to as the [Multichannel Transport (MCT) model](https://cadet.github.io/v5.1.X/modelling/unit_operations/multi_channel_transport_model.html) in CADET, is implemented in this repository using [CADET-Python](https://cadet.github.io/v5.1.X/developer_guide/cadet_python.html), which provides a direct Python interface to the CADET simulation engine.

A newer and more flexible implementation of the MCT model is now primarily available in [CADET-Process](https://cadet-process.readthedocs.io/en/latest/reference/generated/CADETProcess.processModel.MCT.html#CADETProcess.processModel.MCT).

This updated implementation enables a more convenient setup of complex and branched transport systems and is well suited for extending the plant transport scenarios considered in this work.


---

## Installation

Requires Python ≥ 3.10 and dependencies as listed in [pyproject.toml](https://github.com/hannahlanzrath/Lanzrath_2025/blob/main/pyproject.toml).

```bash
git clone https://github.com/hannahlanzrath/Lanzrath_2025
pip install -e Lanzrath_2025
```

---

## Usage

Run all studies and save figures to `figures/`:

```bash
python run_all_studies.py
```

Run a single study:

```bash
python -m studies.01_illustrative_examples.fig2_decay_correction
python -m studies.01_illustrative_examples.fig5_spatial_temporal_distribution
python -m studies.01_illustrative_examples.fig7_transport_types

python -m studies.02_artificial_data_studies.fig3_data_driven_methods
python -m studies.02_artificial_data_studies.fig8_transport_velocity_analysis_in_silico
python -m studies.02_artificial_data_studies.fig9_intercept_offset_study

python -m studies.03_plant_data_studies.fig10_transport_velocity_analysis_tomato
python -m studies.03_plant_data_studies.fig11_transport_velocity_analysis_barley
python -m studies.03_plant_data_studies.fig12_transport_velocity_analysis_phaseolus
```

Pass `--optimize` to the plant data studies to run least-squares fitting for the Bühler model before extracting velocity (slower, prints fitted parameters):

```bash
python -m studies.03_plant_data_studies.fig10_transport_velocity_analysis_tomato --optimize
```

---

## Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 2 | `fig2_decay_correction.py` | Effect of radioactive decay correction |
| Fig. 3 | `fig3_data_driven_methods.py` | Data-driven reference-time methods on simulated signals |
| Fig. 5 | `fig5_spatial_temporal_distribution.py` | Spatiotemporal tracer distribution |
| Fig. 7 | `fig7_transport_types.py` | Activity profiles for transport models M01, M02, M13 |
| Fig. 8 | `fig8_transport_velocity_analysis_in_silico.py` | Velocity benchmark on clean and noisy in-silico data |
| Fig. 9 | `fig9_intercept_offset_study.py` | RANSAC sensitivity to ROI index offset |
| Fig. 10 | `fig10_transport_velocity_analysis_tomato.py` | Velocity analysis — tomato |
| Fig. 11 | `fig11_transport_velocity_analysis_barley.py` | Velocity analysis — barley |
| Fig. 12 | `fig12_transport_velocity_analysis_phaseolus.py` | Velocity analysis — *Phaseolus vulgaris* |
