from __future__ import annotations

import numpy as np
import pandas as pd
from addict import Dict
from scipy import interpolate


# ── Errors ────────────────────────────────────────────────────────────────────

class ModelError(Exception):
    pass


# ── Unit builders ─────────────────────────────────────────────────────────────

def get_inlet_unit(params) -> Dict:
    unit = Dict()
    unit.inlet_type = 'PIECEWISE_CUBIC_POLY'
    unit.unit_type = 'INLET'
    unit.ncomp = params.ncomp
    return unit


def get_multi_channel_transport_unit(params) -> Dict:
    unit = Dict()
    unit.unit_type = 'MULTI_CHANNEL_TRANSPORT'
    unit.ncomp = 1
    unit.col_length = params.col_length
    unit.par_radius = 0.0
    unit.par_porosity = 0.0
    unit.col_dispersion = params.col_dispersion
    unit.col_dispersion_radial = 0
    unit.film_diffusion = 0
    unit.par_diffusion = 0
    unit.par_surfdiffusion = 0
    unit.init_c = params.init_c
    unit.init_cp = [0]
    unit.init_q = [0]
    unit.exchange_matrix = params.e
    decay = params.decay
    unit.reaction_model = 'MASS_ACTION_LAW'
    unit.reaction_bulk.mal_kfwd_bulk = [decay]
    unit.reaction_bulk.mal_kbwd_bulk = [0]
    unit.reaction_bulk.mal_stoichiometry_bulk = [-1]
    unit.reaction_parameters.mal_exponents_bulk_fwd = [0, 1, 1, 0]
    unit.discretization.use_analytic_jacobian = 1
    unit.discretization.nbound = [0] * unit.ncomp
    unit.discretization.schur_safety = 1.0e-8
    unit.discretization.weno.boundary_model = 0
    unit.discretization.weno.weno_eps = 1e-10
    unit.discretization.weno.weno_order = 3
    unit.discretization.gs_type = 1
    unit.discretization.max_krylov = 0
    unit.discretization.max_restarts = 10
    unit.discretization.ncol = params.ncol
    unit.discretization.par_disc_type = ['EQUIDISTANT_PAR']
    unit.discretization.npar = 1
    unit.discretization.nbound = 0
    unit.discretization.radial_disc_type = 'EQUIVOLUME'
    return unit


# ── Solver settings ───────────────────────────────────────────────────────────

def configure_solver(cadet, n_units: int) -> None:
    cadet.root.input.model.nunits = n_units

    cadet.root.input['return'].split_components_data = False
    cadet.root.input['return'].split_ports_data = 0
    cadet.root.input['return'].unit_000.write_solution_inlet = 1
    cadet.root.input['return'].unit_000.write_solution_outlet = 1
    cadet.root.input['return'].unit_000.write_solution_bulk = 1
    cadet.root.input['return'].unit_000.write_solution_particle = 1
    cadet.root.input['return'].unit_000.write_solution_solid = 1
    cadet.root.input['return'].unit_000.write_solution_flux = 1
    cadet.root.input['return'].unit_000.write_solution_volume = 1
    cadet.root.input['return'].unit_000.write_coordinates = 1
    cadet.root.input['return'].unit_000.write_sens_outlet = 1
    cadet.root.input['return'].unit_000.write_soldot_bulk = 1

    for unit in range(n_units):
        cadet.root.input['return']['unit_{0:03d}'.format(unit)] = cadet.root.input['return'].unit_000

    cadet.root.input.solver.time_integrator.abstol = 1e-10
    cadet.root.input.solver.time_integrator.algtol = 1e-10
    cadet.root.input.solver.time_integrator.reltol = 1e-6
    cadet.root.input.solver.time_integrator.init_step_size = 1e-8
    cadet.root.input.solver.time_integrator.max_steps = 1000000
    cadet.root.input.model.solver.gs_type = 1
    cadet.root.input.model.solver.max_krylov = 0
    cadet.root.input.model.solver.max_restarts = 10
    cadet.root.input.model.solver.schur_safety = 1e-8
    cadet.root.input.solver.nthreads = 1


# ── Gaussian inlet profile ────────────────────────────────────────────────────

def _set_spline_input(cadet, gauss_t, sol_t, y) -> None:
    cu = interpolate.CubicSpline(gauss_t, y)
    n_sections = len(gauss_t) - 1

    for i in range(n_sections):
        sec = 'sec_{0:03d}'.format(i)
        cadet.root.input.model.unit_000[sec].cube_coeff = cu.c[:, i][0]
        cadet.root.input.model.unit_000[sec].quad_coeff = cu.c[:, i][1]
        cadet.root.input.model.unit_000[sec].lin_coeff = cu.c[:, i][2]
        cadet.root.input.model.unit_000[sec].const_coeff = cu.c[:, i][3]

    last = 'sec_{0:03d}'.format(n_sections)
    cadet.root.input.model.unit_000[last].cube_coeff = [0.0]
    cadet.root.input.model.unit_000[last].quad_coeff = [0.0]
    cadet.root.input.model.unit_000[last].lin_coeff = [0.0]
    cadet.root.input.model.unit_000[last].const_coeff = [0.0]

    secs = np.append(gauss_t, sol_t[-1])
    cadet.root.input.solver.sections.section_times = secs
    cadet.root.input.solver.sections.nsec = n_sections + 1
    cadet.root.input.solver.sections.section_continuity = [1]


def generate_gaussian_input(cadet, sol_t, t0, sigma, p4) -> None:
    """Build a Gaussian (or stretched-Gaussian) inlet profile and write it to cadet."""
    width = sol_t[-1]
    sections = 100
    gauss_t = np.linspace(0, width, sections)

    if p4 is None:
        y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((gauss_t - t0) / sigma) ** 2)
    else:
        m = (gauss_t - sigma) / (t0 - sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.nan_to_num(np.exp(-0.5 * (np.log(m) / np.log(p4)) ** 2))

    # Reverse decay correction — input is assumed to be decay-corrected
    y_uncorr = y * np.exp(-0.000567 * 60 * gauss_t)
    _set_spline_input(cadet, gauss_t, sol_t, y_uncorr)


# ── CSV data helper (used by CadetContainer) ─────────────────────────────────

def get_xi_t_and_ROIs(data: pd.DataFrame, decay_correct: bool = False):
    t = data.iloc[:, 0].to_numpy().reshape(-1, 1)
    ROIs = []
    xi = []
    decay_rate = 0.000567 * 60

    for i in range(1, len(data.columns)):
        xi.append(float(data.columns[i]))
        ROI = data.iloc[:, i].to_numpy().reshape(-1, 1)
        if decay_correct:
            ROI = ROI * np.exp(-decay_rate * t)
        ROIs.append(ROI)

    ROIs = np.concatenate(tuple(ROIs), axis=1)
    ROIs = ROIs / np.max(ROIs)
    return np.array(xi), t, ROIs


# ── Model configuration ───────────────────────────────────────────────────────

def get_model_props(container):
    model = container.model
    cadet = container.cadet

    try:
        if model == "M01":
            cadet.root.input.model.unit_001.nchannel = 1
            cadet.root.input.model.unit_002.nchannel = 1
            cadet.root.input.model.unit_001.channel_cross_section_areas = np.full((1), 1).tolist()
            cadet.root.input.model.unit_002.channel_cross_section_areas = np.full((1), 1).tolist()
            n_var = 1 + 4
            if container.gauss_type == "stretched":
                lb     = [0.0,  0.0,   0.00,  0.00,  0.0]
                ub     = [40.0, 100.0, 100.0, 99.0,  50.0]
                ub_GA  = [15.0, 60.0,  60.0,  20.0,  50.0]
                scale  = [10, 5, 5, 1, 5]
            else:
                lb     = [0.0,  0.0,   0.00,  0.00]
                ub     = [40.0, 100.0, 100.0, 99.0]
                ub_GA  = [15.0, 60.0,  60.0,  20.0]
                scale  = [10, 5, 1, 1]

        elif model == "M02":
            cadet.root.input.model.unit_001.nchannel = 2
            cadet.root.input.model.unit_002.nchannel = 2
            cadet.root.input.model.unit_001.channel_cross_section_areas = np.full((2), 1).tolist()
            cadet.root.input.model.unit_002.channel_cross_section_areas = np.full((2), 1).tolist()
            if container.gauss_type == "stretched":
                lb     = [0.0,  0.0,   0.00,  0.00,  0.0,  0.0]
                ub     = [40.0, 100.0, 100.0, 99.0,  50.0, 1.0]
                ub_GA  = [15.0, 60.0,  60.0,  20.0,  50.0, 0.01]
                scale  = [10, 5, 5, 1, 5, 0.1]
            else:
                lb     = [0.0,  0.0,   0.00,  0.00,  0.0]
                ub     = [40.0, 100.0, 100.0, 99.0,  1.0]
                ub_GA  = [15.0, 60.0,  60.0,  20.0,  0.01]
                scale  = [10, 5, 1, 1, 0.1]
            n_var = len(lb)

        elif model == "M13":
            cadet.root.input.model.unit_001.nchannel = 3
            cadet.root.input.model.unit_002.nchannel = 3
            cadet.root.input.model.unit_001.channel_cross_section_areas = np.full((3), 1).tolist()
            cadet.root.input.model.unit_002.channel_cross_section_areas = np.full((3), 1).tolist()
            if container.gauss_type == "stretched":
                lb     = [0.0,  0.0,   0.0,   0.00,  0.00, 0.0,  0.0,  0.0]
                ub     = [40.0, 100.0, 100.0, 99.0,  40.0, 1.0,  1.0,  1.0]
                ub_GA  = [15.0, 60.0,  60.0,  50.0,  20.0, 0.01, 0.01, 0.01]
                scale  = [10, 5, 5, 1, 5, 0.1, 0.1, 0.1]
            else:
                lb     = [0.0,  0.0,   0.0,   0.00,  0.0,  0.0,  0.0]
                ub     = [40.0, 100.0, 100.0, 99.0,  1.0,  1.0,  1.0]
                ub_GA  = [15.0, 60.0,  60.0,  40.0,  0.01, 0.01, 0.01]
                scale  = [10, 5, 5, 1, 0.1, 0.1, 0.1]
            n_var = len(lb)

        else:
            raise ModelError

    except ModelError:
        print("Please select a valid model number.")

    return n_var, lb, ub, scale, ub_GA


def adjust_model_parameters(self, p) -> None:
    if self.gauss_type == "stretched":
        Q, x0, t0, sigma, p4, *exchange_parameters = p
    else:
        Q, x0, t0, sigma, *exchange_parameters = p
        p4 = None

    cadet = self.cadet
    model = self.model
    self.x0 = x0

    self.cadet.root.input.model.unit_001.col_length = self.col_length_pre + x0

    delta = 5
    self.t = np.arange(0, self.t[-1] + delta, delta)
    cadet.root.input.solver.user_solution_times = self.t
    self.t_input_end = int(np.ceil(sigma * 3.5))

    generate_gaussian_input(cadet, self.t, t0, sigma, p4)

    if model == "M01":
        cadet.root.input.model.connections.switch_000.connections = [
            0, 1, -1, -1, Q,
            1, 2, -1, -1, Q,
            2, 3, -1, -1, Q,
        ]
        cadet.root.input.model.unit_001.exchange_matrix = [0]
        cadet.root.input.model.unit_002.exchange_matrix = [0]

    if model == "M02":
        a12 = exchange_parameters[0]
        cadet.root.input.model.connections.switch_000.connections = [
            0, 1, 0, 0, -1, -1, Q,
            1, 2, 0, 0, -1, -1, Q,
            2, 3, 0, 0, -1, -1, Q,
        ]
        cadet.root.input.model.unit_001.exchange_matrix = [0, a12, 0, 0]
        cadet.root.input.model.unit_002.exchange_matrix = [0, a12, 0, 0]

    if model == "M13":
        a12, a21, a23 = exchange_parameters
        cadet.root.input.model.connections.switch_000.connections = [
            0, 1, 0, 0, -1, -1, Q,
            1, 2, 0, 0, -1, -1, Q,
            2, 3, 0, 0, -1, -1, Q,
        ]
        cadet.root.input.model.unit_001.exchange_matrix = [0, a12, 0, a21, 0, a23, 0, 0, 0]
        cadet.root.input.model.unit_002.exchange_matrix = [0, a12, 0, a21, 0, a23, 0, 0, 0]
