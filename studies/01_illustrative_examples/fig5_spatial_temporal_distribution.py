from cadet import Cadet

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from scipy.interpolate import CubicSpline

from Lanzrath_2025.tools.utils import save_figure


def get_mct_unit(params: Dict) -> Dict:
    """Create the MULTI_CHANNEL_TRANSPORT unit configuration.

    This function builds the main transport unit of the model, including
    geometric parameters, exchange dynamics, discretization settings, and
    an optional first-order decay reaction implemented via a mass-action law.

    Args:
        params: Parameter dictionary containing the unit configuration.
            Expected keys are:
                col_length: Length of the transport domain.
                channel_cross_section_areas: Cross-sectional areas of channels.
                col_dispersion: Axial dispersion coefficient(s).
                init_c: Initial concentration values.
                e: Exchange matrix.
                ncol: Number of axial cells.
                nchannel: Number of channels.
                decay: First-order decay rate.

    Returns:
        Dict: CADET unit configuration for a MULTI_CHANNEL_TRANSPORT unit.
    """
    unit = Dict()

    # Model parameters
    unit.unit_type = "MULTI_CHANNEL_TRANSPORT"
    unit.ncomp = 1
    unit.col_length = params.col_length
    unit.channel_cross_section_areas = params.channel_cross_section_areas
    unit.col_dispersion = params.col_dispersion
    unit.init_c = params.init_c
    unit.exchange_matrix = params.e
    unit.nchannel = params.nchannel

    # Discretization parameters
    unit.discretization.use_analytic_jacobian = 1
    unit.discretization.weno.boundary_model = 0
    unit.discretization.weno.weno_eps = 1e-10
    unit.discretization.weno.weno_order = 3
    unit.discretization.max_restarts = 10
    unit.discretization.ncol = params.ncol

    # Reaction model for optional tracer decay
    decay = params.decay
    unit.reaction_model = "MASS_ACTION_LAW"
    unit.reaction_bulk.mal_kfwd_bulk = [decay]
    unit.reaction_bulk.mal_kbwd_bulk = [0]
    unit.reaction_bulk.mal_stoichiometry_bulk = [-1]
    unit.mal_exponents_bulk_fwd = [
        0, 1,
        1, 0,
    ]

    return unit


def get_inlet_unit(params: Dict | None = None) -> Dict:
    """Create the inlet unit configuration.

    The inlet is defined as a piecewise cubic polynomial source term.

    Args:
        params: Unused placeholder for interface consistency.

    Returns:
        Dict: CADET unit configuration for an INLET unit.
    """
    del params

    unit = Dict()
    unit.inlet_type = "PIECEWISE_CUBIC_POLY"
    unit.unit_type = "INLET"
    unit.ncomp = 1
    return unit


def configure_solver_and_output(cadet: Cadet, n_units: int) -> None:
    """Configure solver tolerances and requested output fields.

    This sets global solver options, time integration tolerances, and enables
    output writing for all model units.

    Args:
        cadet: CADET model instance to configure.
        n_units: Number of units in the model.

    Returns:
        None
    """
    cadet.root.input.model.nunits = n_units

    cadet.root.input["return"].split_components_data = False
    cadet.root.input["return"].split_ports_data = 0
    cadet.root.input["return"].unit_000.write_solution_inlet = 1
    cadet.root.input["return"].unit_000.write_solution_outlet = 1
    cadet.root.input["return"].unit_000.write_solution_bulk = 1
    cadet.root.input["return"].unit_000.write_solution_particle = 1
    cadet.root.input["return"].unit_000.write_solution_solid = 1
    cadet.root.input["return"].unit_000.write_solution_flux = 1
    cadet.root.input["return"].unit_000.write_solution_volume = 1
    cadet.root.input["return"].unit_000.write_coordinates = 1
    cadet.root.input["return"].unit_000.write_sens_outlet = 1

    for unit_idx in range(n_units):
        cadet.root.input["return"][f"unit_{unit_idx:03d}"] = (
            cadet.root.input["return"].unit_000
        )

    cadet.root.input.solver.time_integrator.abstol = 1e-10
    cadet.root.input.solver.time_integrator.algtol = 1e-12
    cadet.root.input.solver.time_integrator.reltol = 1e-10
    cadet.root.input.solver.time_integrator.init_step_size = 1e-10
    cadet.root.input.solver.time_integrator.max_steps = 1_000_000

    cadet.root.input.model.solver.gs_type = 1
    cadet.root.input.model.solver.max_krylov = 0
    cadet.root.input.model.solver.max_restarts = 10
    cadet.root.input.model.solver.schur_safety = 1e-8

    cadet.root.input.solver.nthreads = 1


def set_spline_input(
    cadet: Cadet,
    section_times: np.ndarray,
    solution_times: np.ndarray,
    values: np.ndarray,
) -> CubicSpline:
    """Create and assign a cubic spline inlet profile.

    The spline is constructed on the provided section times and written into
    the inlet unit as piecewise cubic polynomial coefficients.

    Args:
        cadet: CADET model instance.
        section_times: Time points defining the spline sections.
        solution_times: Time points at which the solution should be evaluated.
        values: Function values at `section_times`.

    Returns:
        CubicSpline: The fitted cubic spline object.
    """
    spline = CubicSpline(section_times, values)
    n_sections = len(section_times) - 1

    for i in range(n_sections):
        section_name = f"sec_{i:03d}"
        cadet.root.input.model.unit_000[section_name].cube_coeff = spline.c[:, i][0]
        cadet.root.input.model.unit_000[section_name].quad_coeff = spline.c[:, i][1]
        cadet.root.input.model.unit_000[section_name].lin_coeff = spline.c[:, i][2]
        cadet.root.input.model.unit_000[section_name].const_coeff = spline.c[:, i][3]

    last_section_name = f"sec_{n_sections:03d}"
    cadet.root.input.model.unit_000[last_section_name].cube_coeff = [0.0]
    cadet.root.input.model.unit_000[last_section_name].quad_coeff = [0.0]
    cadet.root.input.model.unit_000[last_section_name].lin_coeff = [0.0]
    cadet.root.input.model.unit_000[last_section_name].const_coeff = [0.0]

    all_section_times = np.append(section_times, solution_times[-1])
    cadet.root.input.solver.sections.section_times = all_section_times
    cadet.root.input.solver.sections.nsec = n_sections + 1
    cadet.root.input.solver.sections.section_continuity = [1]
    cadet.root.input.solver.user_solution_times = solution_times

    return spline


def stretched_gaussian_input(
    time: np.ndarray,
    sigma: float,
    t0: float,
    base: float = 2.0,
    amplitude: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Generate the stretched Gaussian-like inlet profile.

    The profile is based on

        exp(-0.5 * (log(m) / log(base))**2)

    with

        m = (time - sigma) / (t0 - sigma)

    Args:
        time: Time grid for evaluating the profile.
        sigma: Shape parameter used in the transformation.
        t0: Peak-location-related parameter.
        base: Logarithm base used in the transformation.
        amplitude: Multiplicative amplitude factor.
        normalize: Whether to normalize the output to a maximum of 1.

    Returns:
        np.ndarray: Profile values on the given time grid.
    """
    m = (time - sigma) / (t0 - sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.nan_to_num(
            amplitude * np.exp(-0.5 * (np.log(m) / np.log(base)) ** 2)
        )

    if normalize and np.max(profile) > 0:
        profile = profile / np.max(profile)

    return profile


def gaussian_input(
    time: np.ndarray,
    sigma: float,
    mu: float,
    amplitude: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a standard Gaussian inlet profile.

    The profile is defined as

        amplitude * exp(-0.5 * ((time - mu)^2 / sigma^2))

    Args:
        time: Time grid for evaluating the profile.
        sigma: Standard deviation of the Gaussian.
        mu: Mean value or peak position of the Gaussian.
        amplitude: Multiplicative amplitude factor.
        normalize: Whether to normalize the output to a maximum of 1.

    Returns:
        np.ndarray: Gaussian profile values on the given time grid.
    """
    exponent = ((time - mu) ** 2) / (sigma ** 2)
    profile = np.nan_to_num(amplitude * np.exp(-0.5 * exponent))

    if normalize and np.max(profile) > 0:
        profile = profile / np.max(profile)

    return profile


def plot_boundary_condition(cadet: Cadet, figure_name: str, title: str) -> None:
    """Plot the outlet concentration of the inlet unit over time.

    Args:
        cadet: Simulated CADET model instance containing output data.
        figure_name: File name used for saving the figure.
        title: Plot title.

    Returns:
        None
    """
    plt.figure()

    time = cadet.root.output.solution.solution_times
    concentration = cadet.root.output.solution.unit_000.solution_outlet

    plt.plot(time, concentration)
    plt.title(title)
    plt.xlabel(r"$time~/~min$")
    plt.ylabel(r"$concentration~/~mol \cdot L^{-1}$")

    save_figure("figures/figure_5", figure_name)
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    plt.close()


def plot_initial_condition(
    cadet: Cadet,
    figure_name: str,
    title: str,
    start_time_index: int = 60,
    end_space_index: int = 100,
    channel_index: int = 0,
    aspect: float = 0.3,
) -> None:
    """Plot the selected bulk concentration field as an initial-condition figure.

    This reproduces the requested heatmap

        cb[60:, 0:100, 0, 0].T

    Args:
        cadet: Simulated CADET model instance containing output data.
        figure_name: File name used for saving the figure.
        title: Plot title.
        start_time_index: First temporal index shown in the plot.
        end_space_index: Last spatial index shown in the plot.
        channel_index: Channel index used in the third axis of solution_bulk.
        aspect: Aspect ratio for imshow.

    Returns:
        None
    """
    cb = cadet.root.output.solution.unit_001.solution_bulk

    plt.figure()
    plt.imshow(
        cb[start_time_index:, 0:end_space_index, channel_index, 0].T,
        cmap="viridis",
        aspect=aspect,
    )
    plt.title(title)
    plt.xlabel(r"$temporal~discretization~[t]$")
    plt.ylabel(r"$spatial~discretization~[x]$")

    save_figure("figures/figure_5", figure_name)
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    plt.close()


def plot_spatiotemporal_distribution(
    cadet: Cadet,
    figure_name: str,
    title: str,
    end_space_index: int = 100,
    aspect: float = 0.5,
) -> None:
    """Plot the summed spatiotemporal concentration distribution.

    The concentrations of all channels are summed and displayed as an image
    over time and space.

    Args:
        cadet: Simulated CADET model instance containing output data.
        figure_name: File name used for saving the figure.
        title: Plot title.
        end_space_index: Last spatial index included in the plot.
        aspect: Aspect ratio for imshow.

    Returns:
        None
    """
    cb = cadet.root.output.solution.unit_001.solution_bulk
    image = np.sum(cb[:, 0:end_space_index, :, 0], axis=2).T

    plt.figure()
    plt.imshow(image, cmap="viridis", aspect=aspect)
    plt.title(title)
    plt.xlabel(r"$temporal~discretization~[t]$")
    plt.ylabel(r"$spatial~discretization~[x]$")

    cbar = plt.colorbar()
    cbar.set_label("Activity [a.u.]")

    save_figure("figures/figure_5", figure_name)
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    plt.close()


def build_model() -> Cadet:
    """Build and configure the CADET model.

    Returns:
        Cadet: Fully configured CADET model instance.
    """
    unit_params = Dict()
    unit_params.col_dispersion = 0
    unit_params.col_length = 500
    unit_params.init_c = [0]
    unit_params.ncol = 200
    unit_params.decay = 0
    unit_params.channel_cross_section_areas = [1, 1, 1]
    unit_params.nchannel = 3

    # Exchange parameters for M13
    e = 1e-2
    a12 = 1.0 * e
    a21 = 0.8 * e
    a23 = 0.5 * e

    unit_params.e = [
        0,   a12, 0,
        a21, 0,   a23,
        0,   0,   0,
    ]

    inlet_params = Dict()

    cadet = Cadet()

    cadet.root.input.model.unit_000 = get_inlet_unit(inlet_params)
    cadet.root.input.model.unit_001 = get_mct_unit(unit_params)
    cadet.root.input.model.unit_002.unit_type = "OUTLET"
    cadet.root.input.model.unit_002.ncomp = 1

    configure_solver_and_output(cadet, n_units=3)

    flow_rate = 5
    cadet.root.input.model.connections.nswitches = 1
    cadet.root.input.model.connections.switch_000.section = 0
    cadet.root.input.model.connections.switch_000.connections = [
        0, 1, 0, 0, -1, -1, flow_rate,
        1, 2, 0, 0, -1, -1, flow_rate,
    ]

    return cadet


def run_simulation_with_input(
    input_type: str,
    solution_times: np.ndarray,
    section_times: np.ndarray,
    sigma: float,
    t0: float,
) -> Cadet:
    """Build and run a model with the selected inlet profile.

    Args:
        input_type: Type of input profile. Supported values are
            "gaussian" and "stretched_gaussian".
        solution_times: Time grid for evaluated solution output.
        section_times: Time grid for spline section definition.
        sigma: Width parameter of the inlet function.
        t0: Peak position parameter of the inlet function.

    Returns:
        Cadet: Simulated CADET object.
    """
    cadet = build_model()

    if input_type == "gaussian":
        inlet_profile = gaussian_input(
            time=section_times,
            sigma=sigma,
            mu=t0,
            amplitude=1.0,
            normalize=True,
        )
    elif input_type == "stretched_gaussian":
        inlet_profile = stretched_gaussian_input(
            time=section_times,
            sigma=sigma,
            t0=t0,
            base=2.0,
            amplitude=1.0,
            normalize=True,
        )
    else:
        raise ValueError(
            f"Unknown input_type '{input_type}'. "
            "Supported values are 'gaussian' and 'stretched_gaussian'."
        )

    set_spline_input(
        cadet=cadet,
        section_times=section_times,
        solution_times=solution_times,
        values=inlet_profile,
    )

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cadet.filename = tmp_path
        cadet.save()
        cadet.run_simulation()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return cadet


def main() -> None:
    """Run two model setups and generate four figures in total.

    Setup 1:
        Standard Gaussian input
        - boundary condition plot
        - initial condition heatmap

    Setup 2:
        Stretched Gaussian input
        - boundary condition plot
        - spatiotemporal distribution heatmap
    """
    solution_times = np.linspace(0, 145, 145)
    section_times = np.linspace(0, 145, 100)
    

    # ------------------------------------------------------------------
    # Setup 1: Normal Gaussian with initial condition plot
    # ------------------------------------------------------------------
    cadet_gaussian = run_simulation_with_input(
        input_type="gaussian",
        solution_times=solution_times,
        section_times=section_times,
        sigma=12,
        t0=40,
    )

    plot_boundary_condition(
        cadet=cadet_gaussian,
        figure_name="fig5_gaussian_boundary_condition",
        title="Gaussian input boundary condition",
    )

    plot_initial_condition(
        cadet=cadet_gaussian,
        figure_name="fig5_gaussian_initial_condition",
        title="Gaussian input initial condition",
        start_time_index=60,
        end_space_index=100,
        channel_index=0,
        aspect=0.3,
    )

    # ------------------------------------------------------------------
    # Setup 2: Stretched Gaussian with boundary condition plot
    # ------------------------------------------------------------------
    cadet_stretched = run_simulation_with_input(
        input_type="stretched_gaussian",
        solution_times=solution_times,
        section_times=section_times,
        sigma=8,
        t0=20,
    )

    plot_boundary_condition(
        cadet=cadet_stretched,
        figure_name="fig5_stretched_gaussian_boundary_condition",
        title="Stretched Gaussian boundary condition",
    )

    plot_spatiotemporal_distribution(
        cadet=cadet_stretched,
        figure_name="fig5_stretched_gaussian_spatiotemporal_distribution",
        title="Stretched Gaussian spatiotemporal distribution",
        end_space_index=100,
        aspect=0.5,
    )


if __name__ == "__main__":
    main()