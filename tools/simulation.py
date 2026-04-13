from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from addict import Dict
from cadet import Cadet
from IPython import display
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.optimize import least_squares

from .models import (
    get_model_props,
    adjust_model_parameters,
    get_inlet_unit,
    get_multi_channel_transport_unit,
    configure_solver,
    get_xi_t_and_ROIs,
)


class CadetContainer:
    def __init__(
        self,
        model,
        filepath,
        csv_readout=None,
        xi=None,
        t=None,
        decay=0.000567 * 60,
        gauss_type=None,
        input_data_is_decay_corrected=False,
    ):
        self.cadet = Cadet()
        self.model = model
        self.filepath = filepath
        self.gauss_type = gauss_type

        if csv_readout is not None:
            xi, t, exp_data = get_xi_t_and_ROIs(csv_readout, decay_correct=input_data_is_decay_corrected)
            self.experimental_data_original = exp_data
            self.experimental_data = exp_data / np.amax(exp_data)
        else:
            self.experimental_data = None

        self.t = t
        self.xi = xi
        self.col_length = xi[-1]
        self.ncol = 100
        self.ncol_GA = 100
        self.ssq_list = []

        self.ncol_pre = 1
        self.col_length_pre = 0.0001

        self.live_plot = False
        self.live_plot_every = 1
        self._live_fig = None
        self._live_axes = None
        self._live_iter = 0

        unit_params_pre = Dict()
        unit_params_pre.col_dispersion = 0
        unit_params_pre.col_length = self.col_length_pre
        unit_params_pre.init_c = [0]
        unit_params_pre.ncol = self.ncol_pre
        unit_params_pre.decay = decay

        unit_params = Dict()
        unit_params.col_dispersion = 0
        unit_params.col_length = self.col_length
        unit_params.init_c = [0]
        unit_params.ncol = self.ncol
        unit_params.decay = decay

        inlet_params = Dict()
        inlet_params.ncomp = 1

        self.cadet.root.input.model.unit_000 = get_inlet_unit(inlet_params)
        self.cadet.root.input.model.unit_001 = get_multi_channel_transport_unit(unit_params_pre)
        self.cadet.root.input.model.unit_002 = get_multi_channel_transport_unit(unit_params)
        self.cadet.root.input.model.unit_003.unit_type = 'OUTLET'
        self.cadet.root.input.model.unit_003.ncomp = 1

        configure_solver(self.cadet, 3)

        self.cadet.root.input.model.connections.nswitches = 1
        self.cadet.root.input.model.connections.switch_000.section = 0

        self.n_var, self.lb, self.ub, self.scale, self.ub_GA = get_model_props(self)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def shift_xi(self, ncol: int) -> None:
        xi_preshifted = np.array(self.xi) * ((ncol - 1) / self.col_length)
        a = np.linspace(0, int(ncol), int(ncol) + 1)
        self.xi_shifted = [int(a[np.abs(a - x).argmin()]) for x in xi_preshifted]

    def run(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.cadet.filename = tmp_path
            self.cadet.save()
            data = self.cadet.run_simulation()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        if data.return_code != 0:
            print(data)
            raise Exception("Simulation failed")

    def norm_forwards(self, p):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return (p - lb) / (ub - lb)

    def norm_backwards(self, p_norm):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return p_norm * (ub - lb) + lb

    # ── Core simulation ───────────────────────────────────────────────────────

    def simulate(self, p) -> np.ndarray:
        adjust_model_parameters(self, p)
        self.cadet.root.input.model.unit_002.discretization.ncol = self.ncol
        self.shift_xi(self.ncol)
        self.run()
        cb = self.cadet.root.output.solution.unit_002.solution_bulk
        calc_data = np.hstack([
            np.array([np.sum(cb[:, :, :, 0], axis=2)[:, x]]).T
            for x in self.xi_shifted
        ])
        return calc_data / np.amax(calc_data)

    # ── Residual functions ────────────────────────────────────────────────────

    def _residual_normed(self, p_norm) -> np.ndarray:
        p = self.norm_backwards(p_norm)
        calc_data = self.simulate(p)
        experimental_data = self.experimental_data
        ssq = np.sum(np.square(calc_data - experimental_data)) / (
            len(experimental_data.ravel()) - len(p)
        )
        self.ssq_list.append(ssq)
        if self.live_plot:
            self._update_live_plot(calc_data, experimental_data, ssq, p)
        return (calc_data - experimental_data).ravel()

    def residual_GA(self, p) -> float:
        calc_data = self.simulate(p)
        experimental_data = self.experimental_data
        ssq = np.sum(np.square(calc_data - experimental_data)) / (
            len(experimental_data.ravel()) - len(p)
        )
        self.ssq_list.append(ssq)
        return ssq

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_p_LS(self, p0, xtol=1e-12, live_plot=False, live_plot_every=1):
        self.ssq_list = []
        self.live_plot = live_plot
        self.live_plot_every = live_plot_every
        self._live_iter = 0

        if self.live_plot:
            self._init_live_plot()

        result = least_squares(
            self._residual_normed,
            self.norm_forwards(p0),
            bounds=(self.norm_forwards(self.lb), self.norm_forwards(self.ub)),
            jac='2-point',
            xtol=xtol,
            method='trf',
            loss='linear',
            diff_step=1e-3,
            ftol=1e-10,
            gtol=1e-6,
            verbose=True,
            max_nfev=100,
            x_scale='jac',
        )

        self._residual_normed(result.x)

        ssq = min(self.ssq_list)
        cov_m = np.linalg.inv(result.jac.T @ result.jac)
        pcov = ssq * cov_m
        rel_std = (np.sqrt(np.diag(pcov)) / result.x) * 100

        return result, self.norm_backwards(result.x), rel_std

    def find_p0(self):
        class _Problem(ElementwiseProblem):
            def __init__(self_, container):
                super().__init__(
                    n_var=container.n_var, n_obj=1, n_constr=0,
                    xl=container.norm_forwards(container.lb),
                    xu=container.norm_forwards(container.ub_GA),
                )
                self_.container = container

            def _evaluate(self_, x, out, *args, **kwargs):
                out["F"] = self_.container.residual_GA(self_.container.norm_backwards(x))

        result = minimize(
            _Problem(self),
            DE(pop_size=8, sampling=LHS(), variant="DE/rand/1/bin", CR=0.6, F=0.8,
               dither="vector", jitter=False),
            get_termination("n_gen", 15),
            seed=1,
            verbose=True,
        )
        return self.norm_backwards(result.X)

    # ── Live plot ─────────────────────────────────────────────────────────────

    def _init_live_plot(self) -> None:
        plt.ion()
        self._live_fig = plt.figure(figsize=(15, 5))
        ax1 = self._live_fig.add_subplot(131)
        ax2 = self._live_fig.add_subplot(132)
        ax3 = self._live_fig.add_subplot(133)
        self._live_axes = (ax1, ax2, ax3)
        self._live_iter = 0
        self._live_fig.tight_layout()
        self._live_fig.canvas.draw()
        self._live_fig.canvas.flush_events()
        plt.show(block=False)

    def _update_live_plot(self, calc_data, experimental_data, ssq, p) -> None:
        self._live_iter += 1
        if self._live_iter % self.live_plot_every != 0:
            return

        cb = self.cadet.root.output.solution.unit_002.solution_bulk
        ax1, ax2, ax3 = self._live_axes
        ax1.clear(); ax2.clear(); ax3.clear()

        velocity = float(p[0]) if len(p) > 0 else np.nan
        ax1.plot(self.t, calc_data, label="Calc")
        ax1.plot(self.t, experimental_data, ".", label="Exp")
        ax1.set_xlabel("Time [min]")
        ax1.set_ylabel("Activity [a.u.]")
        ax1.set_title(f"v = {velocity:.5f}")
        ax1.legend()

        ax2.imshow(np.sum(cb[:, :, :, 0], axis=2).T, cmap="viridis", aspect="auto")
        for x in self.xi_shifted:
            ax2.axhline(x, color="orange")
        ax2.set_title("Bulk signal")

        ax3.plot(self.ssq_list, marker="o", linestyle="-")
        ax3.set_title(f"iter={self._live_iter}, MSE={ssq:.4e}")
        ax3.set_xlabel("Evaluation")
        ax3.set_ylabel("MSE")

        self._live_fig.tight_layout()
        self._live_fig.canvas.draw()
        self._live_fig.canvas.flush_events()
        plt.pause(0.001)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def plot_inlet(self) -> None:
        plt.figure()
        time = self.cadet.root.output.solution.solution_times
        c = self.cadet.root.output.solution.unit_000.solution_outlet
        plt.plot(time, c)
        plt.title('Inlet')
        plt.xlabel('Time [min]')
        plt.ylabel('Concentration [mol/L]')
        plt.show()

    def plot_outlet(self) -> None:
        plt.figure()
        time = self.cadet.root.output.solution.solution_times
        c = self.cadet.root.output.solution.unit_002.solution_outlet
        plt.plot(time, c[:, 0, :])
        plt.plot(time, c[:, 1, :])
        plt.title('Outlet')
        plt.xlabel('Time [min]')
        plt.ylabel('Concentration [mol/L]')
        plt.show()

    def save_in_csv(self, xi, p, filename: str) -> str:
        a = self.simulate(p)
        b = np.concatenate((xi.reshape(1, -1), a))
        np.savetxt(filename + ".csv", b, delimiter=";")
        return "File saved"


# ── Convenience function ──────────────────────────────────────────────────────

def simulate_transport_case(
    model_name: str = "M02",
    model_file: str = "model.h5",
    xi: list[float] | None = None,
    t: np.ndarray | None = None,
    params: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single transport simulation and return (positions, t, data)."""
    xi = xi or [0, 20, 40]
    t = t if t is not None else np.linspace(0, 145, 30)
    params = params or [5.0, 100, 40, 10, 2, 0.0]

    sim = CadetContainer(model_name, model_file, xi=xi, t=t, gauss_type="stretched")
    data = sim.simulate(params)
    return np.array(xi, dtype=float), t, data
