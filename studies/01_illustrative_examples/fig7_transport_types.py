from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from Lanzrath_2025.tools.simulation import CadetContainer
from Lanzrath_2025.tools.utils import save_figure


def plot_transport_types(
    steps: int = 30,
    xi: list[float] | None = None,
    t_start: float = 0.0,
    t_end: float = 145.0,
    decay_constant: float = 0.000567 * 60,
) -> None:
    """Plot decay-corrected ROI signals for multiple transport models.

    Args:
        steps: Number of simulation time points.
        xi: Spatial positions. Defaults to [0, 20, 40, 60, 80].
        t_start: Start time in minutes.
        t_end: End time in minutes.
        decay_constant: Exponential decay constant in 1/min.
    """
    xi = xi or [0, 20, 40, 60, 80]
    t = np.linspace(t_start, t_end, steps)
    decay_factor = np.exp(-decay_constant * t)

    configs = {
        "M01": ("M01", [5.0, 20, 30, 8, 3]),
        "M02": ("M02", [5.0, 20, 30, 5, 2, 0.06]),
        "M13": ("M13", [5.0, 20, 30, 5, 2, 0.5, 0.2, 0.01]),
    }

    for label, (model_name, params) in configs.items():
        plt.figure(figsize=(5.5, 5))
        plt.xlabel("Time [min]", fontsize=11)

        sim = CadetContainer(
            model_name,
            "model.h5",
            xi=xi,
            t=t,
            gauss_type="stretched",
        )
        data = sim.simulate(params)

        norm = np.max(data[:, 0] / decay_factor)

        plt.plot(
            t,
            data[:, 0] / decay_factor / norm,
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            label="ROI 1",
        )
        plt.plot(
            t,
            data[:, 1] / decay_factor / norm,
            marker="s",
            linestyle="none",
            markerfacecolor="none",
            label="ROI 2",
        )
        plt.plot(
            t,
            data[:, 2] / decay_factor / norm,
            marker="^",
            linestyle="none",
            markerfacecolor="none",
            label="ROI 3",
        )

        plt.ylim([-0.15, 1.1])
        plt.tick_params(axis="y", which="both", labelleft=False)
        plt.title(label)
        plt.legend()
        
        save_figure("figures/figure_7", f"fig7_transport_types_{label}")


def main() -> None:
    plot_transport_types()


if __name__ == "__main__":
    main()