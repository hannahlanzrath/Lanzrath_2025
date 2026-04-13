from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from Lanzrath_2025.tools.simulation import CadetContainer
from Lanzrath_2025.tools.utils import save_figure


def plot_decay_correction_comparison(
    steps: int = 30,
    xi: list[float] | None = None,
    t_start: float = 0.0,
    t_end: float = 145.0,
    decay_constant: float = 0.000567 * 60,
    base_error: float = 0.007,
) -> None:
    """Plot original and decay-corrected activity curves with identical styling.

    Args:
        steps: Number of simulation time points.
        xi: Spatial positions. Defaults to [0, 20, 40, 60, 80].
        t_start: Start time in minutes.
        t_end: End time in minutes.
        decay_constant: Exponential decay constant in 1/min.
        base_error: Constant error for original signal.
    """
    xi = xi or [0, 20, 40, 60, 80]
    t = np.linspace(t_start, t_end, steps)

    sim = CadetContainer(
        "M01",
        "model.h5",
        xi=xi,
        t=t,
        gauss_type="stretched",
    )
    data = sim.simulate([5.0, 20, 30, 5, 3])

    signal = data[:, 0]
    decay_factor = np.exp(-decay_constant * t)

    y_orig = signal / np.max(signal)
    y_corr = (signal / decay_factor) / np.max(signal / decay_factor)

    yerr_orig = np.full(steps, base_error)
    yerr_corr = base_error / decay_factor

    plt.errorbar(
        t,
        y_orig,
        marker="o",
        linestyle="none",
        yerr=yerr_orig,
        markerfacecolor="none",
        label="Original data",
    )
    plt.errorbar(
        t,
        y_corr,
        marker="s",
        linestyle="none",
        yerr=yerr_corr,
        markerfacecolor="none",
        label="Decay corrected data",
    )

    plt.xlabel("Time [min]")
    plt.ylabel("Activity [a.u.]")
    plt.ylim([-0.25, 1.2])
    plt.legend()
    
    save_figure("figures/figure_2", "fig2_decay_correction")


def main() -> None:
    """Run the plot."""
    plot_decay_correction_comparison()


if __name__ == "__main__":
    main()