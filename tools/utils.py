from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Lanzrath_2025


def save_figure(
    folder: str,
    name: str,
    ext: str = "png",
    dpi: int = 300,
) -> None:
    """Save the current matplotlib figure inside the Lanzrath package directory.

    Args:
        folder: Relative path from the package root (e.g. ``"figures/figure_2"``).
        name: File stem (without extension).
        ext: File extension. Defaults to ``"png"``.
        dpi: Resolution in dots per inch. Defaults to 300.
    """
    base_path = Path(Lanzrath_2025.__file__).resolve().parent
    path = base_path / folder
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")


def save_figure_local(save_folder: str | Path, filename: str, dpi: int = 300) -> None:
    """Save the current matplotlib figure to a local path.

    Args:
        save_folder: Destination directory (created if it does not exist).
        filename: File stem (without extension). A ``.png`` extension is appended.
        dpi: Resolution in dots per inch. Defaults to 300.
    """
    ensure_dir(save_folder)
    out = Path(save_folder) / f"{filename}.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")


def ensure_dir(path: str | Path) -> None:
    """Create a directory and all parents if they do not already exist.

    Args:
        path: Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    """Convert a plant or file name to a safe ASCII lowercase string.

    Replaces German umlauts and spaces so the result can be used as a filename.

    Args:
        name: Input string (may contain umlauts or spaces).

    Returns:
        Sanitized lowercase string suitable for use as a filename.
    """
    return (
        name.lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace(" ", "_")
    )


def add_gaussian_noise(data: np.ndarray, noise_std: float) -> np.ndarray:
    """Add zero-mean Gaussian noise to an array.

    Args:
        data: Input data array.
        noise_std: Standard deviation of the noise.

    Returns:
        Array of the same shape as ``data`` with added noise.
    """
    noise = np.random.normal(0.0, noise_std, data.size).reshape(data.shape)
    return data + noise


def infer_xi_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric xi (position) values from a DataFrame's column names.

    Columns whose names can be parsed as floats are treated as spatial positions.

    Args:
        df: DataFrame whose column names encode spatial positions.

    Returns:
        Array of parsed position values in column order.
    """
    xi = []
    for col in df.columns:
        try:
            xi.append(float(str(col).strip()))
        except ValueError:
            continue
    return np.array(xi, dtype=float)
