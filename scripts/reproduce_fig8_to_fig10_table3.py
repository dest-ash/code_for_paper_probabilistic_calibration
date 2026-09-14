#!/usr/bin/env python3
"""
Generate Figures 8, 9(a-b), and 10 and the numerical results reported in Table 3.

This script runs a reproducible simultaneous radiocarbon calibration experiment
for two measurements. It:

1. performs joint calibration with the BNN calibration curve using MCMC;
2. generates trace plots for the two MCMC components (Figure 8a);
3. reconstructs and plots the unnormalized joint target density (Figure 8b);
4. performs the corresponding individual calibrations;
5. compares individual posterior densities with the marginals of the joint
   calibration distribution (Figures 9a and 9b);
6. compares the distributions of calibrated-date differences (Figure 10);
7. computes dependence measures after discretizing calibrated dates to integer
   years (Table 3);
8. writes numerical summaries and execution metadata for reproducibility auditing.

Recommended environment
-----------------------
Python 3.9.13 and a compatible installation of `bnn_for_14C_calibration`.

Example
-------
python scripts/reproduce_fig8_to_fig10_table3.py --output-dir outputs/outputs_fig8_to_fig10_table3 --seed 1234 --dpi 300

For interactive display in addition to saving the figures:
python scripts/reproduce_fig8_to_fig10_table3.py --output-dir outputs/outputs_fig8_to_fig10_table3 --seed 1234 --dpi 300 --show
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

# Set TensorFlow determinism-related environment variables before importing
# libraries that may import TensorFlow internally.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

# Use a non-GUI backend for unattended runs. Interactive mode is enabled only
# when --show is explicitly requested.
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mutual_info_score

if "--show" not in sys.argv:
    plt.ioff()

import bnn_for_14C_calibration as bnn
from bnn_for_14C_calibration.bnn_models_built_in_utils import (
    bnn_make_predictions_,
)
from bnn_for_14C_calibration.utils import (
    d14c_to_f14c,
    minimax_scaling_reciproque,
)


# ---------------------------------------------------------------------------
# Experimental settings
# ---------------------------------------------------------------------------

DEFAULT_SEED = 1234
DEFAULT_DPI = 300
CM_TO_INCH = 1.0 / 2.54
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "outputs_fig8_to_fig10_table3"

C14_AGES = np.array([9647, 10033])
C14_SIGMAS = np.array([27, 33])
KNOWN_CALENDAR_DATES = np.array([11100, 11635])

ALPHA = 0.05
BURN_IN = 10_000
CHAIN_LENGTH = 160_000
TARGET_THINNED_SAMPLE_SIZE = 30_000
THINNING_STEP = int((CHAIN_LENGTH - BURN_IN) / TARGET_THINNED_SAMPLE_SIZE)

THEORETICAL_DENSITY_BNN_DRAWS = 500
CALIBRATION_MIN_AGE = -4
CALIBRATION_MAX_AGE = 12_310

GRID_SIZE = 100
GRID_PADDING = 10

TRACE_FIGSIZE = (17 * CM_TO_INCH, 9 * CM_TO_INCH)
COMPARISON_FIGSIZE = (38 * CM_TO_INCH, 12 * CM_TO_INCH)
JOINT_DENSITY_FIGSIZE = (8, 6)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figures 8–10 and Table 3 for the simultaneous "
            "radiocarbon calibration experiment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and numerical outputs are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution in dots per inch (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow when available.

    Exact bitwise reproducibility can still depend on software versions,
    hardware, BLAS implementations, and package internals.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass


def package_version(package_name: str) -> str:
    """Return an installed package version, or 'unknown' if unavailable."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def save_figure(fig: plt.Figure, path: Path, dpi: int, show: bool) -> None:
    """Save a Matplotlib figure and close it in non-interactive mode."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_json_serializable(obj: Any) -> Any:
    """Convert NumPy and pandas objects recursively to JSON-compatible objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return {
            "index": obj.index.tolist(),
            "columns": obj.columns.tolist(),
            "data": obj.to_numpy().tolist(),
        }
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def compact_individual_calibration_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the individual-calibration quantities useful for auditing results."""
    keys = [
        "calage_posterior_mode",
        "calage_posterior_mean",
        "calage_posterior_std",
        "connexe_HPD_intervals_unscaled_round",
        "HPD_region_length",
        "alpha",
    ]
    return {
        key: make_json_serializable(results[key])
        for key in keys
        if key in results
    }


# ---------------------------------------------------------------------------
# Joint posterior reconstruction
# ---------------------------------------------------------------------------

def build_joint_posterior_density(
    measurements: np.ndarray,
    lab_errors: np.ndarray,
    bnn_model: object,
    min_age: float,
    max_age: float,
    nb_curves: int = THEORETICAL_DENSITY_BNN_DRAWS,
    batch_size: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct an unnormalized joint posterior density for calibrated dates.

    The dates supplied to the returned function must be scaled to [0, 1].
    An independent uniform prior over [0, 1] is used for each date. The BNN
    predictive expectation is approximated by Monte Carlo averaging.

    Parameters
    ----------
    measurements : np.ndarray
        Radiocarbon measurements in the F14C domain, shape (n_dates,).
    lab_errors : np.ndarray
        Laboratory standard deviations in the F14C domain, shape (n_dates,).
    bnn_model : object
        Trained BNN calibration-curve model predicting Delta14C.
    min_age, max_age : float
        Calendar-age limits used to reverse minimax scaling.
    nb_curves : int
        Number of BNN predictive draws used in the Monte Carlo approximation.
    batch_size : int or None
        Optional prediction batch size.

    Returns
    -------
    callable
        Function evaluating the unnormalized joint posterior density.
    """
    measurements = np.asarray(measurements, dtype=float)
    lab_errors = np.asarray(lab_errors, dtype=float)
    n_dates = measurements.shape[0]

    measurements_broadcasted = measurements.repeat(nb_curves).reshape(
        n_dates, nb_curves
    )
    errors_broadcasted = lab_errors.repeat(nb_curves).reshape(
        n_dates, nb_curves
    )

    def prior_density(dates_scaled: np.ndarray) -> np.ndarray:
        dates_scaled = np.asarray(dates_scaled)
        inside = (0.0 <= dates_scaled) & (dates_scaled <= 1.0)
        return inside.astype(np.float64).prod(axis=1)

    def predict(dates_scaled: np.ndarray) -> np.ndarray:
        predictions = bnn_make_predictions_(
            bnn_model=bnn_model,
            X_test=dates_scaled.reshape((-1, 1)),
            iterations=nb_curves,
            batch_size=batch_size,
        )
        return predictions.reshape((-1, n_dates, nb_curves))

    def density(dates_scaled: np.ndarray) -> np.ndarray:
        dates_scaled = np.asarray(dates_scaled, dtype=float)

        calendar_ages = minimax_scaling_reciproque(
            x=dates_scaled.reshape((-1, 1)),
            Max=max_age,
            Min=min_age,
        )
        calendar_ages = calendar_ages.repeat(nb_curves).reshape(
            (-1, n_dates, nb_curves)
        )

        predicted_f14c = d14c_to_f14c(
            d14c=predict(dates_scaled),
            teta=calendar_ages,
        )

        likelihood = np.exp(
            -(
                measurements_broadcasted - predicted_f14c
            ) ** 2
            / (2 * errors_broadcasted**2)
        ).prod(axis=1, dtype=np.float64).mean(axis=1, dtype=np.float64)

        normalizing_term = (
            lab_errors.prod() * np.sqrt(2 * np.pi) ** n_dates
        )
        return prior_density(dates_scaled) * likelihood / normalizing_term

    return density


# ---------------------------------------------------------------------------
# Dependence measures for Table 3
# ---------------------------------------------------------------------------

def normalized_discrete_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute normalized mutual information for discrete variables.

    NMI(X, Y) = MI(X, Y) / min(MI(X, X), MI(Y, Y)).
    For discrete variables, MI(X, X) and MI(Y, Y) equal the corresponding
    entropies.
    """
    mi_xy = mutual_info_score(x, y)
    mi_xx = mutual_info_score(x, x)
    mi_yy = mutual_info_score(y, y)

    if mi_xx == 0 or mi_yy == 0:
        return float("nan")

    return float(mi_xy / min(mi_xx, mi_yy))


def compute_table3(
    first_individual_sample: np.ndarray,
    second_individual_sample: np.ndarray,
    joint_sample: np.ndarray,
) -> pd.DataFrame:
    """
    Compute correlation, mutual information, and normalized mutual information.

    Before calculation, all calibrated-date samples are discretized by taking
    their floor, so dates are represented as integer calendar years.
    """
    x_individual = np.floor(
        np.asarray(first_individual_sample)
    ).astype(int)
    y_individual = np.floor(
        np.asarray(second_individual_sample)
    ).astype(int)

    joint_sample = np.floor(np.asarray(joint_sample)).astype(int)
    x_joint = joint_sample[:, 0]
    y_joint = joint_sample[:, 1]

    values = {
        "individual_calibration_samples": [
            np.corrcoef(x_individual, y_individual)[0, 1],
            mutual_info_score(x_individual, y_individual),
            normalized_discrete_mutual_information(
                x_individual, y_individual
            ),
        ],
        "joint_calibration_samples": [
            np.corrcoef(x_joint, y_joint)[0, 1],
            mutual_info_score(x_joint, y_joint),
            normalized_discrete_mutual_information(x_joint, y_joint),
        ],
    }

    return pd.DataFrame(
        values,
        index=[
            "correlation",
            "mutual information",
            "normalized mutual information",
        ],
    )


# ---------------------------------------------------------------------------
# Numerical experiment
# ---------------------------------------------------------------------------

def run_joint_calibration() -> Dict[str, Any]:
    """Run simultaneous calibration of the two radiocarbon measurements."""
    return bnn.joint_calibration(
        c14ages=C14_AGES,
        c14sigs=C14_SIGMAS,
        alpha=ALPHA,
        compute_calage_posterior_mode=True,
        compute_calage_posterior_mean_and_std=True,
        chaine_length=CHAIN_LENGTH,
    )


def thin_joint_chain(chain: np.ndarray) -> np.ndarray:
    """Discard burn-in and thin the MCMC chain to the target sample size."""
    return chain[:, range(BURN_IN, CHAIN_LENGTH, THINNING_STEP)]


def estimate_joint_kde(
    chain_thinned: np.ndarray,
) -> Tuple[gaussian_kde, np.ndarray, np.ndarray]:
    """
    Estimate the joint density by Gaussian KDE and build the grid required later.

    Scott's bandwidth rule is used. Only quantities required by Figures 8b,
    9a, and 9b or by the reproducibility audit are computed here.
    """
    kde = gaussian_kde(chain_thinned, bw_method="scott")

    mins = chain_thinned.min(axis=1) - GRID_PADDING
    maxs = chain_thinned.max(axis=1) + GRID_PADDING

    xx = np.linspace(mins[0], maxs[0], GRID_SIZE)
    yy = np.linspace(mins[1], maxs[1], GRID_SIZE)
    x_grid, y_grid = np.meshgrid(xx, yy)

    return kde, x_grid, y_grid


def reconstruct_theoretical_joint_density(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the unnormalized BNN joint target density on the plotting grid.
    """
    bnn_model = bnn.bnn_models_built_in.bnn_load_model_part_1()

    density_scaled = build_joint_posterior_density(
        measurements=bnn.utils.c14_to_f14c(c14=C14_AGES),
        lab_errors=bnn.utils.c14sig_to_f14csig(
            c14=C14_AGES,
            c14sig=C14_SIGMAS,
        ),
        bnn_model=bnn_model,
        max_age=CALIBRATION_MAX_AGE,
        min_age=CALIBRATION_MIN_AGE,
        nb_curves=THEORETICAL_DENSITY_BNN_DRAWS,
        batch_size=None,
    )

    def density_unscaled(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dates = np.column_stack([x, y])
        dates_scaled = bnn.utils.minimax_scaling(
            dates,
            Min=CALIBRATION_MIN_AGE,
            Max=CALIBRATION_MAX_AGE,
        )
        return density_scaled(dates_scaled)

    return density_unscaled(
        x_grid.ravel(),
        y_grid.ravel(),
    ).reshape(x_grid.shape)


def run_individual_calibrations(
    sample_size: int,
) -> list[Dict[str, Any]]:
    """Calibrate the two measurements independently."""
    results = []

    for c14_age, c14_sigma in zip(C14_AGES, C14_SIGMAS):
        results.append(
            bnn.individual_calibration(
                c14age=c14_age,
                c14sig=c14_sigma,
                compute_calage_posterior_mean_and_std=True,
                sample_size=sample_size,
                alpha=ALPHA,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure_8a(
    chain: np.ndarray,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """Generate Figure 8a: traces of the two MCMC components."""
    colors = ["blue", "orange"]
    n_components = chain.shape[0]

    fig, axs = plt.subplots(
        nrows=n_components,
        ncols=1,
        figsize=TRACE_FIGSIZE,
        sharex=True,
        squeeze=False,
    )

    iterations = range(chain.shape[1])
    for i in range(n_components):
        ax = axs[i, 0]
        ax.plot(iterations, chain[i, :], color=colors[i])
        ax.grid()
        ax.set_title(f"Dim {i + 1} of the chain")

    fig.tight_layout()
    save_figure(fig, output_dir / "Fig8a.png", dpi, show)


def generate_figure_8b(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theoretical_density: np.ndarray,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """Generate Figure 8b: contour map of the unnormalized joint target density."""
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=JOINT_DENSITY_FIGSIZE,
        squeeze=True,
    )

    contour = ax.contourf(
        x_grid,
        y_grid,
        theoretical_density,
        levels=15,
        cmap="viridis",
    )
    fig.colorbar(
        contour,
        ax=ax,
        label="Unnormalized joint density values",
    )

    ax.scatter(
        KNOWN_CALENDAR_DATES[0],
        KNOWN_CALENDAR_DATES[1],
        marker="d",
        s=10,
        alpha=0.5,
        color="red",
        label="Known true calendar dates",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(10850, 11200)
    ax.set_ylim(11400, 11850)
    ax.grid(True)
    ax.set_xlabel(
        "Calibrated ages (cal BP) for first measurement"
    )
    ax.set_ylabel(
        "Calibrated ages (cal BP) for second measurement"
    )
    ax.set_title(
        "Simultaneous calibration:\n"
        "Unnormalized joint density contour lines"
    )
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir / "Fig8b.png", dpi, show)


def add_individual_posterior_density(
    calibration_results: Dict[str, Any],
    ax: plt.Axes,
    color: str = "cyan",
    eps: float = 1e-7,
) -> None:
    """
    Add the normalized individual calibrated-date posterior density to an axis.

    The normalization accounts for the different grid spacings used on the two
    parts of the calibration horizon.
    """
    max_part_1 = 12_310
    min_part_1 = -4
    max_part_2 = 55_000
    min_part_2 = 12_310

    middle_points = calibration_results["middle_points"]
    middle_points_density = calibration_results["middle_points_density"]

    idx_part_1 = np.where(middle_points < max_part_1)[0]
    idx_part_2 = np.where(middle_points > max_part_1)[0]

    delta_part_1 = (
        (max_part_1 - min_part_1) / idx_part_1.shape[0]
    )
    delta_part_2 = (
        (max_part_2 - min_part_2) / idx_part_2.shape[0]
    )

    normalization_factor = (
        delta_part_1 * middle_points_density[idx_part_1].sum()
        + delta_part_2 * middle_points_density[idx_part_2].sum()
    )
    density = middle_points_density / normalization_factor

    visible_idx = np.where(density > eps)[0]
    min_age = middle_points[visible_idx].min()
    max_age = middle_points[visible_idx].max()

    ax.plot(
        middle_points,
        density,
        label="posterior density of\nthe calibrated date",
        color=color,
    )
    ax.set_ylabel("Probability")
    ax.set_xlabel("calibrated dates (in years cal BP)")
    ax.set_xlim(min_age, max_age)
    ax.invert_xaxis()


def generate_figure_9(
    individual_results: list[Dict[str, Any]],
    chain_thinned: np.ndarray,
    joint_kde: gaussian_kde,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """
    Generate Figures 9a and 9b.

    Each panel compares an individual calibration density with the corresponding
    marginal density obtained from the joint-calibration sample.
    """
    colors = ["blue", "orange"]

    for i in range(2):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=COMPARISON_FIGSIZE,
            squeeze=True,
        )

        ax.hist(
            individual_results[i]["calage_sample"],
            color="cyan",
            alpha=0.3,
            density=True,
            bins="auto",
            label=(
                "histogram obtained from sampling\n"
                "individual calibration density"
            ),
        )

        add_individual_posterior_density(
            calibration_results=individual_results[i],
            ax=ax,
        )

        ax.hist(
            chain_thinned[i, :],
            color=colors[i],
            alpha=0.5,
            density=True,
            bins="auto",
            label=(
                f"Dim {i + 1} of the chain: histogram\n"
                "obtained from joint density sample"
            ),
        )

        xbound = ax.get_xbound()
        xbound = (
            min(xbound[0], chain_thinned[i, :].min()),
            max(xbound[1], chain_thinned[i, :].max()),
        )
        ax.set_xbound(xbound)

        x_marginal = np.linspace(xbound[0], xbound[1], 200)
        y_marginal = joint_kde.marginal(i)(x_marginal)

        ax.plot(
            x_marginal,
            y_marginal,
            lw=2,
            color=colors[i],
            label=(
                f"Dim {i + 1} of the chain:\n"
                "Marginal density estimated\n"
                "from the Gaussian KDE of the joint density"
            ),
        )

        ax.set_title(
            f"Dim {i + 1}: Individual calibration density\n"
            "vs\n"
            "Marginalized joint density"
        )
        ax.legend(
            loc="center right",
            fancybox=True,
            framealpha=0.4,
            facecolor="gray",
            bbox_to_anchor=(1.8, 0.4),
        )

        fig.tight_layout()
        save_figure(
            fig,
            output_dir / f"Fig9{'a' if i == 0 else 'b'}.png",
            dpi,
            show,
        )


def generate_figure_10(
    individual_results: list[Dict[str, Any]],
    chain_thinned: np.ndarray,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """
    Generate Figure 10: distributions of calibrated-date differences.
    """
    colors = ["green", "blue"]

    individual_differences = (
        individual_results[1]["calage_sample"]
        - individual_results[0]["calage_sample"]
    )
    joint_differences = chain_thinned[1, :] - chain_thinned[0, :]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=COMPARISON_FIGSIZE,
        squeeze=True,
    )

    ax.hist(
        individual_differences,
        color=colors[0],
        alpha=0.6,
        density=True,
        bins="auto",
        label=(
            "Histogram of sample differences\n"
            "from individual calibration densities"
        ),
    )

    ax.hist(
        joint_differences,
        color=colors[1],
        alpha=0.6,
        density=True,
        bins="auto",
        label=(
            "Histogram of sample differences\n"
            "from the joint density"
        ),
    )

    true_difference = (
        KNOWN_CALENDAR_DATES[1] - KNOWN_CALENDAR_DATES[0]
    )
    ax.plot(
        [true_difference] * 2,
        [*ax.get_ybound()],
        color="red",
        label="Marker showing the true date difference",
    )

    ax.set_xlabel("Date differences")
    ax.set_title(
        "Comparison of sample-difference distributions from\n"
        "individual calibration densities and the joint density"
    )
    ax.legend(
        loc="center right",
        fancybox=True,
        framealpha=0.4,
        facecolor="gray",
        bbox_to_anchor=(1.8, 0.4),
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "Fig10.png", dpi, show)


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def write_table3(table3: pd.DataFrame, output_dir: Path) -> None:
    """Write Table 3 values to CSV and JSON files."""
    table3.to_csv(output_dir / "Table3.csv")

    (output_dir / "Table3.json").write_text(
        json.dumps(make_json_serializable(table3), indent=2),
        encoding="utf-8",
    )


def write_numerical_results(
    output_dir: Path,
    joint_results: Dict[str, Any],
    chain_thinned: np.ndarray,
    joint_kde: gaussian_kde,
    individual_results: list[Dict[str, Any]],
    table3: pd.DataFrame,
) -> None:
    """Write compact numerical summaries of the experiment."""
    individual_differences = (
        individual_results[1]["calage_sample"]
        - individual_results[0]["calage_sample"]
    )
    joint_differences = chain_thinned[1, :] - chain_thinned[0, :]

    results = {
        "inputs": {
            "c14_ages": C14_AGES,
            "c14_sigmas": C14_SIGMAS,
            "known_calendar_dates_cal_BP": KNOWN_CALENDAR_DATES,
        },
        "mcmc": {
            "alpha": ALPHA,
            "chain_length": CHAIN_LENGTH,
            "burn_in": BURN_IN,
            "thinning_step": THINNING_STEP,
            "thinned_sample_size": chain_thinned.shape[1],
            "acceptance_rate": joint_results.get("acceptance_rate"),
            "marginal_acceptance_rates": joint_results.get(
                "marginal_acceptance_rates"
            ),
            "posterior_mode_cal_BP": joint_results.get(
                "calage_posterior_mode"
            ),
            "posterior_mean_cal_BP": joint_results.get(
                "calage_posterior_mean"
            ),
            "posterior_std_years": joint_results.get(
                "calage_posterior_std"
            ),
        },
        "joint_kde": {
            "bandwidth_rule": "scott",
            "bandwidth_factor": joint_kde.factor,
            "covariance": joint_kde.covariance,
        },
        "individual_calibrations": [
            compact_individual_calibration_summary(result)
            for result in individual_results
        ],
        "difference_variances": {
            "individual_calibration_samples": np.var(
                individual_differences
            ),
            "joint_calibration_samples": np.var(joint_differences),
            "sum_individual_marginal_variances": (
                np.var(individual_results[0]["calage_sample"])
                + np.var(individual_results[1]["calage_sample"])
            ),
            "sum_joint_marginal_variances": (
                np.var(chain_thinned[0, :])
                + np.var(chain_thinned[1, :])
            ),
        },
        "table3": table3,
    }

    (output_dir / "numerical_results_fig8_to_fig10_table3.json").write_text(
        json.dumps(make_json_serializable(results), indent=2),
        encoding="utf-8",
    )


def write_run_metadata(
    output_dir: Path,
    seed: int,
    dpi: int,
    elapsed_seconds: float,
) -> None:
    """Write privacy-safe execution metadata for reproducibility auditing."""
    metadata_dict = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "seed": seed,
        "dpi": dpi,
        "elapsed_seconds": elapsed_seconds,
        "bnn_for_14C_calibration_version": package_version(
            "bnn_for_14C_calibration"
        ),
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "matplotlib_version": package_version("matplotlib"),
        "scipy_version": package_version("scipy"),
        "scikit_learn_version": package_version("scikit-learn"),
        "tensorflow_version": package_version("tensorflow"),
        "tensorflow_probability_version": package_version(
            "tensorflow-probability"
        ),
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_dict, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete simultaneous-calibration experiment."""
    args = parse_args()
    start_time = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    print("Running joint calibration...")
    mcmc_start = time.time()
    joint_results = run_joint_calibration()
    mcmc_elapsed = time.time() - mcmc_start
    print(
        f"Joint-calibration MCMC completed in "
        f"{mcmc_elapsed:.2f} s ({mcmc_elapsed / 60:.2f} min)."
    )

    chain = joint_results["chaine"]

    print("Generating Figure 8a...")
    generate_figure_8a(
        chain,
        args.output_dir,
        args.dpi,
        args.show,
    )

    chain_thinned = thin_joint_chain(chain)
    print(
        f"Thinned MCMC sample shape: {chain_thinned.shape}; "
        f"thinning step: {THINNING_STEP}."
    )

    print("Estimating the joint Gaussian KDE...")
    joint_kde, x_grid, y_grid = estimate_joint_kde(
        chain_thinned
    )

    print("Reconstructing the unnormalized joint target density...")
    density_start = time.time()
    theoretical_density = reconstruct_theoretical_joint_density(
        x_grid,
        y_grid,
    )
    density_elapsed = time.time() - density_start
    print(
        f"Joint-density reconstruction completed in "
        f"{density_elapsed:.2f} s."
    )

    print("Generating Figure 8b...")
    generate_figure_8b(
        x_grid,
        y_grid,
        theoretical_density,
        args.output_dir,
        args.dpi,
        args.show,
    )

    print("Running individual calibrations...")
    individual_results = run_individual_calibrations(
        sample_size=chain_thinned.shape[1]
    )

    print("Generating Figures 9a and 9b...")
    generate_figure_9(
        individual_results,
        chain_thinned,
        joint_kde,
        args.output_dir,
        args.dpi,
        args.show,
    )

    print("Generating Figure 10...")
    generate_figure_10(
        individual_results,
        chain_thinned,
        args.output_dir,
        args.dpi,
        args.show,
    )

    print("Computing Table 3...")
    table3 = compute_table3(
        first_individual_sample=individual_results[0]["calage_sample"],
        second_individual_sample=individual_results[1]["calage_sample"],
        joint_sample=chain_thinned.T,
    )
    print(table3)

    write_table3(table3, args.output_dir)
    write_numerical_results(
        args.output_dir,
        joint_results,
        chain_thinned,
        joint_kde,
        individual_results,
        table3,
    )

    elapsed = time.time() - start_time
    write_run_metadata(
        args.output_dir,
        args.seed,
        args.dpi,
        elapsed,
    )

    print(
        "\nCompleted successfully.\n"
        f"Output directory: {args.output_dir.resolve()}\n"
        f"Elapsed time: {elapsed:.2f} s ({elapsed / 60:.2f} min)\n"
    )


if __name__ == "__main__":
    main()
