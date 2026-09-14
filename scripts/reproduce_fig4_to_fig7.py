#!/usr/bin/env python3
"""
Generate Figures 4, 5, 6(a-b), and 7(a-b) for the radiocarbon calibration experiments.

Purpose
-------
This script provides a reproducible, documented workflow that:

1. generates Figure 4 (BNN vs IntCal20 calibration curves),
2. generates Figure 5 (68% and 95% BNN credible intervals),
3. calibrates the first radiocarbon measurement and generates Figures 6a/6b,
4. calibrates the second radiocarbon measurement using the full BNN predictive
   distribution and its Gaussian approximation, and generates Figures 7a/7b,
5. writes compact numerical calibration summaries and run metadata.

Recommended environment
-----------------------
Python 3.9.13 and a compatible installation of `bnn_for_14C_calibration`.

Example
-------
python scripts/reproduce_fig4_to_fig7.py --output-dir outputs/outputs_fig4_to_fig7 --seed 1234 --dpi 300

For interactive mode with plots when allowed, try
python scripts/reproduce_fig4_to_fig7.py --output-dir outputs/outputs_fig4_to_fig7 --seed 1234 --dpi 300 --show
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
from typing import Any, Dict

# Set TensorFlow determinism-related environment variables before importing
# libraries that may import TensorFlow internally.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

# In non-interactive mode (the default), force a non-GUI Matplotlib backend.
# This prevents package-level plotting calls from opening Tkinter/TkAgg windows
# and blocking batch execution. Passing --show leaves the user's normal
# interactive backend unchanged.
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if "--show" not in sys.argv:
    plt.ioff()

import bnn_for_14C_calibration as bnn


# ---------------------------------------------------------------------------
# Reproducibility settings
# ---------------------------------------------------------------------------

DEFAULT_SEED = 1234
DEFAULT_DPI = 300
CM_TO_INCH = 1.0 / 2.54
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "outputs_fig4_to_fig7"

CURVE_COLORS = {
    "BNN_part_1": "blue",
    "IntCal20": "green",
}

# Experimental inputs.
FIG6_C14_AGE = 8977
FIG6_C14_SIGMA = 31
FIG6_KNOWN_TRUE_DATE = 10183  # reference calendar date, years cal BP

FIG7_C14_AGE = 4001
FIG7_C14_SIGMA = 23.4
FIG7_KNOWN_TRUE_DATE = 4471  # reference calendar date, years cal BP

SAMPLE_SIZE = 10_000





def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Figures 4–7 for the radiocarbon calibration experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and reports are written.",
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

    Notes
    -----
    Exact bitwise reproducibility can still depend on the TensorFlow version,
    CPU/GPU implementation, BLAS backend, and the internals of the
    `bnn_for_14C_calibration` package.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Available in newer TensorFlow versions; harmlessly skipped otherwise.
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
    """Save a Matplotlib figure with publication-oriented defaults."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_json_serializable(obj: Any) -> Any:
    """Convert NumPy objects recursively so they can be written as JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def compact_calibration_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract calibration quantities used to summarize posterior results.

    The complete object returned by the package can be large. This function
    keeps posterior modes and highest posterior density (HPD) summaries.
    """
    keys = [
        "calage_posterior_mode",
        "calage_posterior_mode_density",
        "connexe_HPD_intervals_density",
        "HPD_threshold",
        "connexe_HPD_intervals_unscaled",
        "connexe_HPD_intervals_unscaled_round",
        "HPD_region_length",
        "alpha",
    ]
    return {
        key: make_json_serializable(results[key])
        for key in keys
        if key in results
    }


def generate_figure_4(output_dir: Path, dpi: int, show: bool) -> None:
    """Generate Figure 4: BNN and IntCal20 curves in three radiocarbon domains."""
    figsize = (29 * CM_TO_INCH, 29 * CM_TO_INCH)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize, squeeze=False)

    domains = ["c14", "f14c", "delta14c"]
    y_labels = [
        r"$^{14}$C ages in years BP",
        r"F$^{14}$C",
        r"$\Delta^{14}$C",
    ]

    for i, domain in enumerate(domains):
        ax = axs[i, 0]
        ax.margins(0)

        bnn.calib_plot_functions.add_individual_calibration_curve_part_1(
            ax=ax,
            domaine=domain,
            color=CURVE_COLORS["BNN_part_1"],
        )

        # Preserve the plotting limits defined by the BNN curve.
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        bnn.calib_plot_functions.add_IntCal20_curve(
            ax=ax,
            domaine=domain,
            color=CURVE_COLORS["IntCal20"],
            alpha=0.4,
        )

        ax.set_xlabel("calendar dates in years BP")
        ax.set_ylabel(y_labels[i])
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.grid()
        ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir / "Fig4.png", dpi, show)


def generate_figure_5(output_dir: Path, dpi: int, show: bool) -> None:
    """Generate Figure 5: 68% and 95% BNN credible intervals."""
    figsize = (29 * CM_TO_INCH / 2, 29 * CM_TO_INCH)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, squeeze=False)

    sigmas = [1, 2]
    levels = [0.68, 0.95]

    for i, (sigma_length, level) in enumerate(zip(sigmas, levels)):
        ax = axs[i, 0]
        bnn.calib_plot_functions.add_individual_calibration_curve_part_1(
            ax=ax,
            color=CURVE_COLORS["BNN_part_1"],
            sigma_length=sigma_length,
            credible_interval=True,
            credible_interval_level=level,
            credible_color="r",
            credible_alpha=0.2,
        )

        ax.set_xlabel("calendar dates in years BP")
        ax.set_ylabel(r"$\Delta^{14}$C")
        ax.grid()
        ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir / "Fig5.png", dpi, show)


def compute_figure_6_calibrations() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the two calibrations required for Figures 6a and 6b."""
    bnn_results = bnn.individual_calibration(
        FIG6_C14_AGE,
        FIG6_C14_SIGMA,
        sample_size=SAMPLE_SIZE,
        compute_calage_posterior_mean_and_std=True,
    )

    intcal20_results = bnn.IntCal20_calibration(
        FIG6_C14_AGE,
        FIG6_C14_SIGMA,
        sample_size=SAMPLE_SIZE,
        compute_calage_posterior_mean_and_std=True,
    )

    return bnn_results, intcal20_results


def generate_figure_6(
    bnn_results: Dict[str, Any],
    intcal20_results: Dict[str, Any],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """
    Generate Figures 6a and 6b.

    Figure 6b is explicitly rescaled to the same horizontal limits as Figure 6a
    to facilitate direct visual comparison.
    """
    figsize = (17 * CM_TO_INCH, 17 * CM_TO_INCH)
    gridspec_kw = {
        "width_ratios": [1, 3],
        "height_ratios": [3, 1],
    }

    fig_a, axs_a = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
    )
    bnn.plot_calib_results(
        figsize=figsize,
        fig=fig_a,
        axs=axs_a,
        calibration_results=bnn_results,
    )
    fig_a.tight_layout()
    fig_a.savefig(output_dir / "Fig6a.png", dpi=dpi, bbox_inches="tight")

    fig_b, axs_b = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
    )
    bnn.plot_calib_results(
        figsize=figsize,
        fig=fig_b,
        axs=axs_b,
        calibration_results=intcal20_results,
    )

    # Use common horizontal limits to facilitate direct comparison.
    axs_b[0, 1].set_xlim(axs_a[0, 1].get_xlim())
    axs_b[1, 1].set_xlim(axs_a[1, 1].get_xlim())

    fig_b.tight_layout()
    fig_b.savefig(output_dir / "Fig6b.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_a)
        plt.close(fig_b)


def compute_figure_7_calibrations() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the full-BNN and Gaussian-approximation calibrations for Figure 7."""
    full_bnn_results = bnn.individual_calibration(
        FIG7_C14_AGE,
        FIG7_C14_SIGMA,
        sample_size=SAMPLE_SIZE,
        compute_calage_posterior_mean_and_std=True,
    )

    gaussian_approx_results = bnn.individual_calibration(
        FIG7_C14_AGE,
        FIG7_C14_SIGMA,
        sample_size=SAMPLE_SIZE,
        mesure_likelihood="curve_gaussian_approximation",
        compute_calage_posterior_mean_and_std=True,
    )

    return full_bnn_results, gaussian_approx_results


def generate_density_hpd_figure(
    calibration_results: Dict[str, Any],
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    """
    Generate and save a calibrated-date density with its 95% HPD region.

    The package plotting function uses a Matplotlib figure created before
    using the package plotting routine.
    """
    figsize = (29 * CM_TO_INCH / 2, 29 * CM_TO_INCH / 2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    bnn.calib_plot_functions.add_cal_date_density_plot_and_HPD_region(
        calibration_results=calibration_results,
        ax = ax,
        plot_HPD_bounds=True,
    )
    fig.tight_layout()
    save_figure(fig, output_path, dpi, show)


def generate_figure_7(
    full_bnn_results: Dict[str, Any],
    gaussian_approx_results: Dict[str, Any],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """Generate Figures 7a and 7b."""
    generate_density_hpd_figure(
        full_bnn_results,
        output_dir / "Fig7a.png",
        dpi,
        show,
    )
    generate_density_hpd_figure(
        gaussian_approx_results,
        output_dir / "Fig7b.png",
        dpi,
        show,
    )


def write_run_metadata(
    output_dir: Path, seed: int, dpi: int, elapsed_seconds: float
) -> None:
    """Write environment and execution metadata for reproducibility auditing."""
    metadata_dict = {
        # Keep reproducibility-relevant Python information without recording
        # the absolute interpreter path, which may expose local usernames or
        # environment directory names.
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "seed": seed,
        "dpi": dpi,
        "sample_size": SAMPLE_SIZE,
        "elapsed_seconds": elapsed_seconds,
        "bnn_for_14C_calibration_version": package_version(
            "bnn_for_14C_calibration"
        ),
        "numpy_version": package_version("numpy"),
        "matplotlib_version": package_version("matplotlib"),
        "tensorflow_version": package_version("tensorflow"),
        "tensorflow_probability_version": package_version(
            "tensorflow-probability"
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_dict, indent=2),
        encoding="utf-8",
    )


def write_numerical_results(
    output_dir: Path,
    fig6_bnn: Dict[str, Any],
    fig6_intcal20: Dict[str, Any],
    fig7_full: Dict[str, Any],
    fig7_gaussian: Dict[str, Any],
) -> None:
    """Write numerical calibration summaries to JSON."""
    results = {
        "figure_6": {
            "input": {
                "c14age": FIG6_C14_AGE,
                "c14sig": FIG6_C14_SIGMA,
                "known_true_date_cal_BP": FIG6_KNOWN_TRUE_DATE,
                "sample_size": SAMPLE_SIZE,
            },
            "BNN_curve": compact_calibration_summary(fig6_bnn),
            "IntCal20_curve": compact_calibration_summary(fig6_intcal20),
        },
        "figure_7": {
            "input": {
                "c14age": FIG7_C14_AGE,
                "c14sig": FIG7_C14_SIGMA,
                "known_true_date_cal_BP": FIG7_KNOWN_TRUE_DATE,
                "sample_size": SAMPLE_SIZE,
            },
            "unsummarized_BNN_curve": compact_calibration_summary(fig7_full),
            "Gaussian_approximation": compact_calibration_summary(fig7_gaussian),
        },
    }
    (output_dir / "calibration_results_fig6_fig7.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete Figure 4–7 generation workflow."""
    args = parse_args()
    start_time = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    print("Generating Figure 4...")
    generate_figure_4(args.output_dir, args.dpi, args.show)

    print("Generating Figure 5...")
    generate_figure_5(args.output_dir, args.dpi, args.show)

    print("Computing Figure 6 calibrations...")
    fig6_bnn, fig6_intcal20 = compute_figure_6_calibrations()
    print("Generating Figures 6a and 6b...")
    generate_figure_6(
        fig6_bnn,
        fig6_intcal20,
        args.output_dir,
        args.dpi,
        args.show,
    )

    print("Computing Figure 7 calibrations...")
    fig7_full, fig7_gaussian = compute_figure_7_calibrations()
    print("Generating Figures 7a and 7b...")
    generate_figure_7(
        fig7_full,
        fig7_gaussian,
        args.output_dir,
        args.dpi,
        args.show,
    )

    write_numerical_results(
        args.output_dir,
        fig6_bnn,
        fig6_intcal20,
        fig7_full,
        fig7_gaussian,
    )
    elapsed = time.time() - start_time
    write_run_metadata(args.output_dir, args.seed, args.dpi, elapsed)

    print(
        "\nCompleted successfully.\n"
        f"Output directory: {args.output_dir.resolve()}\n"
        f"Elapsed time: {elapsed:.2f} s ({elapsed / 60:.2f} min)\n"
    )


if __name__ == "__main__":
    main()
