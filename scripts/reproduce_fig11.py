#!/usr/bin/env python3
"""
Generate Figure 11: parallel MCMC chains for simultaneous radiocarbon calibration.

The experiment runs ten independent joint-calibration MCMC chains for two
radiocarbon measurements and plots both chain components for convergence
assessment. In addition to the figure, the script writes compact numerical
summaries and privacy-safe execution metadata for reproducibility auditing.

Example
-------
python scripts/reproduce_fig11.py --output-dir outputs/outputs_fig11 --seed 1234 --dpi 300

For interactive display in addition to saving the figure:
python scripts/reproduce_fig11.py --output-dir outputs/outputs_fig11 --seed 1234 --dpi 300 --show
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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if "--show" not in sys.argv:
    plt.ioff()

import bnn_for_14C_calibration as bnn


DEFAULT_SEED = 1234
DEFAULT_DPI = 300
CM_TO_INCH = 1.0 / 2.54
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "outputs_fig11"

ALPHA = 0.05
CHAIN_LENGTH = 10_000
N_CHAINS = 10

C14_AGES = np.array([5089, 7228])
C14_SIGMAS = np.array([16.9, 31.68])
KNOWN_CALENDAR_DATES = np.array([5800.0, 8004.0])

FIGURE_SIZE = (30 * CM_TO_INCH, 7 * CM_TO_INCH * N_CHAINS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 11 from ten parallel MCMC joint-calibration chains."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure and audit outputs are written.",
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
        help="Display the figure interactively in addition to saving it.",
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


def make_json_serializable(obj: Any) -> Any:
    """Convert NumPy objects recursively to JSON-compatible objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def run_parallel_calibrations() -> list[Dict[str, Any]]:
    """Run the ten joint-calibration chains used in Figure 11."""
    results = []

    for chain_index in range(N_CHAINS):
        print(f"Running chain {chain_index + 1}/{N_CHAINS}...")
        result = bnn.joint_calibration(
            c14ages=C14_AGES,
            c14sigs=C14_SIGMAS,
            alpha=ALPHA,
            chaine_length=CHAIN_LENGTH,
            compute_calage_posterior_mode=True,
            compute_calage_posterior_mean_and_std=True,
        )
        results.append(result)

    return results


def generate_figure_11(
    calibration_results: list[Dict[str, Any]],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """Generate Figure 11 from the two components of the ten MCMC chains."""
    colors = ["blue", "orange"]

    fig, axs = plt.subplots(
        nrows=N_CHAINS,
        ncols=2,
        figsize=FIGURE_SIZE,
        sharex="col",
        sharey="col",
        squeeze=False,
    )

    for i, result in enumerate(calibration_results):
        chain = np.asarray(result["chaine"])
        iterations = range(chain.shape[1])

        ax = axs[i, 0]
        ax.plot(iterations, chain[0, :], color=colors[0])
        ax.grid()
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(f"Dim 1 of the chain : iteration {i + 1}")

        ax = axs[i, 1]
        ax.plot(iterations, chain[1, :], color=colors[1])
        ax.grid()
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(f"Dim 2 of the chain : iteration {i + 1}")

    figure_path = output_dir / "Fig11.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_audit_summary(
    calibration_results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build compact numerical summaries useful for reproducibility auditing."""
    chains = [np.asarray(result["chaine"]) for result in calibration_results]

    per_chain = []
    for i, (chain, result) in enumerate(zip(chains, calibration_results), start=1):
        per_chain.append(
            {
                "chain": i,
                "shape": chain.shape,
                "mean_cal_BP": chain.mean(axis=1),
                "median_cal_BP": np.median(chain, axis=1),
                "acceptance_rate": result.get("acceptance_rate"),
                "marginal_acceptance_rates": result.get(
                    "marginal_acceptance_rates"
                ),
                "posterior_mode_cal_BP": result.get("calage_posterior_mode"),
                "posterior_mean_cal_BP": result.get("calage_posterior_mean"),
                "posterior_std_years": result.get("calage_posterior_std"),
            }
        )

    stacked = np.stack(chains, axis=0)

    return {
        "inputs": {
            "c14_ages": C14_AGES,
            "c14_sigmas": C14_SIGMAS,
            "known_calendar_dates_cal_BP": KNOWN_CALENDAR_DATES,
            "alpha": ALPHA,
            "n_chains": N_CHAINS,
            "chain_length": CHAIN_LENGTH,
        },
        "per_chain": per_chain,
        "across_chain_means": {
            "mean_of_chain_means_cal_BP": stacked.mean(axis=2).mean(axis=0),
            "std_of_chain_means_years": stacked.mean(axis=2).std(axis=0),
        },
    }


def write_audit_summary(
    output_dir: Path,
    calibration_results: list[Dict[str, Any]],
) -> None:
    """Write compact numerical results used to audit Figure 11 generation."""
    summary = build_audit_summary(calibration_results)
    (output_dir / "figure11_results.json").write_text(
        json.dumps(make_json_serializable(summary), indent=2),
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


def main() -> None:
    """Run the complete Figure 11 experiment and write reproducibility outputs."""
    args = parse_args()
    start_time = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    calibration_results = run_parallel_calibrations()

    print("Generating Figure 11...")
    generate_figure_11(
        calibration_results,
        args.output_dir,
        args.dpi,
        args.show,
    )

    write_audit_summary(args.output_dir, calibration_results)

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
