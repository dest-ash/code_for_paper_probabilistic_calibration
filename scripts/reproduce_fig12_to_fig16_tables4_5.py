#!/usr/bin/env python3
"""
Generate Figures 12–16 and Tables 4–5 for the Brehm et al. tree-ring experiment.

The script reproduces the complete published analysis in one execution:

1. load the two cleaned annual tree-ring datasets;
2. generate the chronology plot used in Figure 12;
3. independently calibrate every radiocarbon measurement with both IntCal20 and
   the Bayesian neural-network (BNN) calibration curve at 68% HPD level;
4. generate Figures 13 and 14 and the coverage results summarized in Table 4;
5. repeat the calibration at 95% HPD level;
6. generate Figures 15 and 16 and the coverage results summarized in Table 5;
7. write tables, compact numerical summaries, input-file hashes, and privacy-safe
   execution metadata for reproducibility auditing.

Expected input files
--------------------
The data directory must contain:

- donnees_traites_fig_a.csv  (first solar event)
- donnees_traites_fig_b.csv  (second solar event)

Each CSV must contain the columns ``calage``, ``c14age``, ``c14sig``, and
``tree``.

Recommended environment
-----------------------
Python 3.9.13 and a compatible installation of ``bnn_for_14C_calibration``.

Example
-------
python scripts/reproduce_fig12_to_fig16_tables4_5.py \
    --output-dir outputs/outputs_fig12_to_fig16_tables4_5 \
    --seed 1234 \
    --dpi 300

For interactive display in addition to saving the figures, add ``--show``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "--show" not in sys.argv:
    plt.ioff()

import bnn_for_14C_calibration as bnn


DEFAULT_SEED = 1234
DEFAULT_DPI = 300
CM_TO_INCH = 1.0 / 2.54
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "brehm"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "outputs_fig12_to_fig16_tables4_5"

EVENT_FILE_NAMES = (
    "donnees_traites_fig_a.csv",
    "donnees_traites_fig_b.csv",
)
EVENT_NAMES = ("first event", "second event")
REQUIRED_COLUMNS = {"calage", "c14age", "c14sig", "tree"}

ALPHA_68 = 1.0 - 0.68
ALPHA_95 = 0.05

FIGURE_12_SIZE = (29 * CM_TO_INCH, 29 * CM_TO_INCH)
CALIBRATION_FIGURE_SIZE = (58 * CM_TO_INCH, 43.5 * CM_TO_INCH)

TREE_COLORS = {
    "Alpine Larch": "blue",
    "Bristlecone Pine (USA)": "green",
    "German Oak": "orange",
    "Irish Oak": "yellow",
    "Sibirian Larch": "red",
}


# ---------------------------------------------------------------------------
# Command-line and general utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figures 12–16 and Tables 4–5 for the Brehm et al. "
            "tree-ring radiocarbon calibration experiment."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing donnees_traites_fig_a.csv and "
            "donnees_traites_fig_b.csv (default: <repo>/data/brehm)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures, tables, and audit outputs are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for each HPD-level experiment (default: {DEFAULT_SEED}).",
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
    """Set Python, NumPy, and TensorFlow random seeds when available."""
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
    """Return an installed package version, or ``unknown`` if unavailable."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_json_serializable(obj: Any) -> Any:
    """Convert NumPy and pandas objects recursively to JSON-compatible values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def save_figure(fig: plt.Figure, path: Path, dpi: int, show: bool) -> None:
    """Save a figure and close it unless interactive display was requested."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------


def load_brehm_datasets(data_dir: Path) -> Tuple[List[pd.DataFrame], List[Path]]:
    """Load and validate the two cleaned Brehm et al. event datasets."""
    paths = [data_dir / file_name for file_name in EVENT_FILE_NAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing required input file(s): {missing_names}. "
            "Expected location: <repo>/data/brehm. Use --data-dir only to override it."
        )

    datasets = []
    for path in paths:
        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        unknown_trees = sorted(set(frame["tree"]) - set(TREE_COLORS))
        if unknown_trees:
            raise ValueError(
                f"{path.name} contains tree labels without configured colors: "
                f"{unknown_trees}"
            )
        datasets.append(frame.copy())

    return datasets, paths


# ---------------------------------------------------------------------------
# Figure 12: chronology and calibration curves
# ---------------------------------------------------------------------------


def generate_figure_12(
    datasets: Sequence[pd.DataFrame],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """Generate Figure 12 in the radiocarbon-age domain."""
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FIGURE_12_SIZE,
        squeeze=False,
    )

    for event_index, dataset in enumerate(datasets):
        ax = axs[event_index, 0]

        for tree_name, group in dataset.groupby("tree"):
            group.plot.scatter(
                x="calage",
                y="c14age",
                yerr="c14sig",
                label=tree_name,
                ax=ax,
                color=TREE_COLORS[tree_name],
            )

        bnn.calib_plot_functions.add_IntCal20_curve(ax=ax, domaine="c14")
        bnn.calib_plot_functions.add_individual_calibration_curve_part_1(
            ax=ax,
            domaine="c14",
            color="b",
        )

        padding = 5
        x_min = dataset["calage"].min() - padding
        x_max = dataset["calage"].max() + padding
        y_min = (dataset["c14age"] - dataset["c14sig"]).min() - padding
        y_max = (dataset["c14age"] + dataset["c14sig"]).max() + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_xaxis()
        ax.set_xlabel("calendar dates in years BP")
        ax.set_ylabel(r"$^{14}$C ages in years BP")
        ax.grid(True)
        ax.legend(fontsize="small")

    save_figure(fig, output_dir / "Fig12.png", dpi, show)


# ---------------------------------------------------------------------------
# Calibration experiment
# ---------------------------------------------------------------------------


def compact_calibration_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only numerical quantities needed by the figures and audit outputs."""
    return {
        "posterior_mode": result["calage_posterior_mode"],
        "posterior_mean": result.get("calage_posterior_mean"),
        "posterior_std": result.get("calage_posterior_std"),
        "hpd_intervals": np.asarray(
            result["connexe_HPD_intervals_unscaled_round"], dtype=float
        ),
        "hpd_region_length": result.get("HPD_region_length"),
    }


def calibrate_measurement(c14age: float, c14sig: float, alpha: float) -> Dict[str, Dict[str, Any]]:
    """Calibrate one measurement with the BNN curve and IntCal20."""
    bnn_result = bnn.individual_calibration(
        c14age=c14age,
        c14sig=c14sig,
        alpha=alpha,
        compute_calage_posterior_mean_and_std=True,
    )
    intcal_result = bnn.IntCal20_calibration(
        c14age=c14age,
        c14sig=c14sig,
        alpha=alpha,
        compute_calage_posterior_mean_and_std=True,
    )
    return {
        "BNN": compact_calibration_result(bnn_result),
        "IntCal20": compact_calibration_result(intcal_result),
    }


def calibrate_datasets(
    datasets: Sequence[pd.DataFrame],
    alpha: float,
    level_label: str,
) -> List[pd.DataFrame]:
    """Calibrate all measurements and attach compact results to copied datasets."""
    calibrated = []

    for event_index, source in enumerate(datasets):
        frame = source.copy()
        total = len(frame)
        print(
            f"Starting {level_label} calibration for {EVENT_NAMES[event_index]} "
            f"({total} measurements)...",
            flush=True,
        )
        event_start = time.perf_counter()

        bnn_mode: List[float] = []
        bnn_mean: List[float] = []
        bnn_std: List[float] = []
        bnn_hpd: List[np.ndarray] = []
        bnn_hpd_length: List[float] = []

        intcal_mode: List[float] = []
        intcal_mean: List[float] = []
        intcal_std: List[float] = []
        intcal_hpd: List[np.ndarray] = []
        intcal_hpd_length: List[float] = []

        for position, (_, row) in enumerate(frame.iterrows(), start=1):
            result = calibrate_measurement(
                c14age=float(row["c14age"]),
                c14sig=float(row["c14sig"]),
                alpha=alpha,
            )

            bnn_summary = result["BNN"]
            intcal_summary = result["IntCal20"]

            bnn_mode.append(bnn_summary["posterior_mode"])
            bnn_mean.append(bnn_summary["posterior_mean"])
            bnn_std.append(bnn_summary["posterior_std"])
            bnn_hpd.append(bnn_summary["hpd_intervals"])
            bnn_hpd_length.append(bnn_summary["hpd_region_length"])

            intcal_mode.append(intcal_summary["posterior_mode"])
            intcal_mean.append(intcal_summary["posterior_mean"])
            intcal_std.append(intcal_summary["posterior_std"])
            intcal_hpd.append(intcal_summary["hpd_intervals"])
            intcal_hpd_length.append(intcal_summary["hpd_region_length"])

            if position % 25 == 0 or position == total:
                print(
                    f"  {EVENT_NAMES[event_index]}: {position}/{total} measurements completed",
                    flush=True,
                )

        frame["BNN_posterior_mode"] = bnn_mode
        frame["BNN_posterior_mean"] = bnn_mean
        frame["BNN_posterior_std"] = bnn_std
        frame["BNN_HPD_region"] = bnn_hpd
        frame["BNN_HPD_region_length"] = bnn_hpd_length

        frame["IntCal20_posterior_mode"] = intcal_mode
        frame["IntCal20_posterior_mean"] = intcal_mean
        frame["IntCal20_posterior_std"] = intcal_std
        frame["IntCal20_HPD_region"] = intcal_hpd
        frame["IntCal20_HPD_region_length"] = intcal_hpd_length

        elapsed = time.perf_counter() - event_start
        print(
            f"Completed {level_label} calibration for {EVENT_NAMES[event_index]} "
            f"in {elapsed:.2f} s.",
            flush=True,
        )
        calibrated.append(frame)

    return calibrated


def value_in_intervals(value: float, intervals: Iterable[Iterable[float]]) -> int:
    """Return 1 if ``value`` belongs to at least one closed interval."""
    for interval in intervals:
        bounds = list(interval)
        if len(bounds) != 2:
            raise ValueError("Each HPD interval must contain exactly two bounds.")
        lower, upper = bounds
        if lower <= value <= upper:
            return 1
    return 0


def add_coverage_columns(calibrated_datasets: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
    """Add boolean coverage indicators for IntCal20 and BNN HPD regions."""
    output = []
    for dataset in calibrated_datasets:
        frame = dataset.copy()
        frame["IntCal20_contains_calendar_date"] = [
            value_in_intervals(calage, intervals)
            for calage, intervals in zip(frame["calage"], frame["IntCal20_HPD_region"])
        ]
        frame["BNN_contains_calendar_date"] = [
            value_in_intervals(calage, intervals)
            for calage, intervals in zip(frame["calage"], frame["BNN_HPD_region"])
        ]
        output.append(frame)
    return output


# ---------------------------------------------------------------------------
# Figures 13–16
# ---------------------------------------------------------------------------


def plot_calibration_method(
    ax: plt.Axes,
    group: pd.DataFrame,
    tree_name: str,
    method: str,
) -> None:
    """Plot posterior modes and HPD intervals for one tree and one curve."""
    if method == "IntCal20":
        mode_column = "IntCal20_posterior_mode"
        hpd_column = "IntCal20_HPD_region"
        legend_label = f"IntCal20 HPD region for {tree_name}"
    elif method == "BNN":
        mode_column = "BNN_posterior_mode"
        hpd_column = "BNN_HPD_region"
        legend_label = f"BNN curve HPD region for {tree_name}"
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    interval_label_used = False
    for _, row in group.iterrows():
        x = float(row["calage"])
        y = float(row[mode_column])
        intervals = np.asarray(row[hpd_column], dtype=float)

        for lower, upper in intervals:
            kwargs: Dict[str, Any] = {
                "x": x,
                "y_min": lower,
                "y_max": upper,
                "ax": ax,
                "color": TREE_COLORS[tree_name],
            }
            if not interval_label_used:
                kwargs["label"] = legend_label
                interval_label_used = True
            bnn.utils.ajoute_segment_vertical(**kwargs)

        ax.scatter(
            x,
            y,
            label=f"_{method} {tree_name} mode",
            color=TREE_COLORS[tree_name],
        )

    x_min, x_max = ax.get_xlim()
    diagonal_x = np.linspace(x_min, x_max, 100)
    ax.plot(diagonal_x, diagonal_x, label="y = x")
    ax.invert_xaxis()
    ax.set_xlabel("calendar dates in years BP")
    ax.set_ylabel("calibrated dates in years BP")
    ax.grid(True)
    ax.legend(fontsize="medium")


def generate_calibration_figure(
    dataset: pd.DataFrame,
    figure_number: int,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """Generate one event-level comparison figure for IntCal20 and BNN."""
    tree_names = sorted(dataset["tree"].unique().tolist())
    fig, axs = plt.subplots(
        nrows=len(tree_names),
        ncols=2,
        figsize=CALIBRATION_FIGURE_SIZE,
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    for row_index, tree_name in enumerate(tree_names):
        group = dataset.loc[dataset["tree"] == tree_name]
        plot_calibration_method(axs[row_index, 0], group, tree_name, "IntCal20")
        plot_calibration_method(axs[row_index, 1], group, tree_name, "BNN")

    save_figure(fig, output_dir / f"Fig{figure_number}.png", dpi, show)


# ---------------------------------------------------------------------------
# Tables 4 and 5
# ---------------------------------------------------------------------------


def proportion_standard_error(successes: int, sample_size: int) -> float:
    """Return the binomial plug-in standard error of a sample proportion."""
    if sample_size <= 0:
        return float("nan")
    proportion = successes / sample_size
    return float(np.sqrt(proportion * (1.0 - proportion) / sample_size))


def build_coverage_table(calibrated_datasets: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Build tree-level, event-level, and overall HPD coverage summaries."""
    rows: List[Dict[str, Any]] = []

    for event_index, dataset in enumerate(calibrated_datasets):
        for tree_name, group in dataset.groupby("tree"):
            n = len(group)
            intcal_count = int(group["IntCal20_contains_calendar_date"].sum())
            bnn_count = int(group["BNN_contains_calendar_date"].sum())
            rows.append(
                {
                    "event": EVENT_NAMES[event_index],
                    "tree": tree_name,
                    "sample_size": n,
                    "IntCal20_count": intcal_count,
                    "IntCal20_percentage": int(round(100.0 * intcal_count / n)),
                    "IntCal20_standard_error": proportion_standard_error(intcal_count, n),
                    "BNN_count": bnn_count,
                    "BNN_percentage": int(round(100.0 * bnn_count / n)),
                    "BNN_standard_error": proportion_standard_error(bnn_count, n),
                }
            )

        n = len(dataset)
        intcal_count = int(dataset["IntCal20_contains_calendar_date"].sum())
        bnn_count = int(dataset["BNN_contains_calendar_date"].sum())
        rows.append(
            {
                "event": EVENT_NAMES[event_index],
                "tree": "sub-total",
                "sample_size": n,
                "IntCal20_count": intcal_count,
                "IntCal20_percentage": int(round(100.0 * intcal_count / n)),
                "IntCal20_standard_error": proportion_standard_error(intcal_count, n),
                "BNN_count": bnn_count,
                "BNN_percentage": int(round(100.0 * bnn_count / n)),
                "BNN_standard_error": proportion_standard_error(bnn_count, n),
            }
        )

    combined = pd.concat(calibrated_datasets, ignore_index=True)
    n = len(combined)
    intcal_count = int(combined["IntCal20_contains_calendar_date"].sum())
    bnn_count = int(combined["BNN_contains_calendar_date"].sum())
    rows.append(
        {
            "event": "the two events",
            "tree": "total",
            "sample_size": n,
            "IntCal20_count": intcal_count,
            "IntCal20_percentage": int(round(100.0 * intcal_count / n)),
            "IntCal20_standard_error": proportion_standard_error(intcal_count, n),
            "BNN_count": bnn_count,
            "BNN_percentage": int(round(100.0 * bnn_count / n)),
            "BNN_standard_error": proportion_standard_error(bnn_count, n),
        }
    )

    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, table_number: int, output_dir: Path) -> None:
    """Write one coverage table as CSV and JSON."""
    table.to_csv(output_dir / f"Table{table_number}.csv", index=False)
    (output_dir / f"Table{table_number}.json").write_text(
        json.dumps(make_json_serializable(table), indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Audit outputs
# ---------------------------------------------------------------------------


def build_level_audit(
    calibrated_datasets: Sequence[pd.DataFrame],
    table: pd.DataFrame,
    alpha: float,
) -> Dict[str, Any]:
    """Build compact calibration summaries for one HPD level."""
    event_summaries = []
    for event_index, dataset in enumerate(calibrated_datasets):
        event_summaries.append(
            {
                "event": EVENT_NAMES[event_index],
                "sample_size": len(dataset),
                "trees": {
                    str(tree): int(count)
                    for tree, count in dataset["tree"].value_counts().sort_index().items()
                },
                "mean_posterior_mode_cal_BP": {
                    "IntCal20": float(dataset["IntCal20_posterior_mode"].mean()),
                    "BNN": float(dataset["BNN_posterior_mode"].mean()),
                },
                "mean_HPD_region_length_years": {
                    "IntCal20": float(dataset["IntCal20_HPD_region_length"].mean()),
                    "BNN": float(dataset["BNN_HPD_region_length"].mean()),
                },
            }
        )

    return {
        "alpha": alpha,
        "nominal_HPD_level": 1.0 - alpha,
        "event_summaries": event_summaries,
        "coverage_table": table,
    }


def write_numerical_results(
    output_dir: Path,
    datasets: Sequence[pd.DataFrame],
    input_paths: Sequence[Path],
    calibrated_68: Sequence[pd.DataFrame],
    table4: pd.DataFrame,
    calibrated_95: Sequence[pd.DataFrame],
    table5: pd.DataFrame,
) -> None:
    """Write a compact JSON audit summary for Figures 12–16 and Tables 4–5."""
    summary = {
        "input_data": [
            {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "rows": len(dataset),
                "columns": dataset.columns.tolist(),
            }
            for path, dataset in zip(input_paths, datasets)
        ],
        "figure_12": {
            "domain": "c14",
            "calendar_date_ranges_cal_BP": [
                [float(dataset["calage"].min()), float(dataset["calage"].max())]
                for dataset in datasets
            ],
        },
        "HPD_68_percent": build_level_audit(calibrated_68, table4, ALPHA_68),
        "HPD_95_percent": build_level_audit(calibrated_95, table5, ALPHA_95),
    }
    (output_dir / "numerical_results_fig12_to_fig16_tables4_5.json").write_text(
        json.dumps(make_json_serializable(summary), indent=2),
        encoding="utf-8",
    )


def write_run_metadata(
    output_dir: Path,
    seed: int,
    dpi: int,
    elapsed_seconds: float,
) -> None:
    """Write privacy-safe execution metadata."""
    metadata_payload = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "matplotlib": package_version("matplotlib"),
            "tensorflow": package_version("tensorflow"),
            "tensorflow-probability": package_version("tensorflow-probability"),
            "bnn_for_14C_calibration": package_version("bnn_for_14C_calibration"),
        },
        "seed": seed,
        "dpi": dpi,
        "elapsed_seconds": elapsed_seconds,
        "determinism_environment": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_level_experiment(
    raw_datasets: Sequence[pd.DataFrame],
    alpha: float,
    level_label: str,
    figure_numbers: Tuple[int, int],
    table_number: int,
    output_dir: Path,
    seed: int,
    dpi: int,
    show: bool,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Run one complete HPD-level experiment and generate its figures/table."""
    set_global_seed(seed)
    calibrated = calibrate_datasets(raw_datasets, alpha=alpha, level_label=level_label)
    calibrated = add_coverage_columns(calibrated)

    generate_calibration_figure(
        calibrated[0], figure_numbers[0], output_dir, dpi, show
    )
    generate_calibration_figure(
        calibrated[1], figure_numbers[1], output_dir, dpi, show
    )

    table = build_coverage_table(calibrated)
    write_table(table, table_number, output_dir)
    return calibrated, table


def main() -> None:
    """Run the complete Figure 12–16 and Table 4–5 reproduction workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    raw_datasets, input_paths = load_brehm_datasets(args.data_dir)

    generate_figure_12(raw_datasets, args.output_dir, args.dpi, args.show)

    calibrated_68, table4 = run_level_experiment(
        raw_datasets=raw_datasets,
        alpha=ALPHA_68,
        level_label="68% HPD",
        figure_numbers=(13, 14),
        table_number=4,
        output_dir=args.output_dir,
        seed=args.seed,
        dpi=args.dpi,
        show=args.show,
    )

    calibrated_95, table5 = run_level_experiment(
        raw_datasets=raw_datasets,
        alpha=ALPHA_95,
        level_label="95% HPD",
        figure_numbers=(15, 16),
        table_number=5,
        output_dir=args.output_dir,
        seed=args.seed,
        dpi=args.dpi,
        show=args.show,
    )

    write_numerical_results(
        args.output_dir,
        raw_datasets,
        input_paths,
        calibrated_68,
        table4,
        calibrated_95,
        table5,
    )

    elapsed_seconds = time.perf_counter() - start_time
    write_run_metadata(args.output_dir, args.seed, args.dpi, elapsed_seconds)

    print(
        f"Completed Figures 12–16 and Tables 4–5 in {elapsed_seconds:.2f} s. "
        f"Outputs written to {args.output_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
