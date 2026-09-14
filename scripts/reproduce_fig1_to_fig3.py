#!/usr/bin/env python3
"""Reproduce Figures 1--3 and the simulation experiments associated with Figure 3.

The workflow follows the scientific sequence

    simulated data -> Bayesian curve estimation -> calibration -> coverage experiment

and writes publication figures together with machine-readable numerical results and
run metadata. By default the script uses a non-interactive Matplotlib backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Deterministic TensorFlow settings must be defined before importing TensorFlow.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
from scipy.optimize import minimize

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
from tensorflow import keras

from bnn_for_14C_calibration.bnn_models_built_in import bnn_reg_model
from bnn_for_14C_calibration.bnn_models_built_in_utils import (
    bnn_make_predictions_,
    gaussian_prior,
    independent_gaussian_posterior,
)
from bnn_for_14C_calibration.calibration_utils import (
    compute_HPD_regions,
    mono_cal_date_exact_approx_quantile_fct,
    optimise_credible_interval,
)
from bnn_for_14C_calibration.utils import minimax_scaling_reciproque

CM_TO_INCH = 1.0 / 2.54
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "outputs_fig1_to_fig3"
DOMAIN_MIN = -20.0
DOMAIN_MAX = 280.0
DATA_RNG_SEED = 2022
ALPHA = 0.05
CALIBRATION_TRUE_VALUE = 155.0
CALIBRATION_LAB_ERROR = 4.0
COVERAGE_SAMPLE_SIZES = (100, 10_000, 100_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and JSON audit outputs are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_RNG_SEED,
        help="Base seed for model fitting and posterior BNN draws. Defaults to 2022. The simulated-data RNG remains fixed at 2022.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Synchronize Python/NumPy/TensorFlow random states for one stochastic stage."""
    # Available in TensorFlow 2.9 and later. It seeds Python, NumPy and TensorFlow.
    tf.keras.utils.set_random_seed(seed)


def array_sha256(array: np.ndarray) -> str:
    """Return a stable SHA-256 digest for a NumPy array including shape and dtype."""
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def arrays_sha256(arrays: list[np.ndarray]) -> str:
    """Return one SHA-256 digest for an ordered collection of NumPy arrays."""
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(np.asarray(array))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def true_calibration_function(x: np.ndarray, a: float = DOMAIN_MIN, b: float = DOMAIN_MAX) -> np.ndarray:
    """Evaluate the monotone synthetic calibration function used in the experiment."""
    x = np.asarray(x)
    g1 = (b - a) * x + a
    g2 = (
        (b - a)
        * (
            np.sin(2.0 * np.pi * x) ** 2
            + np.cos((1.0 - 2.0 * x) * np.pi / 2.0)
            + np.cos(2.0 * np.pi * x) ** 2 * x**3
        )
        / 3.0
        + a
    )
    return (g1 + g2) / 2.0


def generate_simulated_data(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate training, validation and test data with heteroscedastic Gaussian noise."""
    train_size, val_size, test_size = 10_000, 5_000, 5_000
    mu, sigma = 0.0, 10.0
    sigma_mult_min, sigma_mult_max = 1.0, 2.0

    x_train = rng.random(train_size)
    x_val = rng.random(val_size)
    x_test = rng.random(test_size)

    y_train = true_calibration_function(x_train)
    y_val = true_calibration_function(x_val)
    y_test = true_calibration_function(x_test)

    def measurement_sd(y: np.ndarray, size: int) -> np.ndarray:
        return sigma + (
            (sigma_mult_max - sigma_mult_min) * rng.random(size) + sigma_mult_min
        ) * np.sqrt(np.abs(y))

    s_train = measurement_sd(y_train, train_size)
    s_val = measurement_sd(y_val, val_size)
    s_test = measurement_sd(y_test, test_size)

    Y_train = y_train + mu + s_train * rng.standard_normal(train_size)
    Y_val = y_val + mu + s_val * rng.standard_normal(val_size)
    Y_test = y_test + mu + s_test * rng.standard_normal(test_size)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "s_train": s_train,
        "s_val": s_val,
        "s_test": s_test,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test,
    }


def save_figure(fig: plt.Figure, path: Path, dpi: int, show: bool) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def make_figure_1(data: dict[str, np.ndarray], output_path: Path, dpi: int, show: bool) -> None:
    """Create Figure 1: synthetic calibration curve and error-prone datasets."""
    fig, ax = plt.subplots(figsize=(20 * CM_TO_INCH, 15 * CM_TO_INCH))
    ax.scatter(
        minimax_scaling_reciproque(data["x_train"], DOMAIN_MAX, DOMAIN_MIN),
        data["Y_train"],
        label="training set data",
        s=0.5,
        color="red",
        alpha=0.5,
    )
    ax.scatter(
        minimax_scaling_reciproque(data["x_val"], DOMAIN_MAX, DOMAIN_MIN),
        data["Y_val"],
        label="validation set data",
        s=0.5,
        color="blue",
        alpha=0.5,
    )
    ax.scatter(
        minimax_scaling_reciproque(data["x_test"], DOMAIN_MAX, DOMAIN_MIN),
        data["Y_test"],
        label="test set data",
        s=0.5,
        color="green",
        alpha=0.5,
    )

    x_all = np.concatenate((data["x_train"], data["x_val"], data["x_test"]))
    order = np.argsort(x_all)
    ax.plot(
        minimax_scaling_reciproque(x_all[order], DOMAIN_MAX, DOMAIN_MIN),
        true_calibration_function(x_all[order]),
        color="black",
        alpha=0.9,
        label="true curve of function g",
    )
    ax.set_title("Graph of the function g and the generated data", fontsize=15)
    ax.set_ylabel("error-prone measurements M", fontsize=16)
    ax.set_xlabel("expected true values d", fontsize=18)
    ax.invert_xaxis()
    ax.legend(fontsize="large")
    fig.tight_layout()
    save_figure(fig, output_path, dpi, show)


def build_and_fit_bnn(data: dict[str, np.ndarray], seed: int) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Fit the hybrid Bayesian neural network used for curve estimation."""
    set_global_seed(seed)
    x_fit = np.concatenate((data["x_train"], data["x_val"])).reshape((-1, 1))
    y_fit = np.concatenate((data["Y_train"], data["Y_val"]))
    s_fit = np.concatenate((data["s_train"], data["s_val"]))

    architecture = {
        "hidden_layers": 3,
        "neurons_per_hidden_layer": [20, 260, 240],
        "hidden_bias": [True, True, False],
        "last_bias": False,
        "hybrid": True,
        "last_hybrid": True,
        "kl_use_exact": False,
        "learning_rate": 0.001,
        "batch_size": 50,
        "max_epochs": 400,
        "early_stopping_monitor": "loss",
        "early_stopping_patience": 100,
        "early_stopping_min_delta": 0.0,
        "restore_best_weights": False,
    }

    model = bnn_reg_model(
        train_size=x_fit.shape[0],
        batch_size=architecture["batch_size"],
        prior=gaussian_prior,
        posterior=independent_gaussian_posterior,
        loss_fn=keras.losses.MeanSquaredError(),
        input_shape=1,
        nb_couches_cachees=architecture["hidden_layers"],
        neurones_par_couches=architecture["neurons_per_hidden_layer"],
        activation="relu",
        use_bias=architecture["hidden_bias"],
        dropout="default",
        last_bias=architecture["last_bias"],
        learning_rate=architecture["learning_rate"],
        hybrid=architecture["hybrid"],
        last_hybrid=architecture["last_hybrid"],
        kl_use_exact=architecture["kl_use_exact"],
        metrics=["mean_squared_error", "mean_absolute_error"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=architecture["early_stopping_monitor"],
        min_delta=architecture["early_stopping_min_delta"],
        patience=architecture["early_stopping_patience"],
        verbose=2,
        mode="min",
        restore_best_weights=architecture["restore_best_weights"],
    )

    # Use an explicitly seeded tf.data pipeline rather than Keras' implicit
    # NumPy shuffling. This preserves epoch-wise shuffling while making the
    # minibatch order reproducible across separate processes.
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_fit, y_fit, 1.0 / s_fit)
    ).shuffle(
        buffer_size=x_fit.shape[0],
        seed=seed,
        reshuffle_each_iteration=True,
    ).batch(architecture["batch_size"], drop_remainder=False)
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (
            data["x_test"].reshape((-1, 1)),
            data["Y_test"],
            1.0 / data["s_test"],
        )
    ).batch(architecture["batch_size"], drop_remainder=False)
    dataset_options = tf.data.Options()
    dataset_options.experimental_deterministic = True
    train_dataset = train_dataset.with_options(dataset_options)
    validation_dataset = validation_dataset.with_options(dataset_options)

    print(f"[BNN] Training started with seed={seed}.", flush=True)
    start = time.time()
    history = model.fit(
        train_dataset,
        epochs=architecture["max_epochs"],
        verbose=2,
        validation_data=validation_dataset,
        callbacks=[early_stopping],
    )
    elapsed = time.time() - start
    print(f"[BNN] Training completed in {elapsed:.2f} s.", flush=True)

    evaluation = model.evaluate(
        x=data["x_test"].reshape((-1, 1)),
        y=data["Y_test"],
        sample_weight=1.0 / data["s_test"],
        return_dict=True,
        verbose=0,
    )
    fit_summary = {
        "model_seed": seed,
        "model_weights_sha256": arrays_sha256(model.get_weights()),
        "training_seconds": elapsed,
        "epochs_completed": len(history.history.get("loss", [])),
        "final_training_loss": float(history.history["loss"][-1]),
    }
    evaluation = {key: float(value) for key, value in evaluation.items()}
    return model, evaluation, {"architecture": architecture, "fit": fit_summary}


def find_quantile_beta_opt(alpha: float, predictions: np.ndarray, method: str = "median_unbiased") -> float:
    """Find the shortest empirical credibility interval at one prediction location."""
    interval_length = lambda beta: (
        np.quantile(predictions, 1.0 - alpha + beta, method=method)
        - np.quantile(predictions, beta, method=method)
    )
    result = minimize(
        fun=interval_length,
        x0=np.array([alpha / 2.0]),
        method="Nelder-Mead",
        bounds=[(0.0, alpha)],
    )
    return float(result.x[0])


def compute_credible_interval_envelope(
    alpha: float, predictions: np.ndarray, method: str = "median_unbiased"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pointwise shortest empirical credibility intervals for BNN predictions."""
    lower, upper = [], []
    for row in predictions:
        beta = find_quantile_beta_opt(alpha, row, method=method)
        lower.append(np.quantile(row, beta, method=method))
        upper.append(np.quantile(row, 1.0 - alpha + beta, method=method))
    return np.asarray(lower), np.asarray(upper)


def calibration_density_on_grid(
    measurement: float,
    lab_error: float,
    middle_points_predictions: np.ndarray,
) -> np.ndarray:
    """Evaluate the approximate posterior calibration density on the fixed midpoint grid."""
    return np.exp(
        -((measurement - middle_points_predictions) ** 2) / (2.0 * lab_error**2)
    ).mean(axis=1, dtype=np.float64) / (lab_error * np.sqrt(2.0 * np.pi))


def calibration_setup(model: Any, seed: int) -> dict[str, np.ndarray]:
    """Pre-compute curve draws on the subdivision used by every calibration experiment."""
    nb_curves = 100
    nb_intervals = 1000
    intervals_bounds = np.linspace(0.0, 1.0, nb_intervals + 1)
    middle_points = (intervals_bounds[1:] + intervals_bounds[:-1]) / 2.0
    set_global_seed(seed)
    predictions = bnn_make_predictions_(
        bnn_model=model,
        X_test=middle_points.reshape((-1, 1)),
        iterations=nb_curves,
        batch_size=50,
    )
    return {
        "intervals_bounds": intervals_bounds,
        "middle_points": middle_points,
        "middle_points_predictions": predictions,
        "prediction_seed": seed,
        "predictions_sha256": array_sha256(predictions),
    }


def unscale(values: np.ndarray | float) -> np.ndarray:
    return np.asarray(minimax_scaling_reciproque(values, DOMAIN_MAX, DOMAIN_MIN))


def compute_single_calibration(
    measurement: float, grid: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Calibrate one error-prone measurement and return interval/HPD summaries."""
    density = calibration_density_on_grid(
        measurement,
        CALIBRATION_LAB_ERROR,
        grid["middle_points_predictions"],
    )
    subdivision = (grid["intervals_bounds"], grid["middle_points"], density)
    exact_quantile = mono_cal_date_exact_approx_quantile_fct(subdivision_components=subdivision)
    beta_result = optimise_credible_interval(quantile=exact_quantile, alpha=ALPHA)
    beta = float(beta_result.x[0])
    ci_scaled = np.array(
        [exact_quantile(alpha=beta), exact_quantile(alpha=1.0 - ALPHA + beta)]
    )

    hpd = compute_HPD_regions(alpha=ALPHA, subdivision_components=subdivision)
    hpd_scaled = np.asarray(hpd["connexe_HPD_intervals"], dtype=float)
    hpd_unscaled = unscale(hpd_scaled)

    return {
        "density": density,
        "credible_interval_scaled": ci_scaled,
        "credible_interval_unscaled": unscale(ci_scaled),
        "hpd_intervals_scaled": hpd_scaled,
        "hpd_intervals_unscaled": hpd_unscaled,
        "hpd_probability_mass": [float(x) for x in hpd["connexe_HPD_intervals_density"]],
        "hpd_threshold": float(hpd["HPD_threshold"]),
        "posterior_mode_scaled": float(hpd["calage_posterior_mode"]),
        "posterior_mode_unscaled": float(unscale(hpd["calage_posterior_mode"])),
    }


def make_figure_2(
    data: dict[str, np.ndarray],
    x_curve: np.ndarray,
    curve_mean: np.ndarray,
    credible_envelope: tuple[np.ndarray, np.ndarray],
    measurement: float,
    single_calibration: dict[str, Any],
    grid: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    """Create Figure 2: estimated curve, measurement distribution and posterior calibration density."""
    lower, upper = credible_envelope
    measure_grid = np.linspace(
        measurement - 5.0 * CALIBRATION_LAB_ERROR,
        measurement + 5.0 * CALIBRATION_LAB_ERROR,
        1000,
    )
    measure_density = np.exp(
        -((measure_grid - measurement) ** 2) / (2.0 * CALIBRATION_LAB_ERROR**2)
    ) / (CALIBRATION_LAB_ERROR * np.sqrt(2.0 * np.pi))

    density = single_calibration["density"]
    min_y = min(
        float(data["Y_test"].min()),
        float(true_calibration_function(x_curve).min()),
        float(lower.min()),
        float(measure_grid.min()),
    )
    max_y = max(
        float(data["Y_test"].max()),
        float(true_calibration_function(x_curve).max()),
        float(upper.max()),
        float(measure_grid.max()),
    )
    measure_peak = (measure_density / measure_density.sum()).max()
    posterior_peak = (density / density.sum()).max()
    posterior_mask = density / density.sum() > 1e-7

    fig, ax = plt.subplots(figsize=(20 * CM_TO_INCH, 15 * CM_TO_INCH))
    ax.fill_between(
        (1.0 - measure_density / (5.0 * measure_peak * measure_density.sum())) * DOMAIN_MAX,
        measurement,
        measure_grid,
        label=(
            f"Distribution of the measure {measurement:.2f} "
            f"with standard deviation {CALIBRATION_LAB_ERROR:.1f}"
        ),
        color="gray",
        alpha=0.3,
    )
    ax.plot(
        unscale(x_curve),
        curve_mean,
        label="estimated mean curve of g from Bayesian neural network",
        color="blue",
    )
    ax.plot(
        unscale(x_curve),
        true_calibration_function(x_curve),
        color="red",
        alpha=0.6,
        label="true curve of function g",
    )
    ax.fill_between(
        unscale(x_curve),
        lower,
        upper,
        label="95% level credibility envelope",
        color="blue",
        alpha=0.3,
    )
    ax.fill_between(
        unscale(grid["middle_points"][posterior_mask]),
        min_y,
        density[posterior_mask] / (5.0 * posterior_peak * density.sum()) * max_y + min_y,
        color="green",
        alpha=0.3,
        label=(
            "posterior distribution of the calibrated value\n"
            f"for the given measurement {measurement:.2f} "
            f"(true known value = {CALIBRATION_TRUE_VALUE:.1f})"
        ),
    )
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylabel("error-prone measurements M", fontsize=16)
    ax.set_xlabel("calibrated values d", fontsize=18)
    ax.invert_xaxis()
    ax.legend(fontsize="medium", loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_path, dpi, show)


def run_coverage_experiment(
    rng: np.random.Generator,
    sample_size: int,
    expected_measurement: float,
    grid: dict[str, np.ndarray],
    keep_segments: bool,
) -> dict[str, Any]:
    """Calibrate repeated measurements generated from one true feature value."""
    measurements = expected_measurement + CALIBRATION_LAB_ERROR * rng.standard_normal(sample_size)
    hits = 0
    intervals_count = 0
    line_segments: list[list[tuple[float, float]]] = []
    upper_caps: list[list[tuple[float, float]]] = []
    lower_caps: list[list[tuple[float, float]]] = []

    print(f"[Coverage] K={sample_size}: calibration started.", flush=True)
    start = time.time()
    progress_step = max(1, sample_size // 10)

    for k, measurement in enumerate(measurements, start=1):
        density = calibration_density_on_grid(
            float(measurement), CALIBRATION_LAB_ERROR, grid["middle_points_predictions"]
        )
        hpd = compute_HPD_regions(
            alpha=ALPHA,
            subdivision_components=(grid["intervals_bounds"], grid["middle_points"], density),
        )
        intervals = unscale(np.asarray(hpd["connexe_HPD_intervals"], dtype=float))
        intervals_count += len(intervals)
        contains_true_value = False

        for lower, upper in intervals:
            if keep_segments:
                line_segments.append([(k, float(lower)), (k, float(upper))])
                upper_caps.append([(k - 0.4, float(upper)), (k + 0.4, float(upper))])
                lower_caps.append([(k - 0.4, float(lower)), (k + 0.4, float(lower))])
            if lower <= CALIBRATION_TRUE_VALUE <= upper:
                contains_true_value = True

        hits += int(contains_true_value)
        if k % progress_step == 0 or k == sample_size:
            print(f"[Coverage] K={sample_size}: {k}/{sample_size} completed.", flush=True)

    elapsed = time.time() - start
    result: dict[str, Any] = {
        "sample_size": sample_size,
        "number_of_hpd_connected_intervals": intervals_count,
        "number_covering_true_value": hits,
        "coverage_rate": hits / sample_size,
        "coverage_percentage": 100.0 * hits / sample_size,
        "elapsed_seconds": elapsed,
    }
    if keep_segments:
        result["line_segments"] = line_segments
        result["upper_caps"] = upper_caps
        result["lower_caps"] = lower_caps

    print(
        f"[Coverage] K={sample_size}: {hits}/{sample_size} = "
        f"{100.0 * hits / sample_size:.2f}% in {elapsed:.2f} s.",
        flush=True,
    )
    return result


def make_figure_3(coverage_100: dict[str, Any], output_path: Path, dpi: int, show: bool) -> None:
    """Create Figure 3 from the K=100 repeated-calibration experiment."""
    segments = coverage_100["line_segments"]
    upper_caps = coverage_100["upper_caps"]
    lower_caps = coverage_100["lower_caps"]
    values = np.asarray(segments, dtype=float)[:, :, 1]
    K = coverage_100["sample_size"]

    fig, ax = plt.subplots()
    ax.add_collection(
        LineCollection(segments, linewidths=1, colors="blue", linestyle="solid", label="HPD intervals")
    )
    ax.add_collection(LineCollection(upper_caps, linewidths=1, colors="blue", linestyle="solid"))
    ax.add_collection(LineCollection(lower_caps, linewidths=1, colors="blue", linestyle="solid"))
    ax.plot(
        range(1, K + 1),
        [CALIBRATION_TRUE_VALUE] * K,
        color="red",
        alpha=0.6,
        label=f"true value = {CALIBRATION_TRUE_VALUE:.1f}",
    )
    ax.set_xlim(0, K + 1)
    ax.set_ylim(values.min() - 1.0, values.max() + 1.0)
    ax.set_title(f"HPD intervals for {K} new measurements generated\nwith the same true value")
    ax.set_xlabel("measurement index", fontsize=18)
    ax.set_ylabel("calibrated values", fontsize=18)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi, show)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.time()

    rng = np.random.default_rng(DATA_RNG_SEED)
    data = generate_simulated_data(rng)
    make_figure_1(data, args.output_dir / "Fig1.png", args.dpi, args.show)

    model_seed = args.seed
    curve_prediction_seed = args.seed + 1
    calibration_prediction_seed = args.seed + 2

    bnn_model, bnn_evaluation, bnn_audit = build_and_fit_bnn(data, seed=model_seed)

    x_curve = np.linspace(0.0, 1.0, 50)
    set_global_seed(curve_prediction_seed)
    curve_draws = bnn_make_predictions_(
        bnn_model=bnn_model,
        X_test=x_curve,
        iterations=100,
    )
    curve_mean = np.mean(curve_draws, axis=1).reshape((-1,))
    credible_envelope = compute_credible_interval_envelope(ALPHA, curve_draws)
    curve_rmse = float(
        np.sqrt(np.mean((curve_mean - true_calibration_function(x_curve)) ** 2))
    )

    grid = calibration_setup(bnn_model, seed=calibration_prediction_seed)
    true_scaled = (CALIBRATION_TRUE_VALUE - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)
    expected_measurement = float(true_calibration_function(np.array([true_scaled]))[0])
    measurement = float(expected_measurement + CALIBRATION_LAB_ERROR * rng.standard_normal())
    single_calibration = compute_single_calibration(measurement, grid)

    make_figure_2(
        data,
        x_curve,
        curve_mean,
        credible_envelope,
        measurement,
        single_calibration,
        grid,
        args.output_dir / "Fig2.png",
        args.dpi,
        args.show,
    )

    coverage_results = []
    for K in COVERAGE_SAMPLE_SIZES:
        coverage_results.append(
            run_coverage_experiment(
                rng=rng,
                sample_size=K,
                expected_measurement=expected_measurement,
                grid=grid,
                keep_segments=(K == 100),
            )
        )
    make_figure_3(coverage_results[0], args.output_dir / "Fig3.png", args.dpi, args.show)

    coverage_json = []
    for result in coverage_results:
        compact = {k: v for k, v in result.items() if k not in {"line_segments", "upper_caps", "lower_caps"}}
        coverage_json.append(compact)

    numerical_results = {
        "simulation": {
            "data_rng_seed": DATA_RNG_SEED,
            "training_data_sha256": arrays_sha256([data["x_train"], data["Y_train"], data["s_train"]]),
            "validation_data_sha256": arrays_sha256([data["x_val"], data["Y_val"], data["s_val"]]),
            "test_data_sha256": arrays_sha256([data["x_test"], data["Y_test"], data["s_test"]]),
            "train_size": len(data["x_train"]),
            "validation_size": len(data["x_val"]),
            "test_size": len(data["x_test"]),
            "domain": [DOMAIN_MIN, DOMAIN_MAX],
        },
        "bayesian_neural_network": {
            "evaluation_on_test_set": bnn_evaluation,
            "curve_rmse_against_true_function_on_50_point_grid": curve_rmse,
            "curve_prediction_seed": curve_prediction_seed,
            "curve_draws_sha256": array_sha256(curve_draws),
            "calibration_prediction_seed": calibration_prediction_seed,
            "calibration_grid_draws_sha256": grid["predictions_sha256"],
            **bnn_audit,
        },
        "single_calibration_example": {
            "true_feature_value": CALIBRATION_TRUE_VALUE,
            "expected_measurement": expected_measurement,
            "generated_measurement": measurement,
            "laboratory_standard_deviation": CALIBRATION_LAB_ERROR,
            "confidence_level": 1.0 - ALPHA,
            "credible_interval_scaled": single_calibration["credible_interval_scaled"],
            "credible_interval_unscaled": single_calibration["credible_interval_unscaled"],
            "hpd_intervals_scaled": single_calibration["hpd_intervals_scaled"],
            "hpd_intervals_unscaled": single_calibration["hpd_intervals_unscaled"],
            "hpd_probability_mass": single_calibration["hpd_probability_mass"],
            "posterior_mode_unscaled": single_calibration["posterior_mode_unscaled"],
        },
        "figure_3_coverage_experiments": coverage_json,
    }
    write_json(args.output_dir / "numerical_results_fig1_to_fig3.json", numerical_results)

    metadata = {
        "script": Path(__file__).name,
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
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
            "tensorflow": package_version("tensorflow"),
            "tensorflow_probability": package_version("tensorflow-probability"),
            "bnn_for_14C_calibration": package_version("bnn-for-14C-calibration"),
        },
        "reproducibility": {
            "data_rng_seed": DATA_RNG_SEED,
            "base_seed": args.seed,
            "model_seed": model_seed,
            "curve_prediction_seed": curve_prediction_seed,
            "calibration_prediction_seed": calibration_prediction_seed,
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "training_input_pipeline": "seeded_tf_data_shuffle_deterministic",
        },
        "figure_dpi": args.dpi,
        "elapsed_seconds": time.time() - run_start,
    }
    write_json(args.output_dir / "run_metadata.json", metadata)
    print(f"All outputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
