# Reproducibility package for the paper "Probabilistic calibration of one-dimensional non-linear functions using Bayesian neural networks and variational inference."

This repository contains the executable experiments used to reproduce the numerical results, tables, and publication figures associated with the manuscript.

The repository is organized around standalone Python scripts.

## Scientific software dependency

The central software dependency is [`bnn-for-14c-calibration`](https://pypi.org/project/bnn-for-14c-calibration/) version **0.2.1**. That package provides the reusable Bayesian neural-network and radiocarbon-calibration implementation. This repository contains the experiment definitions, parameter choices, numerical analyses, audit outputs, and figure/table generation used for the article.

Figures 1--3 use generic Bayesian neural-network modelling components from the package on simulated data. The remaining experiments use the package more directly for radiocarbon calibration analyses.

## Reference environment

The reference environment is:

-   Python **3.9.13**
-   `bnn-for-14c-calibration==0.2.1`

The repository is agnostic to the Python environment manager. Use any isolated environment mechanism you prefer (`venv`, Conda, another virtual-environment tool, etc.).

Install the scientific dependency with:

``` bash
python -m pip install -r requirements.txt
```

The transitive scientific stack used by the experiments (including TensorFlow, TensorFlow Probability, Keras, NumPy, SciPy, scikit-learn, pandas, and Matplotlib) is installed through `bnn-for-14c-calibration`.

## Brehm et al. input data

Two cleaned CSV files from Brehm et al. paper are required for Figures 12--16 and Tables 4--5. In this repository, they are placed exactly at:

``` text
data/brehm/donnees_traites_fig_a.csv
data/brehm/donnees_traites_fig_b.csv
```

The corresponding script resolves these paths relative to the repository itself rather than relative to the shell's current working directory.

## Run the complete workflow

From the repository root, run:

``` bash
python run_all.py
```

The blocks execute in publication order:

1.  Figures 1--3
2.  Figures 4--7
3.  Figures 8--10 + Table 3
4.  Figure 11
5.  Figures 12--16 + Tables 4--5

`run_all.py` reports the start, completion, duration, and failure status of every block. By default it clears only the managed `images/` directory and the five managed `outputs/outputs_*` directories before starting, which prevents stale files from being mistaken for results from the current run. Use `--no-clean` only when that behavior is explicitly desired.

A structural smoke test that does not import the scientific dependencies or execute the experiments is available with:

``` bash
python run_all.py --check-only
```

## Running one experiment block

Each public script can also be executed independently. For example:

``` bash
python scripts/reproduce_fig1_to_fig3.py
python scripts/reproduce_fig4_to_fig7.py
python scripts/reproduce_fig8_to_fig10_table3.py
python scripts/reproduce_fig11.py
python scripts/reproduce_fig12_to_fig16_tables4_5.py
```

Each script has a repository-relative default output directory under `outputs/`. Common options include `--output-dir`, `--seed`, `--dpi`, and `--show`. The Brehm script additionally accepts `--data-dir` as an override, although it is unnecessary for the standard layout.

## Reproducibility and seeds

Some experiments are stochastic but explicitly seeded. For Figures 1--3, the simulated-data seed is fixed at **2022** and the script's default model/BNN seed is also **2022**. Important stochastic stages remain explicitly seeded, and SHA-256 digests of simulated data, fitted weights, and posterior/predictive draws are retained in the audit outputs. The repeated coverage experiments supporting Figure 3 are run for `K = 100`, `K = 10,000`, and `K = 100,000`.

The other experiment scripts retain their validated default seed of **1234**. The complete workflow preserves those per-script defaults when `python run_all.py` is used without additional arguments. An explicit global override is available when needed:

``` bash
python run_all.py --seed 1234
```

The global manifest records the effective seed used for every block.

## Figures and manuscript assets

Every script generates each publication figure exactly once inside its own audit output directory. After a block succeeds, `run_all.py` copies the resulting publication PNG files byte-for-byte into `images/` and verifies equality with SHA-256.

The manuscript assets are:

``` text
Fig1.png
Fig2.png
Fig3.png
Fig4.png
Fig5.png
Fig6a.png  Fig6b.png
Fig7a.png  Fig7b.png
Fig8a.png  Fig8b.png
Fig9a.png  Fig9b.png
Fig10.png
Fig11.png
Fig12.png
Fig13.png
Fig14.png
Fig15.png
Fig16.png
```

The panel filenames match the manuscript figure includes directly; no second rendering is produced for `images/`.

## Audit outputs

The five experiment directories are:

``` text
outputs/outputs_fig1_to_fig3/
outputs/outputs_fig4_to_fig7/
outputs/outputs_fig8_to_fig10_table3/
outputs/outputs_fig11/
outputs/outputs_fig12_to_fig16_tables4_5/
```

Depending on the block, they contain publication figures, numerical JSON results, CSV/JSON tables, execution metadata, seed information, package versions, elapsed times, input-data hashes, and other audit information produced by the validated scripts.

The complete workflow also creates:

``` text
outputs/run_manifest.json
```

The global manifest records the run timestamps, Python version, `bnn-for-14c-calibration` version, seed configuration, script order, per-block status and duration, generated files, source-to-`images/` mappings, and SHA-256 hashes of all final publication figure assets.

## Runtime

Several parts of the workflow are computationally intensive, including the repeated Figure 3 coverage experiment with `K = 100,000`, joint-calibration MCMC calculations, posterior-density reconstruction, and the Brehm calibration experiments. A complete reference run can therefore take substantial time depending on the machine and installed numerical stack.
