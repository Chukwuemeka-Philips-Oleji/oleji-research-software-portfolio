# Dr. Oleji Chukwuemeka Philips — Research Software Portfolio

A practical, reproducible portfolio demonstrating Python workflows in statistical modelling, machine learning, clustering, and numerical optimisation.

**Repository owner:** Dr. Oleji Chukwuemeka Philips  
**GitHub:** [Chukwuemeka-Philips-Oleji](https://github.com/Chukwuemeka-Philips-Oleji)

> **Scope and integrity note:** These are educational software demonstrations using seeded synthetic data or datasets bundled with scikit-learn. They are not presented as findings from a published study, and no research performance claims are made. Update this profile with verified biography, publications, and project-specific results as appropriate.

## Projects

### 1. Statistical modelling — `statistical-modelling/`
A seeded synthetic dataset is used to demonstrate exploratory summaries, ordinary least-squares regression, residual diagnostics, and evaluation with MAE and R-squared. Because the data are simulated, the exercise illustrates workflow rather than making claims about a real population.

Run:

```bash
python -m pip install -r requirements.txt
python statistical-modelling/regression_demo.py
```

### 2. Machine learning — `machine-learning/`
A reproducible regression workflow uses scikit-learn's bundled diabetes dataset to demonstrate a train/test split, standardisation within a pipeline, model fitting, and held-out evaluation. The example is for learning and is not a clinical tool.

Run: `python machine-learning/regression_pipeline.py`

### 3. Clustering and optimisation — `optimisation-clustering/`
Demonstrates K-means clustering on generated blobs and bounded numerical optimisation of a simple sphere function using SciPy's differential evolution. Fixed random seeds support repeatability.

Run: `python optimisation-clustering/clustering_optimisation_demo.py`

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the scripts from the repository root. Each script prints summary metrics or optimisation output; plots are optional and are not saved unless the script is extended to do so.

## Tests

```bash
python -m pytest
```

Tests check that demonstration functions return expected shapes and finite values. They do not validate any real-world scientific hypothesis.

## Reproducibility

- Random seeds are set explicitly where supported.
- Machine-learning preprocessing is fitted inside a pipeline to reduce data leakage.
- Examples use synthetic data or datasets distributed with the relevant library; no external dataset download is required.
- Package versions are bounded by broad minimum versions in `requirements.txt`; for archival reproducibility, freeze a tested environment after running the examples.

## Suggested extensions

- Add notebooks with documented assumptions and visualisations.
- Add confidence intervals and sensitivity analysis to the regression example.
- Compare clustering metrics across several values of *k* without treating them as ground truth.
- Add a benchmark protocol before making performance comparisons.
- Link verified publications or datasets only after confirming their bibliographic details and permissions.

## License

MIT License. See `LICENSE`.
