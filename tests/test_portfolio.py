"""Smoke tests for the reproducible portfolio demonstrations."""

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_synthetic_regression_outputs_are_finite():
    module = load_module("statistical-modelling/regression_demo.py", "regression_demo")
    result = module.run_demo(seed=7)
    assert result["n_samples"] == 240
    assert len(result["coefficients"]) == 2
    assert math.isfinite(result["mae"])
    assert math.isfinite(result["r2"])


def test_bundled_dataset_regression_outputs_are_finite():
    module = load_module("machine-learning/regression_pipeline.py", "regression_pipeline")
    result = module.run_demo(seed=7)
    assert result["train_samples"] > result["test_samples"] > 0
    assert result["mae"] >= 0
    assert math.isfinite(result["r2"])


def test_clustering_and_optimisation_return_expected_structure():
    module = load_module(
        "optimisation-clustering/clustering_optimisation_demo.py", "clustering_optimisation_demo"
    )
    clustered = module.run_clustering(seed=7)
    optimised = module.run_optimisation(seed=7)
    assert clustered["n_samples"] == 300
    assert clustered["n_clusters_found"] == 4
    assert optimised["success"]
    assert optimised["objective_value"] >= 0
    assert optimised["objective_value"] < 1e-4
