"""Reproducible ordinary least-squares workflow on synthetic data."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def make_synthetic_data(n_samples: int = 240, seed: int = 42) -> pd.DataFrame:
    """Create a simple simulated dataset with known linear structure."""
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(50, 100, n_samples)
    noise = rng.normal(0, 4, n_samples)
    score = 12 + 4.2 * study_hours + 0.25 * attendance + noise
    return pd.DataFrame({"study_hours": study_hours, "attendance": attendance, "score": score})


def run_demo(seed: int = 42) -> dict:
    """Fit and evaluate a model; return metrics and residual diagnostics."""
    data = make_synthetic_data(seed=seed)
    features = data[["study_hours", "attendance"]]
    target = data["score"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed
    )
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    residuals = y_test.to_numpy() - predictions
    return {
        "n_samples": int(len(data)),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
    }


if __name__ == "__main__":
    result = run_demo()
    print("Synthetic-data regression demonstration (not empirical research)")
    for key, value in result.items():
        print(f"{key}: {value}")
