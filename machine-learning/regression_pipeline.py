"""Train/test regression pipeline on scikit-learn's bundled diabetes dataset."""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_demo(seed: int = 42) -> dict:
    """Fit a Ridge regression pipeline and return held-out metrics."""
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=seed
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "dataset": "scikit-learn bundled diabetes regression dataset",
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }


if __name__ == "__main__":
    print("Educational benchmark only; not a clinical tool.")
    for key, value in run_demo().items():
        print(f"{key}: {value}")
