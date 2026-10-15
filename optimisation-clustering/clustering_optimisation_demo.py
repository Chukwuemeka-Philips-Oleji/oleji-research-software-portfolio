"""K-means on synthetic blobs and differential evolution on a test function."""

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def run_clustering(seed: int = 42) -> dict:
    """Cluster simulated points and report an internal silhouette score."""
    points, _ = make_blobs(
        n_samples=300, centers=4, n_features=2, cluster_std=0.75, random_state=seed
    )
    model = KMeans(n_clusters=4, n_init=10, random_state=seed)
    labels = model.fit_predict(points)
    return {
        "n_samples": int(points.shape[0]),
        "n_clusters_requested": 4,
        "n_clusters_found": int(np.unique(labels).size),
        "silhouette_score": float(silhouette_score(points, labels)),
        "inertia": float(model.inertia_),
    }


def run_optimisation(seed: int = 42) -> dict:
    """Minimise a sphere function over [-5, 5]^3 with differential evolution."""
    def sphere(vector):
        return float(np.sum(np.square(vector)))

    result = differential_evolution(
        sphere, bounds=[(-5, 5)] * 3, seed=seed, polish=True, tol=1e-7
    )
    return {
        "objective": "sum(x_i ** 2)",
        "success": bool(result.success),
        "solution": result.x.tolist(),
        "objective_value": float(result.fun),
        "evaluations": int(result.nfev),
    }


if __name__ == "__main__":
    print("Synthetic K-means demonstration:")
    print(run_clustering())
    print("Bounded numerical optimisation demonstration:")
    print(run_optimisation())
