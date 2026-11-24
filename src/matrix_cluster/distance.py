import numpy as np


def pairwise_squared_euclidean(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Compute squared Euclidean distances between rows of X (N, D) and centroids (K, D)."""
    X_arr = np.asarray(X, dtype=np.float64)
    C_arr = np.asarray(centroids, dtype=np.float64)

    if X_arr.ndim != 2 or C_arr.ndim != 2:
        raise ValueError("X and centroids must both be 2D arrays.")
    if X_arr.shape[1] != C_arr.shape[1]:
        raise ValueError("Feature dimension D must match between X and centroids.")

    # Shapes: x_sq -> (N, 1), c_sq -> (1, K), cross -> (N, K)
    x_sq = np.sum(X_arr * X_arr, axis=1, keepdims=True)
    c_sq = np.sum(C_arr * C_arr, axis=1, keepdims=True).T
    cross = X_arr @ C_arr.T

    d2 = x_sq + c_sq - 2.0 * cross
    if clip_nonnegative:
        # Guard against tiny negative values (~ -1e-15) from floating-point cancellation
        d2 = np.maximum(d2, 0.0)
    return d2