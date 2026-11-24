from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterConfig:
    n_clusters: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    random_seed: int | None = 42