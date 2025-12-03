"""Binning utilities for continuous features."""

import numpy as np
from sklearn.cluster import KMeans


def create_bins(data, n_bins=10, method='quantile'):
    """
    Create bins for continuous data.

    Args:
        data: 1D array of continuous values
        n_bins: Number of bins
        method: Binning method ('quantile', 'uniform', 'kmeans')

    Returns:
        Tuple of (bin_edges, bin_assignments)
    """
    data = np.asarray(data).ravel()

    if method == 'quantile':
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, quantiles)
    elif method == 'uniform':
        bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        kmeans.fit(data.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.ravel())
        bin_edges = [data.min()]
        for i in range(len(centers) - 1):
            bin_edges.append((centers[i] + centers[i + 1]) / 2)
        bin_edges.append(data.max())
        bin_edges = np.array(bin_edges)
    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Ensure unique edges
    bin_edges = np.unique(bin_edges)

    # Assign bins
    bin_assignments = np.digitize(data, bin_edges[1:-1])

    return bin_edges, bin_assignments
