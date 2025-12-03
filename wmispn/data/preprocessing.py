"""Data preprocessing utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union


def identify_variable_types(
    data: Union[np.ndarray, pd.DataFrame],
    categorical_threshold: int = 10,
    unique_ratio_threshold: float = 0.05
) -> Dict[int, str]:
    """
    Automatically identify variable types as categorical or continuous.

    Args:
        data: Input data
        categorical_threshold: Max unique values for categorical
        unique_ratio_threshold: Max unique ratio for categorical

    Returns:
        Dict mapping column index to type ('categorical' or 'continuous')
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_features = data.shape[1]
    variable_types = {}

    for i in range(n_features):
        col = data[:, i]
        col_clean = col[~pd.isna(col)]

        if len(col_clean) == 0:
            variable_types[i] = 'continuous'
            continue

        n_unique = len(np.unique(col_clean))
        unique_ratio = n_unique / len(col_clean)

        if n_unique <= categorical_threshold or unique_ratio <= unique_ratio_threshold:
            variable_types[i] = 'categorical'
        else:
            variable_types[i] = 'continuous'

    return variable_types


def encode_categorical(data: np.ndarray, n_categories: int = None) -> Tuple[np.ndarray, Dict]:
    """
    Encode categorical data to integers.

    Args:
        data: 1D array of categorical values
        n_categories: Number of categories (optional)

    Returns:
        Tuple of (encoded_data, encoding_map)
    """
    unique_vals = np.unique(data[~pd.isna(data)])

    if n_categories is None:
        n_categories = len(unique_vals)

    encoding_map = {val: i for i, val in enumerate(unique_vals)}
    decoding_map = {i: val for val, i in encoding_map.items()}

    encoded = np.array([encoding_map.get(val, -1) for val in data])

    return encoded, {'encoding': encoding_map, 'decoding': decoding_map, 'n_categories': n_categories}


def preprocess_data(
    data: Union[np.ndarray, pd.DataFrame],
    variable_types: Dict[int, str] = None,
    normalize_continuous: bool = True,
    handle_missing: str = 'mean'
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess data for WMISPN learning.

    Args:
        data: Input data
        variable_types: Dict mapping column index to type
        normalize_continuous: Whether to normalize continuous features
        handle_missing: How to handle missing values ('mean', 'median', 'drop')

    Returns:
        Tuple of (preprocessed_data, metadata)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values.copy()
    else:
        data = np.array(data, dtype=float)

    # Auto-detect types if not provided
    if variable_types is None:
        variable_types = identify_variable_types(data)

    metadata = {
        'variable_types': variable_types,
        'categorical_encodings': {},
        'continuous_stats': {}
    }

    # Handle missing values
    if handle_missing == 'drop':
        # Remove rows with missing values
        mask = ~np.any(pd.isna(data), axis=1)
        data = data[mask]
    else:
        # Impute missing values
        for i in range(data.shape[1]):
            col = data[:, i]
            if np.any(pd.isna(col)):
                if variable_types[i] == 'continuous':
                    if handle_missing == 'mean':
                        fill_value = np.nanmean(col)
                    elif handle_missing == 'median':
                        fill_value = np.nanmedian(col)
                    else:
                        fill_value = 0
                    data[pd.isna(col), i] = fill_value
                else:
                    # For categorical, use mode
                    unique, counts = np.unique(col[~pd.isna(col)], return_counts=True)
                    if len(unique) > 0:
                        fill_value = unique[np.argmax(counts)]
                    else:
                        fill_value = 0
                    data[pd.isna(col), i] = fill_value

    # Process each feature
    for i in range(data.shape[1]):
        if variable_types[i] == 'categorical':
            # Encode categorical
            encoded, encoding_info = encode_categorical(data[:, i])
            data[:, i] = encoded
            metadata['categorical_encodings'][i] = encoding_info

        elif variable_types[i] == 'continuous':
            # Normalize continuous if requested
            if normalize_continuous:
                col = data[:, i]
                col_min = col.min()
                col_max = col.max()
                col_mean = col.mean()
                col_std = col.std()

                metadata['continuous_stats'][i] = {
                    'min': col_min,
                    'max': col_max,
                    'mean': col_mean,
                    'std': col_std
                }

                # Min-max normalization to [0, 1]
                if col_max > col_min:
                    data[:, i] = (col - col_min) / (col_max - col_min)

    return data, metadata


def discretize_continuous(
    data: np.ndarray,
    n_bins: int = 10,
    method: str = 'quantile'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous data into bins.

    Args:
        data: 1D array of continuous values
        n_bins: Number of bins
        method: Binning method ('quantile', 'uniform', 'kmeans')

    Returns:
        Tuple of (bin_indices, bin_edges)
    """
    data = np.asarray(data).ravel()

    if method == 'quantile':
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, quantiles)
    elif method == 'uniform':
        bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_bins, random_state=42)
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

    # Assign bin indices
    bin_indices = np.digitize(data, bin_edges[1:-1])

    return bin_indices, bin_edges
