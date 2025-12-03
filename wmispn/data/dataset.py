"""Dataset handling for WMISPN."""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional


class Dataset:
    """
    Dataset wrapper for WMISPN learning and inference.

    Handles both discrete and continuous features.
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        variable_types: Optional[Dict[int, str]] = None,
        categorical_indices: Optional[List[int]] = None,
        continuous_indices: Optional[List[int]] = None
    ):
        """
        Initialize dataset.

        Args:
            data: Data array or DataFrame
            feature_names: List of feature names
            variable_types: Dict mapping feature index to type ('categorical' or 'continuous')
            categorical_indices: List of categorical feature indices
            continuous_indices: List of continuous feature indices
        """
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns) if feature_names is None else feature_names
            self.data = data.values
        else:
            self.data = np.asarray(data)
            self.feature_names = feature_names or [f"X{i}" for i in range(self.data.shape[1])]

        self.n_samples, self.n_features = self.data.shape

        # Determine variable types
        if variable_types is not None:
            self.variable_types = variable_types
        elif categorical_indices is not None and continuous_indices is not None:
            self.variable_types = {}
            for idx in categorical_indices:
                self.variable_types[idx] = 'categorical'
            for idx in continuous_indices:
                self.variable_types[idx] = 'continuous'
        else:
            # Auto-detect
            self.variable_types = self._auto_detect_types()

        # Store indices
        self.categorical_indices = [i for i, t in self.variable_types.items() if t == 'categorical']
        self.continuous_indices = [i for i, t in self.variable_types.items() if t == 'continuous']

    def _auto_detect_types(self, max_unique_ratio=0.05):
        """
        Auto-detect variable types based on unique value ratio.

        Args:
            max_unique_ratio: Max ratio of unique values for categorical

        Returns:
            Dict mapping feature index to type
        """
        variable_types = {}

        for i in range(self.n_features):
            col = self.data[:, i]

            # Remove NaN for counting
            col_clean = col[~pd.isna(col)]

            if len(col_clean) == 0:
                variable_types[i] = 'continuous'
                continue

            n_unique = len(np.unique(col_clean))
            unique_ratio = n_unique / len(col_clean)

            # Heuristic: if few unique values or low ratio, treat as categorical
            if n_unique <= 10 or unique_ratio <= max_unique_ratio:
                variable_types[i] = 'categorical'
            else:
                variable_types[i] = 'continuous'

        return variable_types

    def get_column(self, idx):
        """Get column by index."""
        return self.data[:, idx]

    def get_feature_values(self, feature_idx, indices=None):
        """
        Get values for a specific feature.

        Args:
            feature_idx: Feature index
            indices: Optional row indices to select

        Returns:
            Feature values
        """
        if indices is None:
            return self.data[:, feature_idx]
        else:
            return self.data[indices, feature_idx]

    def split(self, train_ratio=0.7, valid_ratio=0.15, random_state=None):
        """
        Split dataset into train/validation/test sets.

        Args:
            train_ratio: Training set ratio
            valid_ratio: Validation set ratio
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
        """
        rng = np.random.RandomState(random_state)
        indices = np.arange(self.n_samples)
        rng.shuffle(indices)

        n_train = int(self.n_samples * train_ratio)
        n_valid = int(self.n_samples * valid_ratio)

        train_indices = indices[:n_train]
        valid_indices = indices[n_train:n_train + n_valid]
        test_indices = indices[n_train + n_valid:]

        train_data = Dataset(
            self.data[train_indices],
            self.feature_names,
            self.variable_types
        )
        valid_data = Dataset(
            self.data[valid_indices],
            self.feature_names,
            self.variable_types
        )
        test_data = Dataset(
            self.data[test_indices],
            self.feature_names,
            self.variable_types
        )

        return train_data, valid_data, test_data

    def to_dict_list(self):
        """Convert to list of dicts for evaluation."""
        return [
            {i: self.data[j, i] for i in range(self.n_features)}
            for j in range(self.n_samples)
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Get sample by index."""
        if isinstance(idx, int):
            return {i: self.data[idx, i] for i in range(self.n_features)}
        else:
            return Dataset(
                self.data[idx],
                self.feature_names,
                self.variable_types
            )

    def __repr__(self):
        return (f"Dataset(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"categorical={len(self.categorical_indices)}, "
                f"continuous={len(self.continuous_indices)})")
