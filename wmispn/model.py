"""Main WMISPN model class with easy-to-use API."""

import numpy as np
import pickle
from typing import Union, Dict, List, Tuple, Optional
from .data.dataset import Dataset
from .data.preprocessing import preprocess_data
from .learning.structure import LearnWMISPN, LearnWMISPNParams
from .inference.engine import InferenceEngine
from .query.interface import QueryInterface, IntervalQuery


class WMISPN:
    """
    Weighted Model Integration Sum-Product Network.

    High-level API for learning and querying probabilistic models
    in mixed discrete-continuous domains.

    Example:
        >>> # Load data
        >>> model = WMISPN()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Query
        >>> prob = model.query({"class": 1}, {"job": 0})
        >>> samples = model.sample(n_samples=100)
        >>>
        >>> # Evaluate
        >>> ll = model.log_likelihood(X_test)
    """

    def __init__(
        self,
        min_instances: int = 50,
        min_features: int = 1,
        g_factor: float = 1.0,
        cluster_penalty: float = 2.0,
        max_clusters: int = 10,
        min_clusters: int = 2,
        n_bins: int = 5,
        poly_degree: int = 2,
        binning_method: str = 'quantile',
        independence_test: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize WMISPN model.

        Args:
            min_instances: Minimum instances for splitting nodes
            min_features: Minimum features for splitting
            g_factor: G-test factor for independence testing (higher = more conservative)
            cluster_penalty: Penalty for creating clusters (higher = fewer clusters)
            max_clusters: Maximum number of clusters per sum node
            min_clusters: Minimum number of clusters
            n_bins: Number of bins for continuous feature discretization
            poly_degree: Polynomial degree for continuous distributions
            binning_method: Binning method ('quantile', 'uniform', 'kmeans')
            independence_test: Whether to perform independence testing
            random_state: Random seed for reproducibility
        """
        self.params = {
            'min_instances': min_instances,
            'min_features': min_features,
            'g_factor': g_factor,
            'cluster_penalty': cluster_penalty,
            'max_clusters': max_clusters,
            'min_clusters': min_clusters,
            'n_bins': n_bins,
            'poly_degree': poly_degree,
            'binning_method': binning_method,
            'independence_test': independence_test,
            'random_state': random_state
        }

        self.spn_graph = None
        self.inference_engine = None
        self.query_interface = None
        self.dataset = None
        self.preprocessing_metadata = None
        self.is_fitted = False

    @classmethod
    def with_preset(cls, preset: str = 'default', **kwargs):
        """
        Create WMISPN with preset hyperparameters.

        Args:
            preset: Preset name ('default', 'aggressive', 'conservative')
            **kwargs: Override specific parameters

        Returns:
            WMISPN instance
        """
        if preset == 'default':
            params = LearnWMISPNParams.default()
        elif preset == 'aggressive':
            params = LearnWMISPNParams.aggressive_splitting()
        elif preset == 'conservative':
            params = LearnWMISPNParams.conservative_splitting()
        else:
            raise ValueError(f"Unknown preset: {preset}")

        # Override with user-specified params
        params.update(kwargs)

        return cls(**params)

    def fit(
        self,
        X,
        feature_names: Optional[List[str]] = None,
        variable_types: Optional[Dict[int, str]] = None,
        preprocess: bool = True,
        normalize: bool = True
    ):
        """
        Learn WMISPN structure from data.

        Args:
            X: Training data (array, DataFrame, or Dataset)
            feature_names: Optional feature names
            variable_types: Optional dict mapping feature index to type
            preprocess: Whether to preprocess data
            normalize: Whether to normalize continuous features

        Returns:
            self
        """
        # Convert to Dataset if needed
        if isinstance(X, Dataset):
            self.dataset = X
        else:
            if preprocess:
                X_processed, metadata = preprocess_data(
                    X,
                    variable_types=variable_types,
                    normalize_continuous=normalize
                )
                self.preprocessing_metadata = metadata
                variable_types = metadata['variable_types']
            else:
                X_processed = X

            self.dataset = Dataset(
                X_processed,
                feature_names=feature_names,
                variable_types=variable_types
            )

        # Initialize learner
        learner = LearnWMISPN(**self.params)

        # Learn structure
        print(f"Learning WMISPN structure...")
        print(f"  Dataset: {self.dataset.n_samples} samples, {self.dataset.n_features} features")
        print(f"  Categorical: {len(self.dataset.categorical_indices)}, Continuous: {len(self.dataset.continuous_indices)}")

        self.spn_graph = learner.learn(self.dataset)

        # Initialize inference and query engines
        self.inference_engine = InferenceEngine(self.spn_graph)
        self.query_interface = QueryInterface(self.spn_graph)

        self.is_fitted = True

        # Print structure info
        info = self.spn_graph.get_structure_info()
        print(f"\nLearned SPN structure:")
        print(f"  Total nodes: {info['n_nodes']}")
        print(f"  Sum nodes: {info['n_sum_nodes']}")
        print(f"  Product nodes: {info['n_product_nodes']}")
        print(f"  Leaf nodes: {info['n_leaf_nodes']}")
        print(f"  Depth: {info['depth']}")
        print(f"  Parameters: {info['n_parameters']}")

        return self

    def log_likelihood(self, X):
        """
        Compute log-likelihood of data.

        Args:
            X: Data array or Dataset

        Returns:
            Array of log-likelihoods
        """
        self._check_fitted()
        return self.inference_engine.log_likelihood(X)

    def average_log_likelihood(self, X):
        """
        Compute average log-likelihood.

        Args:
            X: Data array or Dataset

        Returns:
            Scalar average log-likelihood
        """
        self._check_fitted()
        return self.inference_engine.average_log_likelihood(X)

    def pseudo_log_likelihood(self, X):
        """
        Compute pseudo log-likelihood.

        Args:
            X: Data array or Dataset

        Returns:
            Array of pseudo log-likelihoods
        """
        self._check_fitted()
        return self.inference_engine.pseudo_log_likelihood(X)

    def query(self, query_vars: Dict[int, any], evidence: Dict[int, any] = None):
        """
        Compute conditional probability P(query | evidence).

        Args:
            query_vars: Dict of variables to query {var_idx: value}
            evidence: Dict of evidence variables {var_idx: value}

        Returns:
            Log conditional probability
        """
        self._check_fitted()

        if evidence is None:
            # Just marginal probability
            return self.spn_graph.marginal(query_vars)
        else:
            # Conditional probability
            return self.spn_graph.conditional(query_vars, evidence)

    def interval_query(
        self,
        intervals: Union[Dict[int, Tuple[float, float]], List[IntervalQuery]],
        evidence: Dict[int, any] = None
    ):
        """
        Query with interval constraints on continuous variables.

        Args:
            intervals: Dict {var_idx: (lower, upper)} or list of IntervalQuery
            evidence: Categorical evidence

        Returns:
            Log probability
        """
        self._check_fitted()

        if isinstance(intervals, dict):
            interval_list = [
                IntervalQuery(var_idx, lower, upper)
                for var_idx, (lower, upper) in intervals.items()
            ]
        else:
            interval_list = intervals

        return self.query_interface.conjunctive_query(interval_list, evidence)

    def predict(self, evidence: Dict[int, any], target_var: int):
        """
        Predict most likely value of target variable given evidence.

        Args:
            evidence: Evidence dict
            target_var: Variable to predict

        Returns:
            Predicted value
        """
        self._check_fitted()
        return self.inference_engine.predict(evidence, target_var)

    def sample(self, n_samples: int = 1, evidence: Dict[int, any] = None, random_state: Optional[int] = None):
        """
        Generate samples from the learned distribution.

        Args:
            n_samples: Number of samples
            evidence: Optional evidence to condition on
            random_state: Random seed

        Returns:
            Array of samples (n_samples, n_features)
        """
        self._check_fitted()
        return self.inference_engine.sample(n_samples, evidence, random_state)

    def evaluate(self, X_test, metrics: List[str] = ['ll', 'pll']):
        """
        Evaluate model on test data with multiple metrics.

        Args:
            X_test: Test data
            metrics: List of metrics to compute ('ll', 'pll', 'cll')

        Returns:
            Dict of metric values
        """
        self._check_fitted()
        return self.inference_engine.evaluate_metrics(X_test, metrics)

    def get_structure_info(self):
        """Get information about learned SPN structure."""
        self._check_fitted()
        return self.spn_graph.get_structure_info()

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save file
        """
        self._check_fitted()

        model_data = {
            'params': self.params,
            'spn_graph': self.spn_graph,
            'dataset_metadata': {
                'feature_names': self.dataset.feature_names if self.dataset else None,
                'variable_types': self.dataset.variable_types if self.dataset else None,
                'n_features': self.dataset.n_features if self.dataset else None,
            },
            'preprocessing_metadata': self.preprocessing_metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            WMISPN instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        model = cls(**model_data['params'])

        # Restore components
        model.spn_graph = model_data['spn_graph']
        model.preprocessing_metadata = model_data['preprocessing_metadata']

        # Recreate inference engines
        model.inference_engine = InferenceEngine(model.spn_graph)
        model.query_interface = QueryInterface(model.spn_graph)

        model.is_fitted = True

        print(f"Model loaded from {filepath}")

        return model

    def _check_fitted(self):
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def __repr__(self):
        if self.is_fitted:
            info = self.spn_graph.get_structure_info()
            return (f"WMISPN(fitted=True, nodes={info['n_nodes']}, "
                    f"depth={info['depth']}, features={info['scope_size']})")
        else:
            return "WMISPN(fitted=False)"
