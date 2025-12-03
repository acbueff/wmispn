"""LearnWMISPN structure learning algorithm."""

import numpy as np
from sklearn.cluster import KMeans
from ..core.nodes import SumNode, ProductNode, CategoricalLeaf, ContinuousLeaf
from ..core.graph import SPNGraph
from .independence import find_independent_set
from ..utils import log_sum_exp


class LearnWMISPN:
    """
    LearnWMISPN structure learning algorithm.

    Learns Sum-Product Networks with Weighted Model Integration
    for mixed discrete-continuous domains.
    """

    def __init__(
        self,
        min_instances=50,
        min_features=1,
        g_factor=1.0,
        cluster_penalty=2.0,
        max_clusters=10,
        min_clusters=2,
        n_bins=5,
        poly_degree=2,
        binning_method='quantile',
        independence_test=True,
        random_state=None
    ):
        """
        Initialize LearnWMISPN algorithm.

        Args:
            min_instances: Minimum instances for splitting
            min_features: Minimum features for splitting
            g_factor: G-test factor for independence testing
            cluster_penalty: Penalty for creating clusters
            max_clusters: Maximum number of clusters
            min_clusters: Minimum number of clusters
            n_bins: Number of bins for continuous features
            poly_degree: Polynomial degree for continuous features
            binning_method: Method for binning ('quantile', 'uniform', 'kmeans')
            independence_test: Whether to perform independence testing
            random_state: Random seed
        """
        self.min_instances = min_instances
        self.min_features = min_features
        self.g_factor = g_factor
        self.cluster_penalty = cluster_penalty
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.n_bins = n_bins
        self.poly_degree = poly_degree
        self.binning_method = binning_method
        self.independence_test = independence_test
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def learn(self, dataset):
        """
        Learn SPN structure from dataset.

        Args:
            dataset: Dataset instance

        Returns:
            SPNGraph instance
        """
        # Extract data and variable types
        data = dataset.data
        variable_types = dataset.variable_types
        variable_indices = list(range(dataset.n_features))

        # Build SPN recursively
        root = self._learn_structure(
            data,
            variable_indices,
            variable_types,
            depth=0
        )

        # Create and return graph
        spn = SPNGraph(root)
        return spn

    def _learn_structure(self, data, var_indices, var_types, depth=0):
        """
        Recursively learn SPN structure.

        Args:
            data: Data subset
            var_indices: Variable indices in this partition
            var_types: Variable types dict
            depth: Current recursion depth

        Returns:
            Root node of learned structure
        """
        n_instances = data.shape[0]
        n_vars = len(var_indices)

        # Base case 1: Single variable
        if n_vars == 1:
            return self._create_leaf_node(data, var_indices[0], var_types[var_indices[0]])

        # Base case 2: Too few instances - create product of leaves
        if n_instances < self.min_instances:
            return self._create_product_of_leaves(data, var_indices, var_types)

        # Try independence-based split (variable partitioning)
        if self.independence_test and n_vars > 1:
            independent_set, dependent_set = find_independent_set(
                data, var_indices, self.g_factor
            )

            if len(independent_set) > 0 and len(dependent_set) > 0:
                # Create product node with independent partitions
                child1 = self._learn_structure(data, independent_set, var_types, depth + 1)
                child2 = self._learn_structure(data, dependent_set, var_types, depth + 1)

                scope = set(independent_set) | set(dependent_set)
                prod_node = ProductNode(scope, [child1, child2])
                return prod_node

        # Try instance clustering (instance partitioning)
        best_clustering, best_score = self._find_best_clustering(data, var_indices)

        if best_clustering is not None and len(best_clustering) >= self.min_clusters:
            # Create sum node with clusters
            scope = set(var_indices)
            sum_node = SumNode(scope)

            for cluster_indices, weight in best_clustering:
                cluster_data = data[cluster_indices]
                child = self._learn_structure(cluster_data, var_indices, var_types, depth + 1)
                sum_node.add_child(child, weight)

            return sum_node

        # Fallback: create product of leaves
        return self._create_product_of_leaves(data, var_indices, var_types)

    def _create_leaf_node(self, data, var_idx, var_type):
        """Create appropriate leaf node for a single variable."""
        scope = {var_idx}
        column_data = data[:, var_idx]

        if var_type == 'categorical':
            leaf = CategoricalLeaf(scope, var_idx)
            leaf.fit(column_data)
        else:  # continuous
            leaf = ContinuousLeaf(scope, var_idx)
            leaf.fit(column_data, self.n_bins, self.poly_degree, self.binning_method)

        return leaf

    def _create_product_of_leaves(self, data, var_indices, var_types):
        """Create product node with leaf for each variable."""
        scope = set(var_indices)
        prod_node = ProductNode(scope)

        for var_idx in var_indices:
            leaf = self._create_leaf_node(data, var_idx, var_types[var_idx])
            prod_node.add_child(leaf)

        return prod_node

    def _find_best_clustering(self, data, var_indices):
        """
        Find best instance clustering using EM with penalized likelihood.

        Args:
            data: Data array
            var_indices: Variable indices

        Returns:
            Tuple of (best_clustering, best_score)
        """
        n_instances = data.shape[0]

        # Don't cluster if too few instances
        if n_instances < 2 * self.min_instances:
            return None, -np.inf

        best_clustering = None
        best_score = -np.inf

        # Try different numbers of clusters
        for n_clusters in range(self.min_clusters, min(self.max_clusters, n_instances // self.min_instances) + 1):
            # Run k-means clustering
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.rng.randint(0, 10000),
                    n_init=10,
                    max_iter=100
                )
                cluster_labels = kmeans.fit_predict(data[:, var_indices])

                # Build clustering with weights
                clustering = []
                for k in range(n_clusters):
                    cluster_mask = cluster_labels == k
                    cluster_size = cluster_mask.sum()

                    if cluster_size > 0:
                        cluster_indices = np.where(cluster_mask)[0]
                        weight = cluster_size / n_instances
                        clustering.append((cluster_indices, weight))

                # Compute penalized score
                # Score = log-likelihood - penalty * n_clusters
                ll = self._compute_clustering_likelihood(data, var_indices, clustering)
                penalty = self.cluster_penalty * n_clusters
                score = ll - penalty

                if score > best_score:
                    best_score = score
                    best_clustering = clustering

            except:
                continue

        return best_clustering, best_score

    def _compute_clustering_likelihood(self, data, var_indices, clustering):
        """
        Compute log-likelihood of clustering.

        Simple estimate based on cluster sizes.
        """
        n_instances = data.shape[0]
        ll = 0

        for cluster_indices, weight in clustering:
            cluster_size = len(cluster_indices)
            if cluster_size > 0:
                # Simple likelihood: log(weight) * cluster_size
                ll += cluster_size * np.log(weight + 1e-10)

        return ll / n_instances


class LearnWMISPNParams:
    """
    Helper class for managing hyperparameters.
    """

    @staticmethod
    def default():
        """Get default parameters."""
        return {
            'min_instances': 50,
            'min_features': 1,
            'g_factor': 1.0,
            'cluster_penalty': 2.0,
            'max_clusters': 10,
            'min_clusters': 2,
            'n_bins': 5,
            'poly_degree': 2,
            'binning_method': 'quantile',
            'independence_test': True,
            'random_state': 42
        }

    @staticmethod
    def aggressive_splitting():
        """Parameters for aggressive splitting (larger networks)."""
        return {
            'min_instances': 20,
            'min_features': 1,
            'g_factor': 0.5,
            'cluster_penalty': 1.0,
            'max_clusters': 15,
            'min_clusters': 2,
            'n_bins': 10,
            'poly_degree': 3,
            'binning_method': 'quantile',
            'independence_test': True,
            'random_state': 42
        }

    @staticmethod
    def conservative_splitting():
        """Parameters for conservative splitting (smaller networks)."""
        return {
            'min_instances': 100,
            'min_features': 2,
            'g_factor': 2.0,
            'cluster_penalty': 5.0,
            'max_clusters': 5,
            'min_clusters': 2,
            'n_bins': 3,
            'poly_degree': 1,
            'binning_method': 'uniform',
            'independence_test': True,
            'random_state': 42
        }
