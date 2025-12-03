"""SPN graph container."""

import numpy as np
from .nodes import Node, SumNode, ProductNode, LeafNode


class SPNGraph:
    """
    Container for Sum-Product Network graph structure.

    Manages the DAG of SPN nodes and provides inference operations.
    """

    def __init__(self, root=None):
        """
        Initialize SPN graph.

        Args:
            root: Root node of the SPN
        """
        self.root = root
        self.nodes = []
        self.leaf_nodes = []
        self.scope = set()

        if root is not None:
            self._collect_nodes(root)

    def _collect_nodes(self, node):
        """Collect all nodes in the graph via DFS."""
        if node in self.nodes:
            return

        self.nodes.append(node)
        node.node_id = len(self.nodes) - 1
        self.scope.update(node.scope)

        if isinstance(node, LeafNode):
            self.leaf_nodes.append(node)
        elif isinstance(node, (SumNode, ProductNode)):
            for child in node.children:
                self._collect_nodes(child)

    def set_root(self, root):
        """Set root node and rebuild node collection."""
        self.root = root
        self.nodes = []
        self.leaf_nodes = []
        self.scope = set()
        self._collect_nodes(root)

    def evaluate(self, data):
        """
        Evaluate SPN on a data instance.

        Args:
            data: Single data instance (dict or array)

        Returns:
            Log probability
        """
        if self.root is None:
            return -np.inf

        return self.root.eval(data)

    def evaluate_batch(self, data_batch):
        """
        Evaluate SPN on a batch of data instances.

        Args:
            data_batch: Array of shape (n_samples, n_features) or list of dicts

        Returns:
            Array of log probabilities
        """
        if isinstance(data_batch, np.ndarray):
            return np.array([self.evaluate(instance) for instance in data_batch])
        else:
            return np.array([self.evaluate(instance) for instance in data_batch])

    def log_likelihood(self, data):
        """
        Compute log-likelihood of dataset.

        Args:
            data: Data array or list of instances

        Returns:
            Total log-likelihood
        """
        log_probs = self.evaluate_batch(data)
        return np.sum(log_probs)

    def average_log_likelihood(self, data):
        """
        Compute average log-likelihood per sample.

        Args:
            data: Data array or list of instances

        Returns:
            Average log-likelihood
        """
        log_probs = self.evaluate_batch(data)
        return np.mean(log_probs)

    def marginal(self, evidence):
        """
        Compute marginal probability given evidence.

        Args:
            evidence: Dict {var_idx: value} of observed variables

        Returns:
            Log marginal probability
        """
        # Create data instance with evidence and None for unobserved
        data = {var: None for var in self.scope}
        data.update(evidence)

        return self.evaluate(data)

    def conditional(self, query_vars, evidence):
        """
        Compute conditional probability P(query_vars | evidence).

        Args:
            query_vars: Dict {var_idx: value} to query
            evidence: Dict {var_idx: value} of evidence

        Returns:
            Log conditional probability
        """
        # P(Q | E) = P(Q, E) / P(E)

        # Joint probability P(Q, E)
        joint_data = evidence.copy()
        joint_data.update(query_vars)
        log_joint = self.evaluate(joint_data)

        # Marginal P(E)
        log_evidence = self.marginal(evidence)

        return log_joint - log_evidence

    def sample(self, n_samples=1, evidence=None, random_state=None):
        """
        Generate samples from the SPN.

        Args:
            n_samples: Number of samples
            evidence: Evidence dict {var_idx: value}
            random_state: Random seed

        Returns:
            Array of samples (n_samples, n_features)
        """
        if self.root is None:
            return np.array([])

        samples_dict = self.root.sample(n_samples, evidence, random_state)

        # Convert to array
        var_indices = sorted(samples_dict.keys())
        samples_array = np.column_stack([samples_dict[var] for var in var_indices])

        return samples_array

    def get_structure_info(self):
        """Get information about the SPN structure."""
        n_sum_nodes = sum(1 for node in self.nodes if isinstance(node, SumNode))
        n_prod_nodes = sum(1 for node in self.nodes if isinstance(node, ProductNode))
        n_leaf_nodes = len(self.leaf_nodes)

        # Compute depth
        depth = self._compute_depth(self.root)

        # Count parameters
        n_params = 0
        for node in self.nodes:
            if isinstance(node, SumNode):
                n_params += len(node.children)  # Mixture weights

        return {
            'n_nodes': len(self.nodes),
            'n_sum_nodes': n_sum_nodes,
            'n_product_nodes': n_prod_nodes,
            'n_leaf_nodes': n_leaf_nodes,
            'depth': depth,
            'n_parameters': n_params,
            'scope_size': len(self.scope)
        }

    def _compute_depth(self, node, current_depth=0):
        """Compute depth of SPN tree."""
        if isinstance(node, LeafNode):
            return current_depth

        if isinstance(node, (SumNode, ProductNode)):
            if len(node.children) == 0:
                return current_depth
            return max(self._compute_depth(child, current_depth + 1) for child in node.children)

        return current_depth

    def __repr__(self):
        info = self.get_structure_info()
        return (f"SPNGraph(nodes={info['n_nodes']}, "
                f"sum={info['n_sum_nodes']}, "
                f"prod={info['n_product_nodes']}, "
                f"leaves={info['n_leaf_nodes']}, "
                f"depth={info['depth']})")
