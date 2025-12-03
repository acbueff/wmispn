"""Inference engine for WMISPN."""

import numpy as np
from .likelihoods import (
    log_likelihood,
    conditional_log_likelihood,
    pseudo_log_likelihood,
    marginal_log_likelihood
)


class InferenceEngine:
    """
    Unified inference engine for WMISPN.

    Provides various inference operations including likelihood computations,
    marginal inference, and conditional inference.
    """

    def __init__(self, spn_graph):
        """
        Initialize inference engine.

        Args:
            spn_graph: SPNGraph instance
        """
        self.spn = spn_graph

    def log_likelihood(self, data):
        """
        Compute log P(X) for each instance.

        Args:
            data: Data array or dataset

        Returns:
            Array of log-likelihoods
        """
        return log_likelihood(self.spn, data)

    def average_log_likelihood(self, data):
        """
        Compute average log-likelihood over dataset.

        Args:
            data: Data array or dataset

        Returns:
            Scalar average log-likelihood
        """
        ll = self.log_likelihood(data)
        return np.mean(ll)

    def conditional_log_likelihood(self, data, condition_vars):
        """
        Compute log P(X_target | X_condition).

        Args:
            data: Data array
            condition_vars: List of variable indices to condition on

        Returns:
            Array of conditional log-likelihoods
        """
        return conditional_log_likelihood(self.spn, data, condition_vars)

    def pseudo_log_likelihood(self, data):
        """
        Compute pseudo log-likelihood.

        PLL = Σ_i log P(X_i | X_{-i})

        Args:
            data: Data array

        Returns:
            Array of pseudo log-likelihoods
        """
        return pseudo_log_likelihood(self.spn, data)

    def marginal_log_likelihood(self, data, marginal_vars):
        """
        Compute log P(X_observed) where some variables are marginalized.

        Args:
            data: Data array
            marginal_vars: List of variable indices to marginalize

        Returns:
            Array of marginal log-likelihoods
        """
        return marginal_log_likelihood(self.spn, data, marginal_vars)

    def marginal_probability(self, evidence):
        """
        Compute P(evidence) by marginalizing unobserved variables.

        Args:
            evidence: Dict {var_idx: value}

        Returns:
            Log marginal probability
        """
        return self.spn.marginal(evidence)

    def conditional_probability(self, query, evidence):
        """
        Compute P(query | evidence).

        Args:
            query: Dict {var_idx: value} to query
            evidence: Dict {var_idx: value} of observations

        Returns:
            Log conditional probability
        """
        return self.spn.conditional(query, evidence)

    def predict(self, evidence, target_var):
        """
        Predict most likely value of target_var given evidence.

        Args:
            evidence: Dict of observed variables
            target_var: Variable index to predict

        Returns:
            Most likely value
        """
        # Find the leaf node for target variable
        target_leaf = None
        for leaf in self.spn.leaf_nodes:
            if leaf.variable_idx == target_var:
                target_leaf = leaf
                break

        if target_leaf is None:
            raise ValueError(f"Target variable {target_var} not found in SPN")

        # For categorical, try all categories
        from ..core.nodes import CategoricalLeaf, ContinuousLeaf

        if isinstance(target_leaf, CategoricalLeaf):
            best_value = None
            best_prob = -np.inf

            for value in range(target_leaf.n_categories):
                query = {target_var: value}
                prob = self.conditional_probability(query, evidence)

                if prob > best_prob:
                    best_prob = prob
                    best_value = value

            return best_value

        elif isinstance(target_leaf, ContinuousLeaf):
            # For continuous, sample from the distribution given evidence
            # This is a simplified approach
            full_instance = evidence.copy()
            full_instance[target_var] = None

            # Sample from marginal
            samples = self.spn.sample(n_samples=100, evidence=evidence)

            # Return mean of samples for target variable
            if target_var in samples:
                return np.mean(samples[target_var])
            else:
                return None

    def sample(self, n_samples=1, evidence=None, random_state=None):
        """
        Generate samples from the model.

        Args:
            n_samples: Number of samples
            evidence: Optional evidence dict
            random_state: Random seed

        Returns:
            Array of samples
        """
        return self.spn.sample(n_samples, evidence, random_state)

    def evaluate_metrics(self, test_data, metrics=['ll', 'pll']):
        """
        Evaluate multiple metrics on test data.

        Args:
            test_data: Test dataset
            metrics: List of metric names ('ll', 'pll', 'cll')

        Returns:
            Dict of metric values
        """
        results = {}

        if 'll' in metrics:
            ll = self.average_log_likelihood(test_data)
            results['log_likelihood'] = ll

        if 'pll' in metrics:
            pll = self.pseudo_log_likelihood(test_data)
            results['pseudo_log_likelihood'] = np.mean(pll)

        if 'cll' in metrics:
            # Conditional on first half of variables
            n_features = test_data.shape[1] if isinstance(test_data, np.ndarray) else len(test_data[0])
            condition_vars = list(range(n_features // 2))
            cll = self.conditional_log_likelihood(test_data, condition_vars)
            results['conditional_log_likelihood'] = np.mean(cll)

        return results
