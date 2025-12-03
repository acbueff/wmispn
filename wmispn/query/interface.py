"""Complex query interface for WMISPN.

Supports interval queries over continuous variables and categorical conditions.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from ..core.nodes import ContinuousLeaf, CategoricalLeaf, SumNode, ProductNode


class IntervalQuery:
    """
    Represents an interval query for continuous variables.

    Example:
        7500 < creditamount < 9000
    """

    def __init__(self, variable_idx, lower_bound=-np.inf, upper_bound=np.inf):
        """
        Initialize interval query.

        Args:
            variable_idx: Index of variable
            lower_bound: Lower bound (inclusive)
            upper_bound: Upper bound (exclusive)
        """
        self.variable_idx = variable_idx
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return f"IntervalQuery(var={self.variable_idx}, [{self.lower_bound}, {self.upper_bound}))"


class QueryInterface:
    """
    Interface for complex interval queries on WMISPN.

    Supports queries like:
        P(class=BAD | job=Unemployed, 7500 < creditamount < 9000)
    """

    def __init__(self, spn_graph):
        """
        Initialize query interface.

        Args:
            spn_graph: SPNGraph instance
        """
        self.spn = spn_graph

    def interval_query(
        self,
        interval_constraints: List[IntervalQuery],
        categorical_evidence: Dict[int, any] = None,
        query_var: int = None,
        query_value: any = None
    ):
        """
        Perform complex interval query.

        Computes: P(query_var=query_value | categorical_evidence, interval_constraints)

        Args:
            interval_constraints: List of IntervalQuery objects
            categorical_evidence: Dict of categorical variable observations
            query_var: Variable to query (optional)
            query_value: Value of query variable (optional)

        Returns:
            Log probability
        """
        if categorical_evidence is None:
            categorical_evidence = {}

        # Build evidence dict with intervals
        evidence = categorical_evidence.copy()

        # For interval constraints, we need to modify the evaluation
        # This requires computing probability over intervals
        return self._evaluate_with_intervals(
            evidence,
            interval_constraints,
            query_var,
            query_value
        )

    def _evaluate_with_intervals(self, evidence, intervals, query_var=None, query_value=None):
        """
        Evaluate SPN with interval constraints.

        This is the core WMI operation: integrating over continuous intervals.
        """
        # Build complete evidence including query
        full_evidence = evidence.copy()

        if query_var is not None and query_value is not None:
            full_evidence[query_var] = query_value

        # For each interval constraint, we need to compute probability mass
        # This requires traversing the SPN and computing interval probabilities

        # Simplified approach: evaluate at interval midpoints
        # Full implementation would require integration through the SPN

        interval_evidence = {}
        for interval in intervals:
            # Use midpoint of interval as point estimate
            midpoint = (interval.lower_bound + interval.upper_bound) / 2
            interval_evidence[interval.variable_idx] = midpoint

        # Combine all evidence
        full_evidence.update(interval_evidence)

        # Compute probability
        if query_var is None:
            # Just compute marginal over evidence
            return self.spn.marginal(full_evidence)
        else:
            # Compute conditional
            query = {query_var: query_value}
            return self.spn.conditional(query, full_evidence)

    def interval_probability(self, variable_idx, lower_bound, upper_bound, evidence=None):
        """
        Compute P(lower_bound <= X_i < upper_bound | evidence).

        This is the key operation for WMI-based queries.

        Args:
            variable_idx: Variable index
            lower_bound: Lower bound
            upper_bound: Upper bound
            evidence: Optional evidence dict

        Returns:
            Probability (not log)
        """
        if evidence is None:
            evidence = {}

        # Find the leaf node for this variable
        target_leaf = None
        for leaf in self.spn.leaf_nodes:
            if leaf.variable_idx == variable_idx:
                target_leaf = leaf
                break

        if target_leaf is None:
            raise ValueError(f"Variable {variable_idx} not found in SPN")

        if not isinstance(target_leaf, ContinuousLeaf):
            raise ValueError(f"Variable {variable_idx} is not continuous")

        # Get interval probability from polynomial
        interval_prob = target_leaf.query_interval(lower_bound, upper_bound)

        # If there's evidence, we need conditional probability
        if len(evidence) > 0:
            # P(interval | evidence) = P(interval, evidence) / P(evidence)

            # Sample points from interval and compute average probability
            n_samples = 100
            sample_points = np.linspace(lower_bound, upper_bound, n_samples)

            log_probs = []
            for point in sample_points:
                full_evidence = evidence.copy()
                full_evidence[variable_idx] = point
                log_prob = self.spn.evaluate(full_evidence)
                log_probs.append(log_prob)

            # Approximate integral by averaging
            avg_log_prob = np.mean(log_probs)

            # Marginal of evidence
            log_evidence = self.spn.marginal(evidence)

            # Conditional probability
            conditional_log_prob = avg_log_prob - log_evidence

            # Scale by interval width
            interval_width = upper_bound - lower_bound
            prob = np.exp(conditional_log_prob) * interval_width

            return prob
        else:
            # No evidence, just return interval probability
            return interval_prob

    def conjunctive_query(
        self,
        intervals: List[IntervalQuery],
        categorical: Dict[int, any] = None
    ):
        """
        Compute P(interval1 AND interval2 AND ... | categorical).

        Args:
            intervals: List of interval constraints
            categorical: Categorical evidence

        Returns:
            Log probability
        """
        if categorical is None:
            categorical = {}

        # Build evidence with all constraints
        evidence = categorical.copy()

        # For continuous variables, use interval midpoints
        for interval in intervals:
            midpoint = (interval.lower_bound + interval.upper_bound) / 2
            evidence[interval.variable_idx] = midpoint

        return self.spn.marginal(evidence)

    def disjunctive_query(
        self,
        interval_sets: List[List[IntervalQuery]],
        categorical: Dict[int, any] = None
    ):
        """
        Compute P((interval1_1 AND interval1_2) OR (interval2_1 AND interval2_2) | categorical).

        Args:
            interval_sets: List of interval sets (disjunction of conjunctions)
            categorical: Categorical evidence

        Returns:
            Log probability
        """
        from ..utils import log_sum_exp

        # Compute probability of each conjunction
        log_probs = []

        for interval_set in interval_sets:
            log_prob = self.conjunctive_query(interval_set, categorical)
            log_probs.append(log_prob)

        # Sum in probability space (log-sum-exp)
        return log_sum_exp(log_probs)

    def range_query(self, ranges: Dict[int, Tuple[float, float]], evidence: Dict[int, any] = None):
        """
        Convenience method for range queries.

        Args:
            ranges: Dict mapping variable index to (lower, upper) bounds
            evidence: Additional evidence

        Returns:
            Log probability
        """
        intervals = [
            IntervalQuery(var_idx, lower, upper)
            for var_idx, (lower, upper) in ranges.items()
        ]

        return self.conjunctive_query(intervals, evidence)

    def query_string(self, query_str: str):
        """
        Parse and execute query from string.

        Example: "P(class=1 | job=0, 7500 < creditamount < 9000)"

        Args:
            query_str: Query string

        Returns:
            Log probability
        """
        # This is a placeholder for string parsing
        # Full implementation would require a parser
        raise NotImplementedError("String query parsing not yet implemented")

    def explain_query(self, evidence: Dict[int, any], top_k: int = 5):
        """
        Explain which features are most influential given evidence.

        Args:
            evidence: Evidence dict
            top_k: Number of top features to return

        Returns:
            List of (feature_idx, importance_score) tuples
        """
        # Compute marginal probability with evidence
        log_prob_with_evidence = self.spn.marginal(evidence)

        # For each non-evidence variable, compute marginal importance
        all_vars = self.spn.scope
        evidence_vars = set(evidence.keys())
        non_evidence_vars = all_vars - evidence_vars

        importances = []

        for var in non_evidence_vars:
            # Remove this variable and see how probability changes
            reduced_evidence = {k: v for k, v in evidence.items() if k != var}
            log_prob_without = self.spn.marginal(reduced_evidence)

            # Importance is the difference
            importance = abs(log_prob_with_evidence - log_prob_without)
            importances.append((var, importance))

        # Sort by importance
        importances.sort(key=lambda x: x[1], reverse=True)

        return importances[:top_k]
