"""SPN node classes for the WMISPN framework."""

import numpy as np
from abc import ABC, abstractmethod
from ..utils import log_sum_exp, log_normalize


class Node(ABC):
    """Abstract base class for all SPN nodes."""

    def __init__(self, scope):
        """
        Initialize node.

        Args:
            scope: Set of variable indices this node covers
        """
        self.scope = set(scope) if not isinstance(scope, set) else scope
        self.log_value = 0.0  # Cached log probability
        self.dirty = True  # Whether cached value needs recomputation
        self.node_id = None  # Assigned by graph

    @abstractmethod
    def eval(self, data):
        """
        Evaluate node on data.

        Args:
            data: Data instance (dict or array)

        Returns:
            Log probability
        """
        pass

    @abstractmethod
    def sample(self, n_samples=1, evidence=None, random_state=None):
        """
        Sample from this node.

        Args:
            n_samples: Number of samples
            evidence: Evidence dict {var_idx: value}
            random_state: Random seed

        Returns:
            Samples dict {var_idx: values}
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={sorted(self.scope)})"


class SumNode(Node):
    """
    Sum node representing a mixture distribution.

    P(X) = Σ w_i * P_i(X)
    """

    def __init__(self, scope, children=None, weights=None):
        """
        Initialize sum node.

        Args:
            scope: Variable scope
            children: List of child nodes
            weights: Mixture weights (will be normalized)
        """
        super().__init__(scope)
        self.children = children if children is not None else []

        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
            # Ensure weights are valid
            self.weights = np.maximum(self.weights, 1e-10)
            self.weights /= self.weights.sum()
            self.log_weights = np.log(self.weights)
        else:
            n = len(self.children)
            self.weights = np.ones(n) / n if n > 0 else np.array([])
            self.log_weights = np.log(self.weights) if n > 0 else np.array([])

    def add_child(self, child, weight=1.0):
        """Add child node with weight."""
        self.children.append(child)
        new_weight = np.append(self.weights, weight)
        new_weight /= new_weight.sum()
        self.weights = new_weight
        self.log_weights = np.log(self.weights)

    def eval(self, data):
        """
        Evaluate sum node: log(Σ w_i * exp(log P_i))

        Args:
            data: Data instance

        Returns:
            Log probability
        """
        if len(self.children) == 0:
            return -np.inf

        log_probs = np.array([child.eval(data) for child in self.children])
        self.log_value = log_sum_exp(self.log_weights + log_probs)
        return self.log_value

    def sample(self, n_samples=1, evidence=None, random_state=None):
        """Sample from mixture by first selecting component."""
        # Handle random state properly
        if isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = np.random.RandomState(random_state)

        if len(self.children) == 0:
            return {var: np.array([]) for var in self.scope}

        # Select components according to weights
        component_indices = rng.choice(
            len(self.children), size=n_samples, p=self.weights
        )

        # Sample from selected components
        all_samples = {var: [] for var in self.scope}

        for i in range(n_samples):
            component = self.children[component_indices[i]]
            # Pass the rng object directly
            sample = component.sample(n_samples=1, evidence=evidence, random_state=rng)
            for var in self.scope:
                if var in sample:
                    all_samples[var].append(sample[var][0])

        return {var: np.array(vals) for var, vals in all_samples.items()}

    def update_weights(self, new_weights):
        """Update mixture weights."""
        self.weights = np.asarray(new_weights, dtype=float)
        self.weights = np.maximum(self.weights, 1e-10)
        self.weights /= self.weights.sum()
        self.log_weights = np.log(self.weights)


class ProductNode(Node):
    """
    Product node representing factorization/independence.

    P(X1, X2, ...) = P(X1) * P(X2) * ...
    """

    def __init__(self, scope, children=None):
        """
        Initialize product node.

        Args:
            scope: Variable scope (union of children's scopes)
            children: List of child nodes with disjoint scopes
        """
        super().__init__(scope)
        self.children = children if children is not None else []

    def add_child(self, child):
        """Add child node."""
        self.children.append(child)
        self.scope.update(child.scope)

    def eval(self, data):
        """
        Evaluate product node: log(Π P_i) = Σ log P_i

        Args:
            data: Data instance

        Returns:
            Log probability
        """
        if len(self.children) == 0:
            return 0.0

        self.log_value = sum(child.eval(data) for child in self.children)
        return self.log_value

    def sample(self, n_samples=1, evidence=None, random_state=None):
        """Sample from product by sampling each factor independently."""
        # Handle random state properly
        if isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = np.random.RandomState(random_state)

        all_samples = {}

        for child in self.children:
            child_samples = child.sample(n_samples=n_samples, evidence=evidence, random_state=rng)
            all_samples.update(child_samples)

        return all_samples


class LeafNode(Node):
    """
    Base class for leaf nodes representing univariate distributions.
    """

    def __init__(self, scope, variable_idx):
        """
        Initialize leaf node.

        Args:
            scope: Variable scope (should be single variable)
            variable_idx: Index of the variable
        """
        super().__init__(scope)
        self.variable_idx = variable_idx

    @abstractmethod
    def fit(self, data):
        """
        Fit distribution to data.

        Args:
            data: 1D array of values
        """
        pass


class CategoricalLeaf(LeafNode):
    """
    Leaf node for discrete/categorical variables.

    Uses smoothed multinomial distribution with Laplace smoothing.
    """

    def __init__(self, scope, variable_idx, n_categories=None, smoothing=1.0):
        """
        Initialize categorical leaf.

        Args:
            scope: Variable scope
            variable_idx: Variable index
            n_categories: Number of categories
            smoothing: Laplace smoothing parameter (default: 1.0)
        """
        super().__init__(scope, variable_idx)
        self.n_categories = n_categories
        self.smoothing = smoothing
        self.probs = None
        self.log_probs = None

    def fit(self, data):
        """
        Fit categorical distribution with Laplace smoothing.

        Args:
            data: 1D array of categorical values (integers)
        """
        data = np.asarray(data).ravel()

        if self.n_categories is None:
            self.n_categories = int(data.max()) + 1

        # Count occurrences with smoothing
        counts = np.zeros(self.n_categories) + self.smoothing
        for val in data:
            if 0 <= val < self.n_categories:
                counts[int(val)] += 1

        # Normalize to probabilities
        self.probs = counts / counts.sum()
        self.log_probs = np.log(self.probs)

    def eval(self, data):
        """
        Evaluate categorical probability.

        Args:
            data: Data instance (dict or array-like)

        Returns:
            Log probability
        """
        if isinstance(data, dict):
            value = data.get(self.variable_idx, None)
        else:
            value = data[self.variable_idx] if self.variable_idx < len(data) else None

        if value is None:
            # Marginalize over all categories
            return 0.0

        value = int(value)
        if 0 <= value < self.n_categories:
            return self.log_probs[value]
        else:
            return -np.inf

    def sample(self, n_samples=1, evidence=None, random_state=None):
        """Sample from categorical distribution."""
        # Handle random state properly
        if isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = np.random.RandomState(random_state)

        if evidence and self.variable_idx in evidence:
            # Return evidence
            samples = np.full(n_samples, evidence[self.variable_idx])
        else:
            # Sample from categorical
            samples = rng.choice(self.n_categories, size=n_samples, p=self.probs)

        return {self.variable_idx: samples}

    def __repr__(self):
        return f"CategoricalLeaf(var={self.variable_idx}, categories={self.n_categories})"


class ContinuousLeaf(LeafNode):
    """
    Leaf node for continuous variables using piecewise polynomial approximation.

    This implements the WMI integration with polynomial weights.
    """

    def __init__(self, scope, variable_idx, polynomial=None):
        """
        Initialize continuous leaf.

        Args:
            scope: Variable scope
            variable_idx: Variable index
            polynomial: PiecewisePolynomial instance
        """
        super().__init__(scope, variable_idx)
        self.polynomial = polynomial

    def fit(self, data, n_pieces=5, degree=2, method='quantile'):
        """
        Fit piecewise polynomial to continuous data.

        Args:
            data: 1D array of continuous values
            n_pieces: Number of polynomial pieces
            degree: Polynomial degree
            method: Binning method ('quantile', 'uniform', 'kmeans')
        """
        from .polynomial import PiecewisePolynomial

        data = np.asarray(data).ravel()
        self.polynomial = PiecewisePolynomial.fit(data, n_pieces, degree, method)

    def eval(self, data):
        """
        Evaluate continuous probability density.

        Args:
            data: Data instance (dict or array-like)

        Returns:
            Log probability density
        """
        if self.polynomial is None:
            return -np.inf

        if isinstance(data, dict):
            value = data.get(self.variable_idx, None)
        else:
            value = data[self.variable_idx] if self.variable_idx < len(data) else None

        if value is None:
            # Marginalize
            return 0.0

        pdf_value = self.polynomial.pdf(value)
        if pdf_value > 0:
            return np.log(pdf_value)
        else:
            return -np.inf

    def sample(self, n_samples=1, evidence=None, random_state=None):
        """Sample from piecewise polynomial distribution."""
        # Handle random state properly
        if isinstance(random_state, np.random.RandomState):
            # Extract seed from random state
            seed = random_state.randint(0, 2**31 - 1)
        else:
            seed = random_state

        if evidence and self.variable_idx in evidence:
            # Return evidence
            samples = np.full(n_samples, evidence[self.variable_idx])
        else:
            # Sample from polynomial
            samples = self.polynomial.sample(n_samples, seed)

        return {self.variable_idx: samples}

    def query_interval(self, lower, upper):
        """
        Compute probability of interval [lower, upper].

        This is the key operation for WMI-based interval queries.

        Args:
            lower: Lower bound
            upper: Upper bound

        Returns:
            Probability mass in interval
        """
        if self.polynomial is None:
            return 0.0

        return self.polynomial.interval_probability(lower, upper)

    def __repr__(self):
        return f"ContinuousLeaf(var={self.variable_idx}, polynomial={self.polynomial})"
