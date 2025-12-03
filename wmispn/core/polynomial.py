"""Piecewise polynomial approximation for continuous distributions."""

import numpy as np
from scipy import integrate
from scipy.optimize import minimize
from ..utils import compute_bic


class PiecewisePolynomial:
    """
    Piecewise polynomial density approximation for continuous features.

    Represents f(x) as:
        f(x) = a_0i + a_1i*x + ... + a_ni*x^n  for x in interval A_i
               0                                 otherwise
    """

    def __init__(self, intervals, coefficients, degree=2):
        """
        Initialize piecewise polynomial.

        Args:
            intervals: List of (lower, upper) interval bounds
            coefficients: List of coefficient arrays for each interval
            degree: Polynomial degree (default: 2 for quadratic)
        """
        self.intervals = np.array(intervals)
        self.coefficients = [np.array(c) for c in coefficients]
        self.degree = degree
        self.n_pieces = len(intervals)

    @classmethod
    def fit(cls, data, n_pieces=5, degree=2, method='quantile'):
        """
        Fit piecewise polynomial to data using BIC for model selection.

        Args:
            data: 1D array of continuous values
            n_pieces: Number of pieces (intervals)
            degree: Maximum polynomial degree to consider
            method: Binning method ('quantile', 'uniform', 'kmeans')

        Returns:
            PiecewisePolynomial instance
        """
        data = np.asarray(data).ravel()

        # Create intervals
        if method == 'quantile':
            quantiles = np.linspace(0, 100, n_pieces + 1)
            edges = np.percentile(data, quantiles)
        elif method == 'uniform':
            edges = np.linspace(data.min(), data.max(), n_pieces + 1)
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_pieces, random_state=42)
            kmeans.fit(data.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.ravel())
            edges = [data.min()]
            for i in range(len(centers) - 1):
                edges.append((centers[i] + centers[i + 1]) / 2)
            edges.append(data.max())
            edges = np.array(edges)
        else:
            raise ValueError(f"Unknown binning method: {method}")

        # Ensure unique edges
        edges = np.unique(edges)
        n_pieces = len(edges) - 1

        intervals = [(edges[i], edges[i + 1]) for i in range(n_pieces)]

        # Try different polynomial degrees and select best by BIC
        best_bic = np.inf
        best_coeffs = None
        best_degree = 0

        for d in range(degree + 1):
            coeffs = []
            log_likelihood = 0
            n_params = 0

            for lower, upper in intervals:
                # Get data in this interval
                mask = (data >= lower) & (data < upper)
                if mask.sum() == 0:
                    # Empty interval - use uniform
                    if d == 0:
                        coeffs.append([1.0 / (upper - lower)])
                    else:
                        coeffs.append([1.0 / (upper - lower)] + [0] * d)
                    n_params += 1
                    continue

                interval_data = data[mask]
                n_params += d + 1

                # Fit polynomial using least squares with normalization constraint
                coeff = cls._fit_interval_polynomial(interval_data, lower, upper, d)
                coeffs.append(coeff)

                # Compute log-likelihood for this interval
                pdf_vals = cls._eval_polynomial(interval_data, coeff)
                pdf_vals = np.maximum(pdf_vals, 1e-10)  # Avoid log(0)
                log_likelihood += np.sum(np.log(pdf_vals))

            bic = compute_bic(log_likelihood, n_params, len(data))

            if bic < best_bic:
                best_bic = bic
                best_coeffs = coeffs
                best_degree = d

        return cls(intervals, best_coeffs, best_degree)

    @staticmethod
    def _fit_interval_polynomial(data, lower, upper, degree):
        """
        Fit polynomial to data in an interval with normalization constraint.

        The polynomial must integrate to 1 over [lower, upper].
        """
        # Use method of moments or simple histogram-based fitting
        # For simplicity, use histogram with polynomial interpolation

        if degree == 0:
            # Uniform distribution
            return [1.0 / (upper - lower)]

        # Normalize data to [0, 1] for numerical stability
        data_norm = (data - lower) / (upper - lower)

        # Fit polynomial using least squares with density estimation
        # We'll use kernel density estimation then fit polynomial to it

        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(data_norm)
            x_samples = np.linspace(0, 1, 100)
            y_samples = kde(x_samples)

            # Fit polynomial to KDE
            coeffs_norm = np.polyfit(x_samples, y_samples, degree)[::-1]  # Reverse for a0, a1, ...

            # Transform back to original scale
            # f(x) = f_norm((x - lower) / (upper - lower)) / (upper - lower)
            coeffs = coeffs_norm / (upper - lower)

            # Normalize to integrate to 1
            integral = PiecewisePolynomial._integrate_polynomial(coeffs, lower, upper)
            if integral > 0:
                coeffs = coeffs / integral

        except:
            # Fallback to uniform
            coeffs = [1.0 / (upper - lower)] + [0] * degree

        return coeffs

    @staticmethod
    def _eval_polynomial(x, coeffs):
        """Evaluate polynomial at points x."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for i, c in enumerate(coeffs):
            result += c * (x ** i)
        return result

    @staticmethod
    def _integrate_polynomial(coeffs, lower, upper):
        """Integrate polynomial over [lower, upper]."""
        result = 0
        for i, c in enumerate(coeffs):
            result += c * (upper ** (i + 1) - lower ** (i + 1)) / (i + 1)
        return result

    def pdf(self, x):
        """
        Evaluate probability density at points x.

        Args:
            x: Points to evaluate (scalar or array)

        Returns:
            Density values
        """
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        result = np.zeros_like(x, dtype=float)

        for i, (lower, upper) in enumerate(self.intervals):
            mask = (x >= lower) & (x < upper)
            if mask.any():
                result[mask] = self._eval_polynomial(x[mask], self.coefficients[i])

        # Ensure non-negative
        result = np.maximum(result, 0)

        return result.item() if scalar_input else result

    def cdf(self, x):
        """
        Evaluate cumulative distribution function at points x.

        Args:
            x: Points to evaluate (scalar or array)

        Returns:
            CDF values
        """
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        result = np.zeros_like(x, dtype=float)

        for i, (lower, upper) in enumerate(self.intervals):
            # Add contribution from all previous intervals
            if i > 0:
                for j in range(i):
                    prev_lower, prev_upper = self.intervals[j]
                    integral = self._integrate_polynomial(
                        self.coefficients[j], prev_lower, prev_upper
                    )
                    result += integral

            # Add contribution from current interval up to x
            mask = x >= lower
            if mask.any():
                x_clip = np.minimum(x[mask], upper)
                for k in range(len(x_clip)):
                    integral = self._integrate_polynomial(
                        self.coefficients[i], lower, x_clip[k]
                    )
                    result[np.where(mask)[0][k]] += integral

        return result.item() if scalar_input else result

    def interval_probability(self, lower_bound, upper_bound):
        """
        Compute probability of interval [lower_bound, upper_bound].

        This is the key operation for WMI-based queries.

        Args:
            lower_bound: Lower bound of query interval
            upper_bound: Upper bound of query interval

        Returns:
            Probability mass in the interval
        """
        probability = 0.0

        for i, (a, b) in enumerate(self.intervals):
            # Find intersection of [lower_bound, upper_bound] with [a, b]
            intersect_lower = max(lower_bound, a)
            intersect_upper = min(upper_bound, b)

            if intersect_lower < intersect_upper:
                # Compute integral over intersection
                prob = self._integrate_polynomial(
                    self.coefficients[i], intersect_lower, intersect_upper
                )
                probability += prob

        return max(0.0, min(1.0, probability))

    def sample(self, n_samples=1, random_state=None):
        """
        Sample from the piecewise polynomial distribution.

        Args:
            n_samples: Number of samples
            random_state: Random seed

        Returns:
            Array of samples
        """
        rng = np.random.RandomState(random_state)

        # Compute probability mass in each interval
        probs = []
        for i, (lower, upper) in enumerate(self.intervals):
            prob = self._integrate_polynomial(self.coefficients[i], lower, upper)
            probs.append(max(0, prob))

        probs = np.array(probs)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)

        # Sample intervals
        interval_indices = rng.choice(len(self.intervals), size=n_samples, p=probs)

        samples = []
        for idx in interval_indices:
            lower, upper = self.intervals[idx]
            # Use rejection sampling within interval
            coeffs = self.coefficients[idx]

            # Find max density in interval
            x_test = np.linspace(lower, upper, 100)
            max_density = np.max(self._eval_polynomial(x_test, coeffs))
            max_density = max(max_density, 1e-10)

            # Rejection sampling
            while True:
                x_proposal = rng.uniform(lower, upper)
                density = self._eval_polynomial(x_proposal, coeffs)
                if rng.uniform(0, max_density) <= density:
                    samples.append(x_proposal)
                    break

        return np.array(samples)

    def __repr__(self):
        return f"PiecewisePolynomial(n_pieces={self.n_pieces}, degree={self.degree})"
