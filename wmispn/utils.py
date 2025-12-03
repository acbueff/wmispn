"""Utility functions for WMISPN."""

import numpy as np
from scipy.special import logsumexp


def log_sum_exp(log_values):
    """
    Numerically stable log-sum-exp computation.

    Args:
        log_values: Array of log values

    Returns:
        log(sum(exp(log_values)))
    """
    return logsumexp(log_values)


def log_normalize(log_weights):
    """
    Normalize log weights to sum to 1 in probability space.

    Args:
        log_weights: Array of log weights

    Returns:
        Normalized log weights
    """
    log_z = log_sum_exp(log_weights)
    return log_weights - log_z


def g_test(contingency_table):
    """
    Compute G-test statistic for independence testing.

    G = 2 * Σ O_ij * log(O_ij / E_ij)

    Args:
        contingency_table: 2D array of observed counts

    Returns:
        G-test statistic value
    """
    observed = np.array(contingency_table, dtype=float)
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    # Expected counts under independence
    expected = (row_sums @ col_sums) / total

    # Avoid log(0) issues
    mask = (observed > 0) & (expected > 0)

    g_value = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))

    return g_value


def compute_bic(log_likelihood, n_params, n_samples):
    """
    Compute Bayesian Information Criterion.

    BIC = -2 * log_likelihood + n_params * log(n_samples)

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters
        n_samples: Number of samples

    Returns:
        BIC score (lower is better)
    """
    return -2 * log_likelihood + n_params * np.log(n_samples)
