"""Independence testing utilities for structure learning."""

import numpy as np
from scipy.stats import chi2
from ..utils import g_test


def g_test_independence(data1, data2, g_factor=1.0, significance=0.05):
    """
    Test independence between two variables using G-test.

    Args:
        data1: 1D array of categorical values for variable 1
        data2: 1D array of categorical values for variable 2
        g_factor: G-test sensitivity multiplier
        significance: Significance level (p-value threshold)

    Returns:
        Boolean indicating whether variables are independent
    """
    data1 = np.asarray(data1).ravel()
    data2 = np.asarray(data2).ravel()

    # Build contingency table
    categories1 = np.unique(data1)
    categories2 = np.unique(data2)

    n_cat1 = len(categories1)
    n_cat2 = len(categories2)

    contingency = np.zeros((n_cat1, n_cat2))

    cat1_map = {val: i for i, val in enumerate(categories1)}
    cat2_map = {val: i for i, val in enumerate(categories2)}

    for v1, v2 in zip(data1, data2):
        i = cat1_map[v1]
        j = cat2_map[v2]
        contingency[i, j] += 1

    # Compute G-test statistic
    g_value = g_test(contingency)

    # Apply g_factor
    g_value *= g_factor

    # Degrees of freedom
    df = (n_cat1 - 1) * (n_cat2 - 1)

    if df <= 0:
        return True  # Degenerate case, assume independent

    # Critical value from chi-squared distribution
    critical_value = chi2.ppf(1 - significance, df)

    # If G-value exceeds critical value, reject independence hypothesis
    is_independent = g_value < critical_value

    return is_independent


def find_independent_set(data, variable_indices, g_factor=1.0, significance=0.05, max_attempts=100):
    """
    Greedy algorithm to find a maximal independent set of variables.

    Args:
        data: Data array (n_samples, n_features)
        variable_indices: List of variable indices to consider
        g_factor: G-test sensitivity factor
        significance: Significance level
        max_attempts: Maximum number of attempts to find independent set

    Returns:
        Tuple of (independent_set, dependent_set)
    """
    if len(variable_indices) <= 1:
        return variable_indices, []

    variable_indices = list(variable_indices)
    np.random.shuffle(variable_indices)  # Randomize order

    independent_set = [variable_indices[0]]
    remaining = variable_indices[1:]

    # Greedy approach: add variables that are independent of all current members
    for var in remaining:
        is_independent = True

        for independent_var in independent_set:
            data1 = data[:, var]
            data2 = data[:, independent_var]

            if not g_test_independence(data1, data2, g_factor, significance):
                is_independent = False
                break

        if is_independent:
            independent_set.append(var)

    dependent_set = [v for v in variable_indices if v not in independent_set]

    return independent_set, dependent_set


def compute_pairwise_dependencies(data, variable_indices, g_factor=1.0):
    """
    Compute pairwise dependency matrix using G-test.

    Args:
        data: Data array
        variable_indices: List of variable indices
        g_factor: G-test factor

    Returns:
        Dependency matrix (higher values = more dependent)
    """
    n_vars = len(variable_indices)
    dependency_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var_i = variable_indices[i]
            var_j = variable_indices[j]

            data1 = data[:, var_i]
            data2 = data[:, var_j]

            # Build contingency table
            categories1 = np.unique(data1)
            categories2 = np.unique(data2)

            contingency = np.zeros((len(categories1), len(categories2)))

            cat1_map = {val: idx for idx, val in enumerate(categories1)}
            cat2_map = {val: idx for idx, val in enumerate(categories2)}

            for v1, v2 in zip(data1, data2):
                contingency[cat1_map[v1], cat2_map[v2]] += 1

            g_value = g_test(contingency) * g_factor

            dependency_matrix[i, j] = g_value
            dependency_matrix[j, i] = g_value

    return dependency_matrix
