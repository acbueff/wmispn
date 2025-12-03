"""Various likelihood functions for WMISPN inference."""

import numpy as np


def log_likelihood(spn_graph, data):
    """
    Compute log-likelihood: log P(X).

    Args:
        spn_graph: SPNGraph instance
        data: Data array or list of instances

    Returns:
        Array of log-likelihoods
    """
    return spn_graph.evaluate_batch(data)


def conditional_log_likelihood(spn_graph, data, condition_vars):
    """
    Compute conditional log-likelihood: log P(X_target | X_condition).

    Args:
        spn_graph: SPNGraph instance
        data: Data array (n_samples, n_features)
        condition_vars: List of variable indices to condition on

    Returns:
        Array of conditional log-likelihoods
    """
    if isinstance(data, np.ndarray):
        n_samples = data.shape[0]
        n_features = data.shape[1]
    else:
        n_samples = len(data)
        n_features = len(data[0]) if isinstance(data[0], dict) else data[0].shape[0]

    all_vars = set(range(n_features))
    target_vars = list(all_vars - set(condition_vars))

    cll = []

    for i in range(n_samples):
        if isinstance(data, np.ndarray):
            instance = {j: data[i, j] for j in range(n_features)}
        else:
            instance = data[i]

        # P(target, condition)
        log_joint = spn_graph.evaluate(instance)

        # P(condition)
        evidence = {j: instance[j] for j in condition_vars}
        log_evidence = spn_graph.marginal(evidence)

        # P(target | condition) = P(target, condition) / P(condition)
        conditional = log_joint - log_evidence

        cll.append(conditional)

    return np.array(cll)


def pseudo_log_likelihood(spn_graph, data):
    """
    Compute pseudo log-likelihood (PLL).

    PLL = Σ_i log P(X_i | X_{-i})

    Args:
        spn_graph: SPNGraph instance
        data: Data array (n_samples, n_features)

    Returns:
        Array of pseudo log-likelihoods (per sample)
    """
    if isinstance(data, np.ndarray):
        n_samples = data.shape[0]
        n_features = data.shape[1]
    else:
        n_samples = len(data)
        n_features = len(data[0]) if isinstance(data[0], dict) else data[0].shape[0]

    pll = []

    for i in range(n_samples):
        if isinstance(data, np.ndarray):
            instance = {j: data[i, j] for j in range(n_features)}
        else:
            instance = data[i]

        sample_pll = 0

        # For each variable, compute P(X_j | X_{-j})
        for j in range(n_features):
            # Create evidence with all variables except j
            evidence = {k: v for k, v in instance.items() if k != j}

            # P(X_j, X_{-j})
            log_joint = spn_graph.evaluate(instance)

            # P(X_{-j})
            log_evidence = spn_graph.marginal(evidence)

            # P(X_j | X_{-j})
            conditional = log_joint - log_evidence

            sample_pll += conditional

        pll.append(sample_pll)

    return np.array(pll)


def marginal_log_likelihood(spn_graph, data, marginal_vars):
    """
    Compute marginal log-likelihood: log P(X_marginal).

    Args:
        spn_graph: SPNGraph instance
        data: Data array (n_samples, n_features)
        marginal_vars: List of variable indices to marginalize over

    Returns:
        Array of marginal log-likelihoods
    """
    if isinstance(data, np.ndarray):
        n_samples = data.shape[0]
        n_features = data.shape[1]
    else:
        n_samples = len(data)
        n_features = len(data[0]) if isinstance(data[0], dict) else data[0].shape[0]

    all_vars = set(range(n_features))
    evidence_vars = list(all_vars - set(marginal_vars))

    mll = []

    for i in range(n_samples):
        if isinstance(data, np.ndarray):
            instance = {j: data[i, j] for j in range(n_features)}
        else:
            instance = data[i]

        # Extract evidence
        evidence = {j: instance[j] for j in evidence_vars if j in instance}

        # Compute marginal
        log_marginal = spn_graph.marginal(evidence)

        mll.append(log_marginal)

    return np.array(mll)


def compute_mpe(spn_graph, evidence):
    """
    Compute Most Probable Explanation (MPE) given evidence.

    Note: This is a simplified greedy approximation.

    Args:
        spn_graph: SPNGraph instance
        evidence: Dict of observed variables {var_idx: value}

    Returns:
        Dict of all variable assignments
    """
    # This is a placeholder for MPE computation
    # Full MPE requires max-product inference which is more complex
    # For now, return evidence
    return evidence
