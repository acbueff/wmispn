"""
Basic usage example for WMISPN.

This script demonstrates:
1. Loading data
2. Fitting a WMISPN model
3. Computing log-likelihood
4. Querying probabilities
5. Generating samples
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from wmispn import WMISPN

def main():
    print("=" * 60)
    print("WMISPN Basic Usage Example")
    print("=" * 60)

    # Generate synthetic mixed data
    np.random.seed(42)
    n_samples = 500

    # Categorical variables (0, 1, 2)
    categorical_1 = np.random.randint(0, 3, n_samples)
    categorical_2 = np.random.randint(0, 2, n_samples)

    # Continuous variables
    continuous_1 = np.random.randn(n_samples)
    continuous_2 = np.random.randn(n_samples) * 2 + 1

    # Combine into dataset
    X = np.column_stack([categorical_1, continuous_1, categorical_2, continuous_2])

    print(f"\nDataset shape: {X.shape}")
    print(f"First 5 rows:\n{X[:5]}")

    # Specify variable types
    variable_types = {
        0: 'categorical',
        1: 'continuous',
        2: 'categorical',
        3: 'continuous'
    }

    # Split data
    n_train = int(0.7 * n_samples)
    X_train = X[:n_train]
    X_test = X[n_train:]

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create and fit model
    print("\n" + "=" * 60)
    print("Training WMISPN Model")
    print("=" * 60)

    model = WMISPN(
        n_bins=5,           # 5 bins for continuous features
        poly_degree=2,      # Quadratic polynomials
        min_instances=30,   # Minimum 30 instances for splitting
        g_factor=1.0,       # G-test factor
        cluster_penalty=2.0 # Cluster penalty
    )

    model.fit(X_train, variable_types=variable_types)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    # Log-likelihood
    test_ll = model.log_likelihood(X_test)
    print(f"\nTest set log-likelihood:")
    print(f"  Mean: {test_ll.mean():.4f}")
    print(f"  Std:  {test_ll.std():.4f}")

    # Pseudo log-likelihood
    test_pll = model.pseudo_log_likelihood(X_test)
    print(f"\nTest set pseudo log-likelihood:")
    print(f"  Mean: {test_pll.mean():.4f}")

    # Query examples
    print("\n" + "=" * 60)
    print("Query Examples")
    print("=" * 60)

    # Query 1: P(cat1=0 | cat2=1, cont1=0.5)
    prob1 = model.query(
        query_vars={0: 0},
        evidence={2: 1, 1: 0.5}
    )
    print(f"\nP(cat1=0 | cat2=1, cont1=0.5) = {np.exp(prob1):.6f}")

    # Query 2: Interval query
    # P(cont1 in [-1, 1] | cat1=0)
    prob2 = model.interval_query(
        intervals={1: (-1.0, 1.0)},
        evidence={0: 0}
    )
    print(f"P(cont1 in [-1, 1] | cat1=0) = {np.exp(prob2):.6f}")

    # Prediction example
    print("\n" + "=" * 60)
    print("Prediction Example")
    print("=" * 60)

    # Predict categorical_1 given other features
    evidence = {1: 0.5, 2: 1, 3: 1.0}
    prediction = model.predict(evidence, target_var=0)
    print(f"\nPrediction: cat1 = {prediction}")
    print(f"Given evidence: cont1=0.5, cat2=1, cont2=1.0")

    # Sampling
    print("\n" + "=" * 60)
    print("Sampling")
    print("=" * 60)

    samples = model.sample(n_samples=10)
    print(f"\n10 samples from learned distribution:")
    print(samples)

    # Conditional sampling
    print("\nConditional sampling (given cat1=0):")
    conditional_samples = model.sample(n_samples=5, evidence={0: 0})
    print(conditional_samples)

    # Model info
    print("\n" + "=" * 60)
    print("Model Structure Info")
    print("=" * 60)

    info = model.get_structure_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Save model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)

    model.save('wmispn_basic_example.pkl')

    # Load model
    print("\nLoading Model...")
    loaded_model = WMISPN.load('wmispn_basic_example.pkl')

    # Verify loaded model works
    loaded_ll = loaded_model.log_likelihood(X_test[:5])
    print(f"Loaded model test LL (5 samples): {loaded_ll.mean():.4f}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
