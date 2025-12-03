"""
Quick test script for WMISPN implementation.

Tests basic functionality on a real dataset.
"""

import numpy as np
import sys
from wmispn import WMISPN
from wmispn.data import Dataset

def load_dataset(name='iris'):
    """Load a benchmark dataset."""
    try:
        data_path = f'data/{name}/{name}.ts.data'
        data = np.loadtxt(data_path)
        print(f"Loaded {name} dataset: {data.shape}")
        return data
    except:
        print(f"Could not load {name} dataset, using synthetic data")
        # Generate synthetic data as fallback
        np.random.seed(42)
        n_samples = 300
        n_features = 5

        # Mixed data
        categorical = np.random.randint(0, 3, (n_samples, 2))
        continuous = np.random.randn(n_samples, 3)

        return np.column_stack([categorical, continuous])


def test_basic_functionality():
    """Test basic WMISPN functionality."""
    print("\n" + "=" * 60)
    print("Testing Basic WMISPN Functionality")
    print("=" * 60)

    # Load data
    data = load_dataset('iris')

    # Split
    n_train = int(0.7 * len(data))
    X_train = data[:n_train]
    X_test = data[n_train:]

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # Create model
    print("\n[1] Creating WMISPN model...")
    model = WMISPN(
        n_bins=3,
        poly_degree=2,
        min_instances=20,
        random_state=42
    )
    print("✓ Model created")

    # Fit model
    print("\n[2] Fitting model...")
    try:
        model.fit(X_train, preprocess=True)
        print("✓ Model fitted successfully")
    except Exception as e:
        print(f"✗ Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get structure info
    print("\n[3] Model structure:")
    info = model.get_structure_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print("✓ Structure info retrieved")

    # Compute log-likelihood
    print("\n[4] Computing log-likelihood...")
    try:
        train_ll = model.log_likelihood(X_train)
        test_ll = model.log_likelihood(X_test)
        print(f"   Train LL: {train_ll.mean():.4f} ± {train_ll.std():.4f}")
        print(f"   Test LL:  {test_ll.mean():.4f} ± {test_ll.std():.4f}")
        print("✓ Log-likelihood computed")
    except Exception as e:
        print(f"✗ Error computing log-likelihood: {e}")
        return False

    # Query
    print("\n[5] Testing query...")
    try:
        instance = {i: X_test[0, i] for i in range(X_test.shape[1])}
        query_vars = {0: instance[0]}
        evidence = {i: instance[i] for i in range(1, X_test.shape[1])}

        prob = model.query(query_vars, evidence)
        print(f"   Query result: {prob:.4f}")
        print("✓ Query executed")
    except Exception as e:
        print(f"✗ Error during query: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Sample
    print("\n[6] Testing sampling...")
    try:
        samples = model.sample(n_samples=10, random_state=42)
        print(f"   Generated samples shape: {samples.shape}")
        print("✓ Sampling successful")
    except Exception as e:
        print(f"✗ Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save and load
    print("\n[7] Testing save/load...")
    try:
        model.save('test_model.pkl')
        loaded_model = WMISPN.load('test_model.pkl')
        loaded_ll = loaded_model.log_likelihood(X_test[:5])
        print(f"   Loaded model LL: {loaded_ll.mean():.4f}")
        print("✓ Save/load successful")
    except Exception as e:
        print(f"✗ Error during save/load: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True


def test_interval_queries():
    """Test interval query functionality."""
    print("\n" + "=" * 60)
    print("Testing Interval Query Functionality")
    print("=" * 60)

    # Generate data with known continuous variables
    np.random.seed(42)
    n_samples = 400

    categorical = np.random.randint(0, 2, (n_samples, 1))
    continuous = np.random.randn(n_samples, 3) * 2

    X = np.column_stack([categorical, continuous])

    variable_types = {
        0: 'categorical',
        1: 'continuous',
        2: 'continuous',
        3: 'continuous'
    }

    X_train = X[:300]
    X_test = X[300:]

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # Fit model
    print("\n[1] Fitting model with explicit variable types...")
    model = WMISPN(n_bins=5, poly_degree=2, min_instances=30)

    try:
        model.fit(X_train, variable_types=variable_types)
        print("✓ Model fitted")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test interval query
    print("\n[2] Testing interval query...")
    try:
        prob = model.interval_query(
            intervals={1: (-1.0, 1.0)},
            evidence={0: 0}
        )
        print(f"   P(-1 < X1 < 1 | X0=0) = {np.exp(prob):.6f}")
        print("✓ Interval query successful")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test multiple intervals
    print("\n[3] Testing multiple intervals...")
    try:
        prob = model.interval_query(
            intervals={1: (-1.0, 1.0), 2: (0.0, 2.0)},
            evidence={0: 1}
        )
        print(f"   P(-1 < X1 < 1 AND 0 < X2 < 2 | X0=1) = {np.exp(prob):.6f}")
        print("✓ Multiple intervals successful")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All interval query tests passed!")
    print("=" * 60)

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WMISPN Implementation Test Suite")
    print("=" * 60)

    success = True

    # Test 1: Basic functionality
    if not test_basic_functionality():
        success = False

    # Test 2: Interval queries
    if not test_interval_queries():
        success = False

    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
