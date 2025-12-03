"""
Advanced query example for WMISPN.

This script demonstrates:
1. Complex interval queries
2. Conjunctive and disjunctive queries
3. Interval probability computation
4. Query explanation
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from wmispn import WMISPN
from wmispn.query import IntervalQuery

def main():
    print("=" * 70)
    print("WMISPN Advanced Query Example")
    print("=" * 70)

    # Generate synthetic credit scoring data
    np.random.seed(42)
    n_samples = 1000

    # Simulate credit scoring scenario
    # Variables:
    # 0: class (0=good, 1=bad)
    # 1: credit_amount (continuous, 0-10000)
    # 2: job_type (0=unemployed, 1=unskilled, 2=skilled, 3=highly_skilled)
    # 3: age (continuous, 18-70)
    # 4: owns_property (0=no, 1=yes)

    # Generate correlated data
    job_type = np.random.randint(0, 4, n_samples)
    owns_property = np.random.randint(0, 2, n_samples)

    credit_amount = np.random.uniform(1000, 10000, n_samples)
    age = np.random.uniform(18, 70, n_samples)

    # Class depends on other variables (simulate realistic scenario)
    class_prob = (
        0.3 +
        0.2 * (job_type >= 2) +
        0.2 * owns_property +
        0.1 * (credit_amount < 5000) +
        0.1 * (age > 30)
    )
    credit_class = (np.random.rand(n_samples) < class_prob).astype(int)

    X = np.column_stack([credit_class, credit_amount, job_type, age, owns_property])

    print(f"\nCredit Scoring Dataset: {X.shape}")
    print(f"Features: class, credit_amount, job_type, age, owns_property")

    # Variable types
    variable_types = {
        0: 'categorical',  # class
        1: 'continuous',   # credit_amount
        2: 'categorical',  # job_type
        3: 'continuous',   # age
        4: 'categorical'   # owns_property
    }

    feature_names = ['class', 'credit_amount', 'job_type', 'age', 'owns_property']

    # Split and train
    X_train = X[:700]
    X_test = X[700:]

    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)

    model = WMISPN(
        n_bins=8,
        poly_degree=2,
        min_instances=40,
        g_factor=0.8
    )

    model.fit(X_train, feature_names=feature_names, variable_types=variable_types)

    # Test evaluation
    test_ll = model.average_log_likelihood(X_test)
    print(f"\nTest average log-likelihood: {test_ll:.4f}")

    # Advanced queries
    print("\n" + "=" * 70)
    print("Advanced Query Examples")
    print("=" * 70)

    # Query 1: Simple interval query
    # P(7500 < credit_amount < 9000)
    print("\n[Query 1] P(7500 < credit_amount < 9000)")
    prob1 = model.interval_query(
        intervals={1: (7500, 9000)}
    )
    print(f"  Result: {np.exp(prob1):.6f}")

    # Query 2: Interval with categorical evidence
    # P(7500 < credit_amount < 9000 | class=GOOD, job=skilled)
    print("\n[Query 2] P(7500 < credit_amount < 9000 | class=GOOD, job=skilled)")
    prob2 = model.interval_query(
        intervals={1: (7500, 9000)},
        evidence={0: 0, 2: 2}
    )
    print(f"  Result: {np.exp(prob2):.6f}")

    # Query 3: Multiple intervals
    # P(7500 < credit_amount < 9000 AND 30 < age < 40 | class=BAD)
    print("\n[Query 3] P(7500 < credit_amount < 9000 AND 30 < age < 40 | class=BAD)")
    prob3 = model.interval_query(
        intervals={1: (7500, 9000), 3: (30, 40)},
        evidence={0: 1}
    )
    print(f"  Result: {np.exp(prob3):.6f}")

    # Query 4: Using IntervalQuery objects
    print("\n[Query 4] Using IntervalQuery objects")
    intervals = [
        IntervalQuery(variable_idx=1, lower_bound=5000, upper_bound=7000),
        IntervalQuery(variable_idx=3, lower_bound=25, upper_bound=35)
    ]

    prob4 = model.query_interface.conjunctive_query(
        intervals=intervals,
        categorical={0: 0, 4: 1}
    )
    print(f"  P(5000 < credit < 7000 AND 25 < age < 35 | class=GOOD, owns_property=YES)")
    print(f"  Result: {np.exp(prob4):.6f}")

    # Query 5: Interval probability for single variable
    print("\n[Query 5] Interval probability with query interface")
    prob5 = model.query_interface.interval_probability(
        variable_idx=1,
        lower_bound=5000,
        upper_bound=8000,
        evidence={0: 0, 2: 3}
    )
    print(f"  P(5000 < credit_amount < 8000 | class=GOOD, job=highly_skilled)")
    print(f"  Result: {prob5:.6f}")

    # Query 6: Disjunctive query
    # P((interval1) OR (interval2) | evidence)
    print("\n[Query 6] Disjunctive query")
    interval_set_1 = [IntervalQuery(1, 1000, 3000)]
    interval_set_2 = [IntervalQuery(1, 8000, 10000)]

    prob6 = model.query_interface.disjunctive_query(
        interval_sets=[interval_set_1, interval_set_2],
        categorical={0: 1}
    )
    print(f"  P((1000 < credit < 3000) OR (8000 < credit < 10000) | class=BAD)")
    print(f"  Result: {np.exp(prob6):.6f}")

    # Compare different credit ranges for good vs bad class
    print("\n" + "=" * 70)
    print("Credit Amount Analysis by Class")
    print("=" * 70)

    credit_ranges = [
        (1000, 3000, "Low"),
        (3000, 5000, "Medium-Low"),
        (5000, 7000, "Medium"),
        (7000, 9000, "Medium-High"),
        (9000, 10000, "High")
    ]

    print("\nP(credit_range | class=GOOD):")
    for lower, upper, label in credit_ranges:
        prob = model.interval_query(
            intervals={1: (lower, upper)},
            evidence={0: 0}
        )
        print(f"  {label:12} [{lower:5}-{upper:5}]: {np.exp(prob):.6f}")

    print("\nP(credit_range | class=BAD):")
    for lower, upper, label in credit_ranges:
        prob = model.interval_query(
            intervals={1: (lower, upper)},
            evidence={0: 1}
        )
        print(f"  {label:12} [{lower:5}-{upper:5}]: {np.exp(prob):.6f}")

    # Age analysis
    print("\n" + "=" * 70)
    print("Age Analysis by Job Type")
    print("=" * 70)

    age_ranges = [
        (18, 25, "Young"),
        (25, 35, "Early Career"),
        (35, 50, "Mid Career"),
        (50, 70, "Senior")
    ]

    job_types = [
        (0, "Unemployed"),
        (1, "Unskilled"),
        (2, "Skilled"),
        (3, "Highly Skilled")
    ]

    print("\nP(age_range | job_type):")
    for job_id, job_name in job_types:
        print(f"\n{job_name}:")
        for lower, upper, label in age_ranges:
            prob = model.interval_query(
                intervals={3: (lower, upper)},
                evidence={2: job_id}
            )
            print(f"  {label:15} [{lower:2}-{upper:2}]: {np.exp(prob):.6f}")

    # Prediction with intervals
    print("\n" + "=" * 70)
    print("Risk Prediction Examples")
    print("=" * 70)

    scenarios = [
        {
            'desc': 'Young, unemployed, low credit',
            'evidence': {1: 2000, 2: 0, 3: 22, 4: 0}
        },
        {
            'desc': 'Middle-aged, skilled, medium credit, owns property',
            'evidence': {1: 6000, 2: 2, 3: 40, 4: 1}
        },
        {
            'desc': 'Senior, highly skilled, high credit',
            'evidence': {1: 9000, 2: 3, 3: 55, 4: 1}
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['desc']}:")

        # Predict class
        predicted_class = model.predict(scenario['evidence'], target_var=0)
        print(f"  Predicted class: {'GOOD' if predicted_class == 0 else 'BAD'}")

        # Probability of each class
        prob_good = model.query({0: 0}, scenario['evidence'])
        prob_bad = model.query({0: 1}, scenario['evidence'])

        print(f"  P(class=GOOD | evidence): {np.exp(prob_good):.6f}")
        print(f"  P(class=BAD | evidence):  {np.exp(prob_bad):.6f}")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
