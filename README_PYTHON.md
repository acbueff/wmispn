# WMISPN: Weighted Model Integration Sum-Product Networks

A Python implementation of the LearnWMISPN framework for learning probabilistic tractable models in mixed discrete-continuous domains.

## Overview

WMISPN integrates Sum-Product Networks (SPNs) with Weighted Model Integration (WMI) to enable:

- **Tractable inference** in hybrid domains (discrete + continuous)
- **Non-parametric distributions** via piecewise polynomial approximations
- **Complex interval queries** over continuous features
- **Exact probabilistic inference** with polynomial time complexity

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy
- scipy
- scikit-learn
- pandas

## Quick Start

### Basic Usage

```python
from wmispn import WMISPN
import numpy as np

# Load your data
X_train = np.random.randn(1000, 5)

# Create and fit model
model = WMISPN(
    n_bins=5,           # Number of bins for continuous features
    poly_degree=2,      # Polynomial degree
    min_instances=50    # Minimum instances for splitting
)

model.fit(X_train)

# Compute log-likelihood
ll = model.log_likelihood(X_test)
print(f"Average log-likelihood: {ll.mean()}")

# Generate samples
samples = model.sample(n_samples=100)
```

### Query Interface

```python
# Conditional probability query
# P(X_0=1 | X_1=0.5, X_2=0.3)
prob = model.query(
    query_vars={0: 1},
    evidence={1: 0.5, 2: 0.3}
)

# Interval query (key feature!)
# P(class=1 | 7500 < creditamount < 9000, job=0)
prob = model.interval_query(
    intervals={1: (7500, 9000)},  # creditamount between 7500 and 9000
    evidence={0: 1, 2: 0}         # class=1, job=0
)

# Predict most likely value
prediction = model.predict(
    evidence={1: 0.5, 2: 0.3},
    target_var=0
)
```

### Hyperparameter Presets

```python
# Default balanced settings
model = WMISPN.with_preset('default')

# Aggressive splitting (larger, more complex networks)
model = WMISPN.with_preset('aggressive')

# Conservative splitting (smaller, simpler networks)
model = WMISPN.with_preset('conservative')

# Or customize specific parameters
model = WMISPN.with_preset('default', n_bins=10, poly_degree=3)
```

## Advanced Usage

### Working with Different Data Types

```python
from wmispn.data import Dataset

# Specify variable types explicitly
variable_types = {
    0: 'categorical',  # Feature 0 is categorical
    1: 'continuous',   # Feature 1 is continuous
    2: 'continuous',
    3: 'categorical'
}

dataset = Dataset(
    data=X,
    feature_names=['class', 'age', 'income', 'job'],
    variable_types=variable_types
)

model.fit(dataset)
```

### Multiple Likelihood Functions

```python
# Log-likelihood: log P(X)
ll = model.log_likelihood(X_test)

# Pseudo log-likelihood: Σ log P(X_i | X_{-i})
pll = model.pseudo_log_likelihood(X_test)

# Evaluate multiple metrics
metrics = model.evaluate(X_test, metrics=['ll', 'pll'])
print(metrics)
```

### Complex Queries

```python
from wmispn.query import IntervalQuery

# Create interval queries
intervals = [
    IntervalQuery(variable_idx=1, lower_bound=7500, upper_bound=9000),
    IntervalQuery(variable_idx=2, lower_bound=0.5, upper_bound=0.8)
]

# Compute P(intervals | evidence)
prob = model.query_interface.conjunctive_query(
    intervals=intervals,
    categorical={0: 1}
)

# Interval probability for single variable
prob = model.query_interface.interval_probability(
    variable_idx=1,
    lower_bound=7500,
    upper_bound=9000,
    evidence={0: 1}
)
```

### Model Persistence

```python
# Save model
model.save('my_wmispn_model.pkl')

# Load model
model = WMISPN.load('my_wmispn_model.pkl')
```

## Hyperparameters

### Structure Learning Parameters

- **`min_instances`** (default: 50): Minimum instances required for splitting a node
- **`g_factor`** (default: 1.0): Sensitivity factor for G-test independence testing
  - Higher values → more conservative → fewer splits
- **`cluster_penalty`** (default: 2.0): Penalty for creating clusters
  - Higher values → fewer clusters → simpler model
- **`max_clusters`** (default: 10): Maximum clusters in sum nodes
- **`min_clusters`** (default: 2): Minimum clusters for creating sum node

### Continuous Feature Parameters

- **`n_bins`** (default: 5): Number of bins/pieces for piecewise polynomials
- **`poly_degree`** (default: 2): Polynomial degree (0=constant, 1=linear, 2=quadratic, etc.)
- **`binning_method`** (default: 'quantile'): Method for creating bins
  - `'quantile'`: Equal-frequency bins
  - `'uniform'`: Equal-width bins
  - `'kmeans'`: K-means based bins

### Other Parameters

- **`independence_test`** (default: True): Whether to test variable independence
- **`random_state`** (default: None): Random seed for reproducibility

## Architecture

```
wmispn/
├── core/               # Core SPN structures
│   ├── nodes.py       # Sum, Product, Leaf nodes
│   ├── graph.py       # SPN graph container
│   └── polynomial.py  # Piecewise polynomial approximation
├── learning/          # Structure learning
│   ├── structure.py   # LearnWMISPN algorithm
│   ├── independence.py # G-test independence testing
│   └── binning.py     # Continuous feature binning
├── inference/         # Inference engine
│   ├── engine.py      # Main inference engine
│   └── likelihoods.py # Various likelihood functions
├── query/             # Query interface
│   └── interface.py   # Complex interval queries
├── data/              # Data handling
│   ├── dataset.py     # Dataset wrapper
│   └── preprocessing.py # Preprocessing utilities
└── model.py           # Main WMISPN API
```

## Examples

### Example 1: Binary Classification

```python
from wmispn import WMISPN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5)
X_with_labels = np.column_stack([y, X])

X_train, X_test = train_test_split(X_with_labels, test_size=0.2)

# Fit model
model = WMISPN(n_bins=5, poly_degree=2)
model.fit(X_train)

# Query: P(y=1 | features)
for i in range(5):
    evidence = {j+1: X_test[i, j] for j in range(10)}
    prob_class_1 = model.query({0: 1}, evidence)
    print(f"P(class=1 | features) = {np.exp(prob_class_1):.4f}")
```

### Example 2: Density Estimation

```python
# Learn density model
model = WMISPN(n_bins=10, poly_degree=2)
model.fit(X_train)

# Evaluate density
log_probs = model.log_likelihood(X_test)
print(f"Average log-likelihood: {log_probs.mean():.4f}")

# Generate new samples
synthetic_data = model.sample(n_samples=500)
```

### Example 3: Mixed Data Types

```python
import pandas as pd

# Mixed categorical and continuous data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 50000, 60000, 80000, 100000],
    'education': [0, 1, 2, 2, 3],  # categorical
    'married': [0, 1, 1, 1, 0]     # binary
})

# Specify types
variable_types = {
    0: 'continuous',  # age
    1: 'continuous',  # income
    2: 'categorical', # education
    3: 'categorical'  # married
}

model = WMISPN()
model.fit(df.values, variable_types=variable_types)

# Query: P(income in [50k, 70k] | education=2, married=1)
prob = model.interval_query(
    intervals={1: (50000, 70000)},
    evidence={2: 2, 3: 1}
)
```

## Theory

### Piecewise Polynomial Approximation

Continuous distributions are approximated using piecewise polynomials:

```
f(x) = a₀ᵢ + a₁ᵢx + ... + aₙᵢxⁿ   for x ∈ [αᵢ, βᵢ]
       0                          otherwise
```

Probability of interval [α*, β*]:

```
P(x ∈ [α*, β*]) = ∫_{α*}^{β*} f(x)dx
```

### G-test Independence

Variables are tested for independence using the G-test statistic:

```
G = 2 × Σᵢⱼ Oᵢⱼ log(Oᵢⱼ / Eᵢⱼ)
```

where Oᵢⱼ are observed counts and Eᵢⱼ are expected counts under independence.

### Structure Learning Algorithm

LearnWMISPN uses recursive partitioning:

1. **Base case**: Single variable → create leaf node
2. **Independence test**: Find independent variable sets → create product node
3. **Instance clustering**: Cluster data instances → create sum node
4. **Recursion**: Apply recursively to sub-problems

## Citation

If you use this implementation, please cite the original paper:

```
@article{zuidberg2020wmispn,
  title={Learning Weighted Model Integration Distributions},
  author={Zuidberg Dos Martires, Pedro and others},
  journal={...},
  year={2020}
}
```

## License

See LICENSE file.

## Contributing

Contributions welcome! Please open an issue or pull request.
