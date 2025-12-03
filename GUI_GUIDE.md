# WMISPN GUI Guide

A modern web interface for the Python WMISPN implementation.

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the GUI:**
   ```bash
   streamlit run wmispn_app.py
   ```

   Or use the launch script:
   ```bash
   ./launch_gui.sh
   ```

3. **Open your browser** - The GUI will automatically open at `http://localhost:8501`

---

## 📋 Features

### 1. Data Loading & Exploration 📊

**Load Data:**
- Upload CSV files
- Use example datasets from `data/` directory
- Generate synthetic data for testing

**Explore Data:**
- **Data Preview**: View your data in a table
- **Statistics**: Comprehensive statistical summary
  - Numeric feature statistics (mean, std, quartiles)
  - Categorical feature distributions
  - Correlation heatmaps
- **Distributions**: Interactive visualizations
  - Histograms for continuous features
  - Bar charts for categorical features
  - Box plots for outlier detection

---

### 2. Model Training 🧠

**Configuration Options:**

#### Presets
- **Default**: Balanced settings for most use cases
- **Aggressive Splitting**: Larger, more complex models
- **Conservative Splitting**: Smaller, simpler models
- **Custom**: Fine-tune all parameters

#### Key Hyperparameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Number of Bins** | 2-10 | 5 | Discretization granularity for continuous features |
| **Polynomial Degree** | 0-3 | 2 | Degree of piecewise polynomials (0=constant, 2=quadratic) |
| **Min Instances** | 10-200 | 50 | Minimum samples needed to split a node |
| **G-Test Factor** | 0.1-5.0 | 1.0 | Independence test sensitivity (higher = more conservative) |
| **Cluster Penalty** | 0.5-10.0 | 2.0 | Penalty for creating clusters (higher = simpler model) |
| **Max Clusters** | 2-20 | 10 | Maximum mixture components per sum node |

#### Data Preprocessing
- **Normalize**: Scale continuous features to [0, 1]
- **Handle Missing**: Choose how to handle missing values (mean, median, drop)
- **Train/Test Split**: Configurable split percentage

**Training Output:**
- Model structure information (nodes, depth, parameters)
- Performance metrics (log-likelihood on train/test)
- Model save/load functionality

---

### 3. Query Interface 🔍

#### Query Types

##### A. Simple Query
Compute `P(query_variables | evidence)`

**Example:**
```
Query: class = 1
Evidence: age = 30, income = 50000
Result: P(class=1 | age=30, income=50000) = 0.73 (73%)
```

**Use Case:** What's the probability of approval given specific conditions?

##### B. Interval Query (Key WMI Feature!)
Compute `P(lower < X < upper | evidence)` for continuous variables

**Example:**
```
Interval: 30000 < income < 50000
Evidence: class = 1, education = 2
Result: P(30k < income < 50k | class=1, edu=2) = 0.42 (42%)
```

**Use Case:** What's the probability that income falls in a specific range?

**Multiple Intervals:**
```
Intervals:
  - 30000 < income < 50000
  - 25 < age < 35
Evidence: class = 1
Result: P(income in range AND age in range | class=1)
```

##### C. Prediction
Predict the most likely value of a target variable

**Example:**
```
Target: income
Evidence: age=35, education=3, employment=2
Result: Predicted income = 65432.12
```

**Use Case:** Given these features, what value should we expect for the target?

##### D. Sampling
Generate new samples from the learned distribution

**Options:**
- Number of samples (1-1000)
- Optional evidence (conditional sampling)
- Random seed for reproducibility

**Output:**
- Samples as a DataFrame
- Downloadable CSV
- Scatter plot visualization

**Use Case:** Generate synthetic data that follows the learned distribution

---

## 🎯 Example Workflows

### Workflow 1: Credit Risk Analysis

1. **Load Data**
   - Upload credit application data (CSV)
   - Features: age, income, credit_score, employment, etc.
   - Check data quality in Statistics tab

2. **Train Model**
   - Use "Default" preset
   - Set test split to 20%
   - Click "Train Model"
   - Review test log-likelihood

3. **Query**
   - **Simple Query**: `P(approval=1 | income=60000, credit_score=720)`
   - **Interval Query**: `P(50000 < income < 75000 | approval=1)`
   - **Prediction**: Predict approval given applicant profile
   - **Sampling**: Generate 100 synthetic approved applicants

### Workflow 2: Exploratory Analysis

1. **Generate Synthetic Data**
   - Click "Generate Synthetic Data"
   - Explore distributions and correlations

2. **Train Multiple Models**
   - Train with "Aggressive" preset → Note test LL
   - Train with "Conservative" preset → Note test LL
   - Compare which performs better

3. **Experiment with Queries**
   - Try different interval ranges
   - Test prediction accuracy
   - Generate and visualize samples

### Workflow 3: Real Dataset Analysis

1. **Load Example Dataset**
   - Select from dropdown (e.g., "australian", "heart")
   - Review data summary

2. **Configure Training**
   - Start with Custom settings
   - Adjust bins based on data distributions
   - Set appropriate cluster penalty

3. **Iterative Refinement**
   - Train model
   - Check test log-likelihood
   - Adjust parameters
   - Retrain and compare

---

## 💡 Tips & Best Practices

### Data Preparation

✅ **DO:**
- Ensure CSV has column headers
- Check for extreme outliers
- Handle missing values appropriately
- Verify data types are correct

❌ **DON'T:**
- Use very sparse datasets (< 100 samples)
- Mix incompatible data types in columns
- Include non-numeric IDs as features

### Model Training

✅ **DO:**
- Start with default parameters
- Use 70/30 or 80/20 train/test split
- Check both train and test log-likelihood
- Try multiple presets to compare

❌ **DON'T:**
- Use too many bins (> 8) for small datasets
- Set min_instances too low (< 20)
- Ignore the test log-likelihood

### Querying

✅ **DO:**
- Start with simple queries to validate
- Use interval queries for continuous features
- Provide reasonable evidence values
- Generate samples to check distributions

❌ **DON'T:**
- Query with values outside data range
- Use contradictory evidence
- Expect exact probabilities for very specific events

---

## 🔧 Troubleshooting

### GUI Won't Start

**Problem:** `streamlit: command not found`

**Solution:**
```bash
pip install streamlit
```

### Model Training Fails

**Problem:** "Error during training: insufficient data"

**Solutions:**
- Reduce `min_instances` parameter
- Use fewer bins for discretization
- Check if data has enough samples (need > 100)

### Slow Training

**Problem:** Training takes very long

**Solutions:**
- Use "Conservative" preset
- Reduce `max_clusters`
- Increase `min_instances`
- Use smaller dataset for testing

### Query Returns NaN or Inf

**Problem:** Query result is NaN or -inf

**Solutions:**
- Check if query values are within data range
- Verify evidence is not contradictory
- Try simpler queries first
- Check if model trained successfully

---

## 📊 Understanding Results

### Log-Likelihood (LL)
- Measures how well model fits data
- Higher is better (less negative)
- Typical range: -10 to -1
- Compare train vs test to check overfitting

### Pseudo Log-Likelihood (PLL)
- Alternative evaluation metric
- Sum of conditional likelihoods
- More robust to model structure
- Higher is better

### Model Structure
- **Total Nodes**: Overall model size
- **Sum Nodes**: Mixture components
- **Product Nodes**: Variable factorizations
- **Depth**: Network depth (affects computation)
- **Parameters**: Number of learned weights

### Probabilities
- All probabilities are between 0 and 1
- Very small probabilities (< 0.01) indicate rare events
- Log probabilities avoid numerical underflow
- Use log-space for comparisons

---

## 🎓 Advanced Features

### Variable Type Detection

The GUI automatically detects variable types:
- **Continuous**: Many unique numeric values (e.g., income, age)
- **Categorical**: Few unique values (e.g., class labels, categories)

You can override auto-detection in Model Training page.

### Piecewise Polynomials

Continuous features are approximated using piecewise polynomials:
- Data is binned into intervals
- Each interval has a polynomial function
- BIC criterion selects best polynomial degree
- Enables efficient integration for WMI

### Interval Queries

The key innovation of WMISPN:
- Can query `P(a < X < b)` exactly
- Integrates polynomial functions over intervals
- Supports multiple intervals simultaneously
- Can be conditioned on evidence

---

## 📚 Additional Resources

### Documentation
- **README_PYTHON.md**: Comprehensive Python API documentation
- **examples/**: Example Python scripts
- **test_wmispn.py**: Test suite showing usage

### Example Datasets
Located in `data/` directory:
- `australian/`: Australian credit approval
- `heart/`: Heart disease diagnosis
- `german/`: German credit data
- And more...

### Citations

If you use WMISPN in research:
```
@article{zuidberg2020wmispn,
  title={Learning Weighted Model Integration Distributions},
  author={Zuidberg Dos Martires, Pedro and others},
  year={2020}
}
```

---

## 🐛 Known Limitations

1. **Large Datasets**: Training can be slow for > 10,000 samples
2. **Many Features**: Performance degrades with > 50 features
3. **Categorical Features**: Limited to discrete values
4. **Interval Queries**: Approximation for very small intervals

---

## 🆘 Support

If you encounter issues:

1. Check this guide first
2. Review error messages in terminal
3. Try simpler configurations
4. Generate synthetic data to test
5. Check Python version (need 3.8+)

---

## 🎨 GUI Customization

The GUI is built with Streamlit and can be customized:

- **Themes**: Use Streamlit's theme settings
- **Layout**: Modify column ratios in code
- **Colors**: Edit CSS in `wmispn_app.py`
- **Features**: Add new pages or query types

---

**Enjoy exploring probabilistic modeling with WMISPN! 🎲**
