# WMI-SPN GUI Interface

A graphical user interface for learning Sum-Product Networks (SPNs) from tabular data and performing probabilistic inference.

## Features

- **Data Upload**: Load CSV datasets or use example datasets
- **Automatic Data Summary**: View comprehensive statistics about your data including:
  - Basic statistics (mean, std, min, max, etc.)
  - Missing value analysis
  - Feature correlations
  - Distribution visualizations
- **Model Learning**: Learn SPN models from data with configurable parameters
- **Probabilistic Queries**: Ask probabilistic questions about your data
- **Interactive Visualizations**: Explore data relationships and distributions

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Java JDK 8+** (for SPN learning backend)

### Setup Steps

1. **Clone or navigate to the repository**:
   ```bash
   cd /path/to/wmispn
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Compile Java code** (if not already compiled):
   ```bash
   mkdir -p bin
   javac -d bin -sourcepath src src/exp/RunSLSPN.java
   ```

## Running the GUI

### Method 1: Streamlit Web Interface (Recommended)

```bash
streamlit run wmi_spn_gui.py
```

This will open a web browser with the interactive interface.

### Quick Start

1. **Load Data**:
   - Click "Browse files" to upload your CSV dataset
   - Or select one of the example datasets from the dropdown

2. **Explore Data**:
   - Navigate to the "Data Preview" tab to see your data
   - Check the "Summary" tab for comprehensive statistics and visualizations

3. **Learn a Model**:
   - Configure learning parameters in the sidebar:
     - **Discretization Method**: How continuous variables are converted to discrete
     - **Number of Bins**: How many bins for discretization (2-10)
     - **Cluster Penalty**: Controls model complexity (0.2-0.8, higher = simpler)
     - **Significance Threshold**: Statistical threshold for independence tests
   - Click "Learn Model"

4. **Run Queries**:
   - Go to the "Query Interface" tab
   - Select variables you want to query
   - Optionally set evidence (known values)
   - Click "Run Query" to get probabilistic answers

## Understanding the Data Summary

The automatic summary includes:

### Basic Information
- Total number of rows and columns
- Missing value counts
- Column types (numeric vs categorical)

### Numeric Statistics
- Mean, standard deviation, min, max
- Quartiles (25%, 50%, 75%)
- Distribution characteristics

### Correlations
- Pearson correlation matrix for numeric features
- Heatmap visualization showing feature relationships
- Strong correlations (positive or negative) indicate related features

### Categorical Statistics
- Number of unique values
- Most common values
- Value distributions

## Example Datasets

The interface includes several example datasets:

1. **Australian Credit**: Credit approval dataset
2. **Heart Disease**: Medical diagnosis data
3. **Credit Approval**: Credit card application data

## Model Learning Parameters

### Discretization Method
- **equal_width_binning**: Divides the range of each feature into equal-width bins
- **mean_split_binary_binning**: Splits based on mean value (binary discretization)

### Number of Bins
- Controls granularity of discretization
- More bins = more detail but potentially more complex model
- Recommended: Start with 2-3 bins

### Cluster Penalty (CP)
- Range: 0.2 to 0.8
- Higher values: Fewer, larger clusters (simpler model)
- Lower values: More, smaller clusters (complex model)
- Recommended: 0.4-0.6

### Significance Threshold (GF)
- Options: 10 or 15
- 10 → p-value of 0.0015 (less strict)
- 15 → p-value of 0.0001 (more strict)
- Controls when variables are considered independent

## Probabilistic Queries

Once a model is learned, you can ask questions like:

1. **Marginal Probability**: What is P(Variable = value)?
2. **Conditional Probability**: What is P(Variable = value | Evidence)?
3. **Joint Probability**: What is P(Var1 = val1, Var2 = val2)?
4. **Most Probable Explanation**: What values are most likely given evidence?

### Query Examples

- "What is the probability of approval given income > 50000?"
- "What is the most likely diagnosis given these symptoms?"
- "What is the joint probability of these conditions occurring together?"

## Architecture

The GUI consists of several components:

1. **Frontend**: Streamlit-based web interface
2. **Python Backend**: Data processing and discretization
3. **Java Backend**: SPN structure learning and inference
4. **Visualization**: Matplotlib/Seaborn for charts

## Data Format

### Input CSV Requirements
- First row should contain column names
- Numeric and categorical data supported
- Missing values should be empty or 'NaN'
- Example format:
  ```
  feature1,feature2,feature3,target
  1.5,2.3,category_a,0
  2.1,3.4,category_b,1
  ```

## Troubleshooting

### Java not found
- Ensure Java JDK is installed: `java -version`
- Add Java to your PATH environment variable

### Module not found errors
- Install requirements: `pip install -r requirements.txt`

### Model learning fails
- Check that Java code is compiled in the `bin` directory
- Verify dataset format (CSV with proper headers)
- Try reducing the number of bins or adjusting parameters

### GUI doesn't load
- Ensure Streamlit is installed: `pip install streamlit`
- Check Python version: `python --version` (should be 3.8+)

## Advanced Usage

### Custom Datasets

To use your own dataset:
1. Prepare data in CSV format
2. Ensure proper column names and data types
3. Handle missing values beforehand if desired
4. Upload through the GUI

### Parameter Tuning

For best results:
1. Start with default parameters
2. Examine the data summary for insights
3. Adjust discretization based on feature distributions
4. Use grid search over CP and GF values for optimal model

## Technical Details

### Discretization
- Continuous features are converted to discrete bins
- One-hot encoding for categorical variables
- Preserves feature relationships

### SPN Learning
- Uses LearnSPN algorithm (Gens & Domingos, 2013)
- Structure learning via recursive decomposition
- G-test for independence testing
- Hard EM for mixture components

### Inference
- Exact inference in linear time
- Supports marginal and conditional queries
- Handles missing data naturally

## Citations

If you use this software in research, please cite:

```
Gens, R., & Domingos, P. (2013).
Learning the structure of sum-product networks.
In International conference on machine learning (pp. 873-880). PMLR.
```

## Support

For issues, questions, or contributions:
- Check the main WMI-SPN repository documentation
- Review the example datasets and queries
- Examine the console output for error messages

## License

This GUI is part of the WMI-SPN project. Please refer to the main project license.
