# WMI-SPN GUI - Quick Start Guide

Get started with the WMI-SPN graphical interface in just a few minutes!

## Installation (One-Time Setup)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Java Installation

The SPN learning backend requires Java. Check if you have it:

```bash
java -version
```

If not installed, download from [https://www.java.com](https://www.java.com)

## Running the GUI

### Option 1: Use the Launcher (Easiest)

```bash
python launch_gui.py
```

Then select your preferred interface:
- **Option 1**: Streamlit Web Interface (recommended for rich visualizations)
- **Option 2**: Tkinter Desktop App (lightweight, traditional UI)

### Option 2: Run Directly

**Streamlit Web Interface:**
```bash
streamlit run wmi_spn_gui.py
```

**Tkinter Desktop App:**
```bash
python wmi_spn_gui_tkinter.py
```

## First Steps

### 1. Load Your Data

- Click **"Load Dataset"** or **"Browse files"**
- Select a CSV file from your computer
- Or choose an **example dataset** to try it out

### 2. Explore Your Data

Navigate to the tabs:
- **Data Preview**: See your data in table format
- **Summary**: Get comprehensive statistics including:
  - Basic stats (mean, median, std deviation)
  - Missing value analysis
  - Feature correlations (with heatmap)
  - Distribution information

### 3. Learn a Model

In the sidebar (Streamlit) or left panel (Tkinter):
1. Choose discretization settings
2. Adjust model parameters:
   - **Bins**: 2-3 for simple models, 4-10 for complex data
   - **Cluster Penalty**: 0.4-0.6 is usually good
3. Click **"Learn Model"**

### 4. Ask Probabilistic Questions

Once the model is learned:
1. Go to the **Query Interface** tab
2. Select variables you want to query
3. Optionally set evidence (known values)
4. Click **"Run Query"**

## Example Workflow

Let's walk through a complete example using the Australian Credit dataset:

### Step 1: Load Example Data
```
1. Launch the GUI
2. Select "Australian Credit" from examples
3. Click "Load Data"
```

### Step 2: Examine the Data
```
1. Check the Summary tab
   - See that there are 690 rows, 15 columns
   - View correlation heatmap
   - Identify numeric vs categorical features
```

### Step 3: Learn the Model
```
1. Set parameters:
   - Discretization: equal_width_binning
   - Bins: 2
   - Cluster Penalty: 0.6
2. Click "Learn Model"
3. Wait for completion (~1-2 minutes)
```

### Step 4: Query the Model
```
1. Go to Query Interface
2. Example query: "What's the probability of approval?"
   - Query variable: approval_status
   - Evidence: income > 50000
3. Run Query
```

## Understanding the Interface

### Data Summary Features

**Basic Information:**
- Row/column counts
- Missing value detection
- Data type identification

**Statistics:**
- Mean, median, mode
- Standard deviation
- Min/max values
- Quartiles (25%, 50%, 75%)

**Visualizations:**
- Correlation heatmap (shows feature relationships)
- Distribution plots (in Streamlit version)

### Model Parameters Explained

**Discretization Method:**
- `equal_width_binning`: Divides value ranges into equal bins
- `mean_split_binary_binning`: Splits at mean value

**Number of Bins:**
- Fewer bins (2-3): Simpler model, faster learning
- More bins (4-10): Captures more detail, slower learning

**Cluster Penalty (0.2-0.8):**
- Lower (0.2-0.4): More clusters, complex model
- Higher (0.6-0.8): Fewer clusters, simple model

**Significance Threshold:**
- 10: Less strict independence testing
- 15: More strict independence testing

### Query Types

You can ask various probabilistic questions:

1. **Marginal Probability**
   - "What is P(outcome = success)?"
   - Query: outcome, No evidence

2. **Conditional Probability**
   - "What is P(outcome = success | feature1 = high)?"
   - Query: outcome, Evidence: feature1 = high

3. **Joint Probability**
   - "What is P(feature1 = A, feature2 = B)?"
   - Query: feature1, feature2, No evidence

4. **MAP (Most Probable Assignment)**
   - "What values are most likely given evidence?"
   - Query: all unknowns, Evidence: known values

## Tips for Best Results

### Data Preparation
- Remove or impute missing values beforehand
- Ensure CSV has column headers
- Check for data quality issues in Summary tab

### Parameter Selection
- Start with defaults (2 bins, 0.6 penalty)
- Increase bins for data with fine-grained distinctions
- Decrease cluster penalty for complex relationships

### Model Learning
- Expect 1-5 minutes for small datasets (<1000 rows)
- Larger datasets may take 5-15 minutes
- Monitor the log/console for progress

### Querying
- Start with simple single-variable queries
- Add evidence incrementally
- Use Summary tab to understand feature distributions

## Troubleshooting

**"No data loaded"**
- Check CSV file format (should have headers)
- Try an example dataset first

**Model learning fails**
- Verify Java is installed (`java -version`)
- Check console/log for error messages
- Try simpler parameters (fewer bins)

**GUI won't start**
- Install dependencies: `pip install -r requirements.txt`
- For Streamlit: `pip install streamlit`
- For Tkinter on Linux: `sudo apt-get install python3-tk`

**Slow performance**
- Reduce number of bins
- Use smaller dataset for testing
- Close other applications

## Example Datasets Included

1. **Australian Credit** (690 rows, 15 features)
   - Credit approval prediction
   - Mix of numeric and categorical features
   - Good for learning classification

2. **Heart Disease** (303 rows, 14 features)
   - Medical diagnosis data
   - Mostly numeric features
   - Good for health analytics

3. **Credit Approval** (690 rows, 16 features)
   - Similar to Australian Credit
   - Different feature set
   - Good for comparison

## Next Steps

After getting comfortable with the basics:

1. **Try Your Own Data**
   - Prepare CSV files
   - Upload and explore

2. **Experiment with Parameters**
   - Compare different discretization methods
   - Try various bin counts
   - Adjust cluster penalties

3. **Advanced Queries**
   - Multiple evidence variables
   - Complex conditional queries
   - Compare query results

4. **Learn More**
   - Read GUI_README.md for detailed documentation
   - Check the SPN literature for theory
   - Explore the code for customization

## Getting Help

- Check **GUI_README.md** for detailed documentation
- Look at the **Log** tab for error messages
- Review example datasets to understand data format
- Consult the main WMI-SPN documentation

## Keyboard Shortcuts (Tkinter version)

- `Ctrl+O`: Open dataset
- `Ctrl+Q`: Quit application
- `F1`: Help/Documentation

## Performance Expectations

**Small datasets** (<1000 rows):
- Loading: Instant
- Summary: <1 second
- Learning: 1-3 minutes

**Medium datasets** (1000-10000 rows):
- Loading: 1-2 seconds
- Summary: 1-2 seconds
- Learning: 3-10 minutes

**Large datasets** (>10000 rows):
- Loading: 2-5 seconds
- Summary: 2-5 seconds
- Learning: 10-30 minutes

## What You Can Learn from Your Data

The summary and model can tell you:

1. **Feature Relationships**
   - Which variables are correlated?
   - Which features are independent?

2. **Distributions**
   - What are typical values?
   - Are there outliers?

3. **Dependencies**
   - How do variables depend on each other?
   - What are conditional relationships?

4. **Predictions**
   - What is likely given evidence?
   - What combinations are probable?

## Best Practices

1. **Always examine the summary first**
   - Understand your data before modeling
   - Check for issues (missing values, weird distributions)

2. **Start simple**
   - Use 2-3 bins initially
   - Default cluster penalty (0.6)
   - Add complexity as needed

3. **Validate results**
   - Do query results make sense?
   - Compare with summary statistics
   - Test multiple queries

4. **Iterate**
   - Try different parameters
   - Compare model performance
   - Refine based on results

---

**Ready to start?** Run `python launch_gui.py` and choose your interface!

For detailed documentation, see **GUI_README.md**
