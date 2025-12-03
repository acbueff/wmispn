# WMISPN GUI - Quick Start

Get started with the WMISPN GUI in 3 easy steps!

## 🚀 Installation & Launch

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Core dependencies (numpy, scipy, scikit-learn, pandas)
- GUI framework (streamlit)
- Visualization libraries (matplotlib, seaborn, plotly)

### Step 2: Launch the GUI

**Linux/Mac:**
```bash
streamlit run wmispn_app.py
```

Or use the launch script:
```bash
./launch_gui.sh
```

**Windows:**
```bash
streamlit run wmispn_app.py
```

Or double-click:
```
launch_gui.bat
```

### Step 3: Open Browser

The GUI will automatically open in your default browser at:
```
http://localhost:8501
```

---

## 📖 First Steps

### 1. Load Some Data

**Option A: Generate Demo Data**
- Navigate to "📊 Data Loading" page
- Select "Generate Synthetic Data"
- Click "Generate Synthetic Dataset"
- Explore the data in the tabs

**Option B: Upload Your CSV**
- Click "Upload CSV"
- Select your CSV file (must have headers)
- View data preview and statistics

### 2. Train a Model

- Navigate to "🧠 Model Training" page
- Leave settings at default (or choose a preset)
- Click "🚀 Train Model"
- Wait for training to complete (may take 30-60 seconds)
- Review model structure and performance

### 3. Run a Query

- Navigate to "🔍 Query Interface" page
- Choose "Simple Query"
- Select variables to query
- Optionally add evidence
- Click "🔍 Run Query"
- See the probability result!

---

## 🎯 Quick Examples

### Example 1: Simple Probability Query

After training on synthetic data:

1. Go to Query Interface → Simple Query
2. Check "Query class" → Select value: 1
3. Check "Condition on age" → Enter: 35
4. Click "Run Query"
5. Result: P(class=1 | age=35) = 0.XX

### Example 2: Interval Query (Cool!)

1. Go to Query Interface → Interval Query
2. Check "Add interval for income"
   - Lower: 30000
   - Upper: 50000
3. Check "Condition on class" → Value: 1
4. Click "Run Interval Query"
5. Result: P(30k < income < 50k | class=1) = 0.XX

### Example 3: Generate Samples

1. Go to Query Interface → Sampling
2. Set number of samples: 100
3. Click "Generate Samples"
4. View and download the generated data
5. Visualize relationships between features

---

## 💡 Tips for Success

### Data
- ✅ Use datasets with at least 200 samples
- ✅ Ensure CSV has column names in first row
- ✅ Mix of continuous and categorical features works best

### Training
- ✅ Start with "Default" preset
- ✅ Check test log-likelihood (higher = better)
- ✅ Try "Aggressive" for complex data, "Conservative" for simple data

### Queries
- ✅ Use interval queries for continuous variables
- ✅ Provide reasonable evidence values
- ✅ Generate samples to validate the model

---

## 🐛 Troubleshooting

### GUI won't start
```bash
# Install streamlit
pip install streamlit>=1.28.0

# Try launching directly
streamlit run wmispn_app.py
```

### "Module not found" errors
```bash
# Reinstall all requirements
pip install -r requirements.txt
```

### Training fails
- Reduce "Min instances" to 30
- Use fewer bins (try 3)
- Check if data has enough samples

### Slow performance
- Use "Conservative" preset
- Reduce max clusters to 5
- Use smaller test set

---

## 📚 Learn More

- **GUI_GUIDE.md**: Comprehensive GUI documentation
- **README_PYTHON.md**: Python API documentation
- **examples/**: Python script examples

---

## 🎓 Understanding the Interface

### Pages

1. **📊 Data Loading**
   - Upload or generate data
   - Explore statistics and distributions
   - Check data quality

2. **🧠 Model Training**
   - Configure hyperparameters
   - Train WMISPN models
   - View model structure and performance

3. **🔍 Query Interface**
   - Simple queries (point probabilities)
   - Interval queries (range probabilities)
   - Predictions (most likely values)
   - Sampling (generate new data)

4. **📚 Help**
   - Getting started guide
   - Feature descriptions
   - Parameter explanations
   - Example use cases

### Key Features

- **Automatic Variable Detection**: Detects continuous vs categorical
- **Interval Queries**: The key WMI feature - query probability over ranges!
- **Multiple Likelihoods**: LL, PLL for evaluation
- **Interactive Visualizations**: Explore your data and results
- **Model Save/Load**: Save trained models for later use

---

## 🎉 That's It!

You're ready to start exploring probabilistic modeling with WMISPN.

**Next Steps:**
1. Try the synthetic data first
2. Experiment with different hyperparameters
3. Load your own dataset
4. Explore interval queries (the coolest feature!)

**Questions?** Check GUI_GUIDE.md for detailed documentation.

**Enjoy! 🎲**
