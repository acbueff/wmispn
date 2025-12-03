"""
WMISPN GUI Application

A modern Streamlit interface for the Python WMISPN implementation.
Supports learning, querying, and visualizing Sum-Product Networks
with Weighted Model Integration for mixed discrete-continuous domains.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import io

from wmispn import WMISPN
from wmispn.data import Dataset
from wmispn.query import IntervalQuery

# Page configuration
st.set_page_config(
    page_title="WMISPN - Probabilistic Modeling",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False
if 'variable_types' not in st.session_state:
    st.session_state.variable_types = None


def load_example_dataset(dataset_name):
    """Load example datasets from the data directory."""
    data_path = Path("data") / dataset_name / f"{dataset_name}.ts.data"

    if data_path.exists():
        try:
            data = np.loadtxt(data_path)

            # Try to get feature names from schema file
            schema_path = Path("data") / dataset_name / f"{dataset_name}.schema"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    lines = f.readlines()
                    feature_names = [line.strip() for line in lines if line.strip()]
            else:
                feature_names = [f"Feature_{i}" for i in range(data.shape[1])]

            df = pd.DataFrame(data, columns=feature_names[:data.shape[1]])
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    else:
        st.warning(f"Dataset file not found: {data_path}")
        return None


def create_synthetic_data():
    """Create synthetic mixed data for demonstration."""
    np.random.seed(42)
    n_samples = 500

    data = {
        'class': np.random.randint(0, 2, n_samples),
        'age': np.random.uniform(18, 70, n_samples),
        'income': np.random.uniform(20000, 100000, n_samples),
        'education': np.random.randint(0, 4, n_samples),
        'credit_score': np.random.uniform(300, 850, n_samples),
        'employment': np.random.randint(0, 3, n_samples)
    }

    return pd.DataFrame(data)


def render_data_page():
    """Render the data loading and exploration page."""
    st.markdown('<p class="main-header">📊 Data Loading & Exploration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Load your dataset and explore its characteristics</p>', unsafe_allow_html=True)

    # Data source selection
    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio(
            "Select data source:",
            ["Upload CSV", "Example Dataset", "Generate Synthetic Data"],
            horizontal=True
        )

    data = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(data)} rows and {len(data.columns)} columns")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    elif data_source == "Example Dataset":
        # List available datasets
        data_dir = Path("data")
        if data_dir.exists():
            datasets = [d.name for d in data_dir.iterdir() if d.is_dir()]
            if datasets:
                dataset_name = st.selectbox("Choose example dataset:", datasets)
                if st.button("Load Dataset"):
                    data = load_example_dataset(dataset_name)
                    if data is not None:
                        st.success(f"✅ Loaded {dataset_name} dataset")
            else:
                st.info("No example datasets found in data/ directory")
        else:
            st.info("data/ directory not found. Using synthetic data option.")

    else:  # Generate Synthetic Data
        if st.button("Generate Synthetic Dataset"):
            data = create_synthetic_data()
            st.success("✅ Generated synthetic dataset with 500 samples")

    # Store data in session state
    if data is not None:
        st.session_state.data = data

    # Display data if loaded
    if st.session_state.data is not None:
        st.markdown("---")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistics", "🔍 Distributions"])

        with tab1:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head(100), use_container_width=True)

            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(st.session_state.data))
            with col2:
                st.metric("Columns", len(st.session_state.data.columns))
            with col3:
                st.metric("Missing Values", st.session_state.data.isna().sum().sum())
            with col4:
                memory_mb = st.session_state.data.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory", f"{memory_mb:.2f} MB")

        with tab2:
            st.subheader("Statistical Summary")

            # Numeric statistics
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Numeric Features:**")
                st.dataframe(st.session_state.data[numeric_cols].describe(), use_container_width=True)

            # Categorical statistics
            categorical_cols = st.session_state.data.select_dtypes(exclude=[np.number]).columns
            if len(categorical_cols) > 0:
                st.write("**Categorical Features:**")
                for col in categorical_cols:
                    with st.expander(f"📊 {col}"):
                        value_counts = st.session_state.data[col].value_counts()
                        st.write(value_counts)

            # Correlation matrix
            if len(numeric_cols) > 1:
                st.write("**Correlation Matrix:**")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = st.session_state.data[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)

        with tab3:
            st.subheader("Feature Distributions")

            # Select feature to visualize
            feature = st.selectbox("Select feature:", st.session_state.data.columns)

            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig, ax = plt.subplots(figsize=(8, 5))
                if st.session_state.data[feature].dtype in [np.float64, np.int64]:
                    ax.hist(st.session_state.data[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {feature}")
                else:
                    value_counts = st.session_state.data[feature].value_counts()
                    ax.bar(range(len(value_counts)), value_counts.values)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45)
                    ax.set_ylabel("Count")
                    ax.set_title(f"Distribution of {feature}")
                st.pyplot(fig)

            with col2:
                # Box plot (for numeric features)
                if st.session_state.data[feature].dtype in [np.float64, np.int64]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.boxplot(st.session_state.data[feature].dropna(), vert=True)
                    ax.set_ylabel(feature)
                    ax.set_title(f"Box Plot of {feature}")
                    st.pyplot(fig)
                else:
                    st.info("Box plot only available for numeric features")


def render_model_page():
    """Render the model training page."""
    st.markdown('<p class="main-header">🧠 Model Training</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure and train your WMISPN model</p>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Loading page")
        return

    # Model configuration in sidebar
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        st.subheader("Data Configuration")

        # Variable type specification
        auto_detect = st.checkbox("Auto-detect variable types", value=True)

        if not auto_detect:
            st.write("**Specify variable types:**")
            variable_types = {}
            for i, col in enumerate(st.session_state.data.columns):
                var_type = st.selectbox(
                    f"{col}:",
                    ["continuous", "categorical"],
                    key=f"vartype_{i}"
                )
                variable_types[i] = var_type
            st.session_state.variable_types = variable_types
        else:
            st.session_state.variable_types = None

        st.markdown("---")
        st.subheader("Hyperparameters")

        # Preset selection
        preset = st.selectbox(
            "Preset:",
            ["Custom", "Default", "Aggressive Splitting", "Conservative Splitting"]
        )

        if preset == "Custom":
            n_bins = st.slider("Number of bins (continuous)", 2, 10, 5)
            poly_degree = st.slider("Polynomial degree", 0, 3, 2)
            min_instances = st.slider("Min instances for split", 10, 200, 50, step=10)
            g_factor = st.slider("G-test factor", 0.1, 5.0, 1.0, step=0.1)
            cluster_penalty = st.slider("Cluster penalty", 0.5, 10.0, 2.0, step=0.5)
            max_clusters = st.slider("Max clusters", 2, 20, 10)
        else:
            preset_map = {
                "Default": "default",
                "Aggressive Splitting": "aggressive",
                "Conservative Splitting": "conservative"
            }
            n_bins = 5
            poly_degree = 2
            min_instances = 50
            g_factor = 1.0
            cluster_penalty = 2.0
            max_clusters = 10

        st.markdown("---")
        st.subheader("Data Preprocessing")
        normalize = st.checkbox("Normalize continuous features", value=True)
        handle_missing = st.selectbox("Handle missing values:", ["mean", "median", "drop"])

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Configuration")

        # Train/test split
        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random seed", 0, 9999, 42)

        # Info box
        st.markdown(f"""
        <div class="info-box">
        <strong>📋 Configuration Summary:</strong><br>
        • Dataset: {len(st.session_state.data)} samples, {len(st.session_state.data.columns)} features<br>
        • Train/Test split: {int((1-test_size)*100)}% / {int(test_size*100)}%<br>
        • Bins: {n_bins}, Polynomial degree: {poly_degree}<br>
        • Min instances: {min_instances}, Cluster penalty: {cluster_penalty}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Actions")

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Split data
                    from sklearn.model_selection import train_test_split
                    train_data, test_data = train_test_split(
                        st.session_state.data.values,
                        test_size=test_size,
                        random_state=random_state
                    )

                    # Create model
                    if preset == "Custom":
                        model = WMISPN(
                            n_bins=n_bins,
                            poly_degree=poly_degree,
                            min_instances=min_instances,
                            g_factor=g_factor,
                            cluster_penalty=cluster_penalty,
                            max_clusters=max_clusters,
                            random_state=random_state
                        )
                    else:
                        model = WMISPN.with_preset(
                            preset_map[preset],
                            random_state=random_state
                        )

                    # Fit model
                    model.fit(
                        train_data,
                        feature_names=list(st.session_state.data.columns),
                        variable_types=st.session_state.variable_types,
                        preprocess=True,
                        normalize=normalize
                    )

                    # Store model
                    st.session_state.model = model
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    st.session_state.model_fitted = True

                    st.success("✅ Model trained successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error during training: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.model_fitted:
            if st.button("💾 Save Model", use_container_width=True):
                model_bytes = pickle.dumps(st.session_state.model)
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name="wmispn_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )

            if st.button("🔄 Reset Model", use_container_width=True):
                st.session_state.model = None
                st.session_state.model_fitted = False
                st.rerun()

    # Display model info if trained
    if st.session_state.model_fitted:
        st.markdown("---")
        st.subheader("📊 Model Information")

        # Model structure
        info = st.session_state.model.get_structure_info()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", info['n_nodes'])
        with col2:
            st.metric("Sum Nodes", info['n_sum_nodes'])
        with col3:
            st.metric("Product Nodes", info['n_product_nodes'])
        with col4:
            st.metric("Leaf Nodes", info['n_leaf_nodes'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Network Depth", info['depth'])
        with col2:
            st.metric("Parameters", info['n_parameters'])
        with col3:
            st.metric("Variables", info['scope_size'])

        # Model evaluation
        st.subheader("📈 Model Performance")

        with st.spinner("Evaluating model..."):
            try:
                train_ll = st.session_state.model.average_log_likelihood(st.session_state.train_data)
                test_ll = st.session_state.model.average_log_likelihood(st.session_state.test_data)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Log-Likelihood", f"{train_ll:.4f}")
                with col2:
                    st.metric("Test Log-Likelihood", f"{test_ll:.4f}")

                # Additional metrics
                if st.checkbox("Show additional metrics"):
                    train_pll = np.mean(st.session_state.model.pseudo_log_likelihood(st.session_state.train_data))
                    test_pll = np.mean(st.session_state.model.pseudo_log_likelihood(st.session_state.test_data))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train Pseudo-LL", f"{train_pll:.4f}")
                    with col2:
                        st.metric("Test Pseudo-LL", f"{test_pll:.4f}")

            except Exception as e:
                st.error(f"Error evaluating model: {e}")


def render_query_page():
    """Render the query interface page."""
    st.markdown('<p class="main-header">🔍 Query Interface</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query probabilities and make predictions</p>', unsafe_allow_html=True)

    if not st.session_state.model_fitted:
        st.warning("⚠️ Please train a model first in the Model Training page")
        return

    model = st.session_state.model
    dataset = model.dataset

    # Query type selection
    query_type = st.radio(
        "Select query type:",
        ["Simple Query", "Interval Query (Continuous)", "Prediction", "Sampling"],
        horizontal=True
    )

    st.markdown("---")

    if query_type == "Simple Query":
        st.subheader("Simple Probability Query")
        st.write("Compute P(query_variables | evidence)")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Query Variables:**")
            query_vars = {}
            for i, feature in enumerate(dataset.feature_names):
                use_query = st.checkbox(f"Query {feature}", key=f"query_{i}")
                if use_query:
                    if dataset.variable_types[i] == 'categorical':
                        n_cats = int(st.session_state.data.iloc[:, i].max()) + 1
                        value = st.selectbox(f"Value for {feature}:", range(n_cats), key=f"qval_{i}")
                    else:
                        value = st.number_input(f"Value for {feature}:", key=f"qval_{i}")
                    query_vars[i] = value

        with col2:
            st.write("**Evidence (Conditions):**")
            evidence = {}
            for i, feature in enumerate(dataset.feature_names):
                if i not in query_vars:
                    use_evidence = st.checkbox(f"Condition on {feature}", key=f"evidence_{i}")
                    if use_evidence:
                        if dataset.variable_types[i] == 'categorical':
                            n_cats = int(st.session_state.data.iloc[:, i].max()) + 1
                            value = st.selectbox(f"Value for {feature}:", range(n_cats), key=f"eval_{i}")
                        else:
                            value = st.number_input(f"Value for {feature}:", key=f"eval_{i}")
                        evidence[i] = value

        if st.button("🔍 Run Query", type="primary"):
            if len(query_vars) == 0:
                st.warning("Please select at least one query variable")
            else:
                try:
                    log_prob = model.query(query_vars, evidence if len(evidence) > 0 else None)
                    prob = np.exp(log_prob)

                    st.markdown(f"""
                    <div class="success-box">
                    <strong>Query Result:</strong><br>
                    Log Probability: {log_prob:.6f}<br>
                    Probability: {prob:.6f} ({prob*100:.4f}%)
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error computing query: {e}")

    elif query_type == "Interval Query (Continuous)":
        st.subheader("Interval Query")
        st.write("Compute P(lower < X < upper | evidence) for continuous variables")

        # Find continuous variables
        continuous_vars = [i for i, t in dataset.variable_types.items() if t == 'continuous']

        if len(continuous_vars) == 0:
            st.warning("No continuous variables in the dataset")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Interval Constraints:**")
            intervals = {}
            for var_idx in continuous_vars:
                feature = dataset.feature_names[var_idx]
                use_interval = st.checkbox(f"Add interval for {feature}", key=f"interval_{var_idx}")
                if use_interval:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        lower = st.number_input(f"Lower ({feature}):", key=f"lower_{var_idx}")
                    with col_b:
                        upper = st.number_input(f"Upper ({feature}):", value=1.0, key=f"upper_{var_idx}")
                    intervals[var_idx] = (lower, upper)

        with col2:
            st.write("**Evidence:**")
            evidence = {}
            for i, feature in enumerate(dataset.feature_names):
                if i not in intervals:
                    use_evidence = st.checkbox(f"Condition on {feature}", key=f"int_evidence_{i}")
                    if use_evidence:
                        if dataset.variable_types[i] == 'categorical':
                            n_cats = int(st.session_state.data.iloc[:, i].max()) + 1
                            value = st.selectbox(f"Value for {feature}:", range(n_cats), key=f"int_eval_{i}")
                        else:
                            value = st.number_input(f"Value for {feature}:", key=f"int_eval_{i}")
                        evidence[i] = value

        if st.button("🔍 Run Interval Query", type="primary"):
            if len(intervals) == 0:
                st.warning("Please add at least one interval constraint")
            else:
                try:
                    log_prob = model.interval_query(intervals, evidence if len(evidence) > 0 else None)
                    prob = np.exp(log_prob)

                    st.markdown(f"""
                    <div class="success-box">
                    <strong>Interval Query Result:</strong><br>
                    Log Probability: {log_prob:.6f}<br>
                    Probability: {prob:.6f} ({prob*100:.4f}%)
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error computing interval query: {e}")

    elif query_type == "Prediction":
        st.subheader("Prediction")
        st.write("Predict the most likely value of a target variable given evidence")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Target Variable:**")
            target_idx = st.selectbox(
                "Select variable to predict:",
                range(len(dataset.feature_names)),
                format_func=lambda i: dataset.feature_names[i]
            )

        with col2:
            st.write("**Evidence:**")
            evidence = {}
            for i, feature in enumerate(dataset.feature_names):
                if i != target_idx:
                    use_evidence = st.checkbox(f"Use {feature}", key=f"pred_evidence_{i}")
                    if use_evidence:
                        if dataset.variable_types[i] == 'categorical':
                            n_cats = int(st.session_state.data.iloc[:, i].max()) + 1
                            value = st.selectbox(f"Value for {feature}:", range(n_cats), key=f"pred_eval_{i}")
                        else:
                            value = st.number_input(f"Value for {feature}:", key=f"pred_eval_{i}")
                        evidence[i] = value

        if st.button("🎯 Predict", type="primary"):
            if len(evidence) == 0:
                st.warning("Please provide some evidence for prediction")
            else:
                try:
                    prediction = model.predict(evidence, target_idx)

                    st.markdown(f"""
                    <div class="success-box">
                    <strong>Prediction Result:</strong><br>
                    Variable: {dataset.feature_names[target_idx]}<br>
                    Predicted Value: {prediction:.4f if dataset.variable_types[target_idx] == 'continuous' else int(prediction)}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

    else:  # Sampling
        st.subheader("Sample Generation")
        st.write("Generate samples from the learned distribution")

        col1, col2 = st.columns(2)

        with col1:
            n_samples = st.slider("Number of samples:", 1, 1000, 10)
            random_seed = st.number_input("Random seed:", 0, 9999, 42)

        with col2:
            st.write("**Optional Evidence:**")
            use_evidence = st.checkbox("Condition on evidence")
            evidence = {}
            if use_evidence:
                for i, feature in enumerate(dataset.feature_names):
                    use_this = st.checkbox(f"Fix {feature}", key=f"sample_evidence_{i}")
                    if use_this:
                        if dataset.variable_types[i] == 'categorical':
                            n_cats = int(st.session_state.data.iloc[:, i].max()) + 1
                            value = st.selectbox(f"Value:", range(n_cats), key=f"sample_eval_{i}")
                        else:
                            value = st.number_input(f"Value:", key=f"sample_eval_{i}")
                        evidence[i] = value

        if st.button("🎲 Generate Samples", type="primary"):
            try:
                samples = model.sample(
                    n_samples=n_samples,
                    evidence=evidence if len(evidence) > 0 else None,
                    random_state=random_seed
                )

                # Convert to DataFrame
                samples_df = pd.DataFrame(samples, columns=dataset.feature_names)

                st.success(f"✅ Generated {n_samples} samples")

                # Display samples
                st.dataframe(samples_df, use_container_width=True)

                # Download button
                csv = samples_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Samples as CSV",
                    data=csv,
                    file_name="wmispn_samples.csv",
                    mime="text/csv"
                )

                # Visualize first few features
                if len(samples_df.columns) >= 2:
                    st.write("**Sample Visualization:**")
                    col1, col2 = st.columns(2)

                    with col1:
                        feat1 = st.selectbox("X-axis:", samples_df.columns, index=0)
                        feat2 = st.selectbox("Y-axis:", samples_df.columns, index=min(1, len(samples_df.columns)-1))

                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(samples_df[feat1], samples_df[feat2], alpha=0.6)
                        ax.set_xlabel(feat1)
                        ax.set_ylabel(feat2)
                        ax.set_title(f"Generated Samples: {feat1} vs {feat2}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating samples: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_help_page():
    """Render the help and documentation page."""
    st.markdown('<p class="main-header">📚 Help & Documentation</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "Features", "Parameters", "Examples"])

    with tab1:
        st.markdown("""
        ## Getting Started with WMISPN

        ### What is WMISPN?

        WMISPN (Weighted Model Integration Sum-Product Networks) is a probabilistic modeling framework
        that combines Sum-Product Networks with Weighted Model Integration for learning tractable models
        in mixed discrete-continuous domains.

        ### Quick Start Guide

        1. **Load Data** (Data Loading page)
           - Upload a CSV file, load an example dataset, or generate synthetic data
           - Explore your data using the statistics and visualization tabs

        2. **Train Model** (Model Training page)
           - Configure hyperparameters or use a preset
           - Click "Train Model" and wait for completion
           - Review model structure and performance metrics

        3. **Query Model** (Query Interface page)
           - Perform various types of probabilistic queries
           - Get predictions and generate samples

        ### Key Features

        - ✅ **Automatic variable type detection** (categorical vs continuous)
        - ✅ **Piecewise polynomial approximation** for continuous features
        - ✅ **Complex interval queries** (e.g., P(5000 < income < 10000 | age > 30))
        - ✅ **Multiple likelihood functions** (LL, PLL, CLL)
        - ✅ **Exact inference** in polynomial time
        - ✅ **Sample generation** from the learned distribution
        """)

    with tab2:
        st.markdown("""
        ## Features

        ### Data Loading & Exploration
        - **CSV Upload**: Load your own datasets
        - **Example Datasets**: Pre-loaded benchmark datasets
        - **Synthetic Data**: Generate demo data for testing
        - **Automatic Summary**: Statistics, correlations, distributions

        ### Model Training
        - **Hyperparameter Presets**: Default, Aggressive, Conservative
        - **Custom Configuration**: Fine-tune all parameters
        - **Train/Test Split**: Automatic data splitting
        - **Model Evaluation**: Multiple likelihood metrics

        ### Query Interface

        #### 1. Simple Query
        Compute P(query | evidence) for specific variable values.

        **Example:** P(class=1 | age=30, income=50000)

        #### 2. Interval Query
        Compute probability over continuous intervals.

        **Example:** P(5000 < income < 10000 | class=1)

        #### 3. Prediction
        Predict most likely value given evidence.

        **Example:** Predict income given age, education, employment

        #### 4. Sampling
        Generate new samples from the learned distribution.
        """)

    with tab3:
        st.markdown("""
        ## Hyperparameters Guide

        ### Continuous Feature Parameters

        **Number of Bins** (n_bins)
        - Controls discretization granularity
        - Range: 2-10
        - Higher = more detail, more complex model
        - Recommended: 3-5

        **Polynomial Degree** (poly_degree)
        - Degree of piecewise polynomials
        - 0 = constant, 1 = linear, 2 = quadratic, 3 = cubic
        - Higher = more flexible distributions
        - Recommended: 2

        ### Structure Learning Parameters

        **Min Instances** (min_instances)
        - Minimum samples needed to split a node
        - Range: 10-200
        - Higher = simpler models
        - Recommended: 50

        **G-Test Factor** (g_factor)
        - Sensitivity for independence testing
        - Range: 0.1-5.0
        - Higher = more conservative (fewer splits)
        - Recommended: 1.0

        **Cluster Penalty** (cluster_penalty)
        - Penalty for creating mixture components
        - Range: 0.5-10.0
        - Higher = fewer clusters, simpler model
        - Recommended: 2.0

        **Max Clusters** (max_clusters)
        - Maximum mixture components per sum node
        - Range: 2-20
        - Recommended: 10

        ### Preprocessing

        **Normalize**: Scale continuous features to [0,1]
        **Handle Missing**: mean, median, or drop rows
        """)

    with tab4:
        st.markdown("""
        ## Example Use Cases

        ### 1. Credit Scoring

        **Data:** Customer information (age, income, credit history, employment)

        **Query Examples:**
        - P(approval=1 | income > 50000, age > 30)
        - P(30000 < income < 50000 | approval=1)
        - Predict approval given customer profile

        ### 2. Medical Diagnosis

        **Data:** Patient symptoms and test results

        **Query Examples:**
        - P(disease=1 | symptom1=yes, symptom2=no)
        - P(test_value in [80, 120] | disease=1)
        - Predict disease given symptoms

        ### 3. Customer Behavior

        **Data:** Demographics and purchase history

        **Query Examples:**
        - P(purchase=1 | age in [25,35], income > 60000)
        - Sample likely customers for marketing
        - Predict purchase probability

        ### Best Practices

        1. **Start Simple**
           - Use default parameters initially
           - Examine data distributions first
           - Start with smaller datasets

        2. **Iterate**
           - Try different parameter settings
           - Compare test log-likelihoods
           - Validate queries make sense

        3. **Validate**
           - Check if predicted probabilities are reasonable
           - Compare samples to original data
           - Test edge cases
        """)


# Main app
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #1f77b4; margin-bottom: 0;'>🎲 WMISPN</h1>
        <p style='font-size: 1.2rem; color: #666;'>Weighted Model Integration Sum-Product Networks</p>
        <p style='font-size: 0.9rem; color: #888;'>Probabilistic Modeling for Mixed Discrete-Continuous Domains</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Data Loading", "🧠 Model Training", "🔍 Query Interface", "📚 Help"],
        label_visibility="collapsed"
    )

    # Route to appropriate page
    if page == "📊 Data Loading":
        render_data_page()
    elif page == "🧠 Model Training":
        render_model_page()
    elif page == "🔍 Query Interface":
        render_query_page()
    elif page == "📚 Help":
        render_help_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8rem; padding: 1rem 0;'>
        WMISPN GUI v1.0 | Built with Streamlit |
        <a href='https://github.com/anthropics/wmispn' target='_blank'>Documentation</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
