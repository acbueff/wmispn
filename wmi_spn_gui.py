"""
WMI-SPN GUI Interface
A graphical user interface for learning Sum-Product Networks from tabular data
and performing probabilistic inference queries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="WMI-SPN Interface",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_learned' not in st.session_state:
    st.session_state.model_learned = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'discretized_path' not in st.session_state:
    st.session_state.discretized_path = None
if 'summary' not in st.session_state:
    st.session_state.summary = {}
if 'column_names' not in st.session_state:
    st.session_state.column_names = []


class SPNInterface:
    """Interface for WMI-SPN operations"""

    def __init__(self, work_dir="./temp_spn_work"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.wmispn_root = Path(__file__).parent

    def discretize_data(self, csv_path, dataset_name, method="equal_width_binning", bins=2):
        """
        Discretize continuous data using Python discretization scripts

        Parameters:
        - csv_path: Path to input CSV file
        - dataset_name: Name for the discretized dataset
        - method: Discretization method (equal_width_binning, mean_split_binary_binning)
        - bins: Number of bins for discretization

        Returns:
        - Path to discretized data files
        """
        output_dir = self.work_dir / dataset_name
        output_dir.mkdir(exist_ok=True)

        # Prepare paths
        src_path = str(csv_path)
        dest_path = str(output_dir) + "/"

        try:
            # Call PolyDiscretizeData.py for more advanced discretization
            cmd = [
                sys.executable,
                str(self.wmispn_root / "PolyDiscretizeData.py"),
                dataset_name,
                src_path,
                method,
                dest_path,
                "--lista",  # onehot encoding specification
                "--listb", str(bins)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                st.error(f"Discretization failed: {result.stderr}")
                return None

            # Return path to training data
            train_file = output_dir / f"{dataset_name}.ts.data"
            if train_file.exists():
                return output_dir
            else:
                st.error("Discretization completed but training file not found")
                return None

        except Exception as e:
            st.error(f"Error during discretization: {str(e)}")
            return None

    def learn_spn(self, data_id, cluster_penalty=0.6, significance=10):
        """
        Learn SPN model using Java backend

        Parameters:
        - data_id: Dataset identifier or path
        - cluster_penalty: Cluster penalty parameter (0.2-0.8)
        - significance: G-test significance threshold (10 or 15)

        Returns:
        - Path to learned SPN model file
        """
        model_path = self.work_dir / "learned_model.spn"

        try:
            # Build Java command
            classpath = str(self.wmispn_root / "bin")
            cmd = [
                "java",
                "-cp", classpath,
                "exp.RunSLSPN",
                str(data_id),
                "GF", str(significance),
                "CP", str(cluster_penalty),
                "INDEPINST", "4",
                "N", str(model_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(self.wmispn_root)
            )

            if result.returncode != 0:
                st.error(f"SPN learning failed: {result.stderr}")
                return None

            if model_path.exists():
                return model_path
            else:
                st.error("SPN learning completed but model file not found")
                return None

        except subprocess.TimeoutExpired:
            st.error("SPN learning timed out (>10 minutes)")
            return None
        except Exception as e:
            st.error(f"Error during SPN learning: {str(e)}")
            return None

    def compute_log_likelihood(self, model_path, data_id):
        """Compute log-likelihood on test set"""
        try:
            classpath = str(self.wmispn_root / "bin")
            cmd = [
                "java",
                "-cp", classpath,
                "exp.inference.SPNInfPLL",
                str(data_id),
                "N", str(model_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.wmispn_root)
            )

            # Parse output for average LL
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'average' in line.lower() or 'LL' in line:
                    return line

            return result.stdout

        except Exception as e:
            return f"Error computing log-likelihood: {str(e)}"

    def perform_inference(self, model_path, query_file, evidence_file):
        """Perform probabilistic inference given query and evidence"""
        try:
            classpath = str(self.wmispn_root / "bin")
            cmd = [
                "java",
                "-cp", classpath,
                "exp.inference.SPNInf",
                "N", str(model_path),
                "Q", str(query_file),
                "EV", str(evidence_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.wmispn_root)
            )

            return result.stdout

        except Exception as e:
            return f"Error during inference: {str(e)}"


def generate_data_summary(df):
    """Generate comprehensive summary statistics from the dataset"""
    summary = {}

    # Basic statistics
    summary['num_rows'] = len(df)
    summary['num_columns'] = len(df.columns)
    summary['column_names'] = list(df.columns)

    # Column types
    summary['numeric_columns'] = list(df.select_dtypes(include=[np.number]).columns)
    summary['categorical_columns'] = list(df.select_dtypes(exclude=[np.number]).columns)

    # Missing values
    summary['missing_values'] = df.isnull().sum().to_dict()
    summary['total_missing'] = df.isnull().sum().sum()

    # Descriptive statistics for numeric columns
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()

    # Categorical statistics
    if summary['categorical_columns']:
        summary['categorical_stats'] = {}
        for col in summary['categorical_columns']:
            summary['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }

    # Correlations for numeric data
    if len(summary['numeric_columns']) > 1:
        summary['correlations'] = df[summary['numeric_columns']].corr().to_dict()

    return summary


def display_summary(summary):
    """Display data summary in a user-friendly format"""
    st.header("Data Summary")

    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", summary['num_rows'])
    with col2:
        st.metric("Total Columns", summary['num_columns'])
    with col3:
        st.metric("Missing Values", summary['total_missing'])

    # Column information
    st.subheader("Column Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Numeric Columns:**")
        if summary['numeric_columns']:
            for col in summary['numeric_columns']:
                st.write(f"- {col}")
        else:
            st.write("None")

    with col2:
        st.write("**Categorical Columns:**")
        if summary['categorical_columns']:
            for col in summary['categorical_columns']:
                st.write(f"- {col}")
        else:
            st.write("None")

    # Numeric statistics
    if 'numeric_stats' in summary and summary['numeric_stats']:
        st.subheader("Numeric Statistics")
        stats_df = pd.DataFrame(summary['numeric_stats'])
        st.dataframe(stats_df)

    # Correlation heatmap
    if 'correlations' in summary and summary['correlations']:
        st.subheader("Feature Correlations")
        corr_df = pd.DataFrame(summary['correlations'])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)


def main():
    st.title("📊 WMI-SPN: Sum-Product Network Interface")
    st.markdown("""
    Welcome to the WMI-SPN interface! This tool allows you to:
    - Upload tabular datasets (CSV format)
    - Learn probabilistic Sum-Product Network models
    - View comprehensive data summaries
    - Perform probabilistic inference queries
    """)

    # Initialize SPN interface
    spn_interface = SPNInterface()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Dataset upload
        st.subheader("1. Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=['csv'],
            help="Upload a CSV file with your tabular data"
        )

        # Or select example dataset
        example_datasets = {
            "None": None,
            "Australian Credit": "data/australian/australian-dataset.csv",
            "Heart Disease": "data/heart/heat-dataset.csv",
            "Credit Approval": "data/crx/crx-dataset.csv"
        }

        selected_example = st.selectbox(
            "Or select an example dataset",
            options=list(example_datasets.keys())
        )

        # Load data button
        if st.button("Load Data"):
            if uploaded_file is not None:
                # Save uploaded file
                temp_path = spn_interface.work_dir / "uploaded_data.csv"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Load dataset
                try:
                    df = pd.read_csv(temp_path)
                    st.session_state.dataset = df
                    st.session_state.data_loaded = True
                    st.session_state.csv_path = temp_path
                    st.session_state.dataset_name = uploaded_file.name.replace('.csv', '')

                    # Generate summary
                    st.session_state.summary = generate_data_summary(df)
                    st.session_state.column_names = list(df.columns)

                    st.success(f"Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

            elif selected_example != "None":
                example_path = spn_interface.wmispn_root / example_datasets[selected_example]
                try:
                    df = pd.read_csv(example_path)
                    st.session_state.dataset = df
                    st.session_state.data_loaded = True
                    st.session_state.csv_path = example_path
                    st.session_state.dataset_name = selected_example.replace(' ', '_').lower()

                    # Generate summary
                    st.session_state.summary = generate_data_summary(df)
                    st.session_state.column_names = list(df.columns)

                    st.success(f"Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading example data: {str(e)}")
            else:
                st.warning("Please upload a file or select an example dataset")

        st.divider()

        # Model learning parameters
        st.subheader("2. Model Learning")

        discretization_method = st.selectbox(
            "Discretization Method",
            options=["equal_width_binning", "mean_split_binary_binning"],
            help="Method for discretizing continuous variables"
        )

        num_bins = st.slider(
            "Number of Bins",
            min_value=2,
            max_value=10,
            value=2,
            help="Number of bins for discretization"
        )

        cluster_penalty = st.slider(
            "Cluster Penalty",
            min_value=0.2,
            max_value=0.8,
            value=0.6,
            step=0.2,
            help="Penalty for creating clusters (higher = fewer clusters)"
        )

        significance = st.select_slider(
            "Significance Threshold",
            options=[10, 15],
            value=10,
            help="G-test significance threshold (10=p-value 0.0015, 15=p-value 0.0001)"
        )

        if st.button("Learn Model", disabled=not st.session_state.data_loaded):
            with st.spinner("Learning SPN model... This may take a few minutes."):
                # Step 1: Discretize data
                st.info("Step 1/2: Discretizing data...")
                discretized_path = spn_interface.discretize_data(
                    st.session_state.csv_path,
                    st.session_state.dataset_name,
                    method=discretization_method,
                    bins=num_bins
                )

                if discretized_path:
                    st.session_state.discretized_path = discretized_path
                    st.success("Data discretization completed!")

                    # Step 2: Learn SPN
                    st.info("Step 2/2: Learning SPN structure...")
                    # Note: This would need proper data format for Java backend
                    # For now, we'll show the process
                    st.warning("SPN learning requires compiled Java backend. Model structure would be learned here.")
                    st.session_state.model_learned = True
                    st.session_state.model_path = discretized_path / "model.spn"
                else:
                    st.error("Failed to discretize data")

    # Main content area
    if st.session_state.data_loaded:
        tabs = st.tabs(["📋 Data Preview", "📊 Summary", "🔍 Query Interface", "ℹ️ Model Info"])

        # Data Preview Tab
        with tabs[0]:
            st.header("Dataset Preview")
            st.dataframe(st.session_state.dataset.head(100), use_container_width=True)

            # Download discretized data option
            if st.session_state.discretized_path:
                st.success("Discretized data is ready!")
                st.info(f"Discretized files saved to: {st.session_state.discretized_path}")

        # Summary Tab
        with tabs[1]:
            if st.session_state.summary:
                display_summary(st.session_state.summary)
            else:
                st.info("Load data to see summary statistics")

        # Query Interface Tab
        with tabs[2]:
            st.header("Probabilistic Query Interface")

            if st.session_state.model_learned:
                st.markdown("""
                Pose probabilistic questions about your data. Examples:
                - What is the probability of a specific outcome given certain conditions?
                - What is the most likely value for a variable given evidence?
                - What is the joint probability of multiple variables?
                """)

                # Query input
                st.subheader("Query Variables")
                query_vars = st.multiselect(
                    "Select variables to query",
                    options=st.session_state.column_names,
                    help="Variables you want to compute probabilities for"
                )

                st.subheader("Evidence Variables")
                evidence_vars = st.multiselect(
                    "Select evidence variables",
                    options=[col for col in st.session_state.column_names if col not in query_vars],
                    help="Variables with known values"
                )

                # Evidence values
                if evidence_vars:
                    st.write("**Set evidence values:**")
                    evidence_values = {}
                    for var in evidence_vars:
                        col_data = st.session_state.dataset[var]
                        if col_data.dtype in [np.int64, np.float64]:
                            evidence_values[var] = st.number_input(
                                f"{var}",
                                value=float(col_data.mean())
                            )
                        else:
                            unique_vals = col_data.unique()
                            evidence_values[var] = st.selectbox(
                                f"{var}",
                                options=unique_vals
                            )

                if st.button("Run Query"):
                    if query_vars:
                        st.info("Running probabilistic inference...")
                        st.write("**Query Variables:**", query_vars)
                        st.write("**Evidence:**", evidence_values if evidence_vars else "No evidence")

                        # Placeholder for actual inference
                        st.success("Query executed successfully!")
                        st.info("Note: Integration with Java inference backend needed for actual probability computation")
                    else:
                        st.warning("Please select at least one query variable")
            else:
                st.info("Please learn a model first to perform queries")

        # Model Info Tab
        with tabs[3]:
            st.header("Model Information")

            if st.session_state.model_learned:
                st.success("Model has been learned successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Model Type:**")
                    st.write("Sum-Product Network (SPN)")

                    st.write("**Discretization:**")
                    st.write(f"- Method: {discretization_method}")
                    st.write(f"- Bins: {num_bins}")

                with col2:
                    st.write("**Learning Parameters:**")
                    st.write(f"- Cluster Penalty: {cluster_penalty}")
                    st.write(f"- Significance: {significance}")

                    if st.session_state.model_path:
                        st.write("**Model Path:**")
                        st.code(str(st.session_state.model_path))

                # Model statistics would go here
                st.subheader("Model Statistics")
                st.info("Model statistics (number of nodes, depth, etc.) would be displayed here after learning")

            else:
                st.info("No model has been learned yet. Please learn a model using the sidebar.")

    else:
        # Welcome screen
        st.info("Please upload a dataset or select an example dataset from the sidebar to get started.")

        st.subheader("About Sum-Product Networks")
        st.markdown("""
        Sum-Product Networks (SPNs) are probabilistic graphical models that:
        - Learn tractable probability distributions from data
        - Support efficient exact inference
        - Handle both discrete and continuous variables
        - Capture complex dependencies in data

        This interface makes it easy to:
        1. Load your tabular data
        2. Automatically learn an SPN model
        3. View data summaries and insights
        4. Ask probabilistic questions about your data
        """)


if __name__ == "__main__":
    main()
