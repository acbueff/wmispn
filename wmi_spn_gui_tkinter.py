"""
WMI-SPN Tkinter GUI Interface
A standalone desktop application for learning Sum-Product Networks
and performing probabilistic inference.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import threading
import json


class SPNInterfaceTkinter:
    """Tkinter-based GUI for WMI-SPN"""

    def __init__(self, root):
        self.root = root
        self.root.title("WMI-SPN Interface")
        self.root.geometry("1200x800")

        # State variables
        self.dataset = None
        self.csv_path = None
        self.dataset_name = None
        self.model_learned = False
        self.work_dir = Path("./temp_spn_work")
        self.work_dir.mkdir(exist_ok=True)
        self.wmispn_root = Path(__file__).parent

        # Create UI
        self.create_menu()
        self.create_main_ui()

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Dataset...", command=self.load_dataset)
        file_menu.add_command(label="Load Example Dataset", command=self.load_example_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Learn Model", command=self.learn_model_thread)
        model_menu.add_command(label="Model Info", command=self.show_model_info)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)

    def create_main_ui(self):
        """Create main UI layout"""

        # Main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Controls
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=1)

        # Right panel: Content
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        self.create_control_panel(left_frame)
        self.create_content_panel(right_frame)

    def create_control_panel(self, parent):
        """Create left control panel"""

        # Data Loading Section
        data_frame = ttk.LabelFrame(parent, text="Data", padding=10)
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            data_frame,
            text="Load Dataset",
            command=self.load_dataset
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            data_frame,
            text="Load Example",
            command=self.load_example_dataset
        ).pack(fill=tk.X, pady=2)

        self.data_status_label = ttk.Label(data_frame, text="No data loaded", foreground="gray")
        self.data_status_label.pack(fill=tk.X, pady=5)

        # Model Parameters Section
        params_frame = ttk.LabelFrame(parent, text="Model Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Discretization method
        ttk.Label(params_frame, text="Discretization:").pack(anchor=tk.W)
        self.disc_method = tk.StringVar(value="equal_width_binning")
        disc_combo = ttk.Combobox(
            params_frame,
            textvariable=self.disc_method,
            values=["equal_width_binning", "mean_split_binary_binning"],
            state="readonly"
        )
        disc_combo.pack(fill=tk.X, pady=2)

        # Number of bins
        ttk.Label(params_frame, text="Number of Bins:").pack(anchor=tk.W, pady=(10, 0))
        self.num_bins = tk.IntVar(value=2)
        bins_scale = ttk.Scale(
            params_frame,
            from_=2,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.num_bins,
            command=lambda v: self.bins_label.config(text=f"{int(float(v))}")
        )
        bins_scale.pack(fill=tk.X)
        self.bins_label = ttk.Label(params_frame, text="2")
        self.bins_label.pack(anchor=tk.W)

        # Cluster penalty
        ttk.Label(params_frame, text="Cluster Penalty:").pack(anchor=tk.W, pady=(10, 0))
        self.cluster_penalty = tk.DoubleVar(value=0.6)
        cp_scale = ttk.Scale(
            params_frame,
            from_=0.2,
            to=0.8,
            orient=tk.HORIZONTAL,
            variable=self.cluster_penalty,
            command=lambda v: self.cp_label.config(text=f"{float(v):.1f}")
        )
        cp_scale.pack(fill=tk.X)
        self.cp_label = ttk.Label(params_frame, text="0.6")
        self.cp_label.pack(anchor=tk.W)

        # Significance
        ttk.Label(params_frame, text="Significance:").pack(anchor=tk.W, pady=(10, 0))
        self.significance = tk.IntVar(value=10)
        ttk.Radiobutton(
            params_frame,
            text="10 (p=0.0015)",
            variable=self.significance,
            value=10
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            params_frame,
            text="15 (p=0.0001)",
            variable=self.significance,
            value=15
        ).pack(anchor=tk.W)

        # Learn button
        self.learn_button = ttk.Button(
            params_frame,
            text="Learn Model",
            command=self.learn_model_thread,
            state=tk.DISABLED
        )
        self.learn_button.pack(fill=tk.X, pady=(10, 0))

        # Query Section
        query_frame = ttk.LabelFrame(parent, text="Quick Query", padding=10)
        query_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(query_frame, text="Query variable:").pack(anchor=tk.W)
        self.query_var = tk.StringVar()
        self.query_combo = ttk.Combobox(
            query_frame,
            textvariable=self.query_var,
            state="readonly"
        )
        self.query_combo.pack(fill=tk.X, pady=2)

        ttk.Button(
            query_frame,
            text="Run Query",
            command=self.run_simple_query
        ).pack(fill=tk.X, pady=(10, 0))

    def create_content_panel(self, parent):
        """Create right content panel with tabs"""

        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Data Preview Tab
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="Data Preview")

        # Create treeview for data display
        tree_scroll_y = ttk.Scrollbar(preview_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.data_tree = ttk.Treeview(
            preview_frame,
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        self.data_tree.pack(fill=tk.BOTH, expand=True)

        tree_scroll_y.config(command=self.data_tree.yview)
        tree_scroll_x.config(command=self.data_tree.xview)

        # Summary Tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")

        self.summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Query Tab
        query_frame = ttk.Frame(self.notebook)
        self.notebook.add(query_frame, text="Query Interface")

        query_top = ttk.Frame(query_frame)
        query_top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(query_top, text="Query Variables (comma-separated):").pack(anchor=tk.W)
        self.query_vars_entry = ttk.Entry(query_top)
        self.query_vars_entry.pack(fill=tk.X, pady=2)

        ttk.Label(query_top, text="Evidence (format: var1=val1,var2=val2):").pack(anchor=tk.W, pady=(10, 0))
        self.evidence_entry = ttk.Entry(query_top)
        self.evidence_entry.pack(fill=tk.X, pady=2)

        ttk.Button(
            query_top,
            text="Execute Query",
            command=self.execute_query
        ).pack(pady=10)

        ttk.Label(query_frame, text="Results:").pack(anchor=tk.W, padx=10)
        self.query_results = scrolledtext.ScrolledText(
            query_frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            height=15
        )
        self.query_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Log Tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg="#f0f0f0"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("WMI-SPN Interface initialized")

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def load_dataset(self):
        """Load dataset from file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.csv_path = Path(filename)
                self.dataset = pd.read_csv(filename)
                self.dataset_name = self.csv_path.stem

                self.display_dataset()
                self.display_summary()
                self.update_ui_after_load()

                self.data_status_label.config(
                    text=f"Loaded: {len(self.dataset)} rows",
                    foreground="green"
                )
                self.log(f"Dataset loaded: {filename}")
                messagebox.showinfo("Success", f"Dataset loaded successfully!\n{len(self.dataset)} rows, {len(self.dataset.columns)} columns")

            except Exception as e:
                self.log(f"Error loading dataset: {str(e)}")
                messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def load_example_dataset(self):
        """Load example dataset"""
        examples = {
            "Australian Credit": "data/australian/australian-dataset.csv",
            "Heart Disease": "data/heart/heat-dataset.csv",
            "Credit Approval": "data/crx/crx-dataset.csv"
        }

        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Example Dataset")
        dialog.geometry("300x200")

        ttk.Label(dialog, text="Choose an example dataset:").pack(pady=10)

        selected = tk.StringVar()

        for name in examples.keys():
            ttk.Radiobutton(
                dialog,
                text=name,
                variable=selected,
                value=name
            ).pack(anchor=tk.W, padx=20)

        def load_selected():
            if selected.get():
                path = self.wmispn_root / examples[selected.get()]
                try:
                    self.csv_path = path
                    self.dataset = pd.read_csv(path)
                    self.dataset_name = selected.get().replace(' ', '_').lower()

                    self.display_dataset()
                    self.display_summary()
                    self.update_ui_after_load()

                    self.data_status_label.config(
                        text=f"Loaded: {len(self.dataset)} rows",
                        foreground="green"
                    )
                    self.log(f"Example dataset loaded: {selected.get()}")
                    dialog.destroy()
                    messagebox.showinfo("Success", f"Dataset loaded successfully!\n{len(self.dataset)} rows, {len(self.dataset.columns)} columns")

                except Exception as e:
                    self.log(f"Error loading example: {str(e)}")
                    messagebox.showerror("Error", f"Failed to load example:\n{str(e)}")
            else:
                messagebox.showwarning("Warning", "Please select a dataset")

        ttk.Button(dialog, text="Load", command=load_selected).pack(pady=10)

    def display_dataset(self):
        """Display dataset in treeview"""
        # Clear existing data
        self.data_tree.delete(*self.data_tree.get_children())

        # Configure columns
        self.data_tree['columns'] = list(self.dataset.columns)
        self.data_tree['show'] = 'headings'

        # Set column headings
        for col in self.dataset.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)

        # Add data (first 100 rows)
        for idx, row in self.dataset.head(100).iterrows():
            self.data_tree.insert('', tk.END, values=list(row))

    def display_summary(self):
        """Display data summary"""
        self.summary_text.delete(1.0, tk.END)

        summary = []
        summary.append("=" * 60)
        summary.append("DATA SUMMARY")
        summary.append("=" * 60)
        summary.append(f"\nDataset: {self.dataset_name}")
        summary.append(f"Rows: {len(self.dataset)}")
        summary.append(f"Columns: {len(self.dataset.columns)}")
        summary.append("\n" + "-" * 60)
        summary.append("COLUMN INFORMATION")
        summary.append("-" * 60)

        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        categorical_cols = self.dataset.select_dtypes(exclude=[np.number]).columns

        summary.append(f"\nNumeric Columns ({len(numeric_cols)}):")
        for col in numeric_cols:
            summary.append(f"  - {col}")

        summary.append(f"\nCategorical Columns ({len(categorical_cols)}):")
        for col in categorical_cols:
            summary.append(f"  - {col}")

        summary.append("\n" + "-" * 60)
        summary.append("MISSING VALUES")
        summary.append("-" * 60)
        missing = self.dataset.isnull().sum()
        if missing.sum() > 0:
            for col, count in missing[missing > 0].items():
                summary.append(f"  {col}: {count}")
        else:
            summary.append("  No missing values")

        summary.append("\n" + "-" * 60)
        summary.append("NUMERIC STATISTICS")
        summary.append("-" * 60)
        summary.append("\n" + str(self.dataset.describe()))

        self.summary_text.insert(1.0, "\n".join(summary))

    def update_ui_after_load(self):
        """Update UI elements after loading data"""
        self.learn_button.config(state=tk.NORMAL)

        # Update query variable combo
        self.query_combo['values'] = list(self.dataset.columns)
        if len(self.dataset.columns) > 0:
            self.query_combo.current(0)

    def learn_model_thread(self):
        """Learn model in separate thread"""
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        # Run in thread to prevent UI freezing
        thread = threading.Thread(target=self.learn_model, daemon=True)
        thread.start()

    def learn_model(self):
        """Learn SPN model"""
        self.log("Starting model learning...")
        self.learn_button.config(state=tk.DISABLED, text="Learning...")

        try:
            # Discretize data
            self.log(f"Discretizing data with method: {self.disc_method.get()}")
            discretized_path = self.discretize_data(
                self.csv_path,
                self.dataset_name,
                method=self.disc_method.get(),
                bins=self.num_bins.get()
            )

            if discretized_path:
                self.log("Data discretization completed successfully")
                self.model_learned = True
                messagebox.showinfo("Success", "Model learning completed!\n(Note: Full SPN learning requires compiled Java backend)")
            else:
                self.log("Data discretization failed")
                messagebox.showerror("Error", "Model learning failed during discretization")

        except Exception as e:
            self.log(f"Error during model learning: {str(e)}")
            messagebox.showerror("Error", f"Model learning failed:\n{str(e)}")

        finally:
            self.learn_button.config(state=tk.NORMAL, text="Learn Model")

    def discretize_data(self, csv_path, dataset_name, method="equal_width_binning", bins=2):
        """Discretize data using Python script"""
        output_dir = self.work_dir / dataset_name
        output_dir.mkdir(exist_ok=True)

        src_path = str(csv_path)
        dest_path = str(output_dir) + "/"

        try:
            cmd = [
                sys.executable,
                str(self.wmispn_root / "PolyDiscretizeData.py"),
                dataset_name,
                src_path,
                method,
                dest_path,
                "--lista",
                "--listb", str(bins)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                train_file = output_dir / f"{dataset_name}.ts.data"
                if train_file.exists():
                    return output_dir

            return None

        except Exception as e:
            self.log(f"Discretization error: {str(e)}")
            return None

    def run_simple_query(self):
        """Run simple query on selected variable"""
        if not self.model_learned:
            messagebox.showinfo("Info", "Please learn a model first")
            return

        var = self.query_var.get()
        if var:
            result = f"Query: P({var})\n\nNote: Inference requires Java backend integration.\n"
            result += f"Selected variable: {var}\n"
            result += f"Unique values: {self.dataset[var].nunique()}\n"
            result += f"Value distribution:\n{self.dataset[var].value_counts()}"

            messagebox.showinfo("Query Result", result)
            self.log(f"Query executed for variable: {var}")

    def execute_query(self):
        """Execute probabilistic query"""
        if not self.model_learned:
            messagebox.showinfo("Info", "Please learn a model first")
            return

        query_vars = self.query_vars_entry.get()
        evidence = self.evidence_entry.get()

        self.query_results.delete(1.0, tk.END)
        self.query_results.insert(tk.END, f"Query Variables: {query_vars}\n")
        self.query_results.insert(tk.END, f"Evidence: {evidence}\n")
        self.query_results.insert(tk.END, "\n" + "-" * 60 + "\n")
        self.query_results.insert(tk.END, "Note: Full inference requires Java backend.\n")
        self.query_results.insert(tk.END, "\nQuery would compute:\n")
        self.query_results.insert(tk.END, f"P({query_vars} | {evidence})\n")

        self.log(f"Query executed: {query_vars} given {evidence}")

    def show_model_info(self):
        """Show model information"""
        if self.model_learned:
            info = f"Model Type: Sum-Product Network\n\n"
            info += f"Parameters:\n"
            info += f"  Discretization: {self.disc_method.get()}\n"
            info += f"  Bins: {self.num_bins.get()}\n"
            info += f"  Cluster Penalty: {self.cluster_penalty.get():.1f}\n"
            info += f"  Significance: {self.significance.get()}\n"
            messagebox.showinfo("Model Information", info)
        else:
            messagebox.showinfo("Model Information", "No model has been learned yet")

    def show_about(self):
        """Show about dialog"""
        about_text = """WMI-SPN Interface v1.0

A graphical interface for learning Sum-Product Networks
from tabular data and performing probabilistic inference.

Features:
- Data loading and exploration
- Automatic data summarization
- SPN model learning
- Probabilistic query interface

Based on LearnSPN algorithm
(Gens & Domingos, 2013)
"""
        messagebox.showinfo("About", about_text)

    def show_documentation(self):
        """Show documentation"""
        doc_text = """Quick Start Guide:

1. Load Data
   - File > Load Dataset or Load Example

2. Explore Data
   - View data in Data Preview tab
   - Check Summary tab for statistics

3. Learn Model
   - Adjust parameters in left panel
   - Click Learn Model button

4. Run Queries
   - Use Query Interface tab
   - Enter query variables and evidence
   - Execute query

For detailed documentation, see GUI_README.md
"""
        messagebox.showinfo("Documentation", doc_text)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SPNInterfaceTkinter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
