from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List
from pydantic import BaseModel, ValidationError, field_validator

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway, multivariate_normal
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import FactorAnalysis

# Safely handle Google Colab's file upload module
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

class PlottingMethods:
    """Handles granular chart generation returning HTML-wrapped figures."""
    
    @staticmethod
    def _to_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    @staticmethod
    def plot_bar(df, x_col, y_col=None, title="Bar Chart"):
        if y_col:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        else:
            fig = px.bar(df[x_col].value_counts().reset_index(), x='index', y=x_col, title=title)
        return PlottingMethods._to_html(fig)

    @staticmethod
    def plot_histogram(df, num_col, bins=30, title="Histogram"):
        fig = px.histogram(df, x=num_col, nbins=bins, title=title)
        return PlottingMethods._to_html(fig)


class DataInspector:
    """Advanced end-to-end tool for ingestion, cleaning, EDA, and statistical insights."""
    
    def __init__(self, df=None):
        self.df = df
        self.plotter = PlottingMethods()
        
    def upload_data(self):
        """Handles local file uploads in Google Colab."""
        if not IN_COLAB:
            print("Not running in Google Colab. Please instantiate with a dataframe directly.")
            return
            
        print("Please upload your CSV file:")
        uploaded = files.upload()
        for filename in uploaded.keys():
            print(f"Loading {filename}...")
            garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
            self.df = pd.read_csv(io.BytesIO(uploaded[filename]), na_values=garbage_strings)
            self._sanitize_types()
            print(f"Data successfully loaded and sanitized. Shape: {self.df.shape}")
            break

    def _sanitize_types(self):
        """Force-converts objects to numeric if it doesn't destroy the column."""
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            self.df[col].replace('', np.nan, inplace=True)
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if not converted.isna().all() or self.df[col].isna().all():
                self.df[col] = converted

    def data_summary(self):
        """Displays row/column counts, types, and a preview."""
        if self.df is None: return print("No data loaded.")
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print("-" * 50)
        print(f"DATASET SUMMARY")
        print("-" * 50)
        print(f"Total Rows: {self.df.shape[0]}")
        print(f"Total Columns: {self.df.shape[1]}")
        from IPython.display import display
        display(self.df.head(20))

    def handle_missing_values(self, strategy='median', constant_val=None):
        """Imputes missing values based on user strategy."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        if strategy == 'constant' and constant_val is not None:
            self.df.fillna(constant_val, inplace=True)
        elif strategy in ['mean', 'median', 'mode']:
            for col in num_cols:
                if strategy == 'mean': val = self.df[col].mean()
                elif strategy == 'median': val = self.df[col].median()
                elif strategy == 'mode': val = self.df[col].mode()[0]
                self.df[col].fillna(val, inplace=True)
            for col in cat_cols:
                if not self.df[col].mode().empty:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print(f"Missing values handled using strategy: '{strategy}'")

    def plot_univariate_subplots(self, col):
        """Generates a 3-panel subplot for a numeric column."""
        if col not in self.df.select_dtypes(include=[np.number]).columns:
            return print("Column must be numeric.")
            
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Distribution (Box)", "Value vs Index", "Histogram"))
        fig.add_trace(go.Box(x=self.df[col], name="Box", orientation='h', marker_color='teal'), row=1, col=1)
        fig.add_trace(go.Scatter(y=self.df[col], x=self.df.index, mode='markers', marker=dict(color='orange', opacity=0.5), name="Scatter"), row=1, col=2)
        fig.add_trace(go.Histogram(x=self.df[col], nbinsx=30, marker_color='purple', name="Hist"), row=1, col=3)
        fig.update_layout(height=400, title_text=f"Univariate Analysis: {col}", showlegend=False)
        fig.show()
