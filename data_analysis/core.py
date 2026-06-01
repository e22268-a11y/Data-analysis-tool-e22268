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
    def plot_pie(df, cat_col, title="Pie Chart"):
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, 'count']
        fig = px.pie(counts, names=cat_col, values='count', title=title)
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
        
    # --- 1. Data Ingestion & Sanitization ---
    def upload_data(self):
        """Handles local file uploads in Google Colab."""
        if not IN_COLAB:
            print("Not running in Google Colab. Please pass a DataFrame directly to the inspector.")
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

    # --- 2. Structural Analysis & Cleaning ---
    def data_summary(self):
        """Displays row/column counts, types, and a preview."""
        if self.df is None: return print("No data loaded.")
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print("-" * 50)
        print("DATASET SUMMARY")
        print("-" * 50)
        print(f"Total Rows: {self.df.shape[0]}")
        print(f"Total Columns: {self.df.shape[1]}")
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("-" * 50)
        print("Data Preview (First 20 Rows):")
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

    def remove_duplicates(self):
        """Removes exact duplicate rows from the dataset."""
        if self.df is None: return
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {before - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns=None, action='flag'):
        """IQR-based outlier detection. Action: 'flag' or 'delete'."""
        if not columns:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        outlier_indices = set()
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            col_outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)].index
            outlier_indices.update(col_outliers)
            
        if action == 'delete':
            self.df.drop(index=list(outlier_indices), inplace=True)
            print(f"Deleted {len(outlier_indices)} outlier rows.")
        else:
            self.df['is_outlier'] = False
            self.df.loc[list(outlier_indices), 'is_outlier'] = True
            print(f"Flagged {len(outlier_indices)} rows as outliers in new 'is_outlier' column.")

    def delete_rows(self, row_indices_str):
        """Interactive pruning: Takes a comma-separated string of indices."""
        indices = [int(i.strip()) for i in row_indices_str.split(',') if i.strip().isdigit()]
        valid_indices = [i for i in indices if i in self.df.index]
        self.df.drop(index=valid_indices, inplace=True)
        print(f"Deleted rows: {valid_indices}")

    def delete_columns(self, cols_str):
        """Interactive pruning: Takes a comma-separated string of column names."""
        cols = [c.strip() for c in cols_str.split(',')]
        valid_cols = [c for c in cols if c in self.df.columns]
        self.df.drop(columns=valid_cols, inplace=True)
        print(f"Deleted columns: {valid_cols}")

    # --- 3. Feature Engineering Preparation ---
    def extract_normalized_numeric_data(self, strategy='standard'):
        """Scales numeric data. strategies: 'minmax', 'standard', 'robust'"""
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if num_df.empty: return pd.DataFrame()
        
        if strategy == 'minmax': scaler = MinMaxScaler()
        elif strategy == 'standard': scaler = StandardScaler()
        elif strategy == 'robust': scaler = RobustScaler()
        else: raise ValueError("Invalid scaling strategy.")
            
        scaled_data = scaler.fit_transform(num_df)
        return pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)

    def extract_normalized_categorical_data(self, strategy='onehot'):
        """Encodes categorical data. strategies: 'onehot', 'ordinal', 'uniform'"""
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if cat_df.empty: return pd.DataFrame()

        if strategy == 'onehot':
            return pd.get_dummies(cat_df, drop_first=True)
        elif strategy == 'ordinal':
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(cat_df)
            return pd.DataFrame(encoded, columns=cat_df.columns, index=cat_df.index)
        elif strategy == 'uniform':
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(cat_df)
            scaler = MinMaxScaler()
            uniform_data = scaler.fit_transform(encoded)
            return pd.DataFrame(uniform_data, columns=cat_df.columns, index=cat_df.index)
        else:
            raise ValueError("Invalid encoding strategy.")

    def merge_features(self, num_strategy='standard', cat_strategy='onehot'):
        """Creates a unified DataFrame with scaled numeric and encoded categorical data."""
        num_prep = self.extract_normalized_numeric_data(strategy=num_strategy)
        cat_prep = self.extract_normalized_categorical_data(strategy=cat_strategy)
        unified_df = pd.concat([num_prep, cat_prep], axis=1)
        print(f"Features merged successfully. Final shape: {unified_df.shape}")
        return unified_df

    # --- 4. Advanced Interactive Visualization ---
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

    def plot_relationship(self, col1, col2):
        """Detects data types and plots the correct interactive chart."""
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Scatter: {col1} vs {col2}")
        elif not is_num1 and not is_num2:
            count_df = self.df.groupby([col1, col2]).size().reset_index(name='Count')
            fig = px.bar(count_df, x=col1, y='Count', color=col2, barmode='group', title=f"Grouped Bar: {col1} by {col2}")
        else:
            cat_c, num_c = (col1, col2) if not is_num1 else (col2, col1)
            fig = px.box(self.df, x=cat_c, y=num_c, points="all", title=f"Boxplot: {num_c} grouped by {cat_c}")
        fig.show()

    def plot_categorical_frequency(self, col):
        """Bar chart displaying raw counts and percentage labels."""
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return print("Column must be categorical.")
            
        counts = self.df[col].value_counts().reset_index()
        counts.columns = [col, 'Count']
        counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)
        counts['TextLabel'] = counts['Count'].astype(str) + " (" + counts['Percentage'].astype(str) + "%)"
        
        fig = px.bar(counts, x=col, y='Count', text='TextLabel', title=f"Frequency & Percentage of {col}")
        fig.update_traces(textposition='outside')
        fig.show()

    # --- 5. Deep Statistical Insights ---
    def plot_all_associations_heatmap(self):
        """Computes Pearson, Cramér's V, and Eta into a unified matrix."""
        df_clean = self.df.dropna()
        cols = df_clean.columns
        n = len(cols)
        matrix = pd.DataFrame(np.ones((n, n)), columns=cols, index=cols)
        
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
            n_tot = confusion_matrix.sum().sum()
            phi2 = chi2 / n_tot
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n_tot-1))
            rcorr = r - ((r-1)**2)/(n_tot-1)
            kcorr = k - ((k-1)**2)/(n_tot-1)
            if min((kcorr-1), (rcorr-1)) == 0: return 0.0
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            
        def correlation_ratio(cat, num):
            fcat, _ = pd.factorize(cat)
            cat_num = np.max(fcat)+1
            y_avg = np.zeros(cat_num)
            n_arr = np.zeros(cat_num)
            for i in range(cat_num):
                cat_measures = num[np.argwhere(fcat == i).flatten()]
                n_arr[i] = len(cat_measures)
                y_avg[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0
            y_tot_avg = np.sum(num)/len(num)
            numerator = np.sum(np.multiply(n_arr, np.power(np.subtract(y_avg, y_tot_avg), 2)))
            denominator = np.sum(np.power(np.subtract(num, y_tot_avg), 2))
            if denominator == 0: return 0.0
            return np.sqrt(numerator/denominator)

        for i in range(n):
            for j in range(n):
                if i == j: continue
                col1, col2 = cols[i], cols[j]
                is_num1 = pd.api.types.is_numeric_dtype(df_clean[col1])
                is_num2 = pd.api.types.is_numeric_dtype(df_clean[col2])
                
                if is_num1 and is_num2:
                    matrix.loc[col1, col2] = scipy.stats.pearsonr(df_clean[col1], df_clean[col2])[0]
                elif not is_num1 and not is_num2:
                    matrix.loc[col1, col2] = cramers_v(df_clean[col1], df_clean[col2])
                else:
                    cat_c, num_c = (col1, col2) if not is_num1 else (col2, col1)
                    matrix.loc[col1, col2] = correlation_ratio(df_clean[cat_c].values, df_clean[num_c].values)

        fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title="Unified Association Heatmap")
        fig.show()
