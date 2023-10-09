from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

from scipy.stats import chi2_contingency, kruskal, f_oneway, spearmanr, pearsonr

def make_column_non_graphic_analysis(
    df: pd.DataFrame,
    value_col: str,
    category_col: str    
):
    """Describes column separated by category"""
    if str(df[value_col].dtype) == 'object':
        display(
            df.groupby(category_col)[value_col].describe().T
        )
    else:
        display(
            df.groupby(category_col)[value_col].describe(
                percentiles=[.05, .25, .5, .75, .95]
            ).T
        )        

def plot_distributions(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    figsize: Tuple[int, int],
    kind: str,  # Puede ser hist o box
    showfliers: bool 
):
    """Plots column distributions by boxplot or histogram divided by category """
    plt.figure(figsize=(figsize))
    ax = plt.gca()
    title = (
        f"Distribution of {value_col} by {category_col}"
        if category_col is not None
        else f"Distribution of {value_col}"
    )
    if showfliers:
        plot_data = df
        title += " with outliers"
    else:
        plot_data = df[
            (df[value_col] <= df[value_col].quantile(0.95))
            & (df[value_col] >= df[value_col].quantile(0.05))
        ]
        title += " without outliers"

    plt.title(title)
    if kind == "box":
        sns.boxplot(
            df,
            x=category_col,
            y=value_col,
            showfliers=showfliers
        )
    else:
        sns.histplot(
            df,
            hue=category_col,
            x=value_col,
            element="step"
        )
    if df[category_col].nunique() > 10:
        plt.xticks(rotation=90)

    plt.show()
    plt.close()


def plot_time_series(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    category_col: str, # Puede ser None si no hay categorias
    figsize: Tuple[int, int]
):
    """Plots column time series"""
    plt.figure(figsize=(figsize))
    ax = plt.gca()
    title = (
        f"Time series of {value_col} by {category_col}"
        if category_col is not None
        else f"Time series of {value_col}"
    )
    plt.title(title)
    sns.lineplot(
        df.sort_values(date_col),
        x=date_col,
        y=value_col,
        hue=category_col,
        ax=ax
    )
    plt.show()
    plt.close()
    
def calculate_chi2(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    significance_level: float
):
    """Calculates chi2 between two categorical columns"""
    df_valid = df[
        (~df[cat_col1].isnull())
        & (~df[cat_col2].isnull())
    ]
    contingency = pd.crosstab(
        df_valid[cat_col1],
        df_valid[cat_col2]
    )
    chi2_results = chi2_contingency(contingency)
    chi2_statistic = chi2_results.statistic
    chi2_pvalue = chi2_results.pvalue
    is_significantly_different = chi2_pvalue > significance_level
    
    display(pd.DataFrame({
        "col_1": cat_col1,
        "col_2": cat_col2,
        "chi2_statistic": [chi2_statistic],
        "chi2_pvalue": [chi2_pvalue],
        "significantly_different": [is_significantly_different]
    }))

def test_population_difference(
    df: pd.DataFrame,
    value_col: str,
    category_col: str, # Puede ser None si no hay categorias
    test: str, # kruskal or anova
    significance_level: float
):
    """Test if populations of categories are significantly different with anova or kruskal"""
    df_valid = df[
        (~df[value_col].isnull())
        & (~df[category_col].isnull())
    ]
    samples = [
        df_group[value_col].tolist()
        for _, df_group in df_valid.groupby(category_col)
    ]

    if test == "kruskal":
        test_result = kruskal(
            *samples
        )
    else:
        test_result = f_oneway(
            *samples
        )
    test_statistic = test_result.statistic
    test_pvalue = test_result.pvalue
    is_significantly_different = test_pvalue < significance_level
    display(pd.DataFrame({
        "value_col": [value_col],
        "category_col": [category_col],
        "test": [test],
        "statistic": [test_statistic],
        "pvalue": [test_pvalue],
        "significantly_different": [is_significantly_different]
    }))
    
def test_correlation(
    df: pd.DataFrame,
    value_col1: str,
    value_col2: str,
    test: str, # pearson or spearman
    significance_level: float
):
    """Tests if variables are correlated"""
    df_valid = df[
        (~df[value_col1].isnull())
        & (~df[value_col2].isnull())
    ]
    if test == "pearson":
        test_result = pearsonr(
            df_valid[value_col1].tolist(),
            df_valid[value_col2].tolist()
        )
    else:
        test_result = spearmanr(
            df_valid[value_col1].tolist(),
            df_valid[value_col2].tolist()
        )
    test_statistic = test_result.statistic
    test_pvalue = test_result.pvalue
    is_significantly_dependent = test_pvalue < significance_level
    return pd.DataFrame({
        "value_col1": [value_col1],
        "value_col2": [value_col2],
        "test": [test],
        "statistic": [test_statistic],
        "pvalue": [test_pvalue],
        "dependent": [is_significantly_dependent]
    })
    
def heatmap_time(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    category_col: str, # Puede ser None si no hay categorias
    figsize: Tuple[int, int],
    central_statistic: str # mean or median
):
    
    df_valid = df[[value_col, time_col, category_col]].copy(deep=True)
    
    df_valid[time_col] = df_valid[time_col]
    
    if central_statistic == "mean":
        df_grouped = df_valid.groupby([category_col, time_col])[value_col].mean().reset_index()
    else:
        df_grouped = df_valid.groupby([category_col, time_col])[value_col].median().reset_index()
        
    df_pivot = df_grouped.pivot(
        columns=time_col,
        index=category_col,
        values=value_col
    )
    df_pivot = df_pivot[sorted(list(df_pivot.columns))]
    plt.figure(figsize=figsize)
    plt.title(f"Temporal accumulations of {category_col} by {time_col}")
    ax = plt.gca()
    sns.heatmap(df_pivot, annot=False, ax=ax)
    plt.show()
    plt.close()
