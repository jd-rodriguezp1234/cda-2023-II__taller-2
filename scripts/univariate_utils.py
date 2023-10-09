from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

sns.set_theme()

pd.set_option("display.max_columns", 100)

def make_not_graphic_analysis(df: pd.DataFrame) -> None:
    """Displays categorical and numerical column statistics"""
    print("-" * 5,"dataset info","-" * 5)
    df.info()
    print("-" * 5,"null count info","-" * 5)
    display(df.isnull().sum() / len(df))
    print("-" * 5,"Object column statistics","-" * 5)
    display(df.describe(include="O"))
    print("-" * 5,"Number column statistics","-" * 5)
    display(df.describe(include=np.number, percentiles=[.05, .25, .5, .75, .95]))
    print("-" * 5,"First five rows","-" * 5)
    
    display(df.head(5))

def plot_column_distribution(df: pd.DataFrame, col: str, figsize: Tuple[int, int], bins: int) -> None:
    """Plots boxplot, histogram and histogram without outliers for a column of a dataframe"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"Distribution of {col}")
    df[col].plot(kind="box", showfliers=False, ax=ax1, title="boxplot")
    df[col].plot(kind="hist", bins=bins, ax=ax2, title="histogram")
    df[
      (df[col] >= df[col].quantile(0.05)) & (df[col] <= df[col].quantile(0.95))
    ][col].plot(kind="hist", bins=bins, ax=ax3, title="histogram without outliers")

    plt.show()
    plt.close()
    
def plot_categories(
    df: pd.DataFrame,
    col: str,
    max_categories: int,
    category_width: float,
    plot_height: int,
    normalize: bool
) -> None:
    """Plots normalized count of categories"""
    df_cat_counts = df[col].value_counts(normalize=normalize)[:max_categories]
    df_cat_counts.plot(
        kind="bar",
        title=f"Count of {col} by category",
        figsize=(category_width * len(df_cat_counts), plot_height)
    )
    plt.show()
    plt.close()
    
def plot_pareto(
    df: pd.DataFrame,
    col: str,
    max_categories: int,
    category_width: float,
    plot_height: int
):
    """Plots pareto count of categories"""
    df_cat_counts = (
        df[col]
        .value_counts(normalize=True)
        .to_frame(name="Percentage")
        .sort_values("Percentage", ascending=False)
    )
    df_cat_counts["Cumulative Percentage"] = df_cat_counts["Percentage"].cumsum()
    plt.figure(figsize=(category_width * len(df_cat_counts), plot_height))
    ax = plt.gca()
    df_cat_counts["Percentage"][:max_categories].plot(
        kind="bar",
        title=f"Pareto distribution of {col} categories",
        color="b",
        label="Percentage",
        ax=ax
    )
    df_cat_counts["Cumulative Percentage"][:max_categories].plot(
        title=f"Pareto distribution of {col} categories",
        color="r",
        label="Acumulative",
        ax=ax
    )
    plt.xticks(rotation=90)

    plt.legend(loc="best")
    plt.show()
    plt.close()