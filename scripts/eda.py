# eda.py
# author: Wesley Beard
# data: 2025-12-03

import click
import altair as alt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_plot import correlation_plot

@click.command()
@click.option('--clean-data', type=str, help="Path to cleaned data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be saved to")
def main(clean_data, plot_to):
    """
    Create two tables: one for summary statistics and the other for counts and dtypes
    of the features. As well, three plots are created to aid in exploratory data analysis:
    univariate distributions of numeric features, distributions of features by wine type,
    and a pairwise correlations plot for the numeric features
    """
    
    train_df = pd.read_csv(clean_data)

    # create and save a csv of the summary statistics of the data
    df_summary = train_df.describe()
    df_summary.to_csv(f"{plot_to}/tables/data_summary.csv")

    # create and save a csv for count and dtypes of features
    df_info = pd.DataFrame({
        "Features": train_df.columns,
        "Total Entries": train_df.count(),
        "Null Values": train_df.isnull().sum(),
        "Data Type": train_df.dtypes
        }).sort_values(by=["Data Type", "Features"])
    df_info.to_csv(f"{plot_to}/tables/data_info.csv", index=False)

    # allow for plotting over 5000 examples and extract relevant columns
    alt.data_transformers.enable("vegafusion")
    numeric_cols = train_df.select_dtypes('number').columns.tolist()

    # create several univariate distributions of numeric features
    plots = []
    for col in numeric_cols:
        plot = alt.Chart(train_df).mark_bar().encode(
            alt.X(
                col, type="quantitative",
                bin=alt.Bin(maxbins=40),
                title=(f"{col} (binned)")
            ),
            alt.Y("count()",title="Count").stack(False),
        ).properties(
            width=140,
            height=110
        )
        plots.append(plot)

    # parse out plots for nicer visualizations
    hist_plot = alt.concat(*plots, columns=3).properties(
        title="Univariate Distributions of Numeric Features"
    ).configure_title(fontSize=20)
    
    hist_plot.save(f"{plot_to}/figures/hist_univariate_distributions.png")

    # create several distributions of features by wine type
    plot_2 = []
    numeric_cols_2 = numeric_cols.copy()
    numeric_cols_2.remove("target")
    target_col = "target"

    for col2 in numeric_cols_2:
        kde_plot = alt.Chart(train_df).transform_density(
            density=col2,
            groupby=[target_col],
            as_=[f"{col2}", "kde_density"],
            steps=200
        ).mark_line().encode(
            alt.X(f"{col2}:Q", title=col2),
            alt.Y("kde_density:Q", title="KDE Density").stack(False),
            alt.Color(f"{target_col}:N", title="Wine Type")
        ).properties(
            width=130,
            height=100
        )
        plot_2.append(kde_plot)

    # group distributions together for nicer plotting
    kde_final = alt.concat(*plot_2, columns=3).resolve_scale(
        x="independent",
        y="independent"
    ).properties(
        title="Distributions of Different Features by Wine Type"
    ).configure_title(fontSize=20)
    
    kde_final.save(f"{plot_to}/figures/distributions_of_features.png") 

    # create a pairwise correlations plot for the numeric features
    correlation_plot(train_df, numeric_cols)

if __name__ == '__main__':
    main()