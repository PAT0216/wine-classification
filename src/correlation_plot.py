import pandas as pd
import altair as alt
import os

def correlation_plot(wine_data, correlation_cols, save_to="results/figures/pairwise_correlations.png"):
    """
    Generate and save a pairwise correlation heatmap for select numeric columns.

    This function computes a correlation matrix for the specified columns of the
    input DataFrame and produces a heatmap. The resulting PNG image is saved to 
    the path provided. The correlation DataFrame is returned for validation and 
    used for testing.

    Parameters
    ----------
    wine_data : pandas.DataFrame
        The dataset containing all wine features.
    correlation_cols : list of str
        A list of numeric column names to compute correlations for.
    save_to : str, default="results/figures/pairwise_correlations.png"
        File path (including filename, e.g. 'directory/corr_plot.png') 
        where the plot should be saved.

    Returns
    -------
    pandas.DataFrame
        A correlation table useful for testing.

    Raises
    ------
    TypeError
        If `wine_data` is not a DataFrame.
        If `correlation_cols` is not a list.
        If any entries in `correlation_cols` are not strings.
        If `save_to` is not a string.

    ValueError
        If `wine_data` is an empty DataFrame.
        If `correlation_cols` is an empty list.
        If any of the columns in `correlation_cols` do not 
        exist in `wine_data`. 
        
    Examples
    --------
    >>> df = pd.DataFrame({"a": [1,2,3], "b": [2,4,6]})
    >>> corr_df = correlation_plot(df, ["a", "b"], "corr.png")
    """
    # check wine_data data type
    if not isinstance(wine_data, pd.DataFrame):
        raise TypeError("The wine_data must be a pandas DataFrame.")

    # check if DataFrame is empty
    if wine_data.empty:
        raise ValueError("The wine_data DataFrame cannot be empty.")

    # check if correlation_cols is a list
    if not isinstance(correlation_cols, list):
        raise TypeError(" The correlation_cols must be a list of column names.")

    # check if correlation_cols is empty
    if len(correlation_cols) == 0:
        raise ValueError("The correlation_cols cannot be an empty list.")

    # check if each entry is a string
    if not all(isinstance(c, str) for c in correlation_cols):
        raise TypeError("All entries in correlation_cols must be strings.")

    # check columns exist in the DataFrame
    missing = set(correlation_cols) - set(wine_data.columns)
    if missing:
        raise ValueError(f"The following columns were not found in the DataFrame: {missing}")

    # check if save_to is a string
    if not isinstance(save_to, str):
        raise TypeError("save_to must be a string file path.")

    # check directory exists
    directory = os.path.dirname(save_to)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # function begins
    corr = wine_data[correlation_cols].corr().reset_index().melt("index")
    corr.columns = ["feature_x", "feature_y", "correlation"]

    # generate and save heatmap
    heatmap = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            alt.X("feature_x:N", title="Feature X"),
            alt.Y("feature_y:N", title="Feature Y"),
            color=alt.Color(
                "correlation:Q", scale=alt.Scale(domain=(-1, 1),scheme="purpleorange"),
                title="Correlation"
            ),
            tooltip=["feature_x", "feature_y", "correlation"]
        )
        .properties(width=300, height=300, title="Correlation Heatmap of Wine Chemical Features")
        .configure_title(fontSize=15)
    )
    
    heatmap.save(save_to)

    return corr