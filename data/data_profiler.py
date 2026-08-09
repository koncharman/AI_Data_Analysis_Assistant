from typing import Any

import pandas as pd


def detect_text_columns(
    dataframe: pd.DataFrame,
    min_avg_words: float = 6.0,
) -> list[str]:
    """
    Detect columns that appear to contain free text.

    The goal is to distinguish columns such as:

        city -> categorical

    from columns such as:

        customer_review -> NLP/text

    A string column is considered text when its non-null
    values contain at least `min_avg_words` words on average.
    """

    text_columns = []

    # Only inspect columns that may contain strings.
    candidate_columns = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in candidate_columns:

        # Remove missing values and convert the remaining
        # values to strings.
        values = (
            dataframe[column]
            .dropna()
            .astype(str)
        )

        # Ignore completely empty columns.
        if values.empty:
            continue

        # Count the number of words in each value.
        word_counts = values.str.split().str.len()

        # Calculate the average text length in words.
        average_words = word_counts.mean()

        # Longer values are likely to represent free text
        # rather than simple categories.
        if average_words >= min_avg_words:
            text_columns.append(column)

    return text_columns


def profile_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze the structure and basic characteristics
    of a pandas DataFrame.

    The generated profile can later be used by:
    - Streamlit
    - the LangGraph agent
    - machine learning tools
    - NLP tools
    - Ollama
    """

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.shape[1] == 0:
        raise ValueError(
            "The DataFrame does not contain any columns."
        )

    # --------------------------------------------------
    # Detect standard column types
    # --------------------------------------------------

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    boolean_columns = dataframe.select_dtypes(
        include="bool"
    ).columns.tolist()

    datetime_columns = dataframe.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # --------------------------------------------------
    # Detect NLP/free-text columns
    # --------------------------------------------------

    text_columns = detect_text_columns(
        dataframe
    )

    # --------------------------------------------------
    # Detect categorical columns
    # --------------------------------------------------

    string_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Remove detected NLP columns from categorical columns.
    categorical_columns = [
        column
        for column in string_columns
        if column not in text_columns
    ]

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    missing_values = (
        dataframe
        .isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    # --------------------------------------------------
    # Unique values
    # --------------------------------------------------

    unique_values = (
        dataframe
        .nunique(dropna=True)
        .astype(int)
        .to_dict()
    )

    # --------------------------------------------------
    # Build detailed information for every column
    # --------------------------------------------------

    column_profiles = {}

    for column in dataframe.columns:

        # Determine our semantic classification.
        if column in text_columns:
            column_type = "text"

        elif column in numeric_columns:
            column_type = "numeric"

        elif column in boolean_columns:
            column_type = "boolean"

        elif column in datetime_columns:
            column_type = "datetime"

        elif column in categorical_columns:
            column_type = "categorical"

        else:
            column_type = "other"

        column_profile = {
            "dtype": str(
                dataframe[column].dtype
            ),

            "column_type": column_type,

            "missing_count": int(
                dataframe[column]
                .isna()
                .sum()
            ),

            "missing_percentage": round(
                float(
                    dataframe[column]
                    .isna()
                    .mean()
                    * 100
                ),
                2,
            ),

            "unique_count": int(
                dataframe[column]
                .nunique(dropna=True)
            ),
        }

        # ----------------------------------------------
        # Add extra information for text columns
        # ----------------------------------------------

        if column in text_columns:

            values = (
                dataframe[column]
                .dropna()
                .astype(str)
            )

            if not values.empty:

                word_counts = (
                    values
                    .str.split()
                    .str.len()
                )

                column_profile[
                    "average_words"
                ] = round(
                    float(word_counts.mean()),
                    2,
                )

                column_profile[
                    "max_words"
                ] = int(
                    word_counts.max()
                )

                column_profile[
                    "average_characters"
                ] = round(
                    float(
                        values.str.len().mean()
                    ),
                    2,
                )

        column_profiles[column] = (
            column_profile
        )

    # --------------------------------------------------
    # Final dataset profile
    # --------------------------------------------------

    profile = {

        "row_count": int(
            dataframe.shape[0]
        ),

        "column_count": int(
            dataframe.shape[1]
        ),

        "duplicate_rows": int(
            dataframe
            .duplicated()
            .sum()
        ),

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "text_columns":
            text_columns,

        "datetime_columns":
            datetime_columns,

        "boolean_columns":
            boolean_columns,

        "missing_values":
            missing_values,

        "unique_values":
            unique_values,

        "columns":
            column_profiles,
    }

    return profile