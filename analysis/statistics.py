from typing import Any

import pandas as pd

from data.data_profiler import profile_dataset


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Validate that the input is a usable pandas DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.shape[1] == 0:
        raise ValueError(
            "The DataFrame does not contain any columns."
        )


def get_numeric_statistics(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Calculate descriptive statistics for numeric columns.

    Statistics include:
    - non-null count
    - missing count
    - mean
    - standard deviation
    - minimum
    - first quartile
    - median
    - third quartile
    - maximum
    - number of unique values
    - number of zeros
    - number of negative values
    """

    result = {}

    for column in columns:
        series = dataframe[column]

        # Convert compatible values to numeric.
        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        non_null_series = numeric_series.dropna()

        column_result = {
            "count": int(non_null_series.count()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(
                non_null_series.nunique()
            ),
        }

        # Avoid calculating statistics on an empty column.
        if non_null_series.empty:
            column_result.update(
                {
                    "mean": None,
                    "standard_deviation": None,
                    "minimum": None,
                    "first_quartile": None,
                    "median": None,
                    "third_quartile": None,
                    "maximum": None,
                    "zero_count": 0,
                    "negative_count": 0,
                }
            )

        else:
            column_result.update(
                {
                    "mean": float(
                        non_null_series.mean()
                    ),
                    "standard_deviation": (
                        float(non_null_series.std())
                        if len(non_null_series) > 1
                        else None
                    ),
                    "minimum": float(
                        non_null_series.min()
                    ),
                    "first_quartile": float(
                        non_null_series.quantile(0.25)
                    ),
                    "median": float(
                        non_null_series.median()
                    ),
                    "third_quartile": float(
                        non_null_series.quantile(0.75)
                    ),
                    "maximum": float(
                        non_null_series.max()
                    ),
                    "zero_count": int(
                        (non_null_series == 0).sum()
                    ),
                    "negative_count": int(
                        (non_null_series < 0).sum()
                    ),
                }
            )

        result[column] = column_result

    return result


def get_categorical_statistics(
    dataframe: pd.DataFrame,
    columns: list[str],
    top_n: int = 10,
) -> dict[str, dict[str, Any]]:
    """
    Calculate descriptive statistics for categorical columns.

    Statistics include:
    - non-null count
    - missing count
    - unique category count
    - most frequent category
    - most frequent category count
    - top category frequencies
    """

    result = {}

    for column in columns:
        series = dataframe[column]
        non_null_series = series.dropna()

        frequencies = (
            non_null_series
            .astype(str)
            .value_counts()
            .head(top_n)
        )

        if frequencies.empty:
            most_frequent = None
            most_frequent_count = 0

        else:
            most_frequent = str(frequencies.index[0])
            most_frequent_count = int(
                frequencies.iloc[0]
            )

        result[column] = {
            "count": int(non_null_series.count()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(
                non_null_series.nunique()
            ),
            "most_frequent": most_frequent,
            "most_frequent_count":
                most_frequent_count,
            "top_values": {
                str(value): int(count)
                for value, count
                in frequencies.items()
            },
        }

    return result


def get_text_statistics(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Calculate descriptive statistics for free-text columns.

    Statistics include:
    - non-null and missing counts
    - unique text count
    - average, minimum and maximum word count
    - average, minimum and maximum character count
    - number of empty strings
    """

    result = {}

    for column in columns:
        original_series = dataframe[column]

        text_series = (
            original_series
            .dropna()
            .astype(str)
        )

        # Strip spaces so strings containing only whitespace
        # can be detected as empty values.
        cleaned_series = text_series.str.strip()

        empty_string_count = int(
            cleaned_series.eq("").sum()
        )

        # Exclude empty strings from length calculations.
        usable_text = cleaned_series[
            cleaned_series.ne("")
        ]

        if usable_text.empty:
            text_result = {
                "average_words": None,
                "minimum_words": None,
                "maximum_words": None,
                "average_characters": None,
                "minimum_characters": None,
                "maximum_characters": None,
            }

        else:
            word_counts = (
                usable_text
                .str.split()
                .str.len()
            )

            character_counts = usable_text.str.len()

            text_result = {
                "average_words": float(
                    word_counts.mean()
                ),
                "minimum_words": int(
                    word_counts.min()
                ),
                "maximum_words": int(
                    word_counts.max()
                ),
                "average_characters": float(
                    character_counts.mean()
                ),
                "minimum_characters": int(
                    character_counts.min()
                ),
                "maximum_characters": int(
                    character_counts.max()
                ),
            }

        result[column] = {
            "count": int(text_series.count()),
            "missing_count": int(
                original_series.isna().sum()
            ),
            "unique_count": int(
                cleaned_series.nunique()
            ),
            "empty_string_count":
                empty_string_count,
            **text_result,
        }

    return result


def get_datetime_statistics(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Calculate descriptive statistics for datetime columns.

    Statistics include:
    - non-null and missing counts
    - earliest and latest date
    - date range in days
    - unique date count
    """

    result = {}

    for column in columns:
        original_series = dataframe[column]

        datetime_series = pd.to_datetime(
            original_series,
            errors="coerce",
        )

        non_null_series = datetime_series.dropna()

        if non_null_series.empty:
            earliest = None
            latest = None
            range_days = None

        else:
            earliest_value = non_null_series.min()
            latest_value = non_null_series.max()

            earliest = earliest_value.isoformat()
            latest = latest_value.isoformat()

            range_days = int(
                (
                    latest_value
                    - earliest_value
                ).days
            )

        result[column] = {
            "count": int(non_null_series.count()),
            "missing_count": int(
                original_series.isna().sum()
            ),
            "invalid_date_count": int(
                datetime_series.isna().sum()
                - original_series.isna().sum()
            ),
            "unique_count": int(
                non_null_series.nunique()
            ),
            "earliest": earliest,
            "latest": latest,
            "range_days": range_days,
        }

    return result


def get_boolean_statistics(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Calculate descriptive statistics for boolean columns.
    """

    result = {}

    for column in columns:
        series = dataframe[column]
        non_null_series = series.dropna()

        true_count = int(
            non_null_series.eq(True).sum()
        )

        false_count = int(
            non_null_series.eq(False).sum()
        )

        total = true_count + false_count

        result[column] = {
            "count": int(non_null_series.count()),
            "missing_count": int(series.isna().sum()),
            "true_count": true_count,
            "false_count": false_count,
            "true_percentage": (
                round(true_count / total * 100, 2)
                if total > 0
                else None
            ),
            "false_percentage": (
                round(false_count / total * 100, 2)
                if total > 0
                else None
            ),
        }

    return result


def get_other_statistics(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Return basic statistics for columns that could not be
    classified as numeric, categorical, text, datetime,
    or boolean.
    """

    result = {}

    for column in columns:
        series = dataframe[column]

        result[column] = {
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(
                series.nunique(dropna=True)
            ),
        }

    return result


def get_dataset_statistics(
    dataframe: pd.DataFrame,
    profile: dict[str,Any]
) -> dict[str, Any]:
    """
    Generate descriptive statistics for every semantic
    variable type detected by data_profiler.py.

    This function does not perform correlation analysis,
    visualization, inference, or predictive modelling.
    """

    _validate_dataframe(dataframe)

    # Reuse the profiler so both modules follow the same
    # variable-type definitions.

    known_columns = set(
        profile["numeric_columns"]
        + profile["categorical_columns"]
        + profile["text_columns"]
        + profile["datetime_columns"]
        + profile["boolean_columns"]
    )

    # Find any columns that were classified as "other".
    other_columns = [
        column
        for column in dataframe.columns
        if column not in known_columns
    ]

    return {
        "dataset": {
            "row_count": profile["row_count"],
            "column_count": profile["column_count"],
            "duplicate_rows":
                profile["duplicate_rows"],
        },
        "numeric": get_numeric_statistics(
            dataframe,
            profile["numeric_columns"],
        ),
        "categorical": get_categorical_statistics(
            dataframe,
            profile["categorical_columns"],
        ),
        "text": get_text_statistics(
            dataframe,
            profile["text_columns"],
        ),
        "datetime": get_datetime_statistics(
            dataframe,
            profile["datetime_columns"],
        ),
        "boolean": get_boolean_statistics(
            dataframe,
            profile["boolean_columns"],
        ),
        "other": get_other_statistics(
            dataframe,
            other_columns,
        ),
    }