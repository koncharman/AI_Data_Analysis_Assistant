import pandas as pd
import pytest

from analysis.statistics import get_dataset_statistics
from data.data_profiler import profile_dataset

def test_dataset_statistics_for_all_supported_types():
    """
    Verify that the statistics module handles:
    - numeric
    - categorical
    - text
    - datetime
    - boolean
    """

    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, None],
            "department": [
                "Sales",
                "IT",
                "Sales",
                "HR",
            ],
            "review": [
                "The service was very good and fast",
                "I really liked this product very much",
                "The delivery was unfortunately very slow",
                None,
            ],
            "created_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-03",
                    "2026-01-05",
                    None,
                ]
            ),
            "active": pd.Series(
                [True, False, True, None],
                dtype="boolean",
            ),
        }
    )

    statistics = get_dataset_statistics(dataframe,profile_dataset(dataframe))

    # Verify general dataset information.
    assert statistics["dataset"]["row_count"] == 4
    assert statistics["dataset"]["column_count"] == 5

    # Verify numeric statistics.
    age_statistics = statistics["numeric"]["age"]

    assert age_statistics["count"] == 3
    assert age_statistics["missing_count"] == 1
    assert age_statistics["mean"] == 30.0
    assert age_statistics["median"] == 30.0
    assert age_statistics["minimum"] == 20.0
    assert age_statistics["maximum"] == 40.0

    # Verify categorical statistics.
    department_statistics = statistics[
        "categorical"
    ]["department"]

    assert department_statistics["count"] == 4
    assert department_statistics["unique_count"] == 3
    assert department_statistics["most_frequent"] == "Sales"
    assert department_statistics[
        "most_frequent_count"
    ] == 2

    # Verify text statistics.
    review_statistics = statistics["text"]["review"]

    assert review_statistics["count"] == 3
    assert review_statistics["missing_count"] == 1
    assert review_statistics["average_words"] > 0
    assert review_statistics["maximum_words"] > 0
    assert review_statistics["average_characters"] > 0

    # Verify datetime statistics.
    date_statistics = statistics[
        "datetime"
    ]["created_at"]

    assert date_statistics["count"] == 3
    assert date_statistics["missing_count"] == 1
    assert date_statistics["range_days"] == 4
    assert date_statistics["earliest"].startswith(
        "2026-01-01"
    )
    assert date_statistics["latest"].startswith(
        "2026-01-05"
    )

    # Verify boolean statistics.
    active_statistics = statistics[
        "boolean"
    ]["active"]

    assert active_statistics["count"] == 3
    assert active_statistics["missing_count"] == 1
    assert active_statistics["true_count"] == 2
    assert active_statistics["false_count"] == 1


def test_numeric_statistics_count_zeros_and_negatives():
    """
    Verify zero and negative-value counts.
    """

    dataframe = pd.DataFrame(
        {
            "value": [-5, 0, 10, 0],
        }
    )

    statistics = get_dataset_statistics(dataframe,profile_dataset(dataframe))

    value_statistics = statistics["numeric"]["value"]

    assert value_statistics["zero_count"] == 2
    assert value_statistics["negative_count"] == 1


def test_text_statistics_detect_empty_strings():
    """
    Verify that whitespace-only text is counted
    as an empty string.
    """

    dataframe = pd.DataFrame(
        {
            "comment": [
                "This is a useful customer comment for me. I do not know",
                "   ",
                "Another detailed customer response what is this? Is this a problem.",
            ]
        }
    )

    statistics = get_dataset_statistics(dataframe,profile_dataset(dataframe))

    comment_statistics = statistics["text"]["comment"]

    assert comment_statistics["empty_string_count"] == 1
    assert comment_statistics["count"] == 3


def test_categorical_statistics_respect_top_n():
    """
    Verify that only the requested number of category
    frequencies is returned.
    """

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "D",
            ]
        }
    )

    # Import directly because get_dataset_statistics
    # uses the default top_n value.
    from analysis.statistics import (
        get_categorical_statistics,
    )

    result = get_categorical_statistics(
        dataframe,
        columns=["category"],
        top_n=2,
    )

    assert len(result["category"]["top_values"]) == 2
    assert result["category"]["top_values"]["A"] == 2
    assert result["category"]["top_values"]["B"] == 2


def test_statistics_reject_non_dataframe():
    """
    Verify that invalid input raises TypeError.
    """

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        get_dataset_statistics(
            ["not", "a", "dataframe"],profile_dataset(["not", "a", "dataframe"])
        )


def test_statistics_reject_dataframe_without_columns():
    """
    Verify that a DataFrame without columns raises ValueError.
    """

    with pytest.raises(
        ValueError,
        match="does not contain any columns",
    ):
        get_dataset_statistics(
            pd.DataFrame(),profile_dataset(pd.DataFrame())
        )


def test_empty_numeric_column_returns_none_statistics():
    """
    Verify that an all-missing numeric column does not
    cause errors.
    """

    dataframe = pd.DataFrame(
        {
            "score": pd.Series(
                [None, None, None],
                dtype="float64",
            )
        }
    )

    statistics = get_dataset_statistics(dataframe,profile_dataset(dataframe))

    score_statistics = statistics["numeric"]["score"]

    assert score_statistics["count"] == 0
    assert score_statistics["missing_count"] == 3
    assert score_statistics["mean"] is None
    assert score_statistics["median"] is None
    assert score_statistics["minimum"] is None
    assert score_statistics["maximum"] is None