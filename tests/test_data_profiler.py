
import pandas as pd
import pytest

from data.data_profiler import profile_dataset


def test_profile_dataset_returns_basic_shape():
    """
    Verify that row and column counts are correct.
    """

    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Athens", "Rome", "Paris"],
        }
    )

    profile = profile_dataset(dataframe)

    assert profile["row_count"] == 3
    assert profile["column_count"] == 2


def test_profile_dataset_detects_column_types():
    """
    Verify that numeric, categorical and boolean columns
    are classified correctly.
    """

    dataframe = pd.DataFrame(
        {
            "age": [20, 30],
            "city": ["Athens", "Rome"],
            "active": [True, False],
        }
    )

    profile = profile_dataset(dataframe)

    assert profile["numeric_columns"] == ["age"]
    assert profile["categorical_columns"] == ["city"]
    assert profile["boolean_columns"] == ["active"]


def test_profile_dataset_counts_missing_values():
    """
    Verify that missing values are counted correctly.
    """

    dataframe = pd.DataFrame(
        {
            "age": [20, None, 40],
            "city": ["Athens", "Rome", None],
        }
    )

    profile = profile_dataset(dataframe)

    assert profile["missing_values"]["age"] == 1
    assert profile["missing_values"]["city"] == 1
    assert profile["columns"]["age"]["missing_percentage"] == 33.33


def test_profile_dataset_counts_duplicates():
    """
    Verify that duplicate rows are detected.
    """

    dataframe = pd.DataFrame(
        {
            "age": [20, 20, 30],
            "city": ["Athens", "Athens", "Rome"],
        }
    )

    profile = profile_dataset(dataframe)

    assert profile["duplicate_rows"] == 1


def test_profile_dataset_rejects_non_dataframe():
    """
    Verify that invalid input raises TypeError.
    """

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        profile_dataset([1, 2, 3])


def test_profile_dataset_rejects_no_columns():
    """
    Verify that a DataFrame with no columns is rejected.
    """

    dataframe = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="does not contain any columns",
    ):
        profile_dataset(dataframe)


def test_profile_dataset_detects_text_column():
    """
    Verify that long free-text values are detected
    as NLP/text columns rather than categorical columns.
    """

    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "country": [
                "Greece",
                "Italy",
                "France",
            ],
            "review": [
                "The service was very good and fast",
                "I really liked this product a lot",
                "The delivery was unfortunately very slow",
            ],
        }
    )

    profile = profile_dataset(dataframe)

    # Age should remain numeric.
    assert "age" in profile["numeric_columns"]

    # Country should be considered categorical.
    assert "country" in profile["categorical_columns"]

    # Review should be recognized as free text.
    assert "review" in profile["text_columns"]

    # A text column should not simultaneously
    # be classified as categorical.
    assert "review" not in profile["categorical_columns"]