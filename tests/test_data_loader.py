from io import BytesIO, StringIO

import pandas as pd
import pytest

from data.data_loader import (
    get_dataset_name,
    get_file_extension,
    read_dataset,
    validate_file_type,
)


class NamedBytesIO(BytesIO):
    """
    In-memory binary file with a filename.

    This imitates uploaded files such as Streamlit's
    UploadedFile object.
    """

    def __init__(
        self,
        content: bytes,
        name: str,
    ):
        super().__init__(content)
        self.name = name


class NamedStringIO(StringIO):
    """
    In-memory text file with a filename.
    """

    def __init__(
        self,
        content: str,
        name: str,
    ):
        super().__init__(content)
        self.name = name


def test_get_file_extension_from_path():
    """
    Verify that extensions are extracted from normal paths.
    """

    result = get_file_extension(
        "data/customer_data.CSV"
    )

    assert result == ".csv"


def test_get_file_extension_from_uploaded_file():
    """
    Verify that extensions can be read from file-like objects.
    """

    uploaded_file = NamedBytesIO(
        b"test",
        "dataset.xlsx",
    )

    result = get_file_extension(uploaded_file)

    assert result == ".xlsx"


def test_validate_file_type_accepts_csv():
    """
    Verify that CSV files are accepted.
    """

    result = validate_file_type("dataset.csv")

    assert result == ".csv"


def test_validate_file_type_rejects_unsupported_format():
    """
    Verify that unsupported formats raise a clear error.
    """

    with pytest.raises(
        ValueError,
        match="Unsupported file format",
    ):
        validate_file_type("dataset.txt")


def test_read_csv_dataset():
    """
    Verify that a CSV file is loaded into a DataFrame.
    """

    csv_file = NamedStringIO(
        "name,age\nAlice,30\nBob,25\n",
        "people.csv",
    )

    dataframe = read_dataset(csv_file)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (2, 2)
    assert dataframe.columns.tolist() == [
        "name",
        "age",
    ]
    assert dataframe.iloc[0]["name"] == "Alice"


def test_read_empty_csv_raises_error():
    """
    Verify that an empty CSV produces a clear ValueError.
    """

    empty_file = NamedStringIO(
        "",
        "empty.csv",
    )

    with pytest.raises(
        ValueError,
        match="dataset is empty",
    ):
        read_dataset(empty_file)


def test_read_dataset_rejects_unsupported_file():
    """
    Verify that unsupported files are rejected before
    pandas attempts to read them.
    """

    text_file = NamedStringIO(
        "some text",
        "notes.txt",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file format",
    ):
        read_dataset(text_file)


def test_get_dataset_name_from_path():
    """
    Verify that the extension is removed from a path.
    """

    result = get_dataset_name(
        "data/customer_churn.csv"
    )

    assert result == "customer_churn"


def test_get_dataset_name_from_uploaded_file():
    """
    Verify that the extension is removed from an uploaded file.
    """

    uploaded_file = NamedBytesIO(
        b"content",
        "sales_2026.xlsx",
    )

    result = get_dataset_name(uploaded_file)

    assert result == "sales_2026"