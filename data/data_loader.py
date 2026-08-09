from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd


# A file input can be:
# - a normal file path;
# - a Path object;
# - an uploaded/in-memory binary file.
FileInput = Union[str, Path, BinaryIO]


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".ods",
    ".parquet",
}


def get_file_extension(file: FileInput) -> str:
    """
    Return the lowercase extension of a file.

    This works with:
    - string paths;
    - pathlib.Path objects;
    - Streamlit UploadedFile objects;
    - other file-like objects that contain a `name` attribute.

    Example:
        dataset.csv -> .csv
    """

    # Normal file paths can be passed directly to pathlib.
    if isinstance(file, (str, Path)):
        return Path(file).suffix.lower()

    # Uploaded files usually have a name attribute.
    file_name = getattr(file, "name", None)

    if not file_name:
        raise ValueError(
            "The uploaded file does not have a filename."
        )

    return Path(file_name).suffix.lower()


def validate_file_type(file: FileInput) -> str:
    """
    Verify that the supplied file format is supported.

    Returns:
        The validated lowercase file extension.

    Raises:
        ValueError: If the file extension is unsupported.
    """

    extension = get_file_extension(file)

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported file format: {extension or 'unknown'}. "
            f"Supported formats are: {supported}."
        )

    return extension


def reset_file_position(file: FileInput) -> None:
    """
    Move an uploaded/in-memory file back to its beginning.

    Streamlit UploadedFile objects behave like file objects.
    After pandas reads them once, their internal position may
    no longer be at the beginning.

    Normal string and Path inputs do not need resetting.
    """

    if isinstance(file, (str, Path)):
        return

    seek_method = getattr(file, "seek", None)

    if callable(seek_method):
        seek_method(0)


def read_dataset(file: FileInput) -> pd.DataFrame:
    """
    Load a supported dataset into a pandas DataFrame.

    Supported formats:
    - CSV
    - Excel (.xlsx and .xls)
    - OpenDocument Spreadsheet (.ods)
    - Parquet

    Args:
        file:
            File path or uploaded/in-memory file object.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        ValueError:
            If the file format is unsupported, the file is empty,
            or the dataset contains no columns.
        RuntimeError:
            If pandas cannot read the file.
    """

    # Determine and validate the file format before reading it.
    extension = validate_file_type(file)

    # Ensure an uploaded file starts from byte position zero.
    reset_file_position(file)

    try:
        # Select the correct pandas function based on file type.
        if extension == ".csv":
            dataframe = pd.read_csv(file)

        elif extension in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(file)

        elif extension == ".ods":
            # Pandas uses odfpy for OpenDocument spreadsheets.
            dataframe = pd.read_excel(
                file,
                engine="odf",
            )

        elif extension == ".parquet":
            dataframe = pd.read_parquet(file)

        else:
            # This should not normally happen because the file
            # type was already validated above.
            raise ValueError(
                f"No loader is configured for {extension}."
            )

    except pd.errors.EmptyDataError as exc:
        # This commonly occurs when a CSV file has no content.
        raise ValueError(
            "The uploaded dataset is empty."
        ) from exc

    except Exception as exc:
        # Convert pandas or engine-specific exceptions into
        # a clearer application-level error.
        raise RuntimeError(
            f"Could not read the dataset: {exc}"
        ) from exc

    # Confirm that pandas returned the expected data structure.
    if not isinstance(dataframe, pd.DataFrame):
        raise RuntimeError(
            "The dataset loader did not return a DataFrame."
        )

    # A dataset without any columns cannot be analyzed.
    if dataframe.shape[1] == 0:
        raise ValueError(
            "The dataset does not contain any columns."
        )

    return dataframe


def get_dataset_name(file: FileInput) -> str:
    """
    Return a clean dataset name without its file extension.

    Example:
        customer_churn.csv -> customer_churn
    """

    if isinstance(file, (str, Path)):
        return Path(file).stem

    file_name = getattr(file, "name", None)

    if not file_name:
        return "uploaded_dataset"

    return Path(file_name).stem