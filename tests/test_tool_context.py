import pandas as pd
import pytest

from agents.tool_context import DatasetContext


def test_context_requires_dataset():
    context = DatasetContext()
    with pytest.raises(RuntimeError):
        context.get_dataframe()
    with pytest.raises(RuntimeError):
        context.get_profile()


def test_context_stores_and_clears_dataset():
    context = DatasetContext()
    dataframe = pd.DataFrame({"x": [1, 2]})
    profile = {"numeric_columns": ["x"]}

    context.set_dataset(dataframe, profile, dataset_id="demo.csv")

    assert context.is_ready() is True
    assert context.get_dataframe().equals(dataframe)
    assert context.get_profile() == profile
    assert context.get_dataset_id() == "demo.csv"

    context.clear()
    assert context.is_ready() is False
