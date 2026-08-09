import pandas as pd

from agents.analysis_tools import (
    calculate_spearman,
    compare_two_groups,
    get_dataset_overview,
)
from agents.tool_context import dataset_context


def setup_module():
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    profile = {
        "numeric_columns": ["x", "y"],
        "categorical_columns": ["group"],
        "boolean_columns": [],
        "text_columns": [],
        "datetime_columns": [],
        "other_columns": [],
    }
    dataset_context.set_dataset(dataframe, profile, dataset_id="unit-test")


def teardown_module():
    dataset_context.clear()


def test_overview_tool():
    result = get_dataset_overview.invoke({})
    assert result["rows"] == 6
    assert "x" in result["numeric_columns"]


def test_spearman_tool():
    result = calculate_spearman.invoke(
        {"first_column": "x", "second_column": "y"}
    )
    assert result["statistic"] == 1.0


def test_two_group_tool():
    result = compare_two_groups.invoke(
        {"numeric_column": "x", "group_column": "group"}
    )
    assert result["method"] == "mann_whitney_u"
