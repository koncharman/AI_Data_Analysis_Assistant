import pandas as pd
import pytest

from analysis.eda import (
    spearman_correlation,
    mann_whitney_test,
    categorical_association,
    binary_association,
    kruskal_wallis_test,
    mann_whitney_posthoc,
    kruskal_wallis_with_posthoc,
)


# ============================================================
# Spearman
# ============================================================

def test_spearman_positive_relationship():
    """
    Perfect monotonic relationship should produce
    a Spearman correlation close to 1.
    """

    dataframe = pd.DataFrame(
        {
            "experience": [1, 2, 3, 4, 5],
            "salary": [20, 40, 60, 80, 100],
        }
    )

    result = spearman_correlation(
        dataframe,
        "experience",
        "salary",
    )

    assert result["method"] == "spearman"
    assert result["n"] == 5
    assert result["statistic"] == pytest.approx(1.0)
    assert result["p_value"] < 0.05


def test_spearman_handles_missing_values():
    """
    Missing values should be removed only for the
    two variables being analyzed.
    """

    dataframe = pd.DataFrame(
        {
            "x": [1, 2, None, 4, 5],
            "y": [10, 20, 30, None, 50],
        }
    )

    result = spearman_correlation(
        dataframe,
        "x",
        "y",
    )

    # Complete pairs:
    # (1,10), (2,20), (5,50)
    assert result["n"] == 3
    assert result["statistic"] == pytest.approx(1.0)


def test_spearman_rejects_constant_variable():
    """
    Correlation cannot be calculated when one variable
    has no variation.
    """

    dataframe = pd.DataFrame(
        {
            "x": [1, 1, 1, 1],
            "y": [10, 20, 30, 40],
        }
    )

    with pytest.raises(
        ValueError,
        match="variation",
    ):
        spearman_correlation(
            dataframe,
            "x",
            "y",
        )


# ============================================================
# Mann-Whitney
# ============================================================

def test_mann_whitney_two_groups():
    """
    Mann-Whitney should compare a numeric variable
    between exactly two groups.
    """

    dataframe = pd.DataFrame(
        {
            "score": [
                1, 2, 3, 4, 5,
                20, 21, 22, 23, 24,
            ],
            "group": [
                "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B",
            ],
        }
    )

    result = mann_whitney_test(
        dataframe,
        "score",
        "group",
    )

    assert result["method"] == "mann_whitney_u"

    assert result["group_1_n"] == 5
    assert result["group_2_n"] == 5

    assert result["p_value"] < 0.05

    # An effect size should also be returned.
    assert -1 <= result["rank_biserial"] <= 1


def test_mann_whitney_rejects_three_groups():
    """
    Mann-Whitney is only appropriate for two groups.
    """

    dataframe = pd.DataFrame(
        {
            "score": [1, 2, 3, 4, 5, 6],
            "group": [
                "A", "A",
                "B", "B",
                "C", "C",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="exactly two groups",
    ):
        mann_whitney_test(
            dataframe,
            "score",
            "group",
        )


# ============================================================
# Categorical ↔ Categorical
# ============================================================

def test_categorical_association():
    """
    Chi-square and Cramer's V should be returned for
    two categorical variables.
    """

    dataframe = pd.DataFrame(
        {
            "department": (
                ["IT"] * 20
                + ["Sales"] * 20
            ),
            "level": (
                ["Senior"] * 18
                + ["Junior"] * 2
                + ["Senior"] * 2
                + ["Junior"] * 18
            ),
        }
    )

    result = categorical_association(
        dataframe,
        "department",
        "level",
    )

    assert result["method"] == "chi_square_cramers_v"

    assert result["n"] == 40

    assert result["p_value"] < 0.05

    assert 0 <= result["cramers_v"] <= 1

    assert "contingency_table" in result


# ============================================================
# Binary ↔ Binary
# ============================================================

def test_binary_association():
    """
    Binary association should return chi-square
    and the Phi effect size.
    """

    dataframe = pd.DataFrame(
        {
            "treatment": (
                [True] * 20
                + [False] * 20
            ),
            "success": (
                [True] * 18
                + [False] * 2
                + [True] * 2
                + [False] * 18
            ),
        }
    )

    result = binary_association(
        dataframe,
        "treatment",
        "success",
    )

    assert result["method"] == "chi_square_phi"

    assert result["p_value"] < 0.05

    assert 0 <= result["phi"] <= 1


def test_binary_association_rejects_non_binary():
    """
    Both variables must contain exactly two categories.
    """

    dataframe = pd.DataFrame(
        {
            "x": ["A", "B", "C", "A", "B", "C"],
            "y": [True, False, True, False, True, False],
        }
    )

    with pytest.raises(
        ValueError,
        match="exactly two groups",
    ):
        binary_association(
            dataframe,
            "x",
            "y",
        )


# ============================================================
# Kruskal-Wallis
# ============================================================

def test_kruskal_wallis_three_groups():
    """
    Kruskal-Wallis should detect clearly different
    numeric distributions across three groups.
    """

    dataframe = pd.DataFrame(
        {
            "score": [
                1, 2, 3, 4, 5,
                20, 21, 22, 23, 24,
                40, 41, 42, 43, 44,
            ],
            "group": (
                ["A"] * 5
                + ["B"] * 5
                + ["C"] * 5
            ),
        }
    )

    result = kruskal_wallis_test(
        dataframe,
        "score",
        "group",
    )

    assert result["method"] == "kruskal_wallis"

    assert result["number_of_groups"] == 3
    assert result["n"] == 15

    assert result["p_value"] < 0.05

    assert "A" in result["groups"]
    assert "B" in result["groups"]
    assert "C" in result["groups"]


def test_kruskal_rejects_two_groups():
    """
    Our design uses Mann-Whitney when there are only
    two groups.
    """

    dataframe = pd.DataFrame(
        {
            "score": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least three groups",
    ):
        kruskal_wallis_test(
            dataframe,
            "score",
            "group",
        )


# ============================================================
# Post-hoc testing
# ============================================================

def test_mann_whitney_posthoc():
    """
    Pairwise post-hoc testing should produce one comparison
    for every pair of groups and apply Holm correction.
    """

    dataframe = pd.DataFrame(
        {
            "score": [
                1, 2, 3, 4, 5,
                20, 21, 22, 23, 24,
                40, 41, 42, 43, 44,
            ],
            "group": (
                ["A"] * 5
                + ["B"] * 5
                + ["C"] * 5
            ),
        }
    )

    result = mann_whitney_posthoc(
        dataframe,
        "score",
        "group",
    )

    assert result["method"] == "pairwise_mann_whitney"
    assert result["correction"] == "holm"

    # Three groups create:
    # A-B, A-C and B-C
    assert len(result["comparisons"]) == 3

    for comparison in result["comparisons"]:

        assert "p_value" in comparison
        assert "adjusted_p_value" in comparison
        assert "significant" in comparison
        assert "rank_biserial" in comparison


def test_kruskal_with_posthoc():
    """
    Significant Kruskal-Wallis results should trigger
    pairwise post-hoc comparisons.
    """

    dataframe = pd.DataFrame(
        {
            "score": [
                1, 2, 3, 4, 5,
                20, 21, 22, 23, 24,
                40, 41, 42, 43, 44,
            ],
            "group": (
                ["A"] * 5
                + ["B"] * 5
                + ["C"] * 5
            ),
        }
    )

    result = kruskal_wallis_with_posthoc(
        dataframe,
        "score",
        "group",
    )

    assert result["global_test"]["p_value"] < 0.05

    assert result["posthoc"] is not None

    assert len(
        result["posthoc"]["comparisons"]
    ) == 3