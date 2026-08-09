from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================
# Helpers
# ============================================================

def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Validate that requested columns exist in the DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    for column in columns:
        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )


def _drop_missing_pair(
    dataframe: pd.DataFrame,
    column_1: str,
    column_2: str,
) -> pd.DataFrame:
    """
    Keep only rows where both requested variables are present.

    Missing values are removed pairwise rather than from the
    entire dataset.
    """

    _validate_columns(
        dataframe,
        [column_1, column_2],
    )

    return dataframe[
        [column_1, column_2]
    ].dropna()


def _holm_correction(
    p_values: list[float],
) -> list[float]:
    """
    Apply Holm's multiple-testing correction.

    This avoids requiring statsmodels only for p-value
    adjustment.
    """

    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return []

    order = np.argsort(p_values)

    adjusted = np.zeros(
        number_of_tests,
        dtype=float,
    )

    previous = 0.0

    for rank, index in enumerate(order):

        multiplier = (
            number_of_tests - rank
        )

        corrected = (
            p_values[index]
            * multiplier
        )

        # Holm adjusted p-values must be monotonic.
        corrected = max(
            corrected,
            previous,
        )

        corrected = min(
            corrected,
            1.0,
        )

        adjusted[index] = corrected
        previous = corrected

    return adjusted.tolist()


# ============================================================
# Numeric ↔ Numeric
# ============================================================

def spearman_correlation(
    dataframe: pd.DataFrame,
    column_1: str,
    column_2: str,
) -> dict[str, Any]:
    """
    Calculate Spearman rank correlation.

    Useful for:
    - ordinal variables;
    - numeric monotonic relationships;
    - situations where Pearson's linear assumptions
      are not appropriate.
    """

    data = _drop_missing_pair(
        dataframe,
        column_1,
        column_2,
    )

    x = pd.to_numeric(
        data[column_1],
        errors="coerce",
    )

    y = pd.to_numeric(
        data[column_2],
        errors="coerce",
    )

    valid = x.notna() & y.notna()

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise ValueError(
            "At least two valid observations are required."
        )

    if x.nunique() < 2 or y.nunique() < 2:
        raise ValueError(
            "Spearman correlation requires variation "
            "in both variables."
        )

    result = stats.spearmanr(
        x,
        y,
    )

    return {
        "method": "spearman",
        "column_1": column_1,
        "column_2": column_2,
        "n": int(len(x)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


# ============================================================
# Binary categorical ↔ Numeric
# ============================================================

def mann_whitney_test(
    dataframe: pd.DataFrame,
    numeric_column: str,
    binary_column: str,
) -> dict[str, Any]:
    """
    Compare a numeric variable between two independent groups
    using the Mann-Whitney U test.

    Also calculate rank-biserial correlation as an effect size.
    """

    data = _drop_missing_pair(
        dataframe,
        numeric_column,
        binary_column,
    )

    data[numeric_column] = pd.to_numeric(
        data[numeric_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[numeric_column]
    )

    groups = list(
        data[binary_column].unique()
    )

    if len(groups) != 2:
        raise ValueError(
            "Mann-Whitney requires exactly two groups."
        )

    group_1 = data.loc[
        data[binary_column] == groups[0],
        numeric_column,
    ]

    group_2 = data.loc[
        data[binary_column] == groups[1],
        numeric_column,
    ]

    if group_1.empty or group_2.empty:
        raise ValueError(
            "Both groups must contain observations."
        )

    result = stats.mannwhitneyu(
        group_1,
        group_2,
        alternative="two-sided",
    )

    n1 = len(group_1)
    n2 = len(group_2)

    # Rank-biserial effect size.
    #
    # Its sign depends on group ordering, so group names
    # are returned alongside the value.
    rank_biserial = (
        (2 * result.statistic)
        / (n1 * n2)
        - 1
    )

    return {
        "method": "mann_whitney_u",
        "numeric_column": numeric_column,
        "group_column": binary_column,
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_n": int(n1),
        "group_2_n": int(n2),
        "group_1_median": float(
            group_1.median()
        ),
        "group_2_median": float(
            group_2.median()
        ),
        "statistic": float(
            result.statistic
        ),
        "p_value": float(
            result.pvalue
        ),
        "rank_biserial": float(
            rank_biserial
        ),
    }


# ============================================================
# Categorical ↔ Categorical
# ============================================================

def categorical_association(
    dataframe: pd.DataFrame,
    column_1: str,
    column_2: str,
) -> dict[str, Any]:
    """
    Analyze association between two categorical variables.

    Returns:
    - chi-square test;
    - Cramer's V effect size;
    - contingency table.
    """

    data = _drop_missing_pair(
        dataframe,
        column_1,
        column_2,
    )

    contingency_table = pd.crosstab(
        data[column_1],
        data[column_2],
    )

    if (
        contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):
        raise ValueError(
            "Both categorical variables must contain "
            "at least two observed categories."
        )

    chi2, p_value, dof, expected = (
        stats.chi2_contingency(
            contingency_table
        )
    )

    n = contingency_table.to_numpy().sum()

    rows, columns = contingency_table.shape

    denominator = min(
        rows - 1,
        columns - 1,
    )

    if denominator == 0 or n == 0:
        cramers_v = None

    else:
        cramers_v = np.sqrt(
            (chi2 / n)
            / denominator
        )

    return {
        "method": "chi_square_cramers_v",
        "column_1": column_1,
        "column_2": column_2,
        "n": int(n),
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": (
            float(cramers_v)
            if cramers_v is not None
            else None
        ),
        "minimum_expected_count": float(
            np.min(expected)
        ),
        "contingency_table":
            contingency_table.to_dict(),
    }


# ============================================================
# Binary ↔ Binary
# ============================================================

def binary_association(
    dataframe: pd.DataFrame,
    column_1: str,
    column_2: str,
) -> dict[str, Any]:
    """
    Analyze two binary variables.

    Returns:
    - chi-square test;
    - Phi coefficient;
    - contingency table.

    A warning is returned when expected cell counts are small,
    because Fisher's exact test may then be preferable.
    """

    data = _drop_missing_pair(
        dataframe,
        column_1,
        column_2,
    )

    if data[column_1].nunique() != 2:
        raise ValueError(
            f"'{column_1}' must contain exactly two groups."
        )

    if data[column_2].nunique() != 2:
        raise ValueError(
            f"'{column_2}' must contain exactly two groups."
        )

    table = pd.crosstab(
        data[column_1],
        data[column_2],
    )

    chi2, p_value, dof, expected = (
        stats.chi2_contingency(table)
    )

    n = table.to_numpy().sum()

    phi = np.sqrt(
        chi2 / n
    )

    minimum_expected = float(
        np.min(expected)
    )

    warning = None

    if minimum_expected < 5:
        warning = (
            "Some expected cell counts are below 5. "
            "Consider Fisher's exact test."
        )

    return {
        "method": "chi_square_phi",
        "column_1": column_1,
        "column_2": column_2,
        "n": int(n),
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "phi": float(phi),
        "minimum_expected_count":
            minimum_expected,
        "warning": warning,
        "contingency_table":
            table.to_dict(),
    }


# ============================================================
# Categorical ↔ Numeric
# ============================================================

def kruskal_wallis_test(
    dataframe: pd.DataFrame,
    numeric_column: str,
    categorical_column: str,
) -> dict[str, Any]:
    """
    Compare a numeric variable across three or more
    independent categorical groups using Kruskal-Wallis.
    """

    data = _drop_missing_pair(
        dataframe,
        numeric_column,
        categorical_column,
    )

    data[numeric_column] = pd.to_numeric(
        data[numeric_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[numeric_column]
    )

    group_names = list(
        data[categorical_column].unique()
    )

    if len(group_names) < 3:
        raise ValueError(
            "Kruskal-Wallis requires at least three groups. "
            "Use Mann-Whitney for two groups."
        )

    groups = [
        data.loc[
            data[categorical_column] == group,
            numeric_column,
        ]
        for group in group_names
    ]

    if any(group.empty for group in groups):
        raise ValueError(
            "Every group must contain observations."
        )

    result = stats.kruskal(
        *groups
    )

    group_summary = {}

    for name, group in zip(
        group_names,
        groups,
    ):
        group_summary[str(name)] = {
            "n": int(len(group)),
            "median": float(
                group.median()
            ),
        }

    return {
        "method": "kruskal_wallis",
        "numeric_column": numeric_column,
        "group_column": categorical_column,
        "n": int(len(data)),
        "number_of_groups": int(
            len(groups)
        ),
        "statistic": float(
            result.statistic
        ),
        "p_value": float(
            result.pvalue
        ),
        "groups": group_summary,
    }


# ============================================================
# Mann-Whitney post-hoc
# ============================================================

def mann_whitney_posthoc(
    dataframe: pd.DataFrame,
    numeric_column: str,
    categorical_column: str,
    alpha: float = 0.05,
    correction: str = "holm",
) -> dict[str, Any]:
    """
    Perform pairwise Mann-Whitney U tests between all groups.

    This function is intended to be used as a post-hoc
    analysis after a significant Kruskal-Wallis test.

    Multiple-testing correction is applied using
    statsmodels.stats.multitest.multipletests.

    Missing values are removed only from the two columns
    involved in the analysis.

    Args:
        dataframe:
            Dataset containing the variables.

        numeric_column:
            Numeric variable being compared.

        categorical_column:
            Categorical grouping variable.

        alpha:
            Significance level used for multiple testing.

        correction:
            Multiple-testing correction method.
            Default is "holm".

    Returns:
        Dictionary containing all pairwise comparisons,
        raw p-values, adjusted p-values, effect sizes,
        and significance decisions.
    """

    # --------------------------------------------------------
    # Remove missing observations
    # --------------------------------------------------------

    data = _drop_missing_pair(
        dataframe,
        numeric_column,
        categorical_column,
    )

    # Ensure that the dependent variable is numeric.
    data[numeric_column] = pd.to_numeric(
        data[numeric_column],
        errors="coerce",
    )

    # Values that could not be converted to numeric become
    # NaN and are removed.
    data = data.dropna(
        subset=[numeric_column]
    )

    # --------------------------------------------------------
    # Identify groups
    # --------------------------------------------------------

    groups = list(
        data[categorical_column].unique()
    )

    if len(groups) < 3:
        raise ValueError(
            "Post-hoc testing requires at least three groups."
        )

    # Store individual comparisons and their raw p-values.
    comparisons = []
    raw_p_values = []

    # --------------------------------------------------------
    # Compare every possible pair of groups
    # --------------------------------------------------------

    for group_1, group_2 in combinations(
        groups,
        2,
    ):

        values_1 = data.loc[
            data[categorical_column] == group_1,
            numeric_column,
        ]

        values_2 = data.loc[
            data[categorical_column] == group_2,
            numeric_column,
        ]

        if values_1.empty or values_2.empty:
            continue

        # Perform two-sided Mann-Whitney U test.
        test_result = stats.mannwhitneyu(
            values_1,
            values_2,
            alternative="two-sided",
        )

        n1 = len(values_1)
        n2 = len(values_2)

        # ----------------------------------------------------
        # Rank-biserial correlation
        # ----------------------------------------------------
        #
        # Provides an effect-size measure in addition
        # to the hypothesis-test p-value.
        #
        # The sign depends on the order of the groups.
        # ----------------------------------------------------

        rank_biserial = (
            (2 * test_result.statistic)
            / (n1 * n2)
            - 1
        )

        raw_p_values.append(
            float(test_result.pvalue)
        )

        comparisons.append(
            {
                "group_1": str(group_1),
                "group_2": str(group_2),

                "group_1_n": int(n1),
                "group_2_n": int(n2),

                "group_1_median": float(
                    values_1.median()
                ),
                "group_2_median": float(
                    values_2.median()
                ),

                "statistic": float(
                    test_result.statistic
                ),

                "p_value": float(
                    test_result.pvalue
                ),

                "rank_biserial": float(
                    rank_biserial
                ),
            }
        )

    # --------------------------------------------------------
    # Multiple-testing correction
    # --------------------------------------------------------

    if raw_p_values:

        reject, adjusted_p_values, _, _ = (
            multipletests(
                raw_p_values,
                alpha=alpha,
                method=correction,
            )
        )

        # Add corrected results back to each comparison.
        for (
            comparison,
            adjusted_p,
            significant,
        ) in zip(
            comparisons,
            adjusted_p_values,
            reject,
        ):

            comparison[
                "adjusted_p_value"
            ] = float(adjusted_p)

            comparison[
                "significant"
            ] = bool(significant)

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "method": "pairwise_mann_whitney",
        "correction": correction,
        "alpha": float(alpha),
        "number_of_comparisons": len(
            comparisons
        ),
        "comparisons": comparisons,
    }


# ============================================================
# Kruskal-Wallis + optional post-hoc
# ============================================================

def kruskal_wallis_with_posthoc(
    dataframe: pd.DataFrame,
    numeric_column: str,
    categorical_column: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Run Kruskal-Wallis first.

    Pairwise Mann-Whitney post-hoc comparisons are only
    performed when the global Kruskal-Wallis test is
    statistically significant.
    """

    global_test = kruskal_wallis_test(
        dataframe,
        numeric_column,
        categorical_column,
    )

    result = {
        "global_test": global_test,
        "posthoc": None,
    }

    if global_test["p_value"] < alpha:

        result["posthoc"] = mann_whitney_posthoc(
            dataframe,
            numeric_column,
            categorical_column,
            alpha=alpha,
        )

    return result

