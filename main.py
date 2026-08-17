import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from data.data_loader import read_dataset
from data.data_profiler import profile_dataset

from analysis.statistics import get_dataset_statistics
from analysis.eda import (
    binary_association,
    categorical_association,
    kruskal_wallis_with_posthoc,
    mann_whitney_test,
    spearman_correlation,
)
from analysis.clustering import (
    find_best_k,
    prepare_clustering_data,
    run_kmeans,
)
from analysis.text_analysis import (
    get_ngrams,
    get_text_statistics,
    get_tfidf_keywords,
    get_word_frequencies,
    run_topic_modeling,
)
from analysis.machine_learning import train_random_forest
from analysis.neural_networks import train_feed_forward_network

from agents.data_analyst_agent import (
    ask_data_analyst,
    create_data_analyst_agent,
)
from agents.tool_context import dataset_context


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.20);
        }

        .app-header {
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            margin-bottom: 1rem;
            background: linear-gradient(
                135deg,
                rgba(76, 110, 245, 0.10),
                rgba(132, 94, 247, 0.06)
            );
        }

        .app-header h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: -0.03em;
        }

        .app-header p {
            margin: 0.35rem 0 0 0;
            opacity: 0.78;
        }

        .section-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 14px;
            margin-bottom: 0.85rem;
        }

        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.65;
            margin-bottom: 0.35rem;
        }

        .muted {
            opacity: 0.72;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 0.9rem;
            border-radius: 12px;
        }

        .status-success {
            padding: 0.75rem 0.9rem;
            border-left: 4px solid #2e7d32;
            background: rgba(46, 125, 50, 0.08);
            border-radius: 8px;
        }

        .status-info {
            padding: 0.75rem 0.9rem;
            border-left: 4px solid #4c6ef5;
            background: rgba(76, 110, 245, 0.08);
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

DEFAULT_STATE: Dict[str, Any] = {
    "dataframe": None,
    "profile": None,
    "statistics_result": None,
    "eda_result": None,
    "clustering_result": None,
    "text_result": None,
    "ml_result": None,
    "nn_result": None,
    "loaded_file_hash": None,
    "loaded_file_name": None,
    "agent_messages": [],
    "agent_trace": None,
}


for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# Helpers
# ============================================================

def reset_analysis_results() -> None:
    """
    Clear all analysis results when a new dataset is loaded.
    """

    for key in [
        "statistics_result",
        "eda_result",
        "clustering_result",
        "text_result",
        "ml_result",
        "nn_result",
    ]:
        st.session_state[key] = None


def calculate_file_hash(uploaded_file: Any) -> str:
    """
    Create a stable hash for the uploaded file contents.
    """

    file_bytes = uploaded_file.getvalue()

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def load_uploaded_dataset(uploaded_file: Any) -> None:
    """
    Load and profile a new file only when its content changes.
    """

    file_hash = calculate_file_hash(
        uploaded_file
    )

    if file_hash == st.session_state.loaded_file_hash:
        return

    dataframe = read_dataset(
        uploaded_file
    )

    profile = profile_dataset(
        dataframe
    )

    st.session_state.dataframe = dataframe
    st.session_state.profile = profile
    st.session_state.loaded_file_hash = file_hash
    st.session_state.loaded_file_name = uploaded_file.name

    dataset_context.set_dataset(
        dataframe,
        profile,
        dataset_id=uploaded_file.name,
    )

    st.session_state.agent_messages = []
    st.session_state.agent_trace = None
    reset_analysis_results()

#
def require_dataset() -> bool:
    """
    Show a friendly message when no dataset is available.
    """

    if st.session_state.dataframe is None:
        st.info(
            "Upload a dataset from the sidebar to begin."
        )

        return False

    return True


def get_profile_list(
    key: str,
) -> List[str]:
    """
    Return one profiled column list safely.
    """

    profile = st.session_state.profile or {}

    return list(
        profile.get(
            key,
            [],
        )
    )


def show_result_dictionary(
    result: Dict[str, Any],
) -> None:
    """
    Display a nested result in a readable expandable block.
    """

    with st.expander(
        "View structured result",
        expanded=False,
    ):
        st.json(
            result,
            expanded=False,
        )


def render_column_type_summary() -> None:
    """
    Render detected semantic column groups.
    """

    profile = st.session_state.profile

    type_groups = [
        (
            "Numeric",
            profile.get(
                "numeric_columns",
                [],
            ),
        ),
        (
            "Categorical",
            profile.get(
                "categorical_columns",
                [],
            ),
        ),
        (
            "Text",
            profile.get(
                "text_columns",
                [],
            ),
        ),
        (
            "Datetime",
            profile.get(
                "datetime_columns",
                [],
            ),
        ),
        (
            "Boolean",
            profile.get(
                "boolean_columns",
                [],
            ),
        ),
        (
            "Other",
            profile.get(
                "other_columns",
                [],
            ),
        ),
    ]

    for label, columns in type_groups:
        with st.expander(
            "{} ({})".format(
                label,
                len(columns),
            ),
            expanded=False,
        ):
            if columns:
                st.write(
                    columns
                )
            else:
                st.caption(
                    "No columns detected."
                )


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <h1>📊 AI Data Analyst</h1>
        <p>
            Upload a dataset, inspect its structure, run statistical analysis,
            explore clusters and text, and train predictive models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "Workspace"
    )

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
            "ods",
            "parquet",
        ],
        help=(
            "Supported formats: CSV, Excel, ODS, "
            "and Parquet."
        ),
    )

    if uploaded_file is not None:

        try:
            with st.spinner(
                "Loading and profiling dataset..."
            ):
                load_uploaded_dataset(
                    uploaded_file
                )

            st.success(
                "Dataset ready"
            )

        except Exception as error:
            st.error(
                "Could not load the dataset."
            )

            st.exception(
                error
            )

    st.divider()

    if st.session_state.dataframe is not None:

        dataframe = st.session_state.dataframe

        st.caption(
            "ACTIVE DATASET"
        )

        st.write(
            "**{}**".format(
                st.session_state.loaded_file_name
            )
        )

        col1, col2 = st.columns(
            2
        )

        col1.metric(
            "Rows",
            "{:,}".format(
                dataframe.shape[0]
            ),
        )

        col2.metric(
            "Columns",
            "{:,}".format(
                dataframe.shape[1]
            ),
        )

        if st.button(
            "Clear workspace",
            use_container_width=True,
        ):
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value

            dataset_context.clear()
            st.rerun()

    else:
        st.caption(
            "No active dataset"
        )

    st.divider()

    st.caption(
        "Execution notes"
    )

    st.markdown(
        """
        - Profiling runs once per uploaded file.
        - Heavy analyses run only after you click a button.
        - Results remain available during the current session.
        """
    )


# ============================================================
# Tabs
# ============================================================

(
    overview_tab,
    statistics_tab,
    eda_tab,
    clustering_tab,
    text_tab,
    ml_tab,
    nn_tab,
    agent_tab,
) = st.tabs(
    [
        "Overview",
        "Statistics",
        "EDA",
        "Clustering",
        "Text Analysis",
        "Random Forest",
        "Neural Network",
        "AI Analyst",
    ]
)


# ============================================================
# Overview
# ============================================================

with overview_tab:

    st.subheader(
        "Dataset overview"
    )

    if require_dataset():

        dataframe = st.session_state.dataframe
        profile = st.session_state.profile

        metric_columns = st.columns(
            4
        )

        metric_columns[0].metric(
            "Rows",
            "{:,}".format(
                dataframe.shape[0]
            ),
        )

        metric_columns[1].metric(
            "Columns",
            "{:,}".format(
                dataframe.shape[1]
            ),
        )

        metric_columns[2].metric(
            "Missing cells",
            "{:,}".format(
                int(
                    dataframe.isna()
                    .sum()
                    .sum()
                )
            ),
        )

        metric_columns[3].metric(
            "Duplicate rows",
            "{:,}".format(
                int(
                    dataframe.duplicated()
                    .sum()
                )
            ),
        )

        st.markdown(
            "### Preview"
        )

        preview_rows = st.slider(
            "Rows to display",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key="overview_preview_rows",
        )

        st.dataframe(
            dataframe.head(
                preview_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

        left_column, right_column = st.columns(
            [
                1,
                1.25,
            ]
        )

        with left_column:

            st.markdown(
                "### Detected column types"
            )

            render_column_type_summary()

        with right_column:

            st.markdown(
                "### Column details"
            )

            selected_column = st.selectbox(
                "Inspect column",
                dataframe.columns.tolist(),
                key="overview_column",
            )

            column_profile = (
                profile.get(
                    "columns",
                    {},
                )
                .get(
                    selected_column,
                    {},
                )
            )

            if column_profile:
                st.json(
                    column_profile,
                    expanded=True,
                )

            else:
                st.caption(
                    "No detailed profile is available "
                    "for this column."
                )


# ============================================================
# Statistics
# ============================================================

with statistics_tab:

    st.subheader(
        "Descriptive statistics"
    )

    st.caption(
        "Generate summaries for numeric, categorical, text, "
        "datetime, boolean, and other variables."
    )

    if require_dataset():

        if st.button(
            "Calculate statistics",
            type="primary",
            key="calculate_statistics",
        ):

            try:
                with st.spinner(
                    "Calculating descriptive statistics..."
                ):
                    st.session_state.statistics_result = (
                        get_dataset_statistics(
                            st.session_state.dataframe,
                            st.session_state.profile,
                        )
                    )

            except Exception as error:
                st.error(
                    "Statistics could not be calculated."
                )

                st.exception(
                    error
                )

        result = st.session_state.statistics_result

        if result is not None:

            st.success(
                "Statistics calculated successfully."
            )

            available_sections = [
                section
                for section in [
                    "numeric",
                    "categorical",
                    "text",
                    "datetime",
                    "boolean",
                    "other",
                ]
                if result.get(
                    section
                )
            ]

            if available_sections:

                selected_section = st.selectbox(
                    "Variable type",
                    available_sections,
                    format_func=lambda value:
                        value.replace(
                            "_",
                            " ",
                        ).title(),
                    key="statistics_section",
                )

                section_result = result[
                    selected_section
                ]

                selected_column = st.selectbox(
                    "Column",
                    list(
                        section_result.keys()
                    ),
                    key="statistics_column",
                )

                st.markdown(
                    "### {} statistics".format(
                        selected_column
                    )
                )

                st.json(
                    section_result[
                        selected_column
                    ],
                    expanded=True,
                )

            show_result_dictionary(
                result
            )


# ============================================================
# EDA
# ============================================================

with eda_tab:

    st.subheader(
        "Exploratory relationships"
    )

    st.caption(
        "Run type-aware statistical analyses while removing "
        "missing values pairwise."
    )

    if require_dataset():

        dataframe = st.session_state.dataframe

        numeric_columns = get_profile_list(
            "numeric_columns"
        )

        categorical_columns = list(
            dict.fromkeys(
                get_profile_list(
                    "categorical_columns"
                )
                + get_profile_list(
                    "boolean_columns"
                )
            )
        )

        analysis_type = st.selectbox(
            "Analysis",
            [
                "Numeric ↔ Numeric — Spearman",
                "Numeric ↔ Binary — Mann–Whitney",
                "Numeric ↔ Categorical — Kruskal–Wallis",
                "Categorical ↔ Categorical — Chi-square / Cramér's V",
                "Binary ↔ Binary — Chi-square / Phi",
            ],
            key="eda_analysis_type",
        )

        eda_parameters: Dict[str, Any] = {}

        if analysis_type == "Numeric ↔ Numeric — Spearman":

            if len(numeric_columns) < 2:
                st.warning(
                    "At least two numeric columns are required."
                )

            else:
                col1, col2 = st.columns(
                    2
                )

                eda_parameters[
                    "column_1"
                ] = col1.selectbox(
                    "First numeric column",
                    numeric_columns,
                    key="eda_spearman_1",
                )

                second_options = [
                    column
                    for column in numeric_columns
                    if column
                    != eda_parameters[
                        "column_1"
                    ]
                ]

                eda_parameters[
                    "column_2"
                ] = col2.selectbox(
                    "Second numeric column",
                    second_options,
                    key="eda_spearman_2",
                )

        elif analysis_type == "Numeric ↔ Binary — Mann–Whitney":

            binary_columns = [
                column
                for column in categorical_columns
                if dataframe[
                    column
                ].dropna().nunique() == 2
            ]

            if (
                not numeric_columns
                or not binary_columns
            ):
                st.warning(
                    "A numeric column and a binary column "
                    "are required."
                )

            else:
                col1, col2 = st.columns(
                    2
                )

                eda_parameters[
                    "numeric_column"
                ] = col1.selectbox(
                    "Numeric column",
                    numeric_columns,
                    key="eda_mw_numeric",
                )

                eda_parameters[
                    "group_column"
                ] = col2.selectbox(
                    "Binary group",
                    binary_columns,
                    key="eda_mw_group",
                )

        elif analysis_type == "Numeric ↔ Categorical — Kruskal–Wallis":

            multi_group_columns = [
                column
                for column in categorical_columns
                if dataframe[
                    column
                ].dropna().nunique() >= 3
            ]

            if (
                not numeric_columns
                or not multi_group_columns
            ):
                st.warning(
                    "A numeric column and a categorical column "
                    "with at least three groups are required."
                )

            else:
                col1, col2 = st.columns(
                    2
                )

                eda_parameters[
                    "numeric_column"
                ] = col1.selectbox(
                    "Numeric column",
                    numeric_columns,
                    key="eda_kw_numeric",
                )

                eda_parameters[
                    "group_column"
                ] = col2.selectbox(
                    "Categorical group",
                    multi_group_columns,
                    key="eda_kw_group",
                )

        elif analysis_type == "Categorical ↔ Categorical — Chi-square / Cramér's V":

            if len(categorical_columns) < 2:
                st.warning(
                    "At least two categorical columns are required."
                )

            else:
                col1, col2 = st.columns(
                    2
                )

                eda_parameters[
                    "column_1"
                ] = col1.selectbox(
                    "First categorical column",
                    categorical_columns,
                    key="eda_cat_1",
                )

                second_options = [
                    column
                    for column in categorical_columns
                    if column
                    != eda_parameters[
                        "column_1"
                    ]
                ]

                eda_parameters[
                    "column_2"
                ] = col2.selectbox(
                    "Second categorical column",
                    second_options,
                    key="eda_cat_2",
                )

        else:

            binary_columns = [
                column
                for column in categorical_columns
                if dataframe[
                    column
                ].dropna().nunique() == 2
            ]

            if len(binary_columns) < 2:
                st.warning(
                    "At least two binary columns are required."
                )

            else:
                col1, col2 = st.columns(
                    2
                )

                eda_parameters[
                    "column_1"
                ] = col1.selectbox(
                    "First binary column",
                    binary_columns,
                    key="eda_bin_1",
                )

                second_options = [
                    column
                    for column in binary_columns
                    if column
                    != eda_parameters[
                        "column_1"
                    ]
                ]

                eda_parameters[
                    "column_2"
                ] = col2.selectbox(
                    "Second binary column",
                    second_options,
                    key="eda_bin_2",
                )

        if st.button(
            "Run EDA analysis",
            type="primary",
            key="run_eda",
        ):

            try:
                with st.spinner(
                    "Running statistical analysis..."
                ):

                    if analysis_type == "Numeric ↔ Numeric — Spearman":
                        result = spearman_correlation(
                            dataframe,
                            eda_parameters[
                                "column_1"
                            ],
                            eda_parameters[
                                "column_2"
                            ],
                        )

                    elif analysis_type == "Numeric ↔ Binary — Mann–Whitney":
                        result = mann_whitney_test(
                            dataframe,
                            eda_parameters[
                                "numeric_column"
                            ],
                            eda_parameters[
                                "group_column"
                            ],
                        )

                    elif analysis_type == "Numeric ↔ Categorical — Kruskal–Wallis":
                        result = kruskal_wallis_with_posthoc(
                            dataframe,
                            eda_parameters[
                                "numeric_column"
                            ],
                            eda_parameters[
                                "group_column"
                            ],
                        )

                    elif analysis_type == "Categorical ↔ Categorical — Chi-square / Cramér's V":
                        result = categorical_association(
                            dataframe,
                            eda_parameters[
                                "column_1"
                            ],
                            eda_parameters[
                                "column_2"
                            ],
                        )

                    else:
                        result = binary_association(
                            dataframe,
                            eda_parameters[
                                "column_1"
                            ],
                            eda_parameters[
                                "column_2"
                            ],
                        )

                    st.session_state.eda_result = result

            except Exception as error:
                st.error(
                    "The EDA analysis could not be completed."
                )

                st.exception(
                    error
                )

        if st.session_state.eda_result is not None:

            st.markdown(
                "### Result"
            )

            st.json(
                st.session_state.eda_result,
                expanded=True,
            )


# ============================================================
# Clustering
# ============================================================

with clustering_tab:

    st.subheader(
        "K-Means clustering"
    )

    st.caption(
        "Prepare numeric and categorical features, evaluate "
        "candidate K values, and fit the selected solution."
    )

    if require_dataset():

        dataframe = st.session_state.dataframe

        numeric_columns = get_profile_list(
            "numeric_columns"
        )

        categorical_columns = list(
            dict.fromkeys(
                get_profile_list(
                    "categorical_columns"
                )
                + get_profile_list(
                    "boolean_columns"
                )
            )
        )

        selected_numeric = st.multiselect(
            "Numeric features",
            numeric_columns,
            default=numeric_columns[:3],
            key="cluster_numeric",
        )

        selected_categorical = st.multiselect(
            "Categorical features",
            categorical_columns,
            default=[],
            key="cluster_categorical",
        )

        col1, col2, col3 = st.columns(
            3
        )

        scaling = col1.selectbox(
            "Numeric scaling",
            [
                "standard",
                "robust",
                "minmax",
                "none",
            ],
            key="cluster_scaling",
        )

        min_k = col2.number_input(
            "Minimum K",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="cluster_min_k",
        )

        max_k = col3.number_input(
            "Maximum K",
            min_value=2,
            max_value=30,
            value=8,
            step=1,
            key="cluster_max_k",
        )

        if st.button(
            "Find best K and run clustering",
            type="primary",
            key="run_clustering",
        ):

            try:
                if (
                    not selected_numeric
                    and not selected_categorical
                ):
                    raise ValueError(
                        "Select at least one feature."
                    )

                with st.spinner(
                    "Preparing features and evaluating K..."
                ):

                    prepared = prepare_clustering_data(
                        dataframe=dataframe,
                        numeric_columns=selected_numeric,
                        categorical_columns=(
                            selected_categorical
                        ),
                        scaling=scaling,
                    )

                    best_k_result = find_best_k(
                        data=prepared[
                            "data"
                        ],
                        min_k=int(
                            min_k
                        ),
                        max_k=int(
                            max_k
                        ),
                    )

                    final_result = run_kmeans(
                        data=prepared[
                            "data"
                        ],
                        n_clusters=best_k_result[
                            "best_k"
                        ],
                    )

                    st.session_state.clustering_result = {
                        "preparation": {
                            "feature_names":
                                prepared[
                                    "feature_names"
                                ],
                            "numeric_columns":
                                prepared[
                                    "numeric_columns"
                                ],
                            "categorical_columns":
                                prepared[
                                    "categorical_columns"
                                ],
                            "removed_columns":
                                prepared[
                                    "removed_columns"
                                ],
                            "scaling":
                                prepared[
                                    "scaling"
                                ],
                        },
                        "best_k_analysis":
                            best_k_result,
                        "final_model":
                            final_result,
                    }

            except Exception as error:
                st.error(
                    "Clustering could not be completed."
                )

                st.exception(
                    error
                )

        result = st.session_state.clustering_result

        if result is not None:

            best_k = result[
                "best_k_analysis"
            ]["best_k"]

            metrics = result[
                "final_model"
            ]["metrics"]

            metric_columns = st.columns(
                4
            )

            metric_columns[0].metric(
                "Best K",
                best_k,
            )

            metric_columns[1].metric(
                "Silhouette",
                "{:.3f}".format(
                    metrics[
                        "silhouette_score"
                    ]
                ),
            )

            metric_columns[2].metric(
                "Calinski–Harabasz",
                "{:.2f}".format(
                    metrics[
                        "calinski_harabasz_score"
                    ]
                ),
            )

            metric_columns[3].metric(
                "Davies–Bouldin",
                "{:.3f}".format(
                    metrics[
                        "davies_bouldin_score"
                    ]
                ),
            )

            evaluation_table = pd.DataFrame(
                result[
                    "best_k_analysis"
                ]["results"]
            )

            st.markdown(
                "### Candidate K values"
            )

            st.dataframe(
                evaluation_table,
                use_container_width=True,
                hide_index=True,
            )

            labels = result[
                "final_model"
            ]["labels"]

            label_counts = (
                pd.Series(
                    labels,
                    name="cluster",
                )
                .value_counts()
                .sort_index()
                .rename_axis(
                    "cluster"
                )
                .reset_index(
                    name="row_count"
                )
            )

            st.markdown(
                "### Cluster sizes"
            )

            st.dataframe(
                label_counts,
                use_container_width=True,
                hide_index=True,
            )

            show_result_dictionary(
                result
            )


# ============================================================
# Text analysis
# ============================================================

with text_tab:

    st.subheader(
        "Text and topic analysis"
    )

    st.caption(
        "Inspect corpus statistics, common terms, n-grams, "
        "TF-IDF keywords, and LDA topics."
    )

    if require_dataset():

        text_columns = get_profile_list(
            "text_columns"
        )

        if not text_columns:
            st.warning(
                "No text columns were detected by the profiler."
            )

        else:
            text_column = st.selectbox(
                "Text column",
                text_columns,
                key="text_column",
            )

            analysis_type = st.selectbox(
                "Analysis",
                [
                    "Text statistics",
                    "Word frequencies",
                    "N-grams",
                    "TF-IDF keywords",
                    "LDA topic modeling",
                ],
                key="text_analysis_type",
            )

            top_n = 20
            ngram_n = 2
            n_topics = 5
            words_per_topic = 10
            min_df = 2

            if analysis_type in {
                "Word frequencies",
                "N-grams",
                "TF-IDF keywords",
            }:
                top_n = st.slider(
                    "Number of results",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    key="text_top_n",
                )

            if analysis_type == "N-grams":
                ngram_n = st.selectbox(
                    "N-gram size",
                    [
                        1,
                        2,
                        3,
                    ],
                    index=1,
                    key="text_ngram_n",
                )

            if analysis_type == "LDA topic modeling":

                col1, col2, col3 = st.columns(
                    3
                )

                n_topics = col1.number_input(
                    "Topics",
                    min_value=2,
                    max_value=30,
                    value=5,
                    step=1,
                    key="text_topics",
                )

                words_per_topic = col2.number_input(
                    "Words per topic",
                    min_value=3,
                    max_value=30,
                    value=10,
                    step=1,
                    key="text_words_per_topic",
                )

                min_df = col3.number_input(
                    "Minimum document frequency",
                    min_value=1,
                    max_value=20,
                    value=2,
                    step=1,
                    key="text_min_df",
                )

            if st.button(
                "Run text analysis",
                type="primary",
                key="run_text_analysis",
            ):

                try:
                    dataframe = (
                        st.session_state.dataframe
                    )

                    with st.spinner(
                        "Analyzing text..."
                    ):

                        if analysis_type == "Text statistics":
                            result = get_text_statistics(
                                dataframe,
                                text_column,
                            )

                        elif analysis_type == "Word frequencies":
                            result = get_word_frequencies(
                                dataframe,
                                text_column,
                                top_n=int(
                                    top_n
                                ),
                            )

                        elif analysis_type == "N-grams":
                            result = get_ngrams(
                                dataframe,
                                text_column,
                                n=int(
                                    ngram_n
                                ),
                                top_n=int(
                                    top_n
                                ),
                            )

                        elif analysis_type == "TF-IDF keywords":
                            result = get_tfidf_keywords(
                                dataframe,
                                text_column,
                                top_n=int(
                                    top_n
                                ),
                            )

                        else:
                            result = run_topic_modeling(
                                dataframe,
                                text_column,
                                n_topics=int(
                                    n_topics
                                ),
                                words_per_topic=int(
                                    words_per_topic
                                ),
                                min_df=int(
                                    min_df
                                ),
                            )

                        st.session_state.text_result = {
                            "analysis_type":
                                analysis_type,
                            "result":
                                result,
                        }

                except Exception as error:
                    st.error(
                        "Text analysis could not be completed."
                    )

                    st.exception(
                        error
                    )

            stored_result = (
                st.session_state.text_result
            )

            if stored_result is not None:

                result = stored_result[
                    "result"
                ]

                st.markdown(
                    "### Result"
                )

                if isinstance(
                    result,
                    list,
                ):
                    st.dataframe(
                        pd.DataFrame(
                            result
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                elif (
                    isinstance(
                        result,
                        dict,
                    )
                    and "topics" in result
                ):
                    for topic in result[
                        "topics"
                    ]:
                        st.markdown(
                            "**Topic {}**".format(
                                topic[
                                    "topic"
                                ]
                            )
                        )

                        st.write(
                            ", ".join(
                                word[
                                    "word"
                                ]
                                for word in topic[
                                    "words"
                                ]
                            )
                        )

                    show_result_dictionary(
                        result
                    )

                else:
                    st.json(
                        result,
                        expanded=True,
                    )


# ============================================================
# Random Forest
# ============================================================

with ml_tab:

    st.subheader(
        "Random Forest"
    )

    st.caption(
        "Train a classifier or regressor using cross-validation, "
        "automatic preprocessing, imbalance handling, and "
        "feature importance."
    )

    if require_dataset():

        dataframe = st.session_state.dataframe
        profile = st.session_state.profile

        supported_features = list(
            dict.fromkeys(
                get_profile_list(
                    "numeric_columns"
                )
                + get_profile_list(
                    "categorical_columns"
                )
                + get_profile_list(
                    "boolean_columns"
                )
            )
        )

        target_column = st.selectbox(
            "Target column",
            dataframe.columns.tolist(),
            key="rf_target",
        )

        available_features = [
            column
            for column in supported_features
            if column != target_column
        ]

        selected_features = st.multiselect(
            "Features",
            available_features,
            default=available_features,
            key="rf_features",
        )

        col1, col2, col3 = st.columns(
            3
        )

        task_type = col1.selectbox(
            "Task type",
            [
                "auto",
                "classification",
                "regression",
            ],
            key="rf_task",
        )

        cv_folds = col2.number_input(
            "Cross-validation folds",
            min_value=2,
            max_value=20,
            value=10,
            step=1,
            key="rf_cv",
        )

        n_estimators = col3.number_input(
            "Trees",
            min_value=10,
            max_value=2000,
            value=300,
            step=50,
            key="rf_trees",
        )

        handle_imbalance = st.checkbox(
            "Handle classification imbalance",
            value=True,
            key="rf_balance",
        )

        if st.button(
            "Train Random Forest",
            type="primary",
            key="train_rf",
        ):

            try:
                if not selected_features:
                    raise ValueError(
                        "Select at least one feature."
                    )

                resolved_task = (
                    None
                    if task_type == "auto"
                    else task_type
                )

                with st.spinner(
                    "Training and cross-validating Random Forest..."
                ):
                    st.session_state.ml_result = (
                        train_random_forest(
                            dataframe=dataframe,
                            profile=profile,
                            target_column=target_column,
                            feature_columns=selected_features,
                            task_type=resolved_task,
                            cv_folds=int(
                                cv_folds
                            ),
                            handle_class_imbalance=(
                                handle_imbalance
                            ),
                            n_estimators=int(
                                n_estimators
                            ),
                        )
                    )

            except Exception as error:
                st.error(
                    "Random Forest training failed."
                )

                st.exception(
                    error
                )

        result = st.session_state.ml_result

        if result is not None:

            metrics = result[
                "cross_validation"
            ]["metrics"]

            metric_columns = st.columns(
                4
            )

            metric_columns[0].metric(
                "Task",
                result[
                    "task_type"
                ].title(),
            )

            metric_columns[1].metric(
                "Primary metric",
                result[
                    "selection_metric"
                ],
            )

            metric_columns[2].metric(
                "Primary score",
                "{:.4f}".format(
                    result[
                        "primary_score"
                    ]
                ),
            )

            metric_columns[3].metric(
                "CV folds",
                result[
                    "actual_cv_folds"
                ],
            )

            if result.get(
                "cv_warning"
            ):
                st.warning(
                    result[
                        "cv_warning"
                    ]
                )

            st.markdown(
                "### Cross-validation metrics"
            )

            metric_table = pd.DataFrame(
                [
                    {
                        "metric": metric_name,
                        "mean": metric_values[
                            "mean"
                        ],
                        "standard_deviation":
                            metric_values[
                                "standard_deviation"
                            ],
                        "minimum":
                            metric_values[
                                "minimum"
                            ],
                        "maximum":
                            metric_values[
                                "maximum"
                            ],
                    }
                    for metric_name, metric_values
                    in metrics.items()
                ]
            )

            st.dataframe(
                metric_table,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "### Feature importance"
            )

            st.dataframe(
                pd.DataFrame(
                    result[
                        "feature_importance"
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# Neural network
# ============================================================

with nn_tab:

    st.subheader(
        "Feed-forward neural network"
    )

    st.caption(
        "Run K-fold cross-validation with internal validation "
        "for early stopping, then fit a final PyTorch model."
    )

    if require_dataset():

        dataframe = st.session_state.dataframe
        profile = st.session_state.profile

        supported_features = list(
            dict.fromkeys(
                get_profile_list(
                    "numeric_columns"
                )
                + get_profile_list(
                    "categorical_columns"
                )
                + get_profile_list(
                    "boolean_columns"
                )
            )
        )

        target_column = st.selectbox(
            "Target column",
            dataframe.columns.tolist(),
            key="nn_target",
        )

        available_features = [
            column
            for column in supported_features
            if column != target_column
        ]

        selected_features = st.multiselect(
            "Features",
            available_features,
            default=available_features,
            key="nn_features",
        )

        top_row = st.columns(
            4
        )

        task_type = top_row[0].selectbox(
            "Task type",
            [
                "auto",
                "classification",
                "regression",
            ],
            key="nn_task",
        )

        cv_folds = top_row[1].number_input(
            "CV folds",
            min_value=2,
            max_value=20,
            value=10,
            step=1,
            key="nn_cv",
        )

        max_epochs = top_row[2].number_input(
            "Maximum epochs",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            key="nn_epochs",
        )

        patience = top_row[3].number_input(
            "Early-stopping patience",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="nn_patience",
        )

        hidden_layers_text = st.text_input(
            "Hidden layers",
            value="64,32",
            help=(
                "Comma-separated layer sizes. Leave empty "
                "for direct input-to-output weights."
            ),
            key="nn_hidden_layers",
        )

        hidden_layers: List[int] = []

        if hidden_layers_text.strip():
            try:
                hidden_layers = [
                    int(
                        value.strip()
                    )
                    for value
                    in hidden_layers_text.split(
                        ","
                    )
                    if value.strip()
                ]
            except ValueError:
                st.error(
                    "Hidden layers must contain only integers."
                )

        option_row = st.columns(
            4
        )

        activation = option_row[0].selectbox(
            "Activation",
            [
                "relu",
                "leaky_relu",
                "elu",
                "gelu",
                "selu",
                "tanh",
                "sigmoid",
                "identity",
            ],
            key="nn_activation",
        )

        optimizer = option_row[1].selectbox(
            "Optimizer",
            [
                "adam",
                "adamw",
                "sgd",
                "rmsprop",
            ],
            key="nn_optimizer",
        )

        learning_rate = option_row[2].number_input(
            "Learning rate",
            min_value=0.000001,
            max_value=1.0,
            value=0.001,
            format="%.6f",
            key="nn_lr",
        )

        batch_size = option_row[3].selectbox(
            "Batch size",
            [
                8,
                16,
                32,
                64,
                128,
                256,
            ],
            index=2,
            key="nn_batch",
        )

        advanced_column_1, advanced_column_2 = st.columns(
            2
        )

        dropout = advanced_column_1.slider(
            "Dropout",
            min_value=0.0,
            max_value=0.8,
            value=0.1,
            step=0.05,
            key="nn_dropout",
        )

        validation_fraction = advanced_column_2.slider(
            "Internal validation fraction",
            min_value=0.05,
            max_value=0.40,
            value=0.15,
            step=0.05,
            key="nn_validation",
        )

        use_batch_norm = st.checkbox(
            "Use batch normalization",
            value=False,
            key="nn_batch_norm",
        )

        handle_imbalance = st.checkbox(
            "Handle classification imbalance",
            value=True,
            key="nn_balance",
        )

        st.info(
            "Neural-network cross-validation can be significantly "
            "slower than Random Forest because the network is "
            "trained once per fold and then fitted again on all data."
        )

        if st.button(
            "Train neural network",
            type="primary",
            key="train_nn",
        ):

            try:
                if not selected_features:
                    raise ValueError(
                        "Select at least one feature."
                    )

                resolved_task = (
                    None
                    if task_type == "auto"
                    else task_type
                )

                with st.spinner(
                    "Training neural network across folds..."
                ):
                    st.session_state.nn_result = (
                        train_feed_forward_network(
                            dataframe=dataframe,
                            profile=profile,
                            target_column=target_column,
                            feature_columns=selected_features,
                            task_type=resolved_task,
                            hidden_layers=hidden_layers,
                            activations=activation,
                            dropout=float(
                                dropout
                            ),
                            use_batch_norm=(
                                use_batch_norm
                            ),
                            optimizer_name=optimizer,
                            learning_rate=float(
                                learning_rate
                            ),
                            batch_size=int(
                                batch_size
                            ),
                            max_epochs=int(
                                max_epochs
                            ),
                            patience=int(
                                patience
                            ),
                            cv_folds=int(
                                cv_folds
                            ),
                            validation_fraction=float(
                                validation_fraction
                            ),
                            handle_class_imbalance=(
                                handle_imbalance
                            ),
                        )
                    )

            except Exception as error:
                st.error(
                    "Neural-network training failed."
                )

                st.exception(
                    error
                )

        result = st.session_state.nn_result

        if result is not None:

            metric_columns = st.columns(
                5
            )

            metric_columns[0].metric(
                "Task",
                result[
                    "task_type"
                ].title(),
            )

            metric_columns[1].metric(
                "Primary metric",
                result[
                    "primary_metric"
                ],
            )

            metric_columns[2].metric(
                "CV score",
                "{:.4f}".format(
                    result[
                        "primary_score"
                    ]
                ),
            )

            metric_columns[3].metric(
                "CV folds",
                result[
                    "actual_cv_folds"
                ],
            )

            metric_columns[4].metric(
                "Final epochs",
                result[
                    "final_epochs"
                ],
            )

            if result.get(
                "cv_warning"
            ):
                st.warning(
                    result[
                        "cv_warning"
                    ]
                )

            st.markdown(
                "### Cross-validation metrics"
            )

            cv_metrics = result[
                "cross_validation"
            ]["metrics"]

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "metric": name,
                            "mean": values[
                                "mean"
                            ],
                            "standard_deviation":
                                values[
                                    "standard_deviation"
                                ],
                            "minimum":
                                values[
                                    "minimum"
                                ],
                            "maximum":
                                values[
                                    "maximum"
                                ],
                        }
                        for name, values
                        in cv_metrics.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "### Architecture"
            )

            st.json(
                result[
                    "architecture"
                ],
                expanded=True,
            )

            direct_weights = result[
                "weights"
            ][
                "direct_input_output_weights"
            ]

            if direct_weights:

                st.markdown(
                    "### Direct feature weights"
                )

                weight_result = result[
                    "weights"
                ][
                    "direct_weights"
                ]

                if (
                    isinstance(
                        weight_result,
                        dict,
                    )
                    and "weights"
                    in weight_result
                ):
                    st.dataframe(
                        pd.DataFrame(
                            weight_result[
                                "weights"
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    selected_output = st.selectbox(
                        "Output class",
                        list(
                            weight_result.keys()
                        ),
                        key="nn_weight_output",
                    )

                    st.dataframe(
                        pd.DataFrame(
                            weight_result[
                                selected_output
                            ]["weights"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            else:
                st.info(
                    "Hidden layers are present, so individual "
                    "input weights are not direct feature-to-output "
                    "effects. Layer matrices remain available in "
                    "the structured result."
                )

            show_result_dictionary(
                result[
                    "weights"
                ]
            )


# ============================================================
# AI Analyst agent
# ============================================================

@st.cache_resource(show_spinner=False)
def load_data_analyst_agent(
    model_name: str,
):
    """Create one cached LangChain 0.3 AgentExecutor per model."""
    return create_data_analyst_agent(
        model_name=model_name,
    )


with agent_tab:

    st.subheader("AI Data Analyst")
    st.caption(
        "Ask natural-language questions about the active dataset. "
        "The agent selects deterministic Python tools and explains their results."
    )

    if require_dataset():

        # Refresh the Python-side dataset context on every Streamlit rerun.
        dataset_context.set_dataset(
            st.session_state.dataframe,
            st.session_state.profile,
            dataset_id=st.session_state.loaded_file_name,
        )

        settings_col, examples_col = st.columns([1, 1.6])

        with settings_col:
            st.markdown("### Agent settings")

            agent_model_name = st.text_input(
                "Ollama model",
                value="llama3.2:3b",
                help="Use a locally installed Ollama model that supports tool calling.",
                key="agent_model_name",
            )

            st.caption(
                "The LangChain 0.3 AgentExecutor manages tool calls internally. "
                "Detailed steps remain visible in the terminal while verbose=True."
            )

            if st.button("Clear conversation", use_container_width=True):
                st.session_state.agent_messages = []
                st.session_state.agent_trace = None
                st.rerun()

        with examples_col:
            st.markdown("### Example questions")
            st.markdown(
                """
                - Which columns have the most missing values?
                - Is age associated with the final grade?
                - Do grades differ between scholarship groups?
                - Segment the observations using the main numeric variables.
                - Train a Random Forest to predict the target and explain importance.
                - Find topics in the review-text column.
                """
            )

        st.divider()

        for message in st.session_state.agent_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask a question about the active dataset...",
            key="agent_question",
        )

        if question:
            prior_conversation = list(st.session_state.agent_messages)

            st.session_state.agent_messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Selecting and running analysis tools..."):
                        agent = load_data_analyst_agent(
                            agent_model_name,
                        )

                        # Convert the Streamlit conversation into LangChain
                        # message objects expected by the 0.3 AgentExecutor.
                        from langchain_core.messages import (
                            AIMessage,
                            HumanMessage,
                        )

                        chat_history = []

                        for item in prior_conversation:
                            role = item.get("role")
                            content = item.get("content", "")

                            if role == "user" and content:
                                chat_history.append(
                                    HumanMessage(content=content)
                                )

                            elif role == "assistant" and content:
                                chat_history.append(
                                    AIMessage(content=content)
                                )

                        answer = ask_data_analyst(
                            agent=agent,
                            question=question,
                            chat_history=chat_history,
                        )

                    if not answer:
                        answer = (
                            "The agent completed the request but did not return "
                            "a final natural-language answer."
                        )

                    st.markdown(answer)
                    st.session_state.agent_trace = None

                st.session_state.agent_messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "The agent could not complete the request. Confirm that Ollama "
                    "is running, the model is installed, and the agent dependencies "
                    "are available."
                )
                st.exception(error)

