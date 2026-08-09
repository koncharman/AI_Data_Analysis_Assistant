from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)


# ============================================================
# Helpers
# ============================================================

def _prepare_text(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Validate and prepare one text column.

    Missing values are removed rather than converted
    into the literal string "nan".
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if column not in dataframe.columns:
        raise ValueError(
            "Column '{}' does not exist.".format(column)
        )

    text = dataframe[column].dropna()

    # Convert valid observations to strings.
    text = text.astype(str)

    # Remove empty strings and strings containing
    # only whitespace.
    text = text[
        text.str.strip() != ""
    ]

    if text.empty:
        raise ValueError(
            "The text column does not contain usable text."
        )

    return text


# ============================================================
# Text cleaning
# ============================================================

def clean_text(
    text: str,
) -> str:
    """
    Perform simple text normalization.

    The cleaning is deliberately conservative because
    vectorizers already perform much of the tokenization.

    Steps:
        - lowercase;
        - remove leading/trailing whitespace;
        - collapse repeated whitespace.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    text = text.lower().strip()

    text = " ".join(
        text.split()
    )

    return text


def clean_text_column(
    dataframe: pd.DataFrame,
    column: str,
) -> List[str]:
    """
    Clean every valid document in a text column.
    """

    text = _prepare_text(
        dataframe,
        column,
    )

    return [
        clean_text(document)
        for document in text
    ]


# ============================================================
# Text statistics
# ============================================================

def get_text_statistics(
    dataframe: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:
    """
    Calculate basic statistics for a text column.
    """

    text = _prepare_text(
        dataframe,
        column,
    )

    word_counts = text.apply(
        lambda value: len(
            value.split()
        )
    )

    character_counts = text.apply(
        len
    )

    return {
        "column": column,
        "document_count": int(
            len(text)
        ),
        "unique_documents": int(
            text.nunique()
        ),
        "duplicate_documents": int(
            text.duplicated().sum()
        ),
        "average_words": float(
            word_counts.mean()
        ),
        "median_words": float(
            word_counts.median()
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
    }


# ============================================================
# Word frequency
# ============================================================

def get_word_frequencies(
    dataframe: pd.DataFrame,
    column: str,
    top_n: int = 20,
    stop_words: Optional[str] = "english",
) -> List[Dict[str, Any]]:
    """
    Return the most frequent words in a text column.

    CountVectorizer is used so tokenization and stop-word
    handling are consistent with the other NLP functions.
    """

    documents = clean_text_column(
        dataframe,
        column,
    )

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 1),
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    words = (
        vectorizer.get_feature_names_out()
    )

    counts = matrix.sum(
        axis=0
    ).A1

    word_counts = list(
        zip(
            words,
            counts,
        )
    )

    word_counts.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "word": word,
            "count": int(count),
        }
        for word, count
        in word_counts[:top_n]
    ]


# ============================================================
# N-grams
# ============================================================

def get_ngrams(
    dataframe: pd.DataFrame,
    column: str,
    n: int = 2,
    top_n: int = 20,
    stop_words: Optional[str] = "english",
) -> List[Dict[str, Any]]:
    """
    Return the most frequent n-grams.

    Examples:
        n=2 -> bigrams
        n=3 -> trigrams
    """

    if n < 1:
        raise ValueError(
            "n must be at least 1."
        )

    documents = clean_text_column(
        dataframe,
        column,
    )

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(n, n),
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    features = (
        vectorizer.get_feature_names_out()
    )

    counts = matrix.sum(
        axis=0
    ).A1

    ngrams = list(
        zip(
            features,
            counts,
        )
    )

    ngrams.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "ngram": ngram,
            "count": int(count),
        }
        for ngram, count
        in ngrams[:top_n]
    ]


# ============================================================
# TF-IDF keywords
# ============================================================

def get_tfidf_keywords(
    dataframe: pd.DataFrame,
    column: str,
    top_n: int = 20,
    stop_words: Optional[str] = "english",
    max_features: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Find important terms across the text corpus using TF-IDF.

    Scores are averaged across documents to obtain
    corpus-level keyword importance.
    """

    documents = clean_text_column(
        dataframe,
        column,
    )

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    features = (
        vectorizer.get_feature_names_out()
    )

    average_scores = (
        matrix.mean(axis=0)
        .A1
    )

    keywords = list(
        zip(
            features,
            average_scores,
        )
    )

    keywords.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "term": term,
            "score": float(score),
        }
        for term, score
        in keywords[:top_n]
    ]


# ============================================================
# Topic modelling - LDA
# ============================================================

def run_topic_modeling(
    dataframe: pd.DataFrame,
    column: str,
    n_topics: int = 5,
    words_per_topic: int = 10,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: Optional[str] = "english",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Discover latent topics using Latent Dirichlet Allocation.

    LDA operates on word counts rather than TF-IDF.

    Returns:
        - important words for each topic;
        - topic weights for each document;
        - dominant topic for each document;
        - fitted vectorizer;
        - fitted LDA model.
    """

    if n_topics < 2:
        raise ValueError(
            "n_topics must be at least 2."
        )

    if words_per_topic < 1:
        raise ValueError(
            "words_per_topic must be at least 1."
        )

    documents = clean_text_column(
        dataframe,
        column,
    )

    if len(documents) < n_topics:
        raise ValueError(
            "The number of topics cannot exceed "
            "the number of documents."
        )

    # --------------------------------------------------------
    # Create document-term matrix
    # --------------------------------------------------------

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    try:
        document_term_matrix = (
            vectorizer.fit_transform(
                documents
            )
        )

    except ValueError as error:
        raise ValueError(
            "Topic modelling could not create a usable "
            "vocabulary. The dataset may be too small "
            "or the filtering settings may be too strict."
        ) from error

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    if len(feature_names) == 0:
        raise ValueError(
            "No usable terms remain for topic modelling."
        )

    # --------------------------------------------------------
    # Fit LDA
    # --------------------------------------------------------

    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
    )

    document_topics = model.fit_transform(
        document_term_matrix
    )

    # --------------------------------------------------------
    # Extract important words from each topic
    # --------------------------------------------------------

    topics = []

    for topic_index, topic_weights in enumerate(
        model.components_
    ):

        # Sort terms from highest to lowest weight.
        top_indices = (
            topic_weights
            .argsort()[::-1][
                :words_per_topic
            ]
        )

        topic_words = [
            {
                "word": str(
                    feature_names[index]
                ),
                "weight": float(
                    topic_weights[index]
                ),
            }
            for index in top_indices
        ]

        topics.append(
            {
                "topic": int(
                    topic_index
                ),
                "words": topic_words,
            }
        )

    # --------------------------------------------------------
    # Dominant topic for each document
    # --------------------------------------------------------

    dominant_topics = (
        document_topics.argmax(
            axis=1
        )
    )

    document_results = []

    for index, (
        dominant_topic,
        probabilities,
    ) in enumerate(
        zip(
            dominant_topics,
            document_topics,
        )
    ):

        document_results.append(
            {
                "document_index": int(index),
                "dominant_topic": int(
                    dominant_topic
                ),
                "topic_probability": float(
                    probabilities[
                        dominant_topic
                    ]
                ),
                "topic_probabilities": [
                    float(value)
                    for value
                    in probabilities
                ],
            }
        )

    return {
        "method": "lda",
        "column": column,
        "document_count": len(
            documents
        ),
        "n_topics": int(
            n_topics
        ),
        "vocabulary_size": int(
            len(feature_names)
        ),
        "topics": topics,
        "documents": document_results,
        "model": model,
        "vectorizer": vectorizer,
    }

