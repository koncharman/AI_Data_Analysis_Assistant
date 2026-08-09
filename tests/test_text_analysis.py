import pandas as pd
import pytest

from analysis.text_analysis import (
    clean_text,
    clean_text_column,
    get_text_statistics,
    get_word_frequencies,
    get_ngrams,
    get_tfidf_keywords,
    run_topic_modeling,
)


# ============================================================
# Test data
# ============================================================

@pytest.fixture
def sample_text_dataframe():
    """
    Create a small text dataset containing two clear topics:
    technology and cooking.

    Missing and empty values are included to test
    text preparation.
    """

    return pd.DataFrame(
        {
            "review": [
                "Python programming is useful for machine learning",
                "Machine learning models analyze data with Python",
                "Artificial intelligence uses data and algorithms",
                "Software developers write Python applications",
                "Data scientists build machine learning models",

                "Cooking pasta requires boiling water and sauce",
                "Fresh pasta tastes delicious with tomato sauce",
                "Cooking recipes use vegetables and fresh ingredients",
                "The kitchen has pasta vegetables and tomato sauce",
                "Fresh ingredients make cooking recipes delicious",

                None,
                "",
                "   ",
            ]
        }
    )


# ============================================================
# Cleaning
# ============================================================

def test_clean_text():
    """
    Text should be lowercased and repeated whitespace
    should be removed.
    """

    result = clean_text(
        "  Machine   Learning   Is Useful  "
    )

    assert result == "machine learning is useful"


def test_clean_text_rejects_non_string():
    """
    clean_text should only accept strings.
    """

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        clean_text(123)


def test_clean_text_column(
    sample_text_dataframe,
):
    """
    Missing and empty documents should be removed.
    """

    result = clean_text_column(
        sample_text_dataframe,
        "review",
    )

    # 10 usable documents remain.
    assert len(result) == 10

    # All returned documents should be cleaned.
    assert all(
        document == document.lower()
        for document in result
    )


# ============================================================
# Text statistics
# ============================================================

def test_text_statistics(
    sample_text_dataframe,
):
    """
    Verify basic corpus statistics.
    """

    result = get_text_statistics(
        sample_text_dataframe,
        "review",
    )

    assert result["column"] == "review"

    assert result["document_count"] == 10

    assert result["unique_documents"] == 10

    assert result["duplicate_documents"] == 0

    assert result["average_words"] > 0

    assert result["maximum_words"] >= (
        result["minimum_words"]
    )

    assert result["average_characters"] > 0


def test_duplicate_documents():
    """
    Duplicate text documents should be detected.
    """

    dataframe = pd.DataFrame(
        {
            "text": [
                "machine learning is useful",
                "machine learning is useful",
                "python programming",
            ]
        }
    )

    result = get_text_statistics(
        dataframe,
        "text",
    )

    assert result["document_count"] == 3
    assert result["unique_documents"] == 2
    assert result["duplicate_documents"] == 1


# ============================================================
# Word frequencies
# ============================================================

def test_word_frequencies(
    sample_text_dataframe,
):
    """
    Word-frequency analysis should return the requested
    number of common words.
    """

    result = get_word_frequencies(
        sample_text_dataframe,
        "review",
        top_n=5,
    )

    assert len(result) == 5

    assert "word" in result[0]
    assert "count" in result[0]

    # Results should be ordered from most to
    # least frequent.
    counts = [
        item["count"]
        for item in result
    ]

    assert counts == sorted(
        counts,
        reverse=True,
    )


def test_word_frequency_known_word():
    """
    Verify that repeated words receive the expected count.
    """

    dataframe = pd.DataFrame(
        {
            "text": [
                "python data python",
                "python machine learning",
                "data python",
            ]
        }
    )

    result = get_word_frequencies(
        dataframe,
        "text",
        top_n=10,
        stop_words=None,
    )

    frequencies = {
        item["word"]: item["count"]
        for item in result
    }

    assert frequencies["python"] == 4
    assert frequencies["data"] == 2


# ============================================================
# N-grams
# ============================================================

def test_bigrams():
    """
    Verify that bigram frequencies are calculated correctly.
    """

    dataframe = pd.DataFrame(
        {
            "text": [
                "machine learning model",
                "machine learning algorithm",
                "machine learning system",
            ]
        }
    )

    result = get_ngrams(
        dataframe,
        "text",
        n=2,
        top_n=10,
        stop_words=None,
    )

    ngrams = {
        item["ngram"]: item["count"]
        for item in result
    }

    assert "machine learning" in ngrams

    assert (
        ngrams["machine learning"]
        == 3
    )


def test_trigrams():
    """
    Verify that the n parameter can also produce trigrams.
    """

    dataframe = pd.DataFrame(
        {
            "text": [
                "artificial intelligence machine learning",
                "artificial intelligence machine learning",
            ]
        }
    )

    result = get_ngrams(
        dataframe,
        "text",
        n=3,
        top_n=10,
        stop_words=None,
    )

    ngrams = {
        item["ngram"]: item["count"]
        for item in result
    }

    assert (
        ngrams[
            "artificial intelligence machine"
        ]
        == 2
    )


def test_invalid_ngram_size(
    sample_text_dataframe,
):
    """
    n must be at least one.
    """

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        get_ngrams(
            sample_text_dataframe,
            "review",
            n=0,
        )


# ============================================================
# TF-IDF
# ============================================================

def test_tfidf_keywords(
    sample_text_dataframe,
):
    """
    TF-IDF should return terms and numeric importance scores.
    """

    result = get_tfidf_keywords(
        sample_text_dataframe,
        "review",
        top_n=5,
    )

    assert len(result) == 5

    assert "term" in result[0]
    assert "score" in result[0]

    assert isinstance(
        result[0]["score"],
        float,
    )


def test_tfidf_ordered_by_score(
    sample_text_dataframe,
):
    """
    TF-IDF results should be ordered from highest
    to lowest importance.
    """

    result = get_tfidf_keywords(
        sample_text_dataframe,
        "review",
        top_n=10,
    )

    scores = [
        item["score"]
        for item in result
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


# ============================================================
# Topic modelling
# ============================================================

def test_topic_modeling(
    sample_text_dataframe,
):
    """
    LDA should return the requested number of topics.
    """

    result = run_topic_modeling(
        sample_text_dataframe,
        "review",
        n_topics=2,
        words_per_topic=5,

        # Important for our small test dataset.
        min_df=1,
        max_df=1.0,
    )

    assert result["method"] == "lda"

    assert result["column"] == "review"

    assert result["document_count"] == 10

    assert result["n_topics"] == 2

    assert len(
        result["topics"]
    ) == 2

    assert result[
        "vocabulary_size"
    ] > 0


def test_topic_model_words(
    sample_text_dataframe,
):
    """
    Every LDA topic should contain the requested
    number of important words.
    """

    result = run_topic_modeling(
        sample_text_dataframe,
        "review",
        n_topics=2,
        words_per_topic=4,
        min_df=1,
        max_df=1.0,
    )

    for topic in result["topics"]:

        assert "topic" in topic
        assert "words" in topic

        assert len(
            topic["words"]
        ) == 4

        for word in topic["words"]:

            assert "word" in word
            assert "weight" in word


def test_topic_model_document_results(
    sample_text_dataframe,
):
    """
    Every usable document should receive a dominant topic
    and topic probabilities.
    """

    result = run_topic_modeling(
        sample_text_dataframe,
        "review",
        n_topics=2,
        min_df=1,
        max_df=1.0,
    )

    documents = result[
        "documents"
    ]

    assert len(documents) == 10

    for document in documents:

        assert document[
            "dominant_topic"
        ] in [0, 1]

        assert (
            0
            <= document[
                "topic_probability"
            ]
            <= 1
        )

        assert len(
            document[
                "topic_probabilities"
            ]
        ) == 2

        # LDA topic probabilities for one document
        # should approximately sum to one.
        assert sum(
            document[
                "topic_probabilities"
            ]
        ) == pytest.approx(
            1.0
        )


def test_topic_model_too_many_topics():
    """
    The number of topics cannot exceed the number
    of usable documents.
    """

    dataframe = pd.DataFrame(
        {
            "text": [
                "python machine learning",
                "cooking fresh pasta",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        run_topic_modeling(
            dataframe,
            "text",
            n_topics=3,
            min_df=1,
            max_df=1.0,
        )


# ============================================================
# Invalid data
# ============================================================

def test_invalid_text_column(
    sample_text_dataframe,
):
    """
    Requesting a column that does not exist should
    produce a clear error.
    """

    with pytest.raises(
        ValueError,
        match="does not exist",
    ):
        get_text_statistics(
            sample_text_dataframe,
            "unknown",
        )


def test_empty_text_column():
    """
    A column containing only missing or empty strings
    cannot be analyzed.
    """

    dataframe = pd.DataFrame(
        {
            "text": [
                None,
                "",
                "   ",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="does not contain usable text",
    ):
        get_text_statistics(
            dataframe,
            "text",
        )