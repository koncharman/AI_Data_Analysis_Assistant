from unittest.mock import patch

import pytest

from models.ollama_model import MODEL_NAME, ask_ollama


# Replace the real ollama.chat function with a mock.
#
# This prevents the unit test from:
# - starting or connecting to the real Ollama server;
# - loading a real model;
# - taking a long time;
# - producing unpredictable text.
@patch("models.ollama_model.ollama.chat")
def test_ask_ollama_returns_content(mock_chat):
    """
    Verify that ask_ollama returns the response content
    produced by Ollama.
    """

    # Configure the fake Ollama response.
    mock_chat.return_value = {
        "message": {
            "content": "This is the model response."
        }
    }

    # Call the function being tested.
    result = ask_ollama("Analyze this dataset.")

    # Verify that the expected text was returned.
    assert result == "This is the model response."

    # Verify that Ollama was called exactly once
    # with the expected model and prompt structure.
    mock_chat.assert_called_once_with(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Analyze this dataset.",
            }
        ],
    )


def test_ask_ollama_removes_prompt_spaces():
    """
    Verify that unnecessary spaces around the prompt
    are removed before sending it to Ollama.
    """

    # Patch Ollama only for this test.
    with patch(
        "models.ollama_model.ollama.chat"
    ) as mock_chat:

        mock_chat.return_value = {
            "message": {
                "content": "Response"
            }
        }

        # The prompt contains extra spaces.
        result = ask_ollama(
            "   Analyze this dataset.   "
        )

        assert result == "Response"

        # Ollama should receive the cleaned prompt.
        mock_chat.assert_called_once_with(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Analyze this dataset.",
                }
            ],
        )


def test_ask_ollama_rejects_empty_prompt():
    """
    Verify that an empty prompt raises ValueError.
    """

    # Spaces alone should be treated as an empty prompt.
    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        ask_ollama("   ")


def test_ask_ollama_rejects_non_string_prompt():
    """
    Verify that a non-string prompt raises TypeError.
    """

    # The function expects text, not an integer.
    with pytest.raises(
        TypeError,
        match="Prompt must be a string",
    ):
        ask_ollama(123)


@patch("models.ollama_model.ollama.chat")
def test_ask_ollama_handles_connection_error(mock_chat):
    """
    Verify that Ollama connection errors are converted
    into a clear RuntimeError.
    """

    # Simulate a failure such as Ollama not running.
    mock_chat.side_effect = ConnectionError(
        "Ollama is not running"
    )

    with pytest.raises(
        RuntimeError,
        match="Could not communicate with Ollama",
    ):
        ask_ollama("Test prompt")


@patch("models.ollama_model.ollama.chat")
def test_ask_ollama_rejects_missing_message(mock_chat):
    """
    Verify that a malformed response without a message
    raises RuntimeError.
    """

    # Simulate an unexpected Ollama response.
    mock_chat.return_value = {}

    with pytest.raises(
        RuntimeError,
        match="without content",
    ):
        ask_ollama("Test prompt")


@patch("models.ollama_model.ollama.chat")
def test_ask_ollama_rejects_empty_response(mock_chat):
    """
    Verify that an empty model response raises RuntimeError.
    """

    # Simulate a response with no generated text.
    mock_chat.return_value = {
        "message": {
            "content": ""
        }
    }

    with pytest.raises(
        RuntimeError,
        match="without content",
    ):
        ask_ollama("Test prompt")


@patch("models.ollama_model.ollama.chat")
def test_ask_ollama_strips_response_spaces(mock_chat):
    """
    Verify that unnecessary spaces around the model response
    are removed before returning it.
    """

    mock_chat.return_value = {
        "message": {
            "content": "   Clean response   "
        }
    }

    result = ask_ollama("Test prompt")

    assert result == "Clean response"