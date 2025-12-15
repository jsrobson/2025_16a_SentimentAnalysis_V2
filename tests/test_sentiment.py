from unittest.mock import patch, MagicMock
from utils import Sentiment

# Test that empty or whitespace-only strings return NEUTRAL
def test_empty_feedback_returns_neutral():
    sentiment = Sentiment()
    result = sentiment.get_feedback_sentiment("")
    assert result == {"label": "NEUTRAL", "score": 0.0}

    result = sentiment.get_feedback_sentiment("   ")
    assert result == {"label": "NEUTRAL", "score": 0.0}


# Test that non-empty strings call the pipeline correctly
@patch("utils.sentiment.get_sentiment_pipeline")
def test_feedback_calls_pipeline(mock_pipeline_fn):
    # Create a fake pipeline returning a dummy sentiment
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [{"label": "POSITIVE", "score": 0.95}]
    mock_pipeline_fn.return_value = fake_pipeline

    sentiment = Sentiment()
    feedback = "I love this product!"
    result = sentiment.get_feedback_sentiment(feedback)

    # Ensure pipeline was called with correct text
    fake_pipeline.assert_called_with(feedback)
    # Ensure returned result matches mocked output
    assert result["label"] == "POSITIVE"
    assert result["score"] == 0.95


# Test multiple feedbacks return expected results
@patch("utils.sentiment.get_sentiment_pipeline")
def test_multiple_feedbacks(mock_pipeline_fn):
    fake_pipeline = MagicMock()
    # simulate different outputs for different inputs
    fake_pipeline.side_effect = [
        [{"label": "POSITIVE", "score": 0.9}],
        [{"label": "NEGATIVE", "score": 0.8}],
    ]
    mock_pipeline_fn.return_value = fake_pipeline

    sentiment = Sentiment()

    res1 = sentiment.get_feedback_sentiment("Great service!")
    res2 = sentiment.get_feedback_sentiment("Terrible experience!")

    assert res1 == {"label": "POSITIVE", "score": 0.9}
    assert res2 == {"label": "NEGATIVE", "score": 0.8}


# Test that only whitespace in string returns neutral without calling pipeline
@patch("utils.sentiment.get_sentiment_pipeline")
def test_whitespace_only_does_not_call_pipeline(mock_pipeline_fn):
    fake_pipeline = MagicMock()
    mock_pipeline_fn.return_value = fake_pipeline

    sentiment = Sentiment()
    result = sentiment.get_feedback_sentiment("   ")
    # Pipeline should never be called
    fake_pipeline.assert_not_called()
    assert result == {"label": "NEUTRAL", "score": 0.0}