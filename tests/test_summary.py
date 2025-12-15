from unittest.mock import patch, MagicMock
from utils.summary import Summary, _bundle_messages

class TestSummary:

    # Test that _bundle_messages returns the correct structure
    def test_bundle_messages_structure(self):
        prompt = "Summarize the topic"
        result = _bundle_messages(prompt)
        assert isinstance(result, list)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["text"] == prompt or result[1]["content"][0]["text"] == "Summarize the topic"

    # Test get_output with mocked pipeline
    @patch("utils.summary.get_topic_pipeline")
    def test_get_output_returns_summary(self, mock_pipeline_fn):
        # Setup mock pipeline
        fake_pipeline = MagicMock()
        fake_pipeline.return_value = [
            {"generated_text": [{"content": "This is a test summary"}]}
        ]
        mock_pipeline_fn.return_value = fake_pipeline

        summary_obj = Summary()
        result = summary_obj.get_output("Topic1", "Generate summary")

        # Ensure pipeline was called
        fake_pipeline.assert_called_with(_bundle_messages("Generate summary"))
        # Ensure returned summary matches mocked output
        assert result == "This is a test summary"

    # Test get_output handles exceptions gracefully
    @patch("utils.summary.get_topic_pipeline")
    def test_get_output_handles_exception(self, mock_pipeline_fn):
        # Mock pipeline to raise an exception
        fake_pipeline = MagicMock(side_effect=Exception("Pipeline error"))
        mock_pipeline_fn.return_value = fake_pipeline

        summary_obj = Summary()
        result = summary_obj.get_output("Topic1", "Generate summary")

        assert result == "Error generating summary"
