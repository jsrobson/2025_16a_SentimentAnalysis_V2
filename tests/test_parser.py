import pytest
from unittest.mock import patch
import pandas as pd

from processor.parser import Parser
from processor.topic_base import Subtopic, Topic

# Sample dataframe
SAMPLE_DF = pd.DataFrame({
    "feedback": [
        "Great product",
        "Needs improvement",
        "Average experience"
    ]
})


@pytest.fixture
def parser_fixture():
    # Patch Sentiment, Summary, and Cluster before instantiating Parser
    with patch("processor.parser.Sentiment") as MockSentiment, \
         patch("processor.parser.Summary") as MockSummary, \
         patch("processor.parser.Cluster") as MockCluster:

        # Mock Sentiment.get_feedback_sentiment
        mock_smt = MockSentiment.return_value
        mock_smt.get_feedback_sentiment.side_effect = lambda fb: {
            "Great product": {"label": "POSITIVE", "score": 0.9},
            "Needs improvement": {"label": "NEGATIVE", "score": 0.8},
            "Average experience": {"label": "NEUTRAL", "score": 0.5}
        }[fb]

        # Mock Summary.get_output
        mock_summary = MockSummary.return_value
        mock_summary.get_output.side_effect = lambda name, prompt: f"Summary for {name}"

        # Mock Cluster methods
        mock_cluster = MockCluster.return_value
        mock_cluster.package_model_data.return_value = {
            1: {"id": 1, "name": "1_clusterA", "count": 1, "feedback": ["Great product"], "tags": ["tag1"]},
            2: {"id": 2, "name": "2_clusterB", "count": 1, "feedback": ["Needs improvement"], "tags": ["tag2"]}
        }
        mock_cluster.assign_topic.side_effect = lambda st_id: "Topic1" if st_id in [1, 2] else "Topic2"

        parser = Parser(SAMPLE_DF, col_name="feedback")
        parser.cluster = mock_cluster  # <<< important: assign mocked cluster
        yield parser


def test_build_subtopics(parser_fixture):
    parser_fixture._build_subtopics()

    subtopics = parser_fixture.subtopics
    assert len(subtopics) == 2
    assert isinstance(subtopics[1], Subtopic)
    assert subtopics[1].name == "clusterA"  # leading number removed by re.sub
    assert subtopics[2].name == "clusterB"

    # Sentiment counts
    assert subtopics[1].sentiment == {"POSITIVE": 1}
    assert subtopics[2].sentiment == {"NEGATIVE": 1}


def test_build_topics(parser_fixture):
    parser_fixture._build_subtopics()
    parser_fixture._build_topics()

    topics = parser_fixture.topics
    assert len(topics) == 1
    assert isinstance(topics[0], Topic)
    assert topics[0].related_sub_topics == [1, 2]


def test_build_topic_names_and_subtopic_info(parser_fixture):
    parser_fixture._build_subtopics()
    parser_fixture._build_topics()

    parser_fixture._build_topic_names()
    parser_fixture._build_subtopic_info()

    # Check subtopics have read_name and summary from Summary mock
    for sub in parser_fixture.subtopics.values():
        assert sub.read_name.startswith("Summary for")
        assert sub.summary.startswith("Summary for")

    # Check topics have read_name from Summary mock
    for t in parser_fixture.topics:
        assert t.read_name.startswith("Summary for")


def test_get_summary(parser_fixture):
    parser_fixture._build_subtopics()
    parser_fixture._build_topics()
    parser_fixture._build_topic_names()
    parser_fixture._build_subtopic_info()

    df_summary = parser_fixture.get_summary()
    assert isinstance(df_summary, pd.DataFrame)
    assert "General Topic" in df_summary.columns
    assert "Subtopic" in df_summary.columns
    assert "Sentiment" in df_summary.columns
    assert "Number of Responses" in df_summary.columns
    assert "Summary" in df_summary.columns
    assert df_summary.shape[0] == 2
