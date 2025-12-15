import pandas as pd
import pytest
from unittest.mock import MagicMock

from utils import Cluster

def test_cluster_with_no_sentences():
    cluster = Cluster(sentences=[])

    assert cluster.topic_model is None

def test_package_model_data(monkeypatch):
    cluster = Cluster.__new__(Cluster)  # bypass __init__

    mock_topic_model = MagicMock()

    # Fake topic info table
    topic_info = pd.DataFrame({
        "Topic": [0, 1, -1],
        "Name": ["Topic A", "Topic B", "Outliers"]
    })

    mock_topic_model.get_topic_info.return_value = topic_info
    mock_topic_model.get_topic_freq.side_effect = lambda t: {0: 5, 1: 3}[t]
    mock_topic_model.get_topic.return_value = [("word1", 0.2), ("word2", 0.1)]
    mock_topic_model.get_representative_docs.return_value = [
        "doc1", "doc2", "doc3", "doc4", "doc5"
    ]

    cluster.topic_model = mock_topic_model

    data = cluster.package_model_data()

    assert 0 in data
    assert 1 in data
    assert -1 not in data

    assert data[0]["name"] == "Topic A"
    assert data[0]["count"] == 5
    assert data[0]["tags"] == ["word1", "word2"]
    assert len(data[0]["feedback"]) == 4

def test_get_subtopic_id():
    cluster = Cluster.__new__(Cluster)

    cluster.topic_model = MagicMock()
    cluster.topic_model.topics_ = [0, 1, 1, 0]

    index = pd.Index(["a", "b", "c", "d"])
    result = cluster.get_subtopic_id(index)

    assert isinstance(result, pd.Series)
    assert result["a"] == 0
    assert result["b"] == 1

def test_assign_topic_returns_parent_name():
    cluster = Cluster.__new__(Cluster)

    cluster.hierarchy = pd.DataFrame({
        "Topics": [[1, 2], [2], [1]],
        "Parent_ID": [10, 20, 5],
        "Parent_Name": ["Parent A", "Parent B", "Parent C"]
    })

    result = cluster.assign_topic(1)

    assert result == "Parent C"

def test_assign_topic_returns_none_if_not_found():
    cluster = Cluster.__new__(Cluster)

    cluster.hierarchy = pd.DataFrame({
        "Topics": [[2], [3]],
        "Parent_ID": [1, 2],
        "Parent_Name": ["A", "B"]
    })

    assert cluster.assign_topic(999) is None