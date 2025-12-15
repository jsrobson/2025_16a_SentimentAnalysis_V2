import pytest
from processor.topic_base import Subtopic, Topic

def normalize_whitespace(s: str) -> str:
    """Normalize whitespace for consistent comparison in tests."""
    return " ".join(s.split())


@pytest.mark.parametrize(
    "name,id,count,tags,feedback,sentiment,read_name,summary",
    [
        ("Cluster X", 1, 5, ["ui"], ["Good"], {"POSITIVE": 5}, "Readable X", "Summary X"),
        ("Cluster Y", 2, 3, ["ux", "feature"], ["Bad experience"], {"NEGATIVE": 3}, "Readable Y", "Summary Y"),
        ("Cluster Z", 3, 0, [], [], {}, None, None),
    ]
)
def test_subtopic_data_dict_and_str(name, id, count, tags, feedback, sentiment, read_name, summary):
    sub = Subtopic(
        name=name,
        id=id,
        count=count,
        tags=tags,
        feedback=feedback,
        sentiment=sentiment,
        read_name=read_name,
        summary=summary
    )

    # Test get_data_dict
    data_dict = sub.get_data_dict()
    expected_label = max(sentiment, key=sentiment.get) if sentiment else None
    assert data_dict["Subtopic"] == read_name
    assert data_dict["Sentiment"] == expected_label
    assert data_dict["Number of Responses"] == count
    assert data_dict["Summary"] == summary

    # Test get_str_data contains key components (normalized)
    str_data_norm = normalize_whitespace(sub.get_str_data())
    assert f"id: {id}" in str_data_norm
    assert f"name: {name}" in str_data_norm
    for tag in tags:
        assert tag in str_data_norm
    for fb in feedback:
        assert fb in str_data_norm
    for label in sentiment.keys():
        assert label in str_data_norm


@pytest.mark.parametrize(
    "topic_name,subtopic_instances,related_ids",
    [
        (
            "MainTopic1",
            [
                Subtopic(name="A", id=1, count=1, tags=["x"], feedback=["ok"], sentiment={"POSITIVE": 1}),
                Subtopic(name="B", id=2, count=2, tags=["y"], feedback=["bad"], sentiment={"NEGATIVE": 2})
            ],
            [1, 2]
        ),
        (
            "MainTopic2",
            [
                Subtopic(name="C", id=3, count=1, tags=["z"], feedback=["meh"], sentiment={"NEUTRAL": 1})
            ],
            [3]
        )
    ]
)
def test_topic_lookup_and_name_prompt(topic_name, subtopic_instances, related_ids):
    # Build dictionary for lookup
    sub_dict = {sub.id: sub for sub in subtopic_instances}

    topic = Topic(name=topic_name, related_sub_topics=related_ids)
    topic.lookup_sub_topic(sub_dict)

    # After lookup, subtopic_data length should match related_ids
    assert len(topic.subtopic_data) == len(related_ids)

    # Normalize whitespace for comparison
    topic_data_norm = [normalize_whitespace(s) for s in topic.subtopic_data]
    for sub in subtopic_instances:
        sub_data_norm = normalize_whitespace(sub.get_str_data())
        assert sub_data_norm in topic_data_norm

    # Check that topic name is in the prompt (normalized)
    prompt_norm = normalize_whitespace(topic.name_prompt())
    assert normalize_whitespace(topic_name.replace("_", " ")) in prompt_norm
    for sub_data in topic.subtopic_data:
        assert normalize_whitespace(sub_data) in prompt_norm
