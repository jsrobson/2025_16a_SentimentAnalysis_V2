"""
Class defines Sentiment, which assesses the sentiment of given
feedback strings.
"""
# == Third party imports ==
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

# constant for specified sentiment analysis model
SMT_MODEL = "tabularisai/multilingual-sentiment-analysis"

def get_sentiment_pipeline() -> pipeline:
    """
    Helper method creates a sentiment analysis pipeline.
    :return: Huggingface transformer pipeline object.
    """
    sentiment_pipeline = pipeline(
        task="text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(SMT_MODEL),
        tokenizer=AutoTokenizer.from_pretrained(SMT_MODEL),
        truncation=True,
        max_length=512
    )
    return sentiment_pipeline

class Sentiment:
    """
    Class for Sentiment object, handles sentiment analysis of given feedback
    strings and produces an associated label (neutral, positive, negative)
    and score.
    """
    def __init__(self):
        # instantiate sentiment analysis pipeline
        self.smt_pipe = get_sentiment_pipeline()

    def get_feedback_sentiment(self, feedback: str) -> dict[str, str]:
        """
        Given a feedback string, method accesses sentiment analysis pipeline to
        associate feedback with a sentiment label and score.
        :param feedback: Given feedback string.
        :return: Dict comprising sentiment label, score, strs.
        """
        if not feedback.strip():
            return {
                "label": "NEUTRAL",
                "score": 0.0
            }
        result = self.smt_pipe(feedback)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }