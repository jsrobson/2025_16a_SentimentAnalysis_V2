"""
Class defines Summary, which handles the LLM topic summarization
pipeline.
"""

# == Standard Library imports ==
import os

# == Third party imports ==
from dotenv import load_dotenv
from torch import bfloat16
from transformers import pipeline

# hide api key
load_dotenv()
ACCESS_TOKEN = os.getenv("HF_API_KEY")

# constant for specified LLM
THEME_MODEL = "google/gemma-3-4b-it"

def get_topic_pipeline() -> pipeline:
    """
    Helper method creates a topic summarization pipeline.
    :return: Huggingface transformer pipeline object.
    """
    topic_pipeline = pipeline(
        task="text-generation",
        model=THEME_MODEL,
        device="cpu",
        dtype=bfloat16,
        token=ACCESS_TOKEN
    )
    return topic_pipeline

def _bundle_messages(prompt: str) -> list[dict[str, list]]:
    """
    Helper method bundles a given prompt string into the expected message
    data structure (list of dicts) for Gemma 3 LLM. Note, if non-Gemma LLM
    used in program, will need to revisit this method!
    :param prompt: Prompt message string to be passed to LLM.
    :return:
    """
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

class Summary:
    """
    Class for Summary object, handles LLM text generation (summarization)
    of deterministic, truncated topic modelling string output into
    human-readable product.
    """
    def __init__(self):
        # instantiate topic summarization pipeline
        self.t_pipe = get_topic_pipeline()

    def get_output(self, name: str, prompt: str) -> str:
        """
        Given a name string and a prompt, method bundles prompt for
        processing by LLM into a summary result.
        :param name: The truncated 'name' of the topic / subtopic.
        :param prompt: The prompt for LLM to perform an action.
        :return: Summary result from LLM text generation.
        """
        try:
            output = self.t_pipe(_bundle_messages(prompt))
            gen_text = output[0]["generated_text"][-1]["content"]
            summary = gen_text.strip()
        except Exception as e:
            print(f"Summary generation failed for '{name}': {e}")
            summary = "Error generating summary"
        return summary