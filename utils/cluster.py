"""
Class defines Cluster, which handles topic modelling via sentence
transformation and text clustering.
"""

# == Third party imports ==
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import pandas as pd

# constant for specified sentence transformer model
ST_MODEL = "all-roberta-large-v1"

def get_sentence_transformer() -> SentenceTransformer:
    """
    Helper method creates a sentence transformer model given constant.
    :return: Sentence transformer object.
    """
    model = SentenceTransformer(ST_MODEL)
    return model

class Cluster:
    """
    Class for Cluster object, handles transformation of natural language
    feedback into transformed sentence objects for clustering and topic
    extraction.
    """
    def __init__(self, sentences: list[str]):
        # natural language feedback, strs
        self.sentences = sentences
        self.st_model = SentenceTransformer(ST_MODEL)
        # cluster model and related topic hierarchy
        self.topic_model = self._build_clusters()
        self.hierarchy = self.topic_model.hierarchical_topics(self.sentences)

    def _build_clusters(self) -> BERTopic | None:
        if not self.sentences:
            return None
        print("Building clusters...")
        # transform text into vector repr that capture semantic meaning
        embeddings = self.st_model.encode(self.sentences, show_progress_bar=True)
        # reduce dimensionality of embeddings using UMAP model and cosine dist
        umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine',
                          random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        # group similar feedback instances based on lower-dimensional embedding
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=4, metric="euclidean")
        hdbscan_model.fit_predict(reduced_embeddings)
        # convert text into numerical features for count vectorizing
        vectorizer_model = CountVectorizer(
            # set low to reduce likelihood that keywords are removed
            max_df=0.4,
            stop_words="english"
        )
        # use repr model and semantic similarity to find most repr topic words
        representation_model = KeyBERTInspired()
        # given the generated model, build topic model and return it
        topic_model = BERTopic(embedding_model=self.st_model,
                               umap_model=umap_model,
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer_model,
                               representation_model=representation_model)
        topic_model.fit_transform(self.sentences, embeddings)
        return topic_model

    def package_model_data(self) -> dict[int, dict]:
        """
        Helper method packages data from the model into a simple data
        structure for use outside the Cluster object; used as input into
        subtopic class objects.
        :return: Dict of int (k: topic id), dict (v: topic vals).
        """
        topic_info = self.topic_model.get_topic_info()
        # build a lookup so we can query topic name by id.
        topic_name_lookup = dict(zip(topic_info["Topic"], topic_info["Name"]))
        data = {}
        for topic_id in self.topic_model.get_topic_info()['Topic']:
            # skip 'uncategorized' items
            if topic_id == -1:
                continue
            feedback = self.topic_model.get_representative_docs(topic_id)
            # build a data record for each topic id
            data[topic_id] = {
                "id": topic_id,
                "name": topic_name_lookup.get(topic_id),
                "count": self.topic_model.get_topic_freq(topic_id),
                "tags": [w for w, _ in self.topic_model.get_topic(topic_id)],
                "feedback": feedback[:4] # sample only 4 feedback items
            }
        return data

    def get_subtopic_id(self, ind) -> pd.Series:
        """
        Return the topic (or subtopic) assignments for a given index
        :param ind: array-like / pandas.Index: Index values to associate
        with each topic label.
        :return: A series mapping each provided index in ind to its
        corresponding topic (or subtopic) ID.
        """
        topics = self.topic_model.topics_
        # return pd.Series(topics, index=self.df.index)
        return pd.Series(topics, index=ind)

    def assign_topic(self, t_id: int) -> None | str:
        """
        Method returns parent topic name for a given topic (subtopic) id,
        based on hierarchical matching rules found in the topic model.
        :param t_id:
        :return:
        """
        # select matching rows where t_id appears in Topics
        condition = self.hierarchy['Topics'].apply(lambda x: t_id in x)
        matching_rows = self.hierarchy.loc[condition]
        # if no matching rows found, simply quit
        if matching_rows.empty:
            return None
        # find the minimum list length among matching rows; this implies
        # closer topic alignment (a larger list implies greater presence of
        # other subtopics).
        min_len = matching_rows['Topics'].apply(len).min()
        # get all rows with the minimum length
        smallest_rows = matching_rows[matching_rows
                                      ['Topics'].apply(len) == min_len]

        # if multiple rows tie, pick one deterministically by sorting and
        # picking the first item
        if len(smallest_rows) > 1:
            smallest_rows = smallest_rows.sort_values(by='Parent_ID',
                                                        ascending=True)
        parent_name = smallest_rows.iloc[0]['Parent_Name']
        return parent_name




