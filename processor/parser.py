"""
Class defines Parser, which handles feedback parsing, clustering, sentiment
assignment, and summarization work.
"""
# == Standard Library imports ==
import re
from collections import defaultdict

# == Third party imports ==
import pandas as pd

# == Local imports ==
from .topic_base import Subtopic, Topic
from utils import Cluster, Sentiment, Summary

# constants for column headers
FB_COL = "Comments"
SMT_LABEL = "smt_label"
SMT_SCORE = "smt_score"
T_ID = "topic_name"
ST_ID = "subtopic_id"

# set maximum columnar output for df
pd.set_option('display.max_columns', None)

class Parser:
    """
    Class for Parser object, coordinates all program operations –
    instantiates data df, conducts sentiment analysis, topic modelling via
    clustering, and text summarization. Instantiates topic and subtopic
    objects for data organization and encapsulation.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.smt = Sentiment()
        self.summary = Summary()
        self.cluster = Cluster(self.df[FB_COL].tolist())
        self.topics: list[Topic] = []
        self.subtopics: dict[int, Subtopic] = {}

    def _get_sentimental(self, feedback: list[str]) -> dict[str, int]:
        """
        Method counts the instance of sentiments (i.e., pos, neg,
        neut) present in the given dataset.
        :param feedback: Comprehensive list of feedback items.
        :return: Dict of str (k = pos / neg / neut), int (v = count)
        """
        sentiment_counts = defaultdict(int)
        # for each feedback str, get feedback label associated with it
        # and update count within dict
        for fb in feedback:
            sentiment = self.smt.get_feedback_sentiment(fb)["label"]
            sentiment_counts[sentiment] += 1
        return dict(sentiment_counts)

    def _build_subtopics(self) -> None:
        """
        Method builds subtopics for packaged model data (which will contain
        >= 1 subtopic). This is assigned to the parser object subtopics ds.
        """
        if not self.cluster.package_model_data():
            return None
        # for each 'topic' id, data instance (subtopic) in the model data,
        for tid, dt in self.cluster.package_model_data().items():
            # get a sentiment count given available feedback
            sentiment_count = self._get_sentimental(dt['feedback'])
            # clean up the name, which usually has a leading number, underscore
            cleaned = re.sub(r'^[-\d_]+', '', dt['name'])
            # build subtopic from data and assign to data structure.
            st = Subtopic(
                name=cleaned,
                id=dt['id'],
                count=dt['count'],
                tags=dt.get('tags', []),
                feedback=dt['feedback'],
                sentiment=sentiment_count
            )
            self.subtopics[tid] = st

    def _build_topics(self) -> None:
        """
        Method builds topics from generalized subtopic combinations present
        in the topic model hierarchy.
        :return:
        """
        t_dict = defaultdict(list)
        # for each subtopic present in the data
        for st_id in self.subtopics.keys():
            # get a generalized topic name by accessing the topic hierarchy
            name = self.cluster.assign_topic(st_id)
            # append the subtopic id as value to the topic name as key
            # multiple subtopics can be appended
            t_dict[name].append(st_id)
        # for each topic / subtopic pairing, generate a topic object and
        # append to ds
        for name, collection in t_dict.items():
            t = Topic(
                name=name,
                related_sub_topics=collection
            )
            self.topics.append(t)

    def _build_topic_names(self) -> None:
        """
        Method 'builds' topic name from raw input using LLM text
        summarization and decoration.
        """
        print("Building topic names...")
        for t in self.topics:
            # if topic already has name, skip
            if t.read_name:
                continue
            # lookup stores subtopic data as flat string within topic object
            t.lookup_sub_topic(self.subtopics)
            # get the readable name from passing name, prompt, st info to LLM
            t.read_name = self.summary.get_output(t.name, t.name_prompt())


    def _build_subtopic_info(self) -> None:
        """
        Method 'builds' subtopic info from raw input using LLM text
        summarization and decoration. Information is human-readable name and
        summarizing text (i.e. what subtopic is about).
        """
        print("Building subtopic information...")
        for st in self.subtopics.values():
            # get readable name from passing name, prompt, st info to LLM
            read_name = self.summary.get_output(st.name, st.name_prompt())
            # get summary information from passing st info to LLM
            summary_txt = self.summary.get_output(st.name, st.summary_prompt())
            st.read_name = read_name
            st.summary = summary_txt

    def get_summary(self) -> pd.DataFrame:
        """
        Method organizes temporary data (i.e., topic, subtopic data
        structures) into persistent dataframe for processing and output.
        :return: Dataframe of summarized topic, subtopic data.
        """
        records = []
        # build a map where the topic readable name corresponds to subtopics
        d_map = {t.read_name: t.related_sub_topics for t in self.topics}
        for st in self.subtopics.values():
            # for every subtopic, get its packaged data in dict
            st_data = st.get_data_dict()
            # find the parent topic to the subtopic through mapping
            parent_topic = next(
                (t_name for t_name, sub_ids in d_map.items() if st.id in
                 sub_ids),
                "None"
            )
            # set the general topic field in subtopic data to its parent
            st_data["General Topic"] = parent_topic
            # ensure this is positioned first in the df for readability
            st_data = {'General Topic': st_data.pop('General Topic'), **st_data}
            records.append(st_data)
        return pd.DataFrame(records)

    def run(self) -> None:
        """
        Method runs parsing / sentiment analysis / topic modelling / LLM
        summarization operations on data.
        """
        # build subtopics and topics from data using topic modelling
        self._build_subtopics()
        self._build_topics()
        # build topic names and subtopic info from topic modelling output
        # using LLM
        self._build_topic_names()
        self._build_subtopic_info()
        # instantiate a summary df and print to CSV
        df = self.get_summary()
        df.to_csv(
            "data/output.csv",
            index=False,
            encoding="utf-8",
            sep=",",
            quoting=1,
        )
