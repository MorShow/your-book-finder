import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from model.title_classifier import TitleClassifier
from model.vectorizer_topic_clusterizer import TopicVectorizerClusterizer
