import json
from ..models.sentiment_model import SentimentModel
from .tfidf import extract_keywords
from .lda_topics import extract_topics


class Analyzer:
    def __init__(self):
        self.sentiment = SentimentModel()

    def analyze(self, items):
        texts = [i["text"] for i in items]

        sentiment_results = self.sentiment.batch_predict(texts)
        keywords = extract_keywords(texts)
        topics = extract_topics(texts)

        return {
            "sentiment": sentiment_results,
            "keywords": keywords,
            "topics": topics,
            "count": len(texts)
        }
