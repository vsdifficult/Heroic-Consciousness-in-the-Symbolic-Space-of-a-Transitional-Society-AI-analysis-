import json
from ..models.sentiment_model import SentimentModel
from .tfidf import extract_keywords
from .lda_topics import extract_topics
from .thematic_analyzer import ThematicAnalyzer


class Analyzer:
    def __init__(self):
        self.sentiment = SentimentModel()
        self.thematic = ThematicAnalyzer()

    def analyze(self, items):
        texts = [i["text"] for i in items]

        sentiment_results = self.sentiment.batch_predict(texts)
        keywords = extract_keywords(texts)
        topics = extract_topics(texts)
        thematic_relevance = self.thematic.analyze_relevance(texts)

        return {
            "sentiment": sentiment_results,
            "keywords": keywords,
            "topics": topics,
            "thematic_relevance": thematic_relevance,
            "count": len(texts)
        }
