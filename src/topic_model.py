from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List
import numpy as np

class OnlineLDA:
    def __init__(self, n_topics=10, max_features=20000, batch_size=1000):
        self.n_topics = n_topics
        self.batch_size = batch_size
        self.vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
        self.lda = LatentDirichletAllocation(n_components=n_topics, learning_method="online", random_state=0)
        self._fitted_vectorizer = False
        self._fitted_lda = False

    def partial_fit(self, texts: List[str]):
        if not self._fitted_vectorizer:
            X = self.vectorizer.fit_transform(texts)
            self._fitted_vectorizer = True
            self.lda.partial_fit(X)
            self._fitted_lda = True
        else:
            X = self.vectorizer.transform(texts)
            self.lda.partial_fit(X)

    def get_topics(self, top_n=10):
        if not self._fitted_lda:
            return []
        features = self.vectorizer.get_feature_names_out()
        topics = []
        for comp in self.lda.components_:
            top_idx = comp.argsort()[-top_n:][::-1]
            topics.append([features[i] for i in top_idx])
        return topics
