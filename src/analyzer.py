from collections import Counter, defaultdict
from typing import Iterable, List, Dict
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
import numpy as np
import joblib
import re
import math

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)

class FrequencyAccumulator:
    def __init__(self):
        self.counter = Counter()

    def add_text(self, text: str):
        if not text:
            return
        tokens = _TOKEN_RE.findall(text.lower())
        self.counter.update(tokens)

    def most_common(self, n=100):
        return self.counter.most_common(n)

    def save(self, path):
        joblib.dump(self.counter, path)

    def load(self, path):
        self.counter = joblib.load(path)

class IncrementalTFIDF:
    def __init__(self, n_features=2**18):
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, token_pattern=r"(?u)\\b\\w+\\b")
        self.tfidf_transformer = TfidfTransformer()
        self.doc_count = 0

    def transform_batch(self, texts: List[str]):
        X = self.vectorizer.transform(texts)
        return X

    def partial_tfidf_fit(self, X):
        # accumulate sums for global idf estimation using transformer.partial_fit if available
        # sklearn.transformers do not have partial_fit for TfidfTransformer: we approximate by fitting on sample batches
        self.tfidf_transformer.fit(X)  # approximate, not perfect for huge corpora

    def toarray_topk(self, X, feature_names_extractor, top_k=10):
        # This helper returns top-k token indices per row (but HashingVectorizer doesn't allow reverse mapping)
        # So prefer FrequencyAccumulator for wordclouds and use TF-IDF for scoring with available token mapping if needed.
        pass
