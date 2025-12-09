from typing import List, Dict, Iterator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import logging

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    
# Transformers optional
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from .config import HF_MODEL_NAME, BATCH_SIZE

logger = logging.getLogger(__name__)

class ISentiment:
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        raise NotImplementedError

class VaderSentiment(ISentiment):
    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        out = []
        for t in texts:
            scores = self._analyzer.polarity_scores(t or "")
            out.append(scores)
        return out

class HFSentiment(ISentiment):
    def __init__(self, model_name: str = HF_MODEL_NAME, device: int = -1):
        if not HF_AVAILABLE:
            raise RuntimeError("Transformers not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
        self.pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=self.tokenizer, device=device)

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        out = []
        n = len(texts)
        batch = BATCH_SIZE
        for i in range(0, n, batch):
            chunk = texts[i:i+batch]
            try:
                res = self.pipe(chunk)
            except Exception as e:
                logger.exception("HF pipeline failed: %s", e)
                # fallback empty
                res = [{"label":"UNKNOWN", "score":0.0} for _ in chunk]
            out.extend(res)
        return out

def get_default_sentiment(use_hf: bool = False, device: int = -1) -> ISentiment:
    if use_hf and HF_AVAILABLE:
        return HFSentiment(device=device)
    return VaderSentiment()
