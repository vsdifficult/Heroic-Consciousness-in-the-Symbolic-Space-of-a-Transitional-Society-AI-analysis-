import torch
import numpy as np
from .loader import load_pretrained_model


class SentimentModel:
    def __init__(self):
        self.tokenizer, self.model = load_pretrained_model()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model(**inputs)
            scores = torch.softmax(output.logits, dim=1).numpy()[0]

        labels = ["negative", "neutral", "positive"]
        return {labels[i]: float(scores[i]) for i in range(3)}

    def batch_predict(self, texts):
        return [self.predict(t) for t in texts]
