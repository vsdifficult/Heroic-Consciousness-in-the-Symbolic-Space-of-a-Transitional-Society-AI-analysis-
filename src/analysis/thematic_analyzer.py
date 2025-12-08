from transformers import pipeline
import torch


class ThematicAnalyzer:
    def __init__(self, theme="Heroic consciousness in the symbolic space of a transitional society: commemorative practices and strategies of transformation"):
        self.theme = theme
        # Use a zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def analyze_relevance(self, texts, labels=["relevant", "not relevant"]):
        """
        Classifies texts as relevant or not to the theme.
        """
        results = []
        for text in texts:
            result = self.classifier(text, candidate_labels=labels, hypothesis_template=f"This text is about {self.theme}.")
            results.append({
                "text": text,
                "label": result["labels"][0],
                "score": result["scores"][0]
            })
        return results

    def summarize_theme(self, texts):
        """
        Summarizes the texts in the context of the theme.
        """
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        combined_text = " ".join(texts)
        summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
        return summary[0]["summary_text"]
