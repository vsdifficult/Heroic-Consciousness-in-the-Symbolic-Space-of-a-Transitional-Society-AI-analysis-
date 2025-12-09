from typing import Iterable, Dict, Optional
from .storage import JSONLStorage
from .sentiment import get_default_sentiment
from .analyzer import FrequencyAccumulator
from .topic_model import OnlineLDA
from tqdm import tqdm
import json
import os

class PipelineOrchestrator:
    def __init__(self, storage_path: str, sentiment_impl="vader", use_hf=False, hf_device=-1):
        self.storage = JSONLStorage(storage_path)
        self.sentiment = get_default_sentiment(use_hf and HF_AVAILABLE, device=hf_device) if 'HF_AVAILABLE' in globals() else get_default_sentiment(False)
        self.freq_acc = FrequencyAccumulator()
        self.lda = OnlineLDA(n_topics=10)
        self.by_source_freq = {}  
        self._temp_text_batch = []
        self._batch_size = 256

    def ingest_stream(self, items: Iterable[Dict]):
        for item in items:
            # store raw
            self.storage.append(item)
            text = item.get("text","")
            self.freq_acc.add_text(text)
            src = item.get("source","all")
            if src not in self.by_source_freq:
                self.by_source_freq[src] = FrequencyAccumulator()
            self.by_source_freq[src].add_text(text)
            self._temp_text_batch.append(text)
            if len(self._temp_text_batch) >= self._batch_size:
                self.lda.partial_fit(self._temp_text_batch)
                self._temp_text_batch = []

        if self._temp_text_batch:
            self.lda.partial_fit(self._temp_text_batch)
            self._temp_text_batch = []

    def analyze_sentiment(self, batch_size=256, use_hf=False):
        out_path = self.storage.path.with_name(self.storage.path.stem + "_sentiment.jsonl")
        tmp = []
        i = 0
        with out_path.open("w", encoding="utf-8") as fout:
            batch = []
            for rec in self.storage.iter():
                txt = rec.get("text","")
                batch.append((i, rec))
                if len(batch) >= batch_size:
                    ids, recs = zip(*batch)
                    texts = [r.get("text","") for r in recs]
                    sent = self.sentiment.analyze_batch(texts)
                    for idx, r, s in zip(ids, recs, sent):
                        r_out = dict(r)
                        r_out["sentiment"] = s
                        fout.write(json.dumps(r_out, ensure_ascii=False) + "\n")
                    batch = []
                i += 1
            if batch:
                ids, recs = zip(*batch)
                texts = [r.get("text","") for r in recs]
                sent = self.sentiment.analyze_batch(texts)
                for idx, r, s in zip(ids, recs, sent):
                    r_out = dict(r)
                    r_out["sentiment"] = s
                    fout.write(json.dumps(r_out, ensure_ascii=False) + "\n")
        return str(out_path)

    def export_reports(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        top_global = self.freq_acc.most_common(200)
        with open(os.path.join(out_dir, "top_global.json"), "w", encoding="utf-8") as f:
            json.dump(top_global, f, ensure_ascii=False, indent=2)
        per_src = {}
        for src, acc in self.by_source_freq.items():
            per_src[src] = acc.most_common(200)
        with open(os.path.join(out_dir, "per_source_top.json"), "w", encoding="utf-8") as f:
            json.dump(per_src, f, ensure_ascii=False, indent=2)
        # topics
        topics = self.lda.get_topics(10)
        with open(os.path.join(out_dir, "topics.json"), "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)
        return {
            "top_global": os.path.join(out_dir, "top_global.json"),
            "per_source": os.path.join(out_dir, "per_source_top.json"),
            "topics": os.path.join(out_dir, "topics.json")
        }
