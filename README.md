# Heroic-Consciousness-in-the-Symbolic-Space-of-a-Transitional-Society-AI-analysis-
Heroic Consciousness in the Symbolic Space of a Transitional Society: Memorial Practices and Strategies of Transformation 

CLI
```
pip install -r requirements.txt

python -m src.text_insights.cli collect-youtube \
  --video-url "https://youtu.be/..." \
  --out data/raw/youtube.jsonl

python -m src.text_insights.cli analyze \
  --storage data/raw/youtube.jsonl \
  --out-dir data/processed \
  --use-hf

python -m src.text_insights.cli generate-report \
  --sentiment-jsonl data/processed/youtube_sentiment.jsonl \
  --out-html data/processed/report.html \
  --top-global data/processed/top_global.json \
  --per-source-top data/processed/per_source_top.json \
  --topics data/processed/topics.json


```

python module
```
from src.text_insights.pipeline import FullPipeline

pipeline = FullPipeline(
    source="youtube",
    url="https://youtube.com/watch?v=....",
    use_hf=True
)

pipeline.run()

```