# Heroic-Consciousness-in-the-Symbolic-Space-of-a-Transitional-Society-AI-analysis-
Heroic Consciousness in the Symbolic Space of a Transitional Society: Memorial Practices and Strategies of Transformation 

command
```
pip install -r requirements.txt

python main.py scrape \
    --source youtube \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --out data/raw/youtube.json

```

python module
```
from src.application.pipeline import FullPipeline

pipeline = FullPipeline("youtube", "https://youtube.com/..")
pipeline.run()

```