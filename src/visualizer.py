from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, Iterable

def make_wordcloud_from_counter(counter, out_path: str, max_words=200):
    wc = WordCloud(width=1600, height=900, background_color="white", max_words=max_words)
    wc.generate_from_frequencies(dict(counter.most_common(max_words)))
    wc.to_file(out_path)
