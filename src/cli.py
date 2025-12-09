import click
from pathlib import Path
from .scrapers.youtube_selenium import YouTubeSeleniumScraper
from .scrapers.rss_html import parse_rss, scrape_html_text
from .orchestrator import PipelineOrchestrator
from .storage import JSONLStorage
from .config import RAW_DIR, PROC_DIR
from .visualizer import make_wordcloud_from_counter
import os

@click.group()
def cli():
    pass

@cli.command()
@click.option("--video-url", required=True, help="YouTube video URL")
@click.option("--out", default=None, help="Output JSONL file (default data/raw/youtube_<id>.jsonl)")
@click.option("--max-comments", type=int, default=None)
def collect_youtube(video_url, out, max_comments):
    os.makedirs(RAW_DIR, exist_ok=True)
    default = Path(RAW_DIR) / "youtube_comments.jsonl"
    out_path = out or str(default)
    storage = JSONLStorage(out_path)
    scraper = YouTubeSeleniumScraper(headless=True)
    print("Start scraping YouTube:", video_url)
    for item in scraper.stream_comments(video_url, max_comments=max_comments):
        storage.append(item)
    print("Saved to", out_path)

@cli.command()
@click.option("--rss-url", required=True)
@click.option("--out", default=None)
def collect_rss(rss_url, out):
    os.makedirs(RAW_DIR, exist_ok=True)
    default = Path(RAW_DIR) / "rss_comments.jsonl"
    out_path = out or str(default)
    items = parse_rss(rss_url)
    storage = JSONLStorage(out_path)
    for it in items:
        storage.append(it)
    print("Saved RSS items to", out_path)

@cli.command()
@click.option("--storage", required=True, help="Path to raw JSONL with comments")
@click.option("--out-dir", required=True, help="Directory to put reports & sentiment file")
@click.option("--use-hf", is_flag=True, help="Use HuggingFace model for sentiment (heavy)")
def analyze(storage, out_dir, use_hf):
    os.makedirs(out_dir, exist_ok=True)
    orchestrator = PipelineOrchestrator(storage, use_hf=use_hf)
    orchestrator.ingest_stream(orchestrator.storage.iter())
    sent_path = orchestrator.analyze_sentiment()
    print("Sentiment enriched file:", sent_path)
    reports = orchestrator.export_reports(out_dir)
    print("Reports saved:", reports)

@cli.command()
@click.option("--top-json", required=True, help="Path to exported top_global.json")
@click.option("--out-png", required=True, help="Output path for wordcloud png")
def wordcloud_top(top_json, out_png):
    import json
    with open(top_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    from collections import Counter
    c = Counter(dict(data))
    make_wordcloud_from_counter(c, out_png)
    print("Wordcloud saved to", out_png)

if __name__ == "__main__":
    cli()
