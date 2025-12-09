
from pathlib import Path
import os

from .storage import JSONLStorage
from .scrapers.youtube_selenium import YouTubeSeleniumScraper
from .scrapers.rss_html import parse_rss
from .orchestrator import PipelineOrchestrator
from .config import RAW_DIR, PROC_DIR
from .report_generator import generate_html_report


class FullPipeline:
    """
    HIGH-LEVEL PIPELINE:
    1) Collect raw comments (YouTube or RSS)
    2) Run sentiment analysis
    3) Run keyword extraction, topics, top-tokens
    4) Generate HTML report
    """

    def __init__(self, source: str, url: str, use_hf: bool = True):
        """
        source: "youtube" | "rss"
        url:   video URL or RSS URL
        """
        self.source = source
        self.url = url
        self.use_hf = use_hf

        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(PROC_DIR, exist_ok=True)

        self.raw_file = Path(RAW_DIR) / f"{source}_raw.jsonl"
        self.sentiment_file = Path(PROC_DIR) / f"{source}_sentiment.jsonl"
        self.out_dir = Path(PROC_DIR)

    def collect(self):
        """Step 1: scrape / parse"""

        if self.source == "youtube":
            scraper = YouTubeSeleniumScraper(headless=True)
            storage = JSONLStorage(self.raw_file)
            print(f"[PIPELINE] Collecting YouTube → {self.url}")

            for item in scraper.stream_comments(self.url):
                storage.append(item)

        elif self.source == "rss":
            print(f"[PIPELINE] Collecting RSS → {self.url}")
            data = parse_rss(self.url)

            storage = JSONLStorage(self.raw_file)
            for item in data:
                storage.append(item)

        else:
            raise ValueError("Unknown source: " + self.source)

        print(f"[PIPELINE] Raw data saved → {self.raw_file}")

    def analyze(self):
        """Step 2–3: sentiment + NLP"""
        print("[PIPELINE] Running analysis...")

        orchestrator = PipelineOrchestrator(
            storage_path=self.raw_file,
            use_hf=self.use_hf
        )

        orchestrator.ingest_stream(
            orchestrator.storage.iter()
        )

        # sentiment results
        sent = orchestrator.analyze_sentiment(out_path=self.sentiment_file)
        print("[PIPELINE] Sentiment saved →", sent)

        # export all NLP reports
        reports = orchestrator.export_reports(self.out_dir)
        print("[PIPELINE] NLP reports saved →", reports)

        return reports

    def report(self):
        """Step 4: generate HTML report"""
        print("[PIPELINE] Building HTML report...")

        out_html = Path(self.out_dir) / f"{self.source}_report.html"

        generate_html_report(
            sentiment_jsonl_path=str(self.sentiment_file),
            out_html=str(out_html),
            top_global_json=str(Path(self.out_dir) / "top_global.json"),
            per_source_top_json=str(Path(self.out_dir) / "per_source_top.json"),
            topics_json=str(Path(self.out_dir) / "topics.json"),
            title=f"Heroic Consciousness — {self.source.capitalize()} Report"
        )

        print("[PIPELINE] Report saved:", out_html)
        return out_html

    def run(self):
        """Execute full pipeline"""
        self.collect()
        self.analyze()
        return self.report()
