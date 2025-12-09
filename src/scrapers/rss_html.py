import requests
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict

def scrape_html_text(url: str, user_agent: str = "TextInsights/1.0") -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)

def parse_rss(rss_url: str, max_items: int = 50) -> List[Dict]:
    feed = feedparser.parse(rss_url)
    items = []
    for e in feed.entries[:max_items]:
        text = (e.get("title","") or "") + "\n" + (e.get("summary","") or "")
        items.append({"source":"rss", "url": e.get("link",""), "text": text})
    return items
