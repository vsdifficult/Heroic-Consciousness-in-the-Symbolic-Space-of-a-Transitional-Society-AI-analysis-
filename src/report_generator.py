"""
report_generator.py

Генерирует HTML-отчёт на основе:
- sentiment-enriched JSONL (каждая строка JSON с ключом "sentiment")
- top_global.json (список пар (token, count))
- per_source_top.json (словарь source -> list(token,count))
- topics.json (список тем)

Встраивает графики (pie/bar) как base64 PNG, создаёт таблицы и секции по источникам.
Работает стримом, не загружая весь JSONL в память.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Iterable
import io
import base64
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def _img_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _safe_open_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _stream_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _detect_sentiment_label(sent_obj) -> Dict[str,float]:
    """
    Normalize various sentiment outputs into dict {'positive':p,'neutral':n,'negative':m}
    Supports:
    - HF pipeline returns {'label': 'POSITIVE'/'NEGATIVE'/'NEUTRAL', 'score': float}
    - Vader returns {'neg':, 'neu':, 'pos':, 'compound':}
    - precomputed dict with keys 'positive' etc.
    """
    if sent_obj is None:
        return {"positive":0.0,"neutral":0.0,"negative":0.0}
    # HF-style list or single-dict
    if isinstance(sent_obj, list) and len(sent_obj)>0:
        obj = sent_obj[0]
    else:
        obj = sent_obj
    if isinstance(obj, dict) and "label" in obj and "score" in obj:
        label = obj["label"].lower()
        score = float(obj.get("score",0.0))
        res = {"positive":0.0,"neutral":0.0,"negative":0.0}
        if "pos" in label:
            res["positive"] = score
        elif "neg" in label:
            res["negative"] = score
        elif "neu" in label:
            res["neutral"] = score
        else:
            # unknown: put score to neutral
            res["neutral"] = score
        return res
    # Vader style
    if isinstance(obj, dict) and ("compound" in obj or "pos" in obj and "neg" in obj):
        # prefer pos/neu/neg keys if present
        if "pos" in obj and "neg" in obj and "neu" in obj:
            return {"positive": float(obj.get("pos",0.0)),
                    "neutral": float(obj.get("neu",0.0)),
                    "negative": float(obj.get("neg",0.0))}
        # fallback: use compound -> map to buckets
        comp = float(obj.get("compound",0.0))
        if comp >= 0.05:
            return {"positive":1.0,"neutral":0.0,"negative":0.0}
        elif comp <= -0.05:
            return {"positive":0.0,"neutral":0.0,"negative":1.0}
        else:
            return {"positive":0.0,"neutral":1.0,"negative":0.0}
    # already normalized?
    if isinstance(obj, dict) and ("positive" in obj or "neutral" in obj or "negative" in obj):
        return {
            "positive": float(obj.get("positive",0.0)),
            "neutral": float(obj.get("neutral",0.0)),
            "negative": float(obj.get("negative",0.0))
        }
    # unknown format
    return {"positive":0.0,"neutral":0.0,"negative":0.0}

def aggregate_sentiment_from_jsonl(path: str) -> Tuple[Dict[str,float], Dict[str,Dict[str,float]], int]:
    """
    Проходит по jsonl и аккумулирует sentiment:
    - total_dist: aggregate sums of positive/neutral/negative (we'll normalize by count)
    - per_source_dist: source -> sums
    - count: total comments
    """
    total = Counter()
    per_source = defaultdict(Counter)
    count = 0
    for rec in _stream_jsonl(path):
        sent = rec.get("sentiment")
        norm = _detect_sentiment_label(sent)
        total["positive"] += norm["positive"]
        total["neutral"] += norm["neutral"]
        total["negative"] += norm["negative"]
        src = rec.get("source","all")
        per_source[src]["positive"] += norm["positive"]
        per_source[src]["neutral"] += norm["neutral"]
        per_source[src]["negative"] += norm["negative"]
        count += 1
    # convert to proportions (safe)
    if count > 0:
        total_prop = {k: float(v)/count for k,v in total.items()}
    else:
        total_prop = {"positive":0.0,"neutral":0.0,"negative":0.0}
    per_src_prop = {}
    for s, c in per_source.items():
        denom = sum(c.values())
        if denom > 0:
            per_src_prop[s] = {k: float(c[k])/denom if denom>0 else 0.0 for k in ["positive","neutral","negative"]}
        else:
            per_src_prop[s] = {"positive":0.0,"neutral":0.0,"negative":0.0}
    return total_prop, per_src_prop, count

def make_pie_chart(dist: Dict[str,float], title: str = "", explode_max=True):
    labels = []
    sizes = []
    for k in ("positive","neutral","negative"):
        labels.append(k.capitalize())
        sizes.append(dist.get(k,0.0))
    fig, ax = plt.subplots(figsize=(5,4))
    # convert to percentages for display, guard small rounding
    total = sum(sizes)
    if total <= 0:
        sizes = [1,1,1]
    ax.pie(sizes, labels=labels, autopct=lambda p: ('%.1f%%' % p) if p>0 else '', startangle=140)
    ax.axis('equal')
    ax.set_title(title)
    return fig

def make_bar_top_tokens(top_list, title="Top tokens", top_n=30):
    items = top_list[:top_n]
    tokens = [t for t,c in items]
    counts = [c for t,c in items]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(range(len(tokens))[::-1], counts, align='center')
    ax.set_yticks(range(len(tokens))[::-1])
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def generate_html_report(
    sentiment_jsonl_path: str,
    out_html: str,
    top_global_json: str = None,
    per_source_top_json: str = None,
    topics_json: str = None,
    title: str = "TextInsights Report"
):
    """
    Main entry. Produces out_html.
    """
    # aggregate sentiment
    total_prop, per_src_prop, total_count = aggregate_sentiment_from_jsonl(sentiment_jsonl_path)

    # load top tokens & topics if available
    top_global = _safe_open_json(top_global_json) or []
    per_source_top = _safe_open_json(per_source_top_json) or {}
    topics = _safe_open_json(topics_json) or []

    # create charts
    pie_fig = make_pie_chart(total_prop, title="Overall sentiment distribution")
    pie_b64 = _img_to_base64(pie_fig)

    tokens_fig_b64 = ""
    if top_global:
        bar_fig = make_bar_top_tokens(top_global, title="Top global tokens")
        tokens_fig_b64 = _img_to_base64(bar_fig)

    # per-source pies
    per_source_imgs = {}
    for s, d in per_src_prop.items():
        fig = make_pie_chart(d, title=f"Sentiment for {s}")
        per_source_imgs[s] = _img_to_base64(fig)

    # topics html
    topics_html = ""
    if topics:
        topics_html += "<div class='topics'>\n"
        for i, t in enumerate(topics):
            topics_html += f"<div class='topic'><h4>Topic {i+1}</h4><p>{', '.join(t)}</p></div>\n"
        topics_html += "</div>"

    # top tokens table
    def top_tokens_table(top_list, top_n=50):
        s = "<table class='tokens'><tr><th>Token</th><th>Count</th></tr>\n"
        for token, cnt in top_list[:top_n]:
            s += f"<tr><td>{token}</td><td>{cnt}</td></tr>\n"
        s += "</table>\n"
        return s

    top_global_html = top_tokens_table(top_global, top_n=50) if top_global else "<p>No global tokens available.</p>"

    # build full html
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body{{font-family: Arial, sans-serif; margin:20px;}}
.header{{display:flex;align-items:center;justify-content:space-between}}
.section{{margin-top:30px;}}
.card{{padding:12px;border:1px solid #ddd;border-radius:6px;background:#fff;}}
.tokens{{border-collapse:collapse;width:100%}}
.tokens th, .tokens td{{border:1px solid #eee;padding:6px;text-align:left}}
.topic{{margin-bottom:10px}}
.imgwrap{{max-width:100%}}
.source-section{{display:flex;flex-wrap:wrap;gap:16px}}
.source-card{{width:320px;padding:8px;border:1px solid #eee;border-radius:6px}}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <div>Generated: {__import__('datetime').datetime.utcnow().isoformat()} UTC</div>
</div>

<div class="section card">
  <h2>Summary</h2>
  <p>Total comments processed: <strong>{total_count}</strong></p>
  <p>Overall sentiment (per comment averaged):</p>
  <div class="imgwrap"><img src="{pie_b64}" alt="sentiment"></div>
</div>

<div class="section card">
  <h2>Top tokens (global)</h2>
  <div class="imgwrap">{('<img src="'+tokens_fig_b64+'"/>') if tokens_fig_b64 else ''}</div>
  {top_global_html}
</div>

<div class="section card">
  <h2>Topics</h2>
  {topics_html if topics else '<p>No topics data.</p>'}
</div>

<div class="section card">
  <h2>Per-source sentiment</h2>
  <div class="source-section">
"""
    for s, img_b64 in per_source_imgs.items():
        html += f"""<div class="source-card"><h3>{s}</h3><img src="{img_b64}" style="max-width:300px;"/>\n"""
        # also include top tokens per source if available
        pst = per_source_top.get(s)
        if pst:
            html += "<h4>Top tokens</h4>\n"
            html += top_tokens_table(pst, top_n=10)
        html += "</div>\n"

    html += """
  </div>
</div>

</body>
</html>
"""
    # write out
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html)
    return str(out_path)
