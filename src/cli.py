import click
import json

from .parsers.youtube_parser import parse_youtube_comments
from .analysis.analyzer import Analyzer
from .visual.wordcloud_gen import generate_wordcloud


@click.group()
def cli():
    pass


@cli.command()
@click.option("--url", required=True)
@click.option("--out", required=True)
@click.option("--limit", type=int)
def youtube(url, out, limit):
    data = parse_youtube_comments(url, limit)
    json.dump(data, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    click.echo(f"Collected {len(data)} YouTube comments")


@cli.command()
@click.option("--inp", required=True)
@click.option("--out", required=True)
def analyze(inp, out):
    items = json.load(open(inp, "r", encoding="utf-8"))
    analyzer = Analyzer()
    res = analyzer.analyze(items)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    click.echo("Analysis complete")


@cli.command()
@click.option("--inp", required=True)
@click.option("--out", required=True)
def wordcloud(inp, out):
    data = json.load(open(inp, "r", encoding="utf-8"))
    texts = [x["text"] for x in data]
    generate_wordcloud("\n".join(texts), out)
    click.echo(f"Saved → {out}")


if __name__ == "__main__":
    cli()
