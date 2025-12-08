import click
import json
from visual.wordcloud_gen import generate_wordcloud
from parsers.jsonc_parser import parse_jsonc


@click.group()
def cli():
    pass


@cli.command()
@click.option("--inp", required=True)
@click.option("--out", required=True)
def wordcloud(inp, out):
    data = json.load(open(inp, "r", encoding="utf-8"))
    texts = [x["text"] for x in data]
    generate_wordcloud("\n".join(texts), out)
    click.echo(f"Saved → {out}")


@cli.command()
@click.option("--inp", required=True)
@click.option("--out", required=True)
def jsonc(inp, out):
    data = parse_jsonc(inp)
    json.dump(data, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    click.echo(f"Parsed {len(data)} comments from JSONC")


if __name__ == "__main__":
    cli()
