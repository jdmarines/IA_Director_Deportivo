import feedparser
from newspaper import Article
import os
import re


SAVE_DIR = "data/articles"
os.makedirs(SAVE_DIR, exist_ok=True)


RSS_FEEDS = {
    "BBC Sport": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "SkySports": "https://www.skysports.com/rss/12040",
    "Premier League": "https://www.premierleague.com/news/rss"  
}

def clean_filename(name: str) -> str:
    name = re.sub(r"[\\/*?\"<>|]", "", name)  
    return name[:80] 
def download_articles(rss_url, source_name):
    feed = feedparser.parse(rss_url)
    print(f" Fuente: {source_name} ({len(feed.entries)} artículos encontrados)")

    for entry in feed.entries:
        url = entry.link
        try:
            article = Article(url)
            article.download()
            article.parse()

            filename = clean_filename(article.title) + ".txt"
            filepath = os.path.join(SAVE_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(article.title + "\n\n")
                f.write(article.text)

            print(f"Guardado: {article.title}")
        except Exception as e:
            print(f"Error con {url}: {e}")


if __name__ == "__main__":
    for name, url in RSS_FEEDS.items():
        download_articles(url, name)

    print("\nDescarga completa. Artículos guardados en:", SAVE_DIR)
