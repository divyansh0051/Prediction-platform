import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from twitter_feed import fetch_tweets
try:
    from sentiment_finbert import score_text as finbert_score
except Exception:
    finbert_score = None
from datetime import datetime
import time

# Simple in-memory cache with TTL to avoid repeated scrapes
CACHE = {}
CACHE_TTL = 300  # seconds

def _cache_get(key):
    ent = CACHE.get(key)
    if not ent:
        return None
    ts, data = ent
    if time.time() - ts > CACHE_TTL:
        try:
            del CACHE[key]
        except Exception:
            pass
        return None
    return data

def _cache_set(key, data):
    CACHE[key] = (time.time(), data)

analyzer = SentimentIntensityAnalyzer()

def fetch_news(symbol, max_articles=12):
    """Fetch recent news headlines for a symbol using Google News RSS (free).
    Returns list of {title, link, source, published, sentiment}
    """
    q = f"{symbol} stock"
    url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    cache_key = f"news:{symbol}:{max_articles}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_articles]:
        title = entry.get('title', '')
        link = entry.get('link', '')
        published = entry.get('published', '')
        try:
            # normalize published to ISO
            published_parsed = entry.get('published_parsed')
            if published_parsed:
                published = datetime.fromtimestamp(time.mktime(published_parsed)).isoformat()
        except Exception:
            pass

        # simple sentiment on title (fast, free)
        try:
            vs = analyzer.polarity_scores(title)
            sentiment = vs['compound']
        except Exception:
            sentiment = 0.0

        # source extraction
        source = ''
        if 'source' in entry and isinstance(entry.source, dict):
            source = entry.source.get('title','')

        items.append({
            'title': title,
            'link': link,
            'published': published,
            'source': source,
            'sentiment': sentiment
        })

    try:
        _cache_set(cache_key, items)
    except Exception:
        pass
    return items

    # unreachable


def fetch_news_and_tweets(symbol, max_articles=12, include_tweets=False, use_finbert=False):
    cache_key = f"news_and_tweets:{symbol}:{max_articles}:{int(include_tweets)}:{int(use_finbert)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    articles = fetch_news(symbol, max_articles=max_articles)
    if include_tweets:
        tweets = fetch_tweets(symbol, max_tweets=max_articles)
    else:
        tweets = []

    # If tweets requested but none were returned (snscrape not installed or no results),
    # synthesize lightweight tweet-like entries from news headlines so UI can show sentiment.
    if include_tweets and (not tweets or len(tweets) == 0):
        synth = []
        for a in articles[:min(len(articles), max_articles//2 or 1)]:
            text = a.get('title') or ''
            synth.append({
                'text': text,
                'url': a.get('link') or '',
                'date': a.get('published') or '',
                'user': 'news',
                'sentiment': a.get('sentiment_finbert') if a.get('sentiment_finbert') is not None else a.get('sentiment', 0.0)
            })
        tweets = synth

    # optionally rescore with finbert if available
    if use_finbert:
        if finbert_score is not None:
            for a in articles:
                try:
                    s = finbert_score(a['title'])
                    if s is not None:
                        a['sentiment_finbert'] = s
                    else:
                        a['sentiment_finbert'] = None
                except Exception:
                    a['sentiment_finbert'] = None
            for t in tweets:
                try:
                    s = finbert_score(t.get('text') or '')
                    if s is not None:
                        t['sentiment_finbert'] = s
                    else:
                        t['sentiment_finbert'] = None
                except Exception:
                    t['sentiment_finbert'] = None
        else:
            # FinBERT not available in this environment; fall back to VADER scores
            for a in articles:
                try:
                    a['sentiment_finbert'] = a.get('sentiment', 0.0)
                except Exception:
                    a['sentiment_finbert'] = None
            for t in tweets:
                try:
                    t['sentiment_finbert'] = t.get('sentiment', 0.0) if t is not None else None
                except Exception:
                    t['sentiment_finbert'] = None

    result = {'articles': articles, 'tweets': tweets}
    try:
        _cache_set(cache_key, result)
    except Exception:
        pass
    return result
