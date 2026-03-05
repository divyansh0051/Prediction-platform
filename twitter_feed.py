import subprocess
import sys
import datetime
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Simple in-memory cache for tweets
TWEET_CACHE = {}
TWEET_CACHE_TTL = 300

def _tweet_cache_get(key):
    ent = TWEET_CACHE.get(key)
    if not ent:
        return None
    ts, data = ent
    if time.time() - ts > TWEET_CACHE_TTL:
        try:
            del TWEET_CACHE[key]
        except Exception:
            pass
        return None
    return data

def _tweet_cache_set(key, data):
    TWEET_CACHE[key] = (time.time(), data)

def fetch_tweets(symbol, max_tweets=25):
    """Fetch recent tweets mentioning the symbol using snscrape (no API key).
    Returns list of {text, url, date, user, sentiment: None}
    If snscrape is not installed, returns empty list.
    """
    cache_key = f"tweets:{symbol}:{max_tweets}"
    cached = _tweet_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        import snscrape.modules.twitter as sntwitter
    except Exception:
        return []

    query = f"{symbol} lang:en"
    items = []
    try:
        analyzer = SentimentIntensityAnalyzer()
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_tweets:
                break
            try:
                vs = analyzer.polarity_scores(tweet.content)
                sentiment = vs.get('compound', 0.0)
            except Exception:
                sentiment = 0.0
            items.append({
                'text': tweet.content,
                'url': f"https://twitter.com/{tweet.user.username}/status/{tweet.id}",
                'date': tweet.date.isoformat(),
                'user': tweet.user.username,
                'sentiment': sentiment
            })
    except Exception:
        return []

    try:
        _tweet_cache_set(cache_key, items)
    except Exception:
        pass

    return items
