import json
from news_feed import fetch_news_and_tweets

print('Fetching and rescoring with FinBERT (if available)...')
res = fetch_news_and_tweets('RELIANCE', max_articles=8, include_tweets=True, use_finbert=True)
print('Articles:', len(res.get('articles', [])))
print('Tweets:', len(res.get('tweets', [])))
print(json.dumps(res, indent=2))
