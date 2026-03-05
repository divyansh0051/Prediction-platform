"""Optional FinBERT scorer using HuggingFace transformers.
This is lazy-loaded and will gracefully return None if transformers/torch aren't installed.
"""

MODEL_NAME = 'yiyanghkust/finbert-tone'

def load_model():
    try:
        from transformers import pipeline
        pipe = pipeline('sentiment-analysis', model=MODEL_NAME)
        return pipe
    except Exception:
        return None

_pipe = None

def score_text(text):
    global _pipe
    if _pipe is None:
        _pipe = load_model()
    if _pipe is None:
        return None
    try:
        res = _pipe(text[:512])
        # result like [{'label': 'positive', 'score': 0.99}]
        if isinstance(res, list) and res:
            r = res[0]
            label = r.get('label')
            score = float(r.get('score', 0.0))
            signed = score if label.lower() in ('positive','pos','bullish') else -score
            return signed
    except Exception:
        return None

def score_texts(texts):
    return [score_text(t) for t in texts]
