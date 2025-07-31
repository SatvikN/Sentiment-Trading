import pytest
from sentiment_analysis import clean_text, sia

@pytest.mark.parametrize("text,expected", [
    ("Check out http://example.com this is amazing!", "check amazing"),
    ("The quick brown fox.", "quick brown fox"),
    ("Visit https://test.com for more info!", "visit info")
])
def test_clean_text_removes_urls_and_stopwords(text, expected):
    cleaned = clean_text(text)
    for word in expected.split():
        assert word in cleaned
    assert 'http' not in cleaned and 'https' not in cleaned


def test_vader_sentiment_positive():
    text = "I love this stock!"
    score = sia.polarity_scores(clean_text(text))
    assert score['compound'] > 0.3

def test_vader_sentiment_negative():
    text = "This is terrible and I hate it."
    score = sia.polarity_scores(clean_text(text))
    assert score['compound'] < -0.3

# Optional: Mock FinBERT for speed
import types
from sentiment_analysis import finbert_score

def test_finbert_score_keys(monkeypatch):
    # Mock the actual model call for speed
    monkeypatch.setattr('sentiment_analysis.finbert_score', lambda text: {'finbert_positive': 0.1, 'finbert_neutral': 0.8, 'finbert_negative': 0.1})
    result = finbert_score("The market is stable.")
    assert set(result.keys()) == {'finbert_positive', 'finbert_neutral', 'finbert_negative'}