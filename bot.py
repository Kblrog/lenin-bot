import os
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Bot

with open('quotes.json', 'r', encoding='utf-8') as f:
    quotes_data = json.load(f)
    LENIN_QUOTES = [q['text'] for q in quotes_data]

def get_top_news(api_key):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'category': 'general',
        'language': 'en',
        'pageSize': 10,
        'apiKey': api_key
    }
    resp = requests.get(url, params=params)
    articles = resp.json().get('articles', [])
    seen = set()
    unique = []
    for a in articles:
        if a.get('title') and a.get('description') and a['title'] not in seen:
            seen.add(a['title'])
            unique.append(a)
    return unique[:3]

def find_best_quote(news_text):
    corpus = LENIN_QUOTES + [news_text]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1000)
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    return LENIN_QUOTES[sims.argmax()]

def generate_post(news, quote):
    return (
        f"🗞 *{news['title']}*\n\n"
        f"Ленинский комментарий (1917):\n> _\"{quote}\"_\n\n"
        f"🔗 [Читать]({news['url']})\n\n"
        f"_P.S. Капитализм всё ещё не отменили. Но Ленин уже всё сказал._"
    )

def main():
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    news_key = os.getenv('NEWS_API_KEY')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    bot = Bot(token=bot_token)
    news_list = get_top_news(news_key)

    for news in news_list:
        full = f"{news['title']} {news['description']}"
        quote = find_best_quote(full)
        post = generate_post(news, quote)
        bot.send_message(chat_id=chat_id, text=post, parse_mode='Markdown', disable_web_page_preview=True)

if __name__ == '__main__':
    main()
