import numpy as np
import faiss
import yfinance as yf
from dotenv import load_dotenv
from newsapi import NewsApiClient
from sec_api import QueryApi
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client  = OpenAI(api_key=api_key)

news_api_key = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=news_api_key)

sec_api_key = os.getenv("SEC_API_KEY")
queryApi = QueryApi(api_key=sec_api_key)

def fetch_financial_data(tickers, period="1y"):
    """
    Download historical closing prices for given tickers.
    Returns a pandas Series/DataFrame of closing prices.
    """
    data = yf.download(tickers, period=period)
    return data['Close'] ## Return the values of closing tickers or markets closing value

def fetch_news(tickers, page_size=5):
    """
    Fetch latest news headlines and descriptions for given tickers.
    Returns a list of strings.
    """
    articles = []
    for ticker in tickers:
        res = newsapi.get_everything(
            q=ticker,
            language='en',
            sort_by='publishedAt',
            page_size=page_size
        )
        for art in res.get('articles', []):
            published = art.get('publishedAt', '')
            title = art.get('title', '')
            desc = art.get('description', '')
            articles.append(f"{published}: {title} - {desc}")
    return articles

def fetch_sec_filings(tickers, size=3):
    """
    Fetch recent SEC filings (10-K, 10-Q) for given tickers.
    Returns a list of strings.
    """
    filings = []
    for ticker in tickers:
        query = {
            "query": {"query_string": {"query": f"{ticker} (10-K OR 10-Q)"}},
            "from": "0",
            "size": str(size),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        resp = queryApi.get_filings(query)
        for f in resp.get('filings', []):
            date = f.get('filedAt', '')
            form = f.get('formType', '')
            comp = f.get('companyName', '')
            text_snip = f.get('text', '').replace('\n', ' ')[:200]
            filings.append(f"{date}: {form} - {comp} - {text_snip}")
    return filings

def embed_text(text):
    """
    Create an embedding for the given text using OpenAI.
    Returns a numpy array of shape (model_dim,).
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(resp.data[0].embedding, dtype="float32")

def create_index(data):
    """
    Build a FAISS index over the provided list of text snippets.
    Returns (index, embeddings_matrix).
    """
    embeddings = [embed_text(str(d)) for d in data] ## Generates vector embeddings using openai embed text
    embeddings = np.vstack(embeddings).astype('float32') ## Converts the embeddings to vstack
    dimension = embeddings.shape[1] ## Returns total numbers of rows
    index = faiss.IndexFlatL2(dimension) ## create indexes equal to total number of rows
    index.add(embeddings)
    return index, embeddings

def setup_data():
    """
    Gather stock, news, and SEC texts; build and return FAISS index.
    """
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    # Gather all text contexts
    stock_df = fetch_financial_data(tickers)
    stock_df.dropna(inplace=True)
    stock_texts = [f"{dt.date()}: {row.to_dict()}" for dt, row in stock_df.iterrows()]
    # NEWS
    news_texts = fetch_news(tickers)
    # SEC FILINGS
    sec_texts = fetch_sec_filings(tickers)
    all_texts = stock_texts + news_texts + sec_texts
    idx, mats = create_index(all_texts)
    return idx, all_texts, mats