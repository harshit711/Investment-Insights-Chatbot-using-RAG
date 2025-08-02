# Investment Insights Chatbot (RAG)

A conversational investment‐analysis API built with FastAPI, FAISS, and OpenAI GPT-3.5. It aggregates live stock prices, news headlines, and SEC filings, semantically indexes them with OpenAI embeddings and FAISS, and returns concise, on-demand investment insights.

## 🔍 Features

- **Real-time Data Ingestion**  
  - Historical closing prices via [yfinance](https://pypi.org/project/yfinance/)  
  - Latest headlines via [NewsAPI-Python](https://pypi.org/project/newsapi-python/)  
  - Recent 10-K/10-Q filings via [sec-api](https://pypi.org/project/sec-api/)  

- **Semantic Retrieval & LLM**  
  - Embeddings generated with OpenAI’s `text-embedding-3-small`  
  - FAISS index for sub-100 ms vector search  
  - GPT-3.5–powered answer synthesis  

- **Scalable API**  
  - Built on [FastAPI](https://fastapi.tiangolo.com/) and served with Uvicorn  
  - Single `/insights` endpoint for natural-language investment queries  

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/investment-insights-chatbot.git
cd investment-insights-chatbot
```

### 2. Create & activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key
SEC_API_KEY=your_sec_api_key
```

### 5. Run the API
```bash
python app.py
```
The server will start on http://localhost:8000

## Usage
`/insights` **(POST)**

Submit an investment question and list of tickers, and receive a concise insight.

### Request
```bash
POST /insights
Content-Type: application/json

{
  "query": "How did AAPL perform today and what news influenced the move?",
  "tickers": ["AAPL","MSFT"]
}
```
### Response
```bash
{
  "insights": "Apple closed down 1.8% today after ..."
}
```
*Built with ❤️ for streamlined, data-driven investment decision support*