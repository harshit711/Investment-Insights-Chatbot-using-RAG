from data_loader import setup_data, embed_text, client, fetch_news, fetch_sec_filings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn

index, data_text, embeddings = setup_data()

class InsightsRequest(BaseModel):
    query: str
    tickers: List[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]

class InsightsResponse(BaseModel):
    insights: str
    # news_insights: str
    # sec_insights: str

app = FastAPI(title="Investment Insights Assistant (RAG)")
def semantic_search(query: str, index, data_text: List[str], embeddings:np.ndarray, top_k= 3) -> List[str]:
    """
    Retrieve top-k relevant text snippets for a query.
    """
    query_embed = embed_text(query).astype('float32')
    distances, indices = index.search(np.array([query_embed]), top_k)
    results = [data_text[i] for i in indices[0]]
    return results

def generate_response(query: str, context: str) -> str:
    """
    Call the LLM with context to generate an insight.
    """
    prompt = f"""
    You are an investment insights assistant. Answer the following question based on the provided financial data context:

    Context:
    {context}

    Question: {query}

    Answer clearly and concisely:
    """
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

@app.post("/insights", response_model=InsightsResponse)
def get_insights(request: InsightsRequest):
    try:
        # 1) fetch relevant contexts
        contexts = semantic_search(request.query, index, data_text, embeddings)
        context_str = "\n".join(contexts)

        # 2) generate the LLM answer
        insights = generate_response(request.query, context_str)

        # news_contexts = fetch_news(request.tickers)
        # news_insights = generate_response(request.query, "\n".join(news_contexts))

        # sec_contexts = fetch_sec_filings(request.tickers)
        # sec_insights = generate_response(request.query, "\n".join(sec_contexts))

        # 4) return structured JSON
        return InsightsResponse(
            insights=insights,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )