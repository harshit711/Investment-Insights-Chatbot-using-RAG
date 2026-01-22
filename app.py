from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn
import time

from data_loader import setup_data, embed_text, client
from metrics.logger import append_jsonl

index, data_text, embeddings = setup_data()

class InsightsRequest(BaseModel):
    query: str
    tickers: List[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]

class InsightsResponse(BaseModel):
    insights: str

app = FastAPI(title="Investment Insights Assistant (RAG)")

def semantic_search_timed(query: str, index, data_text: List[str], top_k=3):
    """
    Returns: (results, retrieval_ms)
    """
    t0 = time.perf_counter()
    query_embed = embed_text(query).astype("float32")
    distances, indices = index.search(np.array([query_embed]), top_k)
    results = [data_text[i] for i in indices[0]]
    retrieval_ms = (time.perf_counter() - t0) * 1000
    return results, retrieval_ms

def generate_response(query: str, context: str) -> str:
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
    e2e_start = time.perf_counter()
    try:
        # 1) retrieval (timed)
        contexts, retrieval_ms = semantic_search_timed(request.query, index, data_text, top_k=3)
        context_str = "\n".join(contexts)

        # 2) generation
        insights = generate_response(request.query, context_str)

        # 3) end-to-end timing
        e2e_ms = (time.perf_counter() - e2e_start) * 1000

        # 4) log metrics (one row per request)
        append_jsonl({
            "ts": time.time(),
            "query_len": len(request.query),
            "tickers": request.tickers,
            "top_k": 3,
            "retrieval_ms": round(retrieval_ms, 2),
            "e2e_ms": round(e2e_ms, 2),
        })

        return InsightsResponse(insights=insights)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)