from fastapi import FastAPI
from chroma_client import get_collection

app = FastAPI(title="AI Moodboard — Backend")

@app.get("/health")
def health():
    _ = get_collection("items")
    return {"status": "ok", "chroma_collection": "items"}
