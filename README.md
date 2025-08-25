# AI Moodboard Builder — Day 1 Starter

Minimal backend scaffold with FastAPI + persistent Chroma client.

## Setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your OPENAI_API_KEY
uvicorn app:app --reload --port 8000
```

Visit: http://localhost:8000/health → should return `{"status":"ok","chroma_collection":"items"}`

## Day 1 Goal
- Run the backend locally
- Verify Chroma persistence works

## Next Steps
- Add `/index` and `/search` routes
- Seed data with `scripts/seed.py`
