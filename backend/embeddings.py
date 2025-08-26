import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "text-embedding-3-small"

def embed(text: str):
    """Turn text into a vector embedding."""
    res = client.embeddings.create(model=MODEL, input=text)
    return res.data[0].embedding