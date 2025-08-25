import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name: str):
    return client.get_or_create_collection(name=name)
