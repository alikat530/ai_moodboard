from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class ItemIn(BaseModel):
    image_url: HttpUrl
    caption: str
    tags: List[str]
    id: Optional[str] = None

class IndexResponse(BaseModel):
    added: int

class SearchResponseItem(BaseModel):
    id: str
    image_url: HttpUrl
    caption: str
    tags: List[str]
    score: float

class SearchResponse(BaseModel):
    items: List[SearchResponseItem]