from fastapi import FastAPI
from chroma_client import get_collection
from fastapi import HTTPException, Query, Body
from embeddings import embed
from models import ItemIn, IndexResponse, SearchResponse, SearchResponseItem
from typing import List

app = FastAPI(title="AI Moodboard — Backend")

@app.get("/health")
def health():
    _ = get_collection("items")
    return {"status": "ok", "chroma_collection": "items"}

COLL = "items"

@app.post("/index", response_model=IndexResponse)
def index_items(items: List[ItemIn]):
    col = get_collection(COLL)

    ids, docs, metas, embeds = [], [], [], []
    for it in items:
        # Simple doc string: caption + tags
        doc = f"{it.caption}\nTAGS: {', '.join(it.tags)}"
        vec = embed(doc)
        _id = it.id or f"itm_{abs(hash(doc))}"

        ids.append(_id)
        docs.append(doc)
        metas.append({"image_url": str(it.image_url),"tags_csv": ", ".join(it.tags)})
        embeds.append(vec)

    try:
        col.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma upsert failed: {e}")

    return IndexResponse(added=len(ids))


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., description="Text query"), k: int = 10):
    col = get_collection(COLL)
    qv = embed(q)

    try:
        res = col.query(
            query_embeddings=[qv],
            n_results=k,
            include=["metadatas","documents","distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    items = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        score = 1.0 / (1.0 + float(dist))
        _id = ids[idx] if idx < len(ids) else f"res_{idx}"
        items.append(SearchResponseItem(
            id=_id,
            image_url=meta["image_url"],
            caption=doc.split("\nTAGS:")[0].strip(),
            tags=meta.get("tags", []) or (meta.get("tags_csv", "").split(", ") if meta.get("tags_csv") else []),
            score=score
        ))
    return SearchResponse(items=items)

@app.post("/blend", response_model=SearchResponse)
def blend(item_ids: List[str] = Body(...), k: int = 10):
    """
    Moodboard blend: take a list of item IDs, average their embeddings,
    and search for nearest neighbors.
    """
    col = get_collection(COLL)

    # Fetch stored embeddings
    res = col.get(ids=item_ids, include=["embeddings"])
    vecs = res.get("embeddings", [])

    if not vecs:
        raise HTTPException(status_code=404, detail="No embeddings found for given IDs")

    # Average vectors (the centroid)
    avg_vec = [sum(vals)/len(vals) for vals in zip(*vecs)]

    # Query with the blend vector
    results = col.query(
        query_embeddings=[avg_vec],
        n_results=k,
        include=["metadatas", "documents", "distances"]
    )

    ids   = results.get("ids", [[]])[0]
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    items_out = []
    
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        score = 1.0 / (1.0 + float(dist))
        _id = ids[idx] if idx < len(ids) else f"res_{idx}"
        items_out.append(SearchResponseItem(
            id=_id,
            image_url=meta["image_url"],
            caption=doc.split("\nTAGS:")[0].strip(),
            tags=meta.get("tags", []) or (meta.get("tags_csv", "").split(", ") if meta.get("tags_csv") else []),
            score=score
        ))
    
    selected = set(item_ids)  # the ids you blended
    filtered = [it for it in items_out if it.id not in selected]
    return SearchResponse(items=filtered[:k])

@app.get("/items")
def items(n: int = 10):
    col = get_collection(COLL)
    peek = col.peek(n)
    # peek returns {"ids":[...], "documents":[...], "metadatas":[...]}
    out = []
    for i, _id in enumerate(peek.get("ids", [])):
        meta = peek["metadatas"][i]
        doc  = peek["documents"][i]
        out.append({
            "id": _id,
            "image_url": meta.get("image_url"),
            "caption": doc.split("\nTAGS:")[0].strip()
        })
    return {"items": out}