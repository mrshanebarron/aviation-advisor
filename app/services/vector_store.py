import chromadb
from app.config import CHROMA_PERSIST_PATH

_client = None
_collection = None


def get_collection():
    """Get or create the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        _collection = _client.get_or_create_collection(
            name="aviation_docs",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(chunks: list[dict], source_filename: str):
    """Add document chunks to the vector store."""
    collection = get_collection()

    # Remove existing chunks from this source first (re-upload support)
    existing = collection.get(where={"source": source_filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{source_filename}_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # ChromaDB handles embedding automatically with its default model
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(chunks)


def query_documents(query: str, n_results: int = 5) -> list[dict]:
    """Query the vector store for relevant document chunks."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return matches


def get_document_sources() -> list[dict]:
    """Get list of all unique document sources and their chunk counts."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    all_data = collection.get()
    sources = {}
    for meta in all_data["metadatas"]:
        source = meta["source"]
        if source not in sources:
            sources[source] = {"filename": source, "chunks": 0}
        sources[source]["chunks"] += 1

    return list(sources.values())


def delete_document(source_filename: str) -> int:
    """Delete all chunks for a given source document."""
    collection = get_collection()

    existing = collection.get(where={"source": source_filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])

    return 0


def get_total_chunks() -> int:
    """Get total number of chunks in the collection."""
    return get_collection().count()
