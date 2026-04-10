from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            client = chromadb.Client()

            # Xóa collection cũ để test luôn sạch và dimension đúng
            try:
                client.delete_collection(name=self._collection_name)
            except Exception:
                pass

            # Tạo collection với embedding_function=None để tránh auto-embed
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=None
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        record = {
            "id": f"{doc.id}_{self._next_index}",
            "doc_id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata or {},
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored_records = []
        for r in records:
            score = _dot(query_embedding, r["embedding"])
            scored_records.append({**r, "score": score})
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        if self._use_chroma and self._collection is not None:
            ids = []
            texts = []
            embeddings = []
            metadatas = []
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                texts.append(record["content"])
                embeddings.append(record["embedding"])
                meta = record["metadata"].copy()
                meta["doc_id"] = record["doc_id"]
                metadatas.append(meta)
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            output = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": -distance,          # <<< FIX: convert distance → similarity
                })
            # Đảm bảo thứ tự giống in-memory path
            output.sort(key=lambda x: x["score"], reverse=True)
            return output

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            where = metadata_filter if metadata_filter else None
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
            output = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": -distance,
                })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output

        # in-memory fallback
        if metadata_filter:
            filtered = [
                r for r in self._store
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            filtered = self._store
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        initial_size = self.get_collection_size()
        if self._use_chroma and self._collection is not None:
            self._collection.delete(where={"doc_id": doc_id})
        else:
            self._store = [r for r in self._store if r.get("doc_id") != doc_id]
        return self.get_collection_size() < initial_size