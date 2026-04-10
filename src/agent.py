from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved_chunks = self._store.search(question, top_k=top_k)
        context = "\n\n".join(chunk.get("text") or chunk.get("content", "") for chunk in retrieved_chunks)
        if not context:
            return "Không tìm thấy thông tin trong tài liệu"
        prompt = f"""Use the following context to answer the question. 
If you don't know, just say you don't know.
Context:
{context}
Question: {question}
Answer:"""
        return self._llm_fn(prompt)
