from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        
        # Tách câu tốt hơn
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            current_chunk.append(sentence.strip())
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["#", "##", "###", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        if not text.split():
            return []
        return self._split(text, self.separators)
        

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []
        
        if not remaining_separators:
            # Fallback: cắt theo ký tự
            return [
                current_text[i:i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]
        
        sep = remaining_separators[0]
        if sep == "":
            return [
                current_text[i:i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]
        
        parts = current_text.split(sep)
        chunks = []
        current = ""
        
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                piece = part + sep
            else:
                piece = part
            
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current:
                    chunks.extend(self._split(current, remaining_separators[1:]))
                current = piece
        
        if current:
            chunks.extend(self._split(current, remaining_separators[1:]))
        
        return [c.strip() for c in chunks if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)
    


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)

            avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

            result[name] = {
                "chunks": chunks,       
                "count": len(chunks),
                "avg_length": avg_length,
            }

        return result

if __name__ == "__main__":
    import json

    # Đọc file dat02.md đến day07.md
    with open("C:\\D\\AI_in_action\\Day_7\\Day-07-Lab-Data-Foundations\\data\\day02.md", "r", encoding="utf-8") as f:
        dat02 = f.read()
    with open("C:\\D\\AI_in_action\\Day_7\\Day-07-Lab-Data-Foundations\\data\\day03.md", "r", encoding="utf-8") as f:
        day03 = f.read()
    with open("C:\\D\\AI_in_action\\Day_7\\Day-07-Lab-Data-Foundations\\data\\day05.md", "r", encoding="utf-8") as f:
        day05 = f.read()
    with open("C:\\D\\AI_in_action\\Day_7\\Day-07-Lab-Data-Foundations\\data\\day06.md", "r", encoding="utf-8") as f:
        day06 = f.read()
    with open("C:\\D\\AI_in_action\\Day_7\\Day-07-Lab-Data-Foundations\\data\\day07.md", "r", encoding="utf-8") as f:
        day07 = f.read()

    comparator = ChunkingStrategyComparator()
    i=1
    for sample_text in [dat02, day03, day05, day06, day07]:
        comparison = comparator.compare(sample_text)
        # Lưu kết quả vào file JSON
        
        with open(f"chunking_comparison_{i}.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        i=i+1
