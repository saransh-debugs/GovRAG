"""Split redacted text into chunks with LlamaIndex's SentenceSplitter.

Chunking happens *after* redaction. Chunking first can cut "John | Smith" across two
chunks, and the name finder then sees neither half as a name. Tokens like
[PERSON_3f9a1c2e4b7d9a01] contain no spaces, so the splitter never cuts one.
"""

from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter


class Chunker:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return [c for c in self._splitter.split_text(text) if c.strip()]
