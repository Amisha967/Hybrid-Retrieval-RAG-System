import re
import uuid
import datetime
from typing import List, Dict, Any, Optional

class TextChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata
        }

class RecursiveCharacterSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        
        final_chunks = []
        raw_splits = self._split_recursive(text, self.separators)
        
        # Merge small pieces with overlap
        current_chunk = []
        current_length = 0
        char_cursor = 0
        
        for piece in raw_splits:
            piece_len = len(piece)
            if not piece.strip():
                continue
                
            if current_length + piece_len > self.chunk_size and current_chunk:
                merged_text = "".join(current_chunk).strip()
                if merged_text:
                    start_pos = text.find(merged_text, max(0, char_cursor - len(merged_text) - 100))
                    if start_pos == -1:
                        start_pos = char_cursor
                    end_pos = start_pos + len(merged_text)
                    final_chunks.append({
                        "text": merged_text,
                        "start_char": start_pos,
                        "end_char": end_pos
                    })
                    char_cursor = end_pos

                # Handle overlap by taking tail tokens
                overlap_buffer = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_buffer.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                current_chunk = overlap_buffer
                current_length = overlap_len

            current_chunk.append(piece)
            current_length += piece_len

        if current_chunk:
            merged_text = "".join(current_chunk).strip()
            if merged_text:
                start_pos = text.find(merged_text, max(0, char_cursor - len(merged_text) - 100))
                if start_pos == -1:
                    start_pos = char_cursor
                end_pos = start_pos + len(merged_text)
                final_chunks.append({
                    "text": merged_text,
                    "start_char": start_pos,
                    "end_char": end_pos
                })

        return final_chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        
        sep = separators[0]
        new_separators = separators[1:]
        
        if sep == "":
            return list(text)
        
        splits = text.split(sep)
        result = []
        for i, s in enumerate(splits):
            piece = s if i == len(splits) - 1 else s + sep
            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                if new_separators:
                    result.extend(self._split_recursive(piece, new_separators))
                else:
                    # Force split if no more separators
                    for j in range(0, len(piece), self.chunk_size):
                        result.append(piece[j : j + self.chunk_size])
        return result

class SlidingWindowSplitter:
    def __init__(self, window_size: int = 500, step_size: int = 400):
        self.window_size = window_size
        self.step_size = max(1, step_size)

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        chunks = []
        length = len(text)
        start = 0
        while start < length:
            end = min(start + self.window_size, length)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end
                })
            start += self.step_size
            if end >= length:
                break
        return chunks

class FixedSizeSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        self.step_size = self.chunk_size - self.chunk_overlap

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        splitter = SlidingWindowSplitter(window_size=self.chunk_size, step_size=self.step_size)
        return splitter.split_text(text)

class DocumentChunker:
    @staticmethod
    def chunk_document(
        text: str,
        doc_id: str,
        filename: str,
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        base_meta = base_metadata or {}
        
        if strategy == "fixed":
            splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy == "sliding_window":
            step = max(1, chunk_size - chunk_overlap)
            splitter = SlidingWindowSplitter(window_size=chunk_size, step_size=step)
        else: # default recursive
            splitter = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        raw_chunks = splitter.split_text(text)
        text_chunks: List[TextChunk] = []

        for idx, item in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_meta = dict(base_meta)
            chunk_meta.update({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "char_length": len(item["text"]),
                "word_count": len(item["text"].split()),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

            text_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=item["text"],
                    chunk_index=idx,
                    start_char=item["start_char"],
                    end_char=item["end_char"],
                    metadata=chunk_meta
                )
            )

        return text_chunks
