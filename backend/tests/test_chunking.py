import pytest
from backend.app.core.chunking import (
    RecursiveCharacterSplitter,
    SlidingWindowSplitter,
    FixedSizeSplitter,
    DocumentChunker
)
from backend.app.core.parsers import extract_text_from_file

def test_recursive_character_splitter():
    text = "Paragraph one with some interesting facts.\n\nParagraph two with another set of information.\n\nParagraph three."
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk["text"]) <= 60
        assert "start_char" in chunk
        assert "end_char" in chunk

def test_sliding_window_splitter():
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
    splitter = SlidingWindowSplitter(window_size=30, step_size=20)
    chunks = splitter.split_text(text)
    
    assert len(chunks) >= 2
    assert chunks[0]["start_char"] == 0

def test_fixed_size_splitter():
    text = "The quick brown fox jumps over the lazy dog." * 5
    splitter = FixedSizeSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1

def test_document_chunker_metadata():
    text = "Cloud computing provides elastic on-demand compute resources across multiple availability zones."
    chunks = DocumentChunker.chunk_document(
        text=text,
        doc_id="doc_123",
        filename="cloud.txt",
        strategy="recursive",
        chunk_size=50,
        chunk_overlap=10,
        base_metadata={"author": "DevOps"}
    )
    
    assert len(chunks) >= 1
    first = chunks[0]
    assert first.doc_id == "doc_123"
    assert first.metadata["filename"] == "cloud.txt"
    assert first.metadata["author"] == "DevOps"
    assert "chunk_id" in first.to_dict()

def test_extract_text_plain():
    raw = b"Sample raw document content for testing text parser."
    text, meta = extract_text_from_file(raw, "sample.txt")
    assert text == "Sample raw document content for testing text parser."
    assert meta["file_extension"] == ".txt"
    assert meta["word_count"] == 8
