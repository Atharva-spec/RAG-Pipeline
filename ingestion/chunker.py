import json
import re
from pathlib import Path
from typing import List, Optional, Set

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_STRATEGY,
    DATA_CHUNKS_DIR,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from ingestion.normalizer import compute_content_hash, normalize_text


def chunk_hierarchical(
    documents: List[dict],
    parent_size: int = PARENT_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
    deduplicate: bool = True,
) -> List[dict]:
    """
    Hierarchical (Parent-Child) Chunking with content deduplication and rich metadata:
    - Large parent context blocks (for LLM context).
    - Small child snippets (for vector search).
    - Deduplicates identical repeated chunks.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    seen_content_hashes: Set[str] = set()
    skipped_duplicates = 0

    for doc in documents:
        doc_source = doc["source"]
        doc_text = doc["text"]
        doc_id = doc.get("doc_id", doc_source)
        doc_hash = doc.get("doc_hash", "")
        file_type = doc.get("file_type", Path(doc_source).suffix.lower())
        file_size = doc.get("file_size_bytes", 0)
        last_modified = doc.get("last_modified", "")

        parent_splits = parent_splitter.split_text(doc_text)
        doc_chunks = []
        global_idx = 0

        for p_idx, parent_text in enumerate(parent_splits):
            clean_parent = normalize_text(parent_text)
            if not clean_parent:
                continue

            parent_id = f"{doc_source}_p{p_idx}"
            child_splits = child_splitter.split_text(clean_parent)

            for c_idx, child_text in enumerate(child_splits):
                clean_child = normalize_text(child_text)
                if not clean_child:
                    continue

                content_hash = compute_content_hash(clean_child)

                # Deduplication filter
                if deduplicate:
                    if content_hash in seen_content_hashes:
                        skipped_duplicates += 1
                        continue
                    seen_content_hashes.add(content_hash)

                chunk_id = f"{doc_source}_p{p_idx}_c{c_idx}"
                doc_chunks.append(
                    {
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "doc_hash": doc_hash,
                        "chunk_index": global_idx,
                        "total_chunks": 0,  # updated below
                        "text": clean_child,
                        "content_hash": content_hash,
                        "parent_id": parent_id,
                        "parent_text": clean_parent,
                        "source": doc_source,
                        "file_type": file_type,
                        "file_size_bytes": file_size,
                        "last_modified": last_modified,
                        "strategy": "hierarchical",
                        "char_count": len(clean_child),
                        "word_count": len(clean_child.split()),
                    }
                )
                global_idx += 1

        # Populate total_chunks
        for ch in doc_chunks:
            ch["total_chunks"] = len(doc_chunks)

        _save_chunks(doc_source, doc_chunks)
        all_chunks.extend(doc_chunks)
        print(f"  [Hierarchical Chunking] {doc_source} ──► {len(parent_splits)} parents, {len(doc_chunks)} unique child chunks")

    if skipped_duplicates > 0:
        print(f"  [Deduplication] Filtered out {skipped_duplicates} duplicate chunk(s)")

    return all_chunks


def chunk_semantic_sections(
    documents: List[dict],
    max_chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    deduplicate: bool = True,
) -> List[dict]:
    """
    Structural / Section-Aware Chunking with metadata & deduplication.
    """
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    header_pattern = r"(?m)^(#{1,6}\s+.+|[A-Z0-9\s]{4,}:|\bModule\s+\d+\b|\bLecture\s+\d+\b)"
    all_chunks = []
    seen_content_hashes: Set[str] = set()
    skipped_duplicates = 0

    for doc in documents:
        doc_source = doc["source"]
        doc_text = doc["text"]
        doc_id = doc.get("doc_id", doc_source)
        doc_hash = doc.get("doc_hash", "")
        file_type = doc.get("file_type", Path(doc_source).suffix.lower())
        file_size = doc.get("file_size_bytes", 0)
        last_modified = doc.get("last_modified", "")

        raw_sections = re.split(header_pattern, doc_text)
        sections = []
        current_header = "General"

        i = 0
        while i < len(raw_sections):
            part = raw_sections[i].strip()
            if not part:
                i += 1
                continue

            if re.match(header_pattern, part):
                current_header = part
                content = raw_sections[i + 1].strip() if i + 1 < len(raw_sections) else ""
                sections.append((current_header, content))
                i += 2
            else:
                sections.append((current_header, part))
                i += 1

        doc_chunks = []
        global_idx = 0

        for section_header, section_content in sections:
            full_section_text = f"{section_header}\n{section_content}".strip() if section_header != "General" else section_content
            clean_section = normalize_text(full_section_text)
            if not clean_section:
                continue

            if len(clean_section) <= max_chunk_size:
                content_hash = compute_content_hash(clean_section)
                if deduplicate:
                    if content_hash in seen_content_hashes:
                        skipped_duplicates += 1
                        continue
                    seen_content_hashes.add(content_hash)

                doc_chunks.append(
                    {
                        "id": f"{doc_source}_{global_idx}",
                        "doc_id": doc_id,
                        "doc_hash": doc_hash,
                        "chunk_index": global_idx,
                        "total_chunks": 0,
                        "text": clean_section,
                        "content_hash": content_hash,
                        "parent_id": None,
                        "parent_text": clean_section,
                        "section_header": section_header,
                        "source": doc_source,
                        "file_type": file_type,
                        "file_size_bytes": file_size,
                        "last_modified": last_modified,
                        "strategy": "semantic_section",
                        "char_count": len(clean_section),
                        "word_count": len(clean_section.split()),
                    }
                )
                global_idx += 1
            else:
                sub_splits = sub_splitter.split_text(clean_section)
                for s in sub_splits:
                    clean_s = normalize_text(s)
                    if not clean_s:
                        continue

                    content_hash = compute_content_hash(clean_s)
                    if deduplicate:
                        if content_hash in seen_content_hashes:
                            skipped_duplicates += 1
                            continue
                        seen_content_hashes.add(content_hash)

                    doc_chunks.append(
                        {
                            "id": f"{doc_source}_{global_idx}",
                            "doc_id": doc_id,
                            "doc_hash": doc_hash,
                            "chunk_index": global_idx,
                            "total_chunks": 0,
                            "text": clean_s,
                            "content_hash": content_hash,
                            "parent_id": None,
                            "parent_text": clean_section,
                            "section_header": section_header,
                            "source": doc_source,
                            "file_type": file_type,
                            "file_size_bytes": file_size,
                            "last_modified": last_modified,
                            "strategy": "semantic_section",
                            "char_count": len(clean_s),
                            "word_count": len(clean_s.split()),
                        }
                    )
                    global_idx += 1

        for ch in doc_chunks:
            ch["total_chunks"] = len(doc_chunks)

        _save_chunks(doc_source, doc_chunks)
        all_chunks.extend(doc_chunks)
        print(f"  [Semantic Section Chunking] {doc_source} ──► {len(doc_chunks)} chunks")

    if skipped_duplicates > 0:
        print(f"  [Deduplication] Filtered out {skipped_duplicates} duplicate chunk(s)")

    return all_chunks


def chunk_recursive(
    documents: List[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    deduplicate: bool = True,
) -> List[dict]:
    """
    Standard recursive character splitter with deduplication and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    seen_content_hashes: Set[str] = set()
    skipped_duplicates = 0

    for doc in documents:
        doc_source = doc["source"]
        doc_id = doc.get("doc_id", doc_source)
        doc_hash = doc.get("doc_hash", "")
        file_type = doc.get("file_type", Path(doc_source).suffix.lower())
        file_size = doc.get("file_size_bytes", 0)
        last_modified = doc.get("last_modified", "")

        splits = splitter.split_text(doc["text"])
        doc_chunks = []

        for i, s in enumerate(splits):
            clean_s = normalize_text(s)
            if not clean_s:
                continue

            content_hash = compute_content_hash(clean_s)
            if deduplicate:
                if content_hash in seen_content_hashes:
                    skipped_duplicates += 1
                    continue
                seen_content_hashes.add(content_hash)

            doc_chunks.append(
                {
                    "id": f"{doc_source}_{i}",
                    "doc_id": doc_id,
                    "doc_hash": doc_hash,
                    "chunk_index": i,
                    "total_chunks": 0,
                    "text": clean_s,
                    "content_hash": content_hash,
                    "parent_id": None,
                    "parent_text": clean_s,
                    "source": doc_source,
                    "file_type": file_type,
                    "file_size_bytes": file_size,
                    "last_modified": last_modified,
                    "strategy": "recursive",
                    "char_count": len(clean_s),
                    "word_count": len(clean_s.split()),
                }
            )

        for ch in doc_chunks:
            ch["total_chunks"] = len(doc_chunks)

        _save_chunks(doc_source, doc_chunks)
        all_chunks.extend(doc_chunks)
        print(f"  [Recursive Chunking] {doc_source} ──► {len(doc_chunks)} chunks")

    if skipped_duplicates > 0:
        print(f"  [Deduplication] Filtered out {skipped_duplicates} duplicate chunk(s)")

    return all_chunks


def chunk_documents(
    documents: List[dict],
    strategy: Optional[str] = None,
    deduplicate: bool = True,
) -> List[dict]:
    """
    Primary chunking dispatcher supporting 'hierarchical', 'semantic_section', and 'recursive'
    with chunk-level deduplication and metadata enrichment.
    """
    Path(DATA_CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    selected_strategy = (strategy or CHUNKING_STRATEGY).lower()

    if selected_strategy == "hierarchical":
        return chunk_hierarchical(documents, deduplicate=deduplicate)
    elif selected_strategy in ("semantic_section", "structural"):
        return chunk_semantic_sections(documents, deduplicate=deduplicate)
    else:
        return chunk_recursive(documents, deduplicate=deduplicate)


def _save_chunks(source: str, chunks: List[dict]):
    stem = Path(source).stem
    out = Path(DATA_CHUNKS_DIR) / f"{stem}_chunks.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
