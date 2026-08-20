import json
import re
from pathlib import Path
from typing import List, Optional

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


def chunk_hierarchical(
    documents: List[dict],
    parent_size: int = PARENT_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> List[dict]:
    """
    Hierarchical (Parent-Child) Chunking:
    - Creates large parent context blocks (for LLM generation).
    - Splits each parent block into small child snippets (for high-precision vector search).
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

    for doc in documents:
        doc_source = doc["source"]
        doc_text = doc["text"]

        parent_splits = parent_splitter.split_text(doc_text)
        doc_chunks = []
        global_idx = 0

        for p_idx, parent_text in enumerate(parent_splits):
            clean_parent = parent_text.strip()
            if not clean_parent:
                continue

            parent_id = f"{doc_source}_p{p_idx}"
            child_splits = child_splitter.split_text(clean_parent)

            for c_idx, child_text in enumerate(child_splits):
                clean_child = child_text.strip()
                if not clean_child:
                    continue

                chunk_id = f"{doc_source}_p{p_idx}_c{c_idx}"
                doc_chunks.append(
                    {
                        "id": chunk_id,
                        "chunk_index": global_idx,
                        "text": clean_child,  # Vector index search target
                        "parent_id": parent_id,
                        "parent_text": clean_parent,  # LLM context target
                        "source": doc_source,
                        "strategy": "hierarchical",
                        "char_count": len(clean_child),
                    }
                )
                global_idx += 1

        _save_chunks(doc_source, doc_chunks)
        all_chunks.extend(doc_chunks)
        print(f"  [Hierarchical Chunking] {doc_source} ──► {len(parent_splits)} parents, {len(doc_chunks)} child chunks")

    return all_chunks


def chunk_semantic_sections(
    documents: List[dict],
    max_chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """
    Structural / Section-Aware Chunking:
    - Splits document on Markdown headers (#, ##, ###) and major section titles.
    - Preserves headings and contextual boundaries.
    """
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    header_pattern = r"(?m)^(#{1,6}\s+.+|[A-Z0-9\s]{4,}:|\bModule\s+\d+\b|\bLecture\s+\d+\b)"
    all_chunks = []

    for doc in documents:
        doc_source = doc["source"]
        doc_text = doc["text"]

        # Split on section headings
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
            if not full_section_text:
                continue

            if len(full_section_text) <= max_chunk_size:
                doc_chunks.append(
                    {
                        "id": f"{doc_source}_{global_idx}",
                        "chunk_index": global_idx,
                        "text": full_section_text,
                        "parent_id": None,
                        "parent_text": full_section_text,
                        "section_header": section_header,
                        "source": doc_source,
                        "strategy": "semantic_section",
                        "char_count": len(full_section_text),
                    }
                )
                global_idx += 1
            else:
                sub_splits = sub_splitter.split_text(full_section_text)
                for s in sub_splits:
                    clean_s = s.strip()
                    if not clean_s:
                        continue
                    doc_chunks.append(
                        {
                            "id": f"{doc_source}_{global_idx}",
                            "chunk_index": global_idx,
                            "text": clean_s,
                            "parent_id": None,
                            "parent_text": full_section_text,
                            "section_header": section_header,
                            "source": doc_source,
                            "strategy": "semantic_section",
                            "char_count": len(clean_s),
                        }
                    )
                    global_idx += 1

        _save_chunks(doc_source, doc_chunks)
        all_chunks.extend(doc_chunks)
        print(f"  [Semantic Section Chunking] {doc_source} ──► {len(doc_chunks)} chunks")

    return all_chunks


def chunk_recursive(
    documents: List[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """
    Standard recursive character splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []

    for doc in documents:
        doc_source = doc["source"]
        splits = splitter.split_text(doc["text"])
        doc_chunks = []

        for i, s in enumerate(splits):
            clean_s = s.strip()
            if not clean_s:
                continue
            doc_chunks.append(
                {
                    "id": f"{doc_source}_{i}",
                    "chunk_index": i,
                    "text": clean_s,
                    "parent_id": None,
                    "parent_text": clean_s,
                    "source": doc_source,
                    "strategy": "recursive",
                    "char_count": len(clean_s),
                }
            )

        _save_chunks(doc_source, doc_chunks)
        all_chunks.extend(doc_chunks)
        print(f"  [Recursive Chunking] {doc_source} ──► {len(doc_chunks)} chunks")

    return all_chunks


def chunk_documents(
    documents: List[dict],
    strategy: Optional[str] = None,
) -> List[dict]:
    """
    Primary chunking dispatcher supporting 'hierarchical', 'semantic_section', and 'recursive'.
    """
    Path(DATA_CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    selected_strategy = (strategy or CHUNKING_STRATEGY).lower()

    if selected_strategy == "hierarchical":
        return chunk_hierarchical(documents)
    elif selected_strategy in ("semantic_section", "structural"):
        return chunk_semantic_sections(documents)
    else:
        return chunk_recursive(documents)


def _save_chunks(source: str, chunks: List[dict]):
    stem = Path(source).stem
    out = Path(DATA_CHUNKS_DIR) / f"{stem}_chunks.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
