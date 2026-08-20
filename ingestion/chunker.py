import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DATA_CHUNKS_DIR


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    Path(DATA_CHUNKS_DIR).mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        chunks = []
        for i, s in enumerate(splits):
            clean_s = s.strip()
            if not clean_s:
                continue
            chunks.append(
                {
                    "id": f"{doc['source']}_{i}",
                    "chunk_index": i,
                    "text": clean_s,
                    "source": doc["source"],
                    "char_count": len(clean_s),
                }
            )
        _save_chunks(doc["source"], chunks)
        all_chunks.extend(chunks)
        print(f" Chunked: {doc['source']} -> {len(chunks)} chunks")

    return all_chunks


def _save_chunks(source: str, chunks: list[dict]):
    stem = Path(source).stem
    out = Path(DATA_CHUNKS_DIR) / f"{stem}_chunks.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
