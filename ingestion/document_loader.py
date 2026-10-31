import datetime
from pathlib import Path
from typing import List
from unstructured.partition.auto import partition

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
from ingestion.normalizer import compute_file_hash, normalize_text

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".xml",
    ".xlsx",
    ".xls",
    ".odt",
    ".rtf",
}


def load_documents(raw_dir: str = DATA_RAW_DIR) -> List[dict]:
    documents = []
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"  Directory not found: {raw_dir}")
        return []

    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    for file in sorted(raw_path.iterdir()):
        if file.name.startswith(".") or not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            # 1. Compute file metadata & SHA-256 hash
            file_stat = file.stat()
            doc_hash = compute_file_hash(file)
            last_modified = datetime.datetime.fromtimestamp(
                file_stat.st_mtime, tz=datetime.timezone.utc
            ).isoformat()

            # 2. Extract layout-aware structured text
            raw_text = _parse_with_unstructured(file)
            if not raw_text:
                print(f"  Warning: No text extracted from {file.name}")
                continue

            # 3. Apply advanced text normalization
            clean_text = normalize_text(raw_text)
            _save_processed(file.stem, clean_text)

            doc_entry = {
                "doc_id": f"{file.stem}_{doc_hash[:8]}",
                "doc_hash": doc_hash,
                "source": file.name,
                "file_type": file.suffix.lower(),
                "file_size_bytes": file_stat.st_size,
                "last_modified": last_modified,
                "text": clean_text,
                "char_count": len(clean_text),
                "word_count": len(clean_text.split()),
            }

            documents.append(doc_entry)
            print(f"  Loaded: {file.name} ({doc_entry['char_count']} chars, {doc_entry['word_count']} words, hash: {doc_hash[:8]})")
        except Exception as e:
            print(f"  Error parsing {file.name}: {e}")

    return documents


def _parse_with_unstructured(path: Path) -> str:
    """
    Partitions documents into layout-aware structured elements (titles, narrative text, tables, lists).
    Includes automatic fallback to pypdf for PDFs if unstructured parser fails or returns empty text.
    """
    try:
        elements = partition(
            filename=str(path),
            strategy="fast",
        )
        parts = []
        for el in elements:
            t = (el.text or "").strip()
            if t:
                parts.append(t)
        if parts:
            return "\n\n".join(parts)
    except Exception as e:
        print(f"  [DocumentLoader Warning] Unstructured parser failed on '{path.name}': {e}")

    # Robust multi-engine fallback for PDFs
    if path.suffix.lower() == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    pages_text.append(f"Page {i+1}\n\n{extracted.strip()}")
            if pages_text:
                print(f"  [DocumentLoader] Fallback parser (pypdf) extracted {len(pages_text)} pages for '{path.name}'.")
                return "\n\n".join(pages_text)
        except Exception as pe:
            print(f"  [DocumentLoader Error] pypdf fallback also failed on '{path.name}': {pe}")

    # Plain-text fallback for formats whose optional Unstructured dependency
    # may be unavailable. This keeps ingestion useful in a minimal install and
    # avoids silently dropping text/markdown/CSV/JSON/XML uploads.
    if path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".rtf"}:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                print(f"  [DocumentLoader] Plain-text fallback used for '{path.name}'.")
                return text
        except Exception as te:
            print(f"  [DocumentLoader Error] Plain-text fallback failed on '{path.name}': {te}")

    return ""


def _save_processed(stem: str, text: str):
    out = Path(DATA_PROCESSED_DIR) / f"{stem}.txt"
    out.write_text(text, encoding="utf-8")
