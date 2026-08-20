import os
from pathlib import Path
from unstructured.partition.auto import partition

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR

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
}


def load_documents(raw_dir: str = DATA_RAW_DIR) -> list[dict]:
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
            text = _parse_with_unstructured(file)
            if not text:
                print(f"  Warning: No text extracted from {file.name}")
                continue

            text = _clean(text)
            _save_processed(file.stem, text)

            documents.append(
                {
                    "source": file.name,
                    "text": text,
                }
            )
            print(f"  Loaded: {file.name} ({len(text)} chars)")
        except Exception as e:
            print(f"  Error parsing {file.name}: {e}")

    return documents


def _parse_with_unstructured(path: Path) -> str:
    """
    Partitions documents into layout-aware structured elements (titles, narrative text, tables, lists).
    """
    elements = partition(
        filename=str(path),
        strategy="fast",
    )
    
    parts = []
    for el in elements:
        t = (el.text or "").strip()
        if t:
            parts.append(t)
            
    return "\n\n".join(parts)


def _clean(text: str) -> str:
    import re

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _save_processed(stem: str, text: str):
    out = Path(DATA_PROCESSED_DIR) / f"{stem}.txt"
    out.write_text(text, encoding="utf-8")