import hashlib
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Standardizes raw extracted document text:
    1. Unicode NFKC normalization (smart quotes, dashes, non-breaking spaces \u00a0, zero-width chars).
    2. Repairs split hyphenated words across line breaks (e.g. 'man-\nager' -> 'manager').
    3. Cleans up whitespace while preserving paragraph breaks.
    4. Strips repeated standalone page numbers / boilerplate markers.
    """
    if not text:
        return ""

    # 1. Unicode Normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace non-standard whitespace / quotes / dashes
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Smart quotes & dashes
    text = re.sub(r'[\u2018\u2019\u201a\u201b`]', "'", text)
    text = re.sub(r'[\u201c\u201d\u201e\u201f]', '"', text)
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)

    # 3. Repair hyphenated word breaks at end of lines (e.g. 'manage-\nment' -> 'management')
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)', r'\1\2', text)

    # 4. Clean line-by-line: trim lines and remove standalone page numbering lines
    cleaned_lines = []
    for line in text.split("\n"):
        trimmed = line.strip()
        # Filter out standalone page numbers like "Page 1 of 10" or just "12"
        if re.match(r'^(page\s+\d+(\s+of\s+\d+)?|\d{1,4})$', trimmed, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(trimmed)

    # Reconstruct text
    text = "\n".join(cleaned_lines)

    # 5. Compress excess blank lines (>2 newlines -> 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. Compress multiple horizontal spaces to a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def compute_content_hash(text: str) -> str:
    """
    Computes a deterministic SHA-256 hash of normalized text for chunk-level deduplication.
    """
    # Normalize for hashing: lowercased and single-spaced
    simplified = " ".join(text.lower().split())
    return hashlib.sha256(simplified.encode("utf-8")).hexdigest()[:16]


def compute_file_hash(file_path: Path) -> str:
    """
    Computes SHA-256 hash of raw file content to detect file changes.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]
