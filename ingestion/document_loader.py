import os
from pathlib import Path
from pypdf import PdfReader
from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR


def load_documents(raw_dir: str = DATA_RAW_DIR) -> list[dict]:
    documents = []

    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    for file in Path(raw_dir).iterdir():
        if file.suffix == ".pdf":
            text = _load_pdf(file)

        elif file.suffix == ".txt":
            text = file.read_text(encoding="utf-8")

        else:
            continue

        text = _clean(text)

        _save_processed(file.stem, text)

        documents.append(
            {
                "source": file.name,
                "text": text,
            }
        )

        print(f"Loaded: {file.name} ({len(text)} chars)")

    return documents


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _clean(text: str) -> str:
    import re

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _save_processed(stem: str, text: str):
    out = Path(DATA_PROCESSED_DIR) / f"{stem}.txt"
    out.write_text(text, encoding="utf-8")