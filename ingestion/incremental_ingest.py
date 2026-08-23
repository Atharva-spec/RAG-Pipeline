import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.settings import DATA_RAW_DIR, POSTGRES_TABLE
from ingestion.chunker import chunk_documents
from ingestion.document_loader import SUPPORTED_EXTENSIONS, load_documents
from ingestion.normalizer import compute_file_hash
from retrieval.vector_store import (
    add_chunks,
    create_index,
    get_connection,
)


def get_indexed_document_hashes() -> Dict[str, str]:
    """
    Queries PostgreSQL to retrieve a mapping of {source_filename: doc_hash}
    for all currently indexed documents.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT to_regclass(%s);",
                    (POSTGRES_TABLE,),
                )
                res = cur.fetchone()
                if not res or not res.get("to_regclass"):
                    return {}

                cur.execute(
                    f"SELECT DISTINCT ON (source) source, doc_hash FROM {POSTGRES_TABLE} WHERE source IS NOT NULL;"
                )
                rows = cur.fetchall()
                return {r["source"]: (r["doc_hash"] or "") for r in rows}
    except Exception as e:
        print(f"  [Incremental Ingest Warning] Could not fetch index state: {e}")
        return {}


def delete_document_chunks(doc_source: str) -> int:
    """
    Deletes all chunks associated with a specific document source from PostgreSQL.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {POSTGRES_TABLE} WHERE source = %s;",
                    (doc_source,),
                )
                return cur.rowcount
    except Exception as e:
        print(f"  [Incremental Ingest Error] Failed to delete chunks for '{doc_source}': {e}")
        return 0


def classify_files(
    raw_dir: str = DATA_RAW_DIR,
    indexed_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Path], List[Path], List[str], List[Path]]:
    """
    Classifies raw directory files against the current OpenSearch state:
    Returns (new_files, modified_files, deleted_sources, unchanged_files).
    """
    if indexed_map is None:
        indexed_map = get_indexed_document_hashes()

    raw_path = Path(raw_dir)
    disk_files = {}
    if raw_path.exists():
        for f in raw_path.iterdir():
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                disk_files[f.name] = (f, compute_file_hash(f))

    new_files = []
    modified_files = []
    unchanged_files = []

    for name, (path_obj, f_hash) in disk_files.items():
        if name not in indexed_map:
            new_files.append(path_obj)
        elif indexed_map[name] != f_hash:
            modified_files.append(path_obj)
        else:
            unchanged_files.append(path_obj)

    # Deleted files: exists in OpenSearch but missing on disk
    deleted_sources = [src for src in indexed_map.keys() if src not in disk_files]

    return new_files, modified_files, deleted_sources, unchanged_files


def sync_incremental(raw_dir: str = DATA_RAW_DIR, verbose: bool = True) -> Dict[str, Any]:
    """
    Performs smart incremental synchronization:
    - Only processes new or modified files.
    - Purges deleted documents from OpenSearch.
    - Skips unchanged documents in <5 milliseconds.
    """
    t_start = datetime.datetime.now()
    create_index()

    indexed_map = get_indexed_document_hashes()
    new_files, modified_files, deleted_sources, unchanged_files = classify_files(raw_dir, indexed_map)

    total_changes = len(new_files) + len(modified_files) + len(deleted_sources)

    if verbose:
        print("\n" + "=" * 65)
        print(" ⚡ Smart Incremental Ingestion Sync")
        print("=" * 65)
        print(f"  • Unchanged (Skipped): {len(unchanged_files)} file(s)")
        print(f"  • New to Index:        {len(new_files)} file(s)")
        print(f"  • Modified:            {len(modified_files)} file(s)")
        print(f"  • Deleted from disk:   {len(deleted_sources)} file(s)")

    if total_changes == 0:
        elapsed = (datetime.datetime.now() - t_start).total_seconds()
        if verbose:
            print(f"  ✓ All documents are up-to-date! Checked in {elapsed:.3f}s")
            print("=" * 65 + "\n")
        return {
            "status": "UP_TO_DATE",
            "new_count": 0,
            "modified_count": 0,
            "deleted_count": 0,
            "unchanged_count": len(unchanged_files),
            "chunks_added": 0,
        }

    # 1. Handle deleted files
    for del_src in deleted_sources:
        count = delete_document_chunks(del_src)
        if verbose:
            print(f"  🗑️ Deleted {count} chunk(s) for removed file: '{del_src}'")

    # 2. Handle modified files (remove stale chunks first)
    for mod_path in modified_files:
        count = delete_document_chunks(mod_path.name)
        if verbose:
            print(f"  🔄 Purged {count} outdated chunk(s) for modified file: '{mod_path.name}'")

    # 3. Ingest new + modified files
    files_to_ingest = new_files + modified_files
    chunks_added = 0

    if files_to_ingest:
        for file_path in files_to_ingest:
            docs = []
            try:
                from ingestion.document_loader import _parse_with_unstructured, _save_processed
                from ingestion.normalizer import normalize_text
                
                raw_text = _parse_with_unstructured(file_path)
                if not raw_text:
                    continue

                clean_text = normalize_text(raw_text)
                _save_processed(file_path.stem, clean_text)
                f_stat = file_path.stat()
                doc_hash = compute_file_hash(file_path)
                last_mod = datetime.datetime.fromtimestamp(f_stat.st_mtime, tz=datetime.timezone.utc).isoformat()

                doc_entry = {
                    "doc_id": f"{file_path.stem}_{doc_hash[:8]}",
                    "doc_hash": doc_hash,
                    "source": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "file_size_bytes": f_stat.st_size,
                    "last_modified": last_mod,
                    "text": clean_text,
                    "char_count": len(clean_text),
                    "word_count": len(clean_text.split()),
                }
                docs.append(doc_entry)
            except Exception as e:
                print(f"  Error loading '{file_path.name}': {e}")
                continue

            if docs:
                chunks = chunk_documents(docs)
                add_chunks(chunks)
                chunks_added += len(chunks)
                if verbose:
                    print(f"  📥 Ingested {len(chunks)} new chunk(s) from '{file_path.name}'")

    elapsed = (datetime.datetime.now() - t_start).total_seconds()
    if verbose:
        print(f"\n  ✓ Sync complete: Added {chunks_added} chunk(s) in {elapsed:.2f}s")
        print("=" * 65 + "\n")

    return {
        "status": "SYNCED",
        "new_count": len(new_files),
        "modified_count": len(modified_files),
        "deleted_count": len(deleted_sources),
        "unchanged_count": len(unchanged_files),
        "chunks_added": chunks_added,
    }
