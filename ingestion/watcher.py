import time
from pathlib import Path
from typing import Dict

from config.settings import DATA_RAW_DIR
from ingestion.incremental_ingest import sync_incremental
from ingestion.normalizer import compute_file_hash
from ingestion.document_loader import SUPPORTED_EXTENSIONS


def get_current_folder_state(raw_dir: str = DATA_RAW_DIR) -> Dict[str, str]:
    """
    Returns a mapping of {filename: sha256_hash} for all supported files in raw_dir.
    """
    raw_path = Path(raw_dir)
    state = {}
    if raw_path.exists():
        for f in raw_path.iterdir():
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    state[f.name] = compute_file_hash(f)
                except Exception:
                    pass
    return state


def start_file_watcher(raw_dir: str = DATA_RAW_DIR, poll_interval: float = 2.0):
    """
    Monitors raw_dir continuously for added, modified, or deleted files.
    Triggers incremental synchronization automatically when changes occur.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" 👀 Live Document Watcher Active")
    print("=" * 70)
    print(f"  • Watching Directory:  '{raw_path.resolve()}'")
    print(f"  • Polling Interval:    {poll_interval}s")
    print("  • Supported Formats:   PDF, DOCX, PPTX, TXT, MD, HTML")
    print("  • Drop, modify, or delete files anytime to auto-sync.")
    print("  • Press Ctrl+C to stop.")
    print("=" * 70 + "\n")

    # Initial sync on startup
    sync_incremental(raw_dir, verbose=True)

    last_state = get_current_folder_state(raw_dir)

    try:
        while True:
            time.sleep(poll_interval)
            current_state = get_current_folder_state(raw_dir)

            if current_state != last_state:
                print(f"\n[Watcher] ⚡ File change detected in '{raw_dir}' at {time.strftime('%X')}")
                sync_incremental(raw_dir, verbose=True)
                last_state = get_current_folder_state(raw_dir)

    except KeyboardInterrupt:
        print("\n\n[Watcher] 🛑 Live Document Watcher stopped by user.\n")
