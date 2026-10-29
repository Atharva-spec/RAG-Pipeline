"""Automatic synchronization for documents dropped into the raw folder.

The manager is shared by the API and CLI query paths. A background thread
handles normal folder changes, while ``ensure_latest`` lets a query catch up
immediately when a document was added just before the question arrived.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import (
    AUTO_SYNC_ENABLED,
    AUTO_SYNC_INTERVAL_SECONDS,
    DATA_RAW_DIR,
)
from ingestion.document_loader import SUPPORTED_EXTENSIONS
from ingestion.incremental_ingest import sync_incremental
from ingestion.normalizer import compute_file_hash

logger = logging.getLogger(__name__)


def get_folder_state(raw_dir: str = DATA_RAW_DIR) -> Dict[str, str]:
    """Return hashes for supported files currently present in ``raw_dir``."""
    raw_path = Path(raw_dir)
    state: Dict[str, str] = {}
    if not raw_path.exists():
        return state

    for path in raw_path.iterdir():
        if (
            path.is_file()
            and not path.name.startswith(".")
            and not path.name.endswith(".uploading")
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            try:
                state[path.name] = compute_file_hash(path)
            except OSError as exc:
                logger.debug("Could not hash %s during auto-sync: %s", path, exc)
    return state


class FolderAutoSync:
    """Thread-safe, change-aware incremental folder synchronizer."""

    def __init__(
        self,
        raw_dir: str = DATA_RAW_DIR,
        interval_seconds: float = AUTO_SYNC_INTERVAL_SECONDS,
    ):
        self.raw_dir = raw_dir
        self.interval_seconds = max(0.25, float(interval_seconds))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_state: Optional[Dict[str, str]] = None
        self._last_sync_at: Optional[float] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """Start the daemon watcher once; repeated calls are harmless."""
        if not AUTO_SYNC_ENABLED:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="rag-folder-auto-sync",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "Automatic document sync enabled for %s (interval %.2fs)",
                self.raw_dir,
                self.interval_seconds,
            )

    def stop(self) -> None:
        """Stop the daemon watcher and wait briefly for an active sync."""
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        with self._lock:
            self._thread = None

    def ensure_latest(self) -> Dict[str, Any]:
        """Synchronize only when the folder differs from the last known state."""
        if not AUTO_SYNC_ENABLED:
            return {"status": "DISABLED"}
        return self._sync_if_changed(force=False)

    def sync_now(self) -> Dict[str, Any]:
        """Force a synchronized update, used after an API upload completes."""
        if not AUTO_SYNC_ENABLED:
            return {"status": "DISABLED"}
        return self._sync_if_changed(force=True)

    def status(self) -> Dict[str, Any]:
        """Return safe operational state for health diagnostics."""
        with self._lock:
            return {
                "enabled": AUTO_SYNC_ENABLED,
                "running": bool(self._thread and self._thread.is_alive()),
                "interval_seconds": self.interval_seconds,
                "last_sync_at": self._last_sync_at,
                "last_result": self._last_result,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        # Initial synchronization makes files already present at API startup
        # available without a separate ingest command.
        self._sync_if_changed(force=False)
        while not self._stop_event.wait(self.interval_seconds):
            self._sync_if_changed(force=False)

    def _sync_if_changed(self, force: bool) -> Dict[str, Any]:
        current_state = get_folder_state(self.raw_dir)
        with self._lock:
            if not force and self._last_state == current_state:
                return {"status": "UNCHANGED", "chunks_added": 0}

            try:
                result = sync_incremental(self.raw_dir, verbose=False)
            except Exception as exc:  # keep the query service available if indexing fails
                self._last_error = str(exc)
                logger.exception("Automatic document sync failed")
                return {"status": "ERROR", "error": str(exc), "chunks_added": 0}

            self._last_state = get_folder_state(self.raw_dir)
            self._last_sync_at = time.time()
            self._last_result = result
            self._last_error = None
            logger.info(
                "Automatic document sync completed: status=%s chunks_added=%s",
                result.get("status"),
                result.get("chunks_added", 0),
            )
            return result


folder_auto_sync = FolderAutoSync()
