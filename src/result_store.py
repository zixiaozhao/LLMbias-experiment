from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set, Tuple


ResultKey = Tuple[int, str]


class ResultStoreError(RuntimeError):
    """Raised when an existing result file cannot be resumed safely."""


class JsonlResultStore:
    """Append durable JSONL progress and atomically publish complete results."""

    def __init__(
        self,
        final_path: Path,
        *,
        sample_count: int,
        orientations: Iterable[str],
        expected_context: Mapping[str, Any],
    ) -> None:
        self.final_path = Path(final_path)
        self.partial_path = Path(f"{self.final_path}.partial")
        self.orientations = tuple(orientations)
        if not self.orientations:
            raise ValueError("At least one orientation is required.")
        if len(set(self.orientations)) != len(self.orientations):
            raise ValueError("Orientations must be unique.")

        self.expected_context = dict(expected_context)
        self.expected_keys: Set[ResultKey] = {
            (sample_index, orientation)
            for sample_index in range(sample_count)
            for orientation in self.orientations
        }
        self.completed: Set[ResultKey] = set()

    def prepare(self, *, overwrite: bool = False) -> bool:
        """Prepare storage; return True when a complete final file already exists."""
        self.final_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite and self.partial_path.exists():
            self.partial_path.unlink()

        # A partial file takes precedence over an older final file so an interrupted
        # explicit overwrite can resume without discarding its new progress.
        if self.partial_path.exists():
            self.completed = self._load_keys(self.partial_path, repair_trailing=True)
            return False

        if self.final_path.exists() and not overwrite:
            self.completed = self._load_keys(self.final_path, repair_trailing=False)
            self._require_complete(self.final_path)
            return True

        self.partial_path.touch()
        self.completed = set()
        return False

    def is_completed(self, sample_index: int, orientation: str) -> bool:
        return (sample_index, orientation) in self.completed

    def append(self, records: Iterable[Dict[str, Any]]) -> None:
        serialized = []
        new_keys: Set[ResultKey] = set()
        for record in records:
            key = self._validate_record(record)
            if key in self.completed or key in new_keys:
                raise ResultStoreError(f"Duplicate result key {key} for {self.partial_path}.")
            new_keys.add(key)
            serialized.append(json.dumps(record, ensure_ascii=False))

        if not serialized:
            return

        with self.partial_path.open("a", encoding="utf-8", newline="\n") as handle:
            for line in serialized:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.update(new_keys)

    def finalize(self) -> None:
        self._require_complete(self.partial_path)
        os.replace(self.partial_path, self.final_path)

    def _load_keys(self, path: Path, *, repair_trailing: bool) -> Set[ResultKey]:
        completed: Set[ResultKey] = set()
        truncate_at = None
        size = path.stat().st_size

        with path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    is_truncated_tail = handle.tell() == size and not raw_line.endswith(b"\n")
                    if repair_trailing and is_truncated_tail:
                        truncate_at = offset
                        break
                    raise ResultStoreError(
                        f"Invalid JSON in {path} at line {line_number}; "
                        "use --overwrite to start a new result file."
                    ) from exc

                key = self._validate_record(record)
                if key in completed:
                    raise ResultStoreError(f"Duplicate result key {key} in {path}.")
                completed.add(key)

        if truncate_at is not None:
            with path.open("r+b") as handle:
                handle.truncate(truncate_at)
        return completed

    def _validate_record(self, record: Mapping[str, Any]) -> ResultKey:
        for field, expected in self.expected_context.items():
            if field not in record or record[field] != expected:
                raise ResultStoreError(
                    f"Result context mismatch for {field!r}: expected {expected!r}, "
                    f"found {record.get(field)!r}. Use --overwrite to start again."
                )

        sample_index = record.get("sample_index")
        orientation = record.get("orientation")
        if not isinstance(sample_index, int) or not isinstance(orientation, str):
            raise ResultStoreError(
                "Each result must contain an integer sample_index and string orientation."
            )

        key = (sample_index, orientation)
        if key not in self.expected_keys:
            raise ResultStoreError(f"Unexpected result key {key} for {self.final_path}.")
        return key

    def _require_complete(self, path: Path) -> None:
        missing = self.expected_keys - self.completed
        extra = self.completed - self.expected_keys
        if missing or extra:
            preview = sorted(missing)[:5]
            raise ResultStoreError(
                f"Result file {path} is incomplete or incompatible: "
                f"missing={len(missing)}, extra={len(extra)}, sample={preview}. "
                "Resume from its .partial file or use --overwrite."
            )
