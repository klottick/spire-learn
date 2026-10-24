"""Append-only JSONL run artifacts in an exclusively created directory."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import tempfile

from spire_learn.artifacts.codec import to_json_value, transition_to_json
from spire_learn.contracts import CaseResult, Transition

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class RunWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        opened = []
        try:
            self._episodes = (run_dir / "episodes.jsonl").open("x", encoding="utf-8")
            opened.append(self._episodes)
            self._metrics = (run_dir / "metrics.jsonl").open("x", encoding="utf-8")
            opened.append(self._metrics)
            self._manifest_updates = (run_dir / "manifest_updates.jsonl").open("x", encoding="utf-8")
            opened.append(self._manifest_updates)
        except BaseException:
            for stream in opened:
                try:
                    stream.close()
                except BaseException:
                    pass
            raise
        self._closed = False

    @classmethod
    def create(cls, root: Path, run_id: str) -> "RunWriter":
        if not isinstance(run_id, str) or _SAFE_ID.fullmatch(run_id) is None or run_id in {".", ".."}:
            raise ValueError("unsafe run_id")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        run_dir.mkdir()
        return cls(run_dir)

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()

    def write_start(self, config: Mapping[str, object], manifest: Mapping[str, object]) -> None:
        self._write_json(self.run_dir / "config.json", {"schema_id": "spire-learn-config-v1", "config": to_json_value(config)})
        try:
            self._write_json(self.run_dir / "manifest.json", {"schema_id": "spire-learn-cases-v1", "manifest": to_json_value(manifest)})
        except BaseException:
            raise

    def _line(self, stream, value: object) -> None:
        json.dump(to_json_value(value), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()

    def write_case(self, result: CaseResult) -> None:
        self._line(self._episodes, {"schema_id": "spire-learn-evaluation-v1", "record": result})

    def write_metric(self, record: Mapping[str, object]) -> None:
        self._line(self._metrics, {"schema_id": record.get("schema_id", "spire-learn-probe-v1"), "record": record})

    def write_transition(self, transition: Transition) -> None:
        self._line(self._metrics, {"schema_id": "spire-learn-transition-v1", "record": transition_to_json(transition)})

    def update_manifest(self, record: Mapping[str, object]) -> None:
        self._line(self._manifest_updates, {"schema_id": "spire-learn-cases-v1", "record": record})

    def finalize(self, summary: Mapping[str, object]) -> None:
        value = {"schema_id": "spire-learn-evaluation-v1", "summary": to_json_value(summary)}
        fd, name = tempfile.mkstemp(prefix=".summary-", suffix=".tmp", dir=self.run_dir)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.run_dir / "summary.json")
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            first_error = None
            for stream in (self._episodes, self._metrics, self._manifest_updates):
                try:
                    stream.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
