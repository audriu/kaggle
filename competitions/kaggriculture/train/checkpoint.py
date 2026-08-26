"""Crash-safe checkpointing.

Training here runs for days on a desktop, so "resumable after interruption" has to
survive the ugly cases: Ctrl-C, a reboot, and `kill -9` landing in the middle of a
write. The scheme is the standard one for that:

  1. Write the payload to `<name>.tmp-<pid>`, flush, `os.fsync` the file.
  2. `os.rename` it into place -- atomic within a filesystem, so a reader sees either
     the whole old file or the whole new one, never a torn mix.
  3. Write a fresh `manifest.json` the same way. The manifest is the commit point:
     a checkpoint only counts once its name appears there.
  4. fsync the *directory* so the renames themselves survive a power cut.

Every payload carries a CRC32 of its bytes, so a checkpoint that somehow lands
corrupt is detected on load and skipped in favour of an older one. `load_latest()`
walks backwards through the manifest until something verifies, so a run can always
resume from *some* consistent point.

The payload is whatever the trainer hands over (`dict`), pickled. RNG streams are
included by `capture_rng()` / `restore_rng()` so a resumed run continues on the same
random schedule rather than silently re-rolling.
"""

import json
import os
import pickle
import random
import shutil
import time
import zlib
from pathlib import Path

MANIFEST = "manifest.json"
FORMAT_VERSION = 1


def _fsync_dir(path: Path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def capture_rng():
    """Snapshot every RNG stream the trainer might use."""
    state = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch

        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    """Keeps the last `keep` checkpoints plus an always-current `best` pointer."""

    def __init__(self, directory, keep=5):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    # ---------- manifest ----------

    def _manifest_path(self):
        return self.dir / MANIFEST

    def read_manifest(self):
        p = self._manifest_path()
        if not p.exists():
            return {"format": FORMAT_VERSION, "checkpoints": [], "best": None}
        try:
            m = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {"format": FORMAT_VERSION, "checkpoints": [], "best": None}
        m.setdefault("checkpoints", [])
        m.setdefault("best", None)
        return m

    def _write_manifest(self, m):
        _atomic_write_bytes(self._manifest_path(), json.dumps(m, indent=1).encode())
        _fsync_dir(self.dir)

    # ---------- save / load ----------

    def save(self, step, payload, metrics=None, is_best=False):
        """Persist one checkpoint. Returns its filename."""
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        crc = zlib.crc32(blob)
        name = f"ckpt-{step:08d}.pkl"
        _atomic_write_bytes(self.dir / name, blob)

        m = self.read_manifest()
        m["format"] = FORMAT_VERSION
        m["checkpoints"] = [c for c in m["checkpoints"] if c["name"] != name]
        m["checkpoints"].append({
            "name": name,
            "step": step,
            "crc32": crc,
            "bytes": len(blob),
            "time": time.time(),
            "metrics": metrics or {},
        })
        m["checkpoints"].sort(key=lambda c: c["step"])

        if is_best:
            best_name = "best.pkl"
            _atomic_write_bytes(self.dir / best_name, blob)
            m["best"] = {"name": best_name, "step": step, "crc32": crc, "metrics": metrics or {}}

        # Prune old checkpoints only AFTER the manifest commit, so a crash mid-prune
        # never leaves the manifest pointing at a deleted file.
        keep_names = {c["name"] for c in m["checkpoints"][-self.keep:]}
        m["checkpoints"] = [c for c in m["checkpoints"] if c["name"] in keep_names]
        self._write_manifest(m)

        for f in self.dir.glob("ckpt-*.pkl"):
            if f.name not in keep_names:
                f.unlink(missing_ok=True)
        for f in self.dir.glob("*.tmp-*"):
            if time.time() - f.stat().st_mtime > 3600:
                f.unlink(missing_ok=True)
        return name

    def _load_entry(self, entry):
        p = self.dir / entry["name"]
        if not p.exists():
            return None
        blob = p.read_bytes()
        if zlib.crc32(blob) != entry.get("crc32"):
            return None
        try:
            return pickle.loads(blob)
        except Exception:
            return None

    def load_latest(self):
        """Newest checkpoint that passes its CRC. Returns (step, payload) or (None, None)."""
        m = self.read_manifest()
        for entry in reversed(m["checkpoints"]):
            payload = self._load_entry(entry)
            if payload is not None:
                return entry["step"], payload
        return None, None

    def load_best(self):
        m = self.read_manifest()
        if not m.get("best"):
            return None, None
        payload = self._load_entry(m["best"])
        return (m["best"]["step"], payload) if payload is not None else (None, None)

    def history(self):
        return self.read_manifest()["checkpoints"]
