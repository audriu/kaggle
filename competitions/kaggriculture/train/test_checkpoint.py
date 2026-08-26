"""Crash-safety tests for CheckpointManager -- including a real `kill -9` mid-write."""

import os
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.checkpoint import CheckpointManager, capture_rng, restore_rng  # noqa: E402


def test_roundtrip(d):
    cm = CheckpointManager(d)
    assert cm.load_latest() == (None, None), "empty dir must report no checkpoint"
    cm.save(1, {"theta": [1, 2, 3]}, metrics={"mean": 100})
    cm.save(2, {"theta": [4, 5, 6]}, metrics={"mean": 200}, is_best=True)
    step, p = cm.load_latest()
    assert (step, p["theta"]) == (2, [4, 5, 6]), (step, p)
    step, p = cm.load_best()
    assert (step, p["theta"]) == (2, [4, 5, 6])
    return "roundtrip"


def test_pruning(d):
    cm = CheckpointManager(d, keep=3)
    for i in range(10):
        cm.save(i, {"i": i})
    files = sorted(f.name for f in Path(d).glob("ckpt-*.pkl"))
    assert len(files) == 3, files
    assert cm.load_latest()[0] == 9
    return f"pruning kept {files}"


def test_corruption(d):
    cm = CheckpointManager(d)
    cm.save(1, {"good": True})
    cm.save(2, {"good": True})
    # Corrupt the newest checkpoint; load_latest must fall back to the older one.
    newest = Path(d) / "ckpt-00000002.pkl"
    newest.write_bytes(b"garbage" + newest.read_bytes()[7:])
    step, p = cm.load_latest()
    assert step == 1, f"expected fallback to step 1, got {step}"
    return "corrupt newest -> fell back to previous"


def test_torn_manifest(d):
    cm = CheckpointManager(d)
    cm.save(1, {"x": 1})
    (Path(d) / "manifest.json").write_text('{"checkpoints": [{"name": "ck')  # truncated
    assert cm.load_latest() == (None, None)
    cm.save(2, {"x": 2})  # must recover and keep working
    assert cm.load_latest()[0] == 2
    return "truncated manifest -> recovered on next save"


CHILD = r'''
import sys, os, time, signal
sys.path.insert(0, {parent!r})
from train.checkpoint import CheckpointManager
cm = CheckpointManager({d!r})
cm.save(1, {{"payload": "first"}})
# Big payload so the write is still in flight when the alarm fires.
big = {{"payload": "x" * (40 * 1024 * 1024), "step": 2}}
signal.setitimer(signal.ITIMER_REAL, 0.004)
signal.signal(signal.SIGALRM, lambda *a: os.kill(os.getpid(), signal.SIGKILL))
cm.save(2, big)
print("SURVIVED")
'''


def test_kill9(d):
    parent = str(Path(__file__).resolve().parent.parent)
    src = CHILD.format(parent=parent, d=d)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=60)
    killed = r.returncode == -9
    cm = CheckpointManager(d)
    step, p = cm.load_latest()
    assert step == 1 and p["payload"] == "first", (
        f"after kill -9 mid-write, expected the committed step-1 checkpoint, got step={step}")
    leftovers = list(Path(d).glob("*.tmp-*"))
    return (f"kill -9 mid-write ({'killed' if killed else 'finished early'}); "
            f"recovered step 1 intact, {len(leftovers)} tmp file(s) left to reap")


def test_rng(d):
    random.seed(7)
    [random.random() for _ in range(5)]
    cm = CheckpointManager(d)
    cm.save(1, {"rng": capture_rng()})
    expected = [random.random() for _ in range(3)]
    random.seed(999)  # clobber
    _, p = cm.load_latest()
    restore_rng(p["rng"])
    got = [random.random() for _ in range(3)]
    assert got == expected, (got, expected)
    return "RNG stream resumes identically"


def main():
    tests = [test_roundtrip, test_pruning, test_corruption, test_torn_manifest, test_kill9, test_rng]
    failures = 0
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                print(f"[ OK ] {t.__name__:20s} {t(d)}")
            except AssertionError as e:
                failures += 1
                print(f"[FAIL] {t.__name__:20s} {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
