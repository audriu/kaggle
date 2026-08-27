"""Tests for agent/vnet_infer.py — the numpy value net the submitted agent runs.

What is on trial:
  1. PARITY: ValueFn.from_checkpoint (torch-free zip unpickler) must reproduce
     torch's V = money + (net((x-x_mean)/x_std)*d_std + d_mean) within $1 at
     float32 on >= 5000 REAL shard rows (measured: $0.0057 max — fp32 matmul
     order noise, 3 orders below the net's own $3.1k val MAE).
  2. ROUND-TRIP: from_embedded(b64(npz_bytes)) == from_npz(same bytes on disk)
     EXACTLY (bitwise arrays, bitwise V outputs) — the packaged main.py uses the
     embedded path, dev tooling the file path; they must be the same function.
  3. LATENCY: single-row v() and v_batch(30) timings, printed for the plan's
     rollout-search budget math (~20-30 rollout+V evals per 0.8 s turn). This
     box runs a 16-proc CEM train, so treat printed numbers as 2-4x pessimistic.
  4. NaN GUARD: all-finite V over 10k+ real rows (a single NaN in a search node
     comparison would silently poison an argmax).

best.pt is read into memory ONCE and both loaders parse that snapshot: the
value-net trainer atomically replaces best.pt between epochs, so two separate
opens during a live run could legitimately compare different checkpoints.

Run directly: python train/test_vnet_infer.py
"""

import base64
import importlib.util
import io
import sys
import tempfile
import time
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.vnet_infer import ValueFn  # noqa: E402

RUN_DIR = ROOT / "train" / "runs" / "vnet_cem2"
DATA_DIR = ROOT / "train" / "data_cem2"
N_PARITY = 5760       # 12 shards x 480 rows >= the 5000-row requirement
N_NAN = 10080         # 21 shards


def load_rows(n_rows):
    files = sorted(glob(str(DATA_DIR / "shard-*.npz")))
    assert files, f"no shards under {DATA_DIR}; run train/collect.py first"
    Xs, total = [], 0
    for f in files:
        Xs.append(np.load(f)["features"])
        total += len(Xs[-1])
        if total >= n_rows:
            break
    X = np.concatenate(Xs).astype(np.float32)
    assert len(X) >= n_rows, f"only {len(X)} rows collected so far"
    return X


def torch_reference(raw_ckpt, X):
    import torch
    torch.set_num_threads(2)  # the cem3 run owns the box; stay off its cores
    sys.path.insert(0, str(ROOT / "train"))
    from value_net import ValueNet
    ck = torch.load(io.BytesIO(raw_ckpt), map_location="cpu", weights_only=False)
    net = ValueNet(ck["d_in"], ck["hidden"])
    net.load_state_dict(ck["model_state"])
    net.eval()
    x_mean = np.asarray(ck["x_mean"], dtype=np.float32)
    x_std = np.asarray(ck["x_std"], dtype=np.float32)
    money = X[:, list(ck["feature_names"]).index("money")]
    with torch.no_grad():
        out = net(torch.from_numpy((X - x_mean) / x_std)).numpy()
    return money + (out * ck["d_std"] + ck["d_mean"])


def main():
    raw = (RUN_DIR / "best.pt").read_bytes()
    vf = ValueFn.from_checkpoint(io.BytesIO(raw))
    X = load_rows(N_NAN)
    Xp = X[:N_PARITY]

    # --- 1. parity vs torch, float32 path -------------------------------
    v_ref = torch_reference(raw, Xp)
    v_np = vf.v_batch(Xp)
    err = np.abs(v_np - v_ref)
    assert err.max() < 1.0, f"float32 parity broke: max ${err.max():.4f}"
    print(f"parity (from_checkpoint, fp32) vs torch: max ${err.max():.4f} "
          f"mean ${err.mean():.4f} over {len(Xp)} rows  OK")
    # v() must agree with v_batch on the same rows (different BLAS path)
    v_single = np.array([vf.v(r) for r in Xp[:200]])
    d = np.abs(v_single - v_np[:200]).max()
    assert d < 0.01, f"v() vs v_batch drift ${d:.4f}"
    print(f"v() vs v_batch: max ${d:.6f} over 200 rows  OK")

    # --- 2. exact npz <-> embedded round-trip ---------------------------
    payload = vf.to_npz_bytes(np.float32)
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "vnet.npz"
        npz_path.write_bytes(payload)
        vf_file = ValueFn.from_npz(npz_path)
    vf_emb = ValueFn.from_embedded(base64.b64encode(payload).decode("ascii"))
    for a, b in zip(vf_file.W + vf_file.B + [vf_file.x_mean, vf_file.x_std],
                    vf_emb.W + vf_emb.B + [vf_emb.x_mean, vf_emb.x_std]):
        assert np.array_equal(a, b), "embedded arrays != npz arrays"
    assert (vf_file.d_mean, vf_file.d_std, vf_file.feature_names) == \
           (vf_emb.d_mean, vf_emb.d_std, vf_emb.feature_names)
    assert np.array_equal(vf_file.v_batch(Xp), vf_emb.v_batch(Xp)), \
        "embedded V != npz V"
    # and the fp32 round-trip is bitwise the checkpoint itself
    assert np.array_equal(vf_emb.v_batch(Xp), v_np), "round-trip changed fp32 V"
    print("from_embedded == from_npz: arrays and V bitwise equal  OK")

    # --- generated agent/vnet_weights.py, if exported -------------------
    weights_py = ROOT / "agent" / "vnet_weights.py"
    if weights_py.exists():
        spec = importlib.util.spec_from_file_location("vnet_weights", weights_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        vf_ship = ValueFn.from_embedded(mod.VNET_B64)
        e = np.abs(vf_ship.v_batch(Xp) - v_ref).max()
        assert e < 5.0, f"shipped weights parity ${e:.2f} >= $5"
        print(f"agent/vnet_weights.py: parity vs torch max ${e:.4f}  OK")
    else:
        vf_ship = vf_emb
        print("agent/vnet_weights.py missing (run scripts/export_vnet.py); "
              "latency/NaN checks use the in-memory round-trip instead")

    # --- 3. latency (loaded-box numbers; 2-4x pessimistic) --------------
    rows = [list(map(float, r)) for r in Xp[:200]]
    t0 = time.perf_counter()
    for _ in range(25):
        for r in rows:
            vf_ship.v(r)
    per_single = (time.perf_counter() - t0) / (25 * len(rows))
    B30 = np.ascontiguousarray(Xp[:30])
    t0 = time.perf_counter()
    for _ in range(2000):
        vf_ship.v_batch(B30)
    per_batch = (time.perf_counter() - t0) / 2000
    print(f"latency: v() {per_single * 1e6:.0f} us/row; "
          f"v_batch(30) {per_batch * 1e6:.0f} us = "
          f"{per_batch / 30 * 1e6:.1f} us/row  (loaded box)")

    # --- 4. NaN guard ----------------------------------------------------
    V = vf_ship.v_batch(X)
    assert len(V) == len(X) and np.isfinite(V).all(), "non-finite V on real rows"
    print(f"NaN guard: {len(X)} rows all finite "
          f"(V range ${V.min():.0f}..${V.max():.0f})  OK")

    print("\nall vnet_infer tests passed")


if __name__ == "__main__":
    main()
