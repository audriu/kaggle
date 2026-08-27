"""Pure-numpy value net for the submitted agent — no torch on the Kaggle worker.

train/value_net.py trains a 2x256 ReLU MLP (torch, CUDA) over the 112 features of
sim/features.py with a delta-residual head: V(state) = money + (net((x - x_mean)
/ x_std) * d_std + d_mean).  Inference-time rollout search (plan section 5 week 3/4)
needs that V inside the submitted single-file main.py, where torch cannot be
assumed — so this module reimplements the forward pass in numpy (three matmuls +
two relus; the checkpoint's numpy port was measured at parity within $0.001 of
torch and ~80 us/eval on the fully loaded training box).

Three loaders, one frozen ValueFn:

  ValueFn.from_checkpoint(path)  dev box: reads the torch ``best.pt`` zip WITHOUT
                                 importing torch — a restricted pickle Unpickler
                                 shims ``torch._utils._rebuild_tensor_v2`` and the
                                 ``torch.*Storage`` persistent ids into numpy
                                 arrays read straight out of the zip.  Any global
                                 outside the small allowlist raises (format drift
                                 fails loudly instead of executing stray pickle).
  ValueFn.from_npz(path_or_file) loads the npz this module itself serialises
                                 (``to_npz_bytes``); exact round-trip of the
                                 checkpoint arrays at float32.
  ValueFn.from_embedded(b64)     the packaged main.py path: decodes a base64 npz
                                 string literal (agent/vnet_weights.py, generated
                                 by scripts/export_vnet.py) — byte-identical to
                                 from_npz on the same payload.

The money index comes from the checkpoint's own feature_names, so a reordered
sim/features.py invalidates old weights at load time instead of silently
mispricing.  Stdlib + numpy only, no __file__, no local imports: the file is
inlined verbatim into dist/main.py, which Kaggle exec()s with cwd only.

Measured (2026-08-27, vnet_cem2 epoch-4 checkpoint, box fully loaded by the
16-proc cem3 run => latencies pessimistic 2-4x): float32 parity vs torch max
$0.0057 / mean $0.0004 over 5760 real shard rows; float16 weights max $5.30 (over
scripts/export_vnet.py's $5 gate, so the export shipped float32). v() single row
40-54 us; v_batch(30) 12-14 us/row with OPENBLAS_NUM_THREADS=1 but ~95 us/row
when numpy's threaded gemm fights the loaded box — if the submitted agent batches
V evals, pin BLAS to 1 thread via env BEFORE numpy import.
"""

from __future__ import annotations

import base64
import io
import pickle
import re
import zipfile

import numpy as np

# torch storage class name (as pickled) -> numpy dtype of its raw bytes.
_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "LongStorage": np.int64,
    "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}

# Non-torch globals a value-net checkpoint legitimately pickles (numpy arrays,
# OrderedDict state_dict). Anything else => refuse to unpickle.
_SAFE_GLOBALS = {
    ("collections", "OrderedDict"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),   # pre-numpy-2 pickles
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _Storage:
    """Placeholder for a torch storage: dtype + key of its raw-bytes zip entry."""

    __slots__ = ("dtype", "key")

    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


def _load_torch_zip(path):
    """torch.save() zip -> python dict, tensors as numpy arrays, without torch.

    Layout (verified against torch 2.13 best.pt): ``<name>/data.pkl`` holds the
    pickled object tree where every tensor is a persistent-id reference
    ``('storage', <Storage class>, key, location, numel)`` plus a
    ``_rebuild_tensor_v2(storage, offset, size, stride, ...)`` call; the storage
    bytes live little-endian at ``<name>/data/<key>``.
    """
    with zipfile.ZipFile(path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]

        def rebuild_tensor(storage, offset, size, stride, *_unused):
            raw = np.frombuffer(zf.read(prefix + "data/" + storage.key),
                                dtype=storage.dtype)
            if not size:                       # 0-d tensor
                return raw[offset].copy()
            itemsize = raw.dtype.itemsize
            view = np.lib.stride_tricks.as_strided(
                raw[offset:], shape=tuple(size),
                strides=tuple(s * itemsize for s in stride))
            return np.ascontiguousarray(view)  # own the memory; zf closes below

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return rebuild_tensor
                if module == "torch" and name in _STORAGE_DTYPES:
                    return _STORAGE_DTYPES[name]
                if (module, name) in _SAFE_GLOBALS:
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(
                    f"unexpected global {module}.{name} in {path}; torch "
                    f"checkpoint format drifted — re-export via scripts/export_vnet.py")

            def persistent_load(self, pid):
                tag, dtype, key = pid[0], pid[1], pid[2]
                if tag != "storage":
                    raise pickle.UnpicklingError(f"unknown persistent id {tag!r}")
                return _Storage(dtype, key)

        return Unpickler(io.BytesIO(zf.read(pkl_name))).load()


class ValueFn:
    """Frozen numpy value function V(features_row) -> predicted final money."""

    def __init__(self, W, B, x_mean, x_std, d_mean, d_std, feature_names,
                 meta=None):
        self.W = [np.ascontiguousarray(w, dtype=np.float32) for w in W]
        self.B = [np.ascontiguousarray(b, dtype=np.float32) for b in B]
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)
        self.d_mean = float(d_mean)
        self.d_std = float(d_std)
        self.feature_names = [str(n) for n in feature_names]
        self.money_idx = self.feature_names.index("money")
        self.meta = dict(meta or {})
        d_in = self.W[0].shape[1]
        if d_in != len(self.feature_names) or d_in != self.x_mean.shape[0]:
            raise ValueError(f"inconsistent widths: W0 {self.W[0].shape}, "
                             f"{len(self.feature_names)} names, "
                             f"x_mean {self.x_mean.shape}")

    # ---- inference -------------------------------------------------------

    def v(self, feats_row):
        """One feature row (list/array of len d_in) -> predicted final money ($)."""
        x = (np.asarray(feats_row, dtype=np.float32) - self.x_mean) / self.x_std
        for w, b in zip(self.W[:-1], self.B[:-1]):
            x = np.maximum(w @ x + b, 0.0)
        out = float((self.W[-1] @ x + self.B[-1])[0])
        return float(feats_row[self.money_idx]) + out * self.d_std + self.d_mean

    def v_batch(self, F):
        """(n, d_in) feature matrix -> (n,) float64 predicted final money ($)."""
        X = np.asarray(F, dtype=np.float32)
        H = (X - self.x_mean) / self.x_std
        for w, b in zip(self.W[:-1], self.B[:-1]):
            H = np.maximum(H @ w.T + b, 0.0)
        out = (H @ self.W[-1].T + self.B[-1])[:, 0].astype(np.float64)
        return X[:, self.money_idx].astype(np.float64) + out * self.d_std + self.d_mean

    # ---- serialisation ---------------------------------------------------

    def to_npz_bytes(self, dtype=np.float32):
        """Serialise to npz bytes; ``dtype`` quantises W/B only (scalers and
        scalars stay full precision — they cost ~1 KB and fp16 there would add
        avoidable error to the standardisation, not just the matmuls)."""
        arrays = {"x_mean": self.x_mean, "x_std": self.x_std,
                  "d_mean": np.float64(self.d_mean),
                  "d_std": np.float64(self.d_std),
                  "feature_names": np.asarray(self.feature_names)}
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            arrays[f"W{i}"] = w.astype(dtype)
            arrays[f"B{i}"] = b.astype(dtype)
        for k, v in self.meta.items():
            arrays[f"meta_{k}"] = np.asarray(str(v))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    # ---- loaders ---------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path):
        """Load train/value_net.py's best.pt (torch zip) without torch."""
        ck = _load_torch_zip(path)
        state = ck["model_state"]
        idxs = sorted({int(m.group(1)) for k in state
                       if (m := re.fullmatch(r"net\.(\d+)\.weight", k))})
        meta = {k: ck[k] for k in ("epoch", "val_mae", "d_in", "hidden")
                if k in ck}
        return cls(W=[state[f"net.{i}.weight"] for i in idxs],
                   B=[state[f"net.{i}.bias"] for i in idxs],
                   x_mean=ck["x_mean"], x_std=ck["x_std"],
                   d_mean=ck["d_mean"], d_std=ck["d_std"],
                   feature_names=ck["feature_names"], meta=meta)

    @classmethod
    def from_npz(cls, path_or_file):
        z = np.load(path_or_file, allow_pickle=False)
        n_layers = sum(1 for k in z.files if re.fullmatch(r"W\d+", k))
        meta = {k[len("meta_"):]: str(z[k]) for k in z.files
                if k.startswith("meta_")}
        return cls(W=[z[f"W{i}"] for i in range(n_layers)],
                   B=[z[f"B{i}"] for i in range(n_layers)],
                   x_mean=z["x_mean"], x_std=z["x_std"],
                   d_mean=float(z["d_mean"]), d_std=float(z["d_std"]),
                   feature_names=z["feature_names"].tolist(), meta=meta)

    @classmethod
    def from_embedded(cls, b64):
        """Decode a VNET_B64 literal (agent/vnet_weights.py) — the packaged
        main.py entry point. Byte-identical to from_npz on the same payload."""
        return cls.from_npz(io.BytesIO(base64.b64decode(b64)))
