from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LABELS = sorted(["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview",
                 "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6",
                 "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"])
K = len(LABELS)
L2I = {l: i for i, l in enumerate(LABELS)}
FAMILY = np.array([l.split("-")[0] for l in LABELS])


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv")
    df["text"] = df.text.astype(str)
    df["y"] = df.model.map(L2I).astype(int)
    return df


def folds(prompt_ids, seed: int = 0, n: int = 5) -> np.ndarray:
    """Prompt-grouped folds, identical to the public recipes: sorted prompt ids shuffled with RandomState(seed), i % n."""
    prompts = np.array(sorted(set(map(str, prompt_ids))))
    rng = np.random.RandomState(seed)
    rng.shuffle(prompts)
    f = {p: i % n for i, p in enumerate(prompts)}
    return np.array([f[str(p)] for p in prompt_ids])


def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def family_accuracy(y, pred) -> float:
    return float(np.mean(FAMILY[np.asarray(pred)] == FAMILY[np.asarray(y)]))


def log_softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temp
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))
