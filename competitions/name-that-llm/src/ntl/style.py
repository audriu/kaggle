"""CPU members: style-aware n-gram LinearSVC (four TF-IDF views) and stylometric features + LightGBM."""
from __future__ import annotations
import re, numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.svm import LinearSVC
from .common import K

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250_000, 1.0), ("word", 1, 3, 180_000, 0.7), ("shape", 2, 6, 60_000, 0.5), ("function", 1, 3, 50_000, 0.5)]


def shape_text(t: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t: str) -> str:
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))


class Stylegram:
    """Four TF-IDF views + LinearSVC. decision() returns (n, K) margins (classes absent in fit get -inf-like scores)."""

    def __init__(self, C: float = 0.5):
        self.C = C

    def _feats(self, fit_texts, pred_texts):
        A, B = [], []
        for an, lo, hi, mx, w in VIEWS:
            tf = shape_text if an == "shape" else function_text if an == "function" else str
            actual = "char" if an == "shape" else "word" if an == "function" else an
            kw = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mx, dtype=np.float32)
            if actual == "word":
                kw["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
            v = TfidfVectorizer(**kw)
            A.append(v.fit_transform([tf(t) for t in fit_texts]) * w)
            B.append(v.transform([tf(t) for t in pred_texts]) * w)
        return hstack(A, format="csr"), hstack(B, format="csr")

    def fit_decision(self, fit_texts, fit_y, pred_texts, sample_weight=None) -> np.ndarray:
        Xa, Xb = self._feats(fit_texts, pred_texts)
        svc = LinearSVC(C=self.C, max_iter=5000, random_state=42).fit(Xa, fit_y, sample_weight=sample_weight)
        d = np.full((Xb.shape[0], K), -10.0, dtype=np.float32)
        d[:, svc.classes_] = svc.decision_function(Xb)
        return d


PUNCT = list(",.;:!?()[]\"'-") + ["—", "–", "‘", "’", "“", "”", "…", "*", "/", "%"]
FUNC = sorted(ENGLISH_STOP_WORDS)
SENT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z’']+")


def stylometric_features(t: str) -> list[float]:
    words = WORD.findall(t)
    nw, nc = max(len(words), 1), max(len(t), 1)
    sents = [x for x in SENT.split(t.strip()) if x]
    sl = np.array([len(WORD.findall(x)) for x in sents]) if sents else np.array([0])
    wl = np.array([len(x) for x in words]) if words else np.array([0])
    lw = [x.lower() for x in words]
    f = [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(), (wl >= 9).mean(), len(set(lw)) / nw,
         sum(c.isupper() for c in t) / nc, sum(c.isdigit() for c in t) / nc, sum(ord(c) > 127 for c in t) / nc, t.count(", ") / nw,
         t.count(" and ") / nw, t.count(" but ") / nw,
         sum(1 for x in sents if x[:1].isupper() and x.split()[0].lower() in ("this", "these", "it", "they")) / max(len(sents), 1),
         sum(1 for x in words if x.endswith("ly")) / nw, sum(1 for x in words if x.endswith("ing")) / nw, sum(1 for x in words if x.endswith("tion")) / nw]
    f += [t.count(q) / nw for q in PUNCT]
    cnt = {}
    for x in lw:
        cnt[x] = cnt.get(x, 0) + 1
    f += [cnt.get(x, 0) / nw for x in FUNC]
    return f


def lgb_proba(fit_texts, fit_y, pred_texts, seed: int = 0) -> np.ndarray:
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=8)
    clf.fit(np.array([stylometric_features(t) for t in fit_texts], dtype=np.float32), fit_y)
    p = np.full((len(pred_texts), K), 1e-6, dtype=np.float32)
    p[:, clf.classes_] = clf.predict_proba(np.array([stylometric_features(t) for t in pred_texts], dtype=np.float32))
    return p / p.sum(1, keepdims=True)
