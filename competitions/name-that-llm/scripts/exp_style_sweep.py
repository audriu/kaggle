#!/usr/bin/env python3
"""Hyper-parameter sweep of the stylegram SVC (prompt-grouped 5-fold balanced accuracy). CPU job for hoody."""
import re, sys, time, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, K
from ntl.style import shape_text, function_text
tr = load_train(); y = tr.y.values; texts = tr.text.to_numpy(dtype=object); fold = folds(tr.prompt_id.values); T0 = time.time()

def views(spec, fit_texts, pred_texts):
    A, B = [], []
    for an, lo, hi, mx, w, lower, min_df in spec:
        tf = shape_text if an == "shape" else function_text if an == "function" else str
        actual = "char" if an == "shape" else "word" if an == "function" else an
        kw = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=min_df, max_features=mx, dtype=np.float32)
        if actual == "word": kw["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**kw); A.append(v.fit_transform([tf(t) for t in fit_texts]) * w); B.append(v.transform([tf(t) for t in pred_texts]) * w)
    return hstack(A, format="csr"), hstack(B, format="csr")

def cv(spec, C=0.5, clf="svc"):
    D = np.zeros((len(y), K), np.float32)
    for k in range(5):
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]; Xa, Xb = views(spec, texts[a], texts[b])
        m = LinearSVC(C=C, max_iter=5000, random_state=42) if clf == "svc" else LogisticRegression(C=C, max_iter=3000)
        m.fit(Xa, y[a]); D[np.ix_(b, m.classes_)] = m.decision_function(Xb)
    return balanced_accuracy(y, D.argmax(1))

BASE = [("char", 3, 5, 250000, 1.0, False, 2), ("word", 1, 3, 180000, 0.7, False, 2), ("shape", 2, 6, 60000, 0.5, False, 2), ("function", 1, 3, 50000, 0.5, False, 2)]
print("baseline", round(cv(BASE), 4), f"({time.time()-T0:.0f}s)", flush=True)
trials = {
    "C=0.25": (BASE, 0.25, "svc"), "C=1": (BASE, 1.0, "svc"), "C=2": (BASE, 2.0, "svc"),
    "LR C=20": (BASE, 20.0, "lr"), "LR C=50": (BASE, 50.0, "lr"),
    "char 2-5": ([("char", 2, 5, 250000, 1.0, False, 2)] + BASE[1:], 0.5, "svc"),
    "char 3-6": ([("char", 3, 6, 300000, 1.0, False, 2)] + BASE[1:], 0.5, "svc"),
    "char 1-5 400k": ([("char", 1, 5, 400000, 1.0, False, 2)] + BASE[1:], 0.5, "svc"),
    "word w=1.0": ([BASE[0], ("word", 1, 3, 180000, 1.0, False, 2)] + BASE[2:], 0.5, "svc"),
    "word w=0.4": ([BASE[0], ("word", 1, 3, 180000, 0.4, False, 2)] + BASE[2:], 0.5, "svc"),
    "word 1-2": ([BASE[0], ("word", 1, 2, 120000, 0.7, False, 2)] + BASE[2:], 0.5, "svc"),
    "shape w=1.0": (BASE[:2] + [("shape", 2, 6, 60000, 1.0, False, 2), BASE[3]], 0.5, "svc"),
    "function w=1.0": (BASE[:3] + [("function", 1, 3, 50000, 1.0, False, 2)], 0.5, "svc"),
    "function 1-4": (BASE[:3] + [("function", 1, 4, 80000, 0.5, False, 2)], 0.5, "svc"),
    "min_df=1": ([(a, lo, hi, mx, w, lower, 1) for a, lo, hi, mx, w, lower, _ in BASE], 0.5, "svc"),
    "min_df=3": ([(a, lo, hi, mx, w, lower, 3) for a, lo, hi, mx, w, lower, _ in BASE], 0.5, "svc"),
    "no caps": ([("char", 3, 5, 250000, 1.0, False, 2), ("word", 1, 3, 180000, 0.7, False, 2), ("shape", 2, 6, 60000, 0.5, False, 2), ("function", 1, 3, 50000, 0.5, False, 2)][:4], 0.5, "svc"),
    "+ char_wb 2-4": (BASE + [("char_wb", 2, 4, 100000, 0.5, False, 2)], 0.5, "svc"),
    "+ punct-only char 1-3": (BASE + [("punct", 1, 3, 20000, 0.5, False, 2)], 0.5, "svc"),
}
def punct_text(t): return re.sub(r"[A-Za-z0-9]+", "w", t)
import ntl.style as st
_orig = st.shape_text
for name, (spec, C, clf) in trials.items():
    spec2 = []
    for v in spec:
        if v[0] == "punct": spec2.append(("shape", v[1], v[2], v[3], v[4], v[5], v[6])); st.shape_text = punct_text
        else: spec2.append(v)
    # 'punct' reuses the shape slot with a punctuation-only transform; restore afterwards
    try:
        if any(v[0] == "punct" for v in spec):
            globals()["shape_text"] = punct_text
        print(f"{name:22s} {cv(spec2, C, clf):.4f} ({time.time()-T0:.0f}s)", flush=True)
    finally:
        globals()["shape_text"] = _orig; st.shape_text = _orig
