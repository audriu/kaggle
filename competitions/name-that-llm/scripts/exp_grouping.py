#!/usr/bin/env python3
"""Experiment: how much is test-time prompt grouping worth?

1. Prompt-grouped 5-fold CV of the public style-aware n-gram LinearSVC (expected ~0.38 balanced accuracy).
2. Oracle grouping: within each held-out prompt (18 passages, one per model) assign labels with the Hungarian
   algorithm on the SVC scores (each model used at most once per prompt), and with Sinkhorn-style balancing.
3. Realistic grouping: cluster held-out passages by content-word TF-IDF (average linkage) and report pair
   precision/recall vs the true prompts, then run the assignment on the recovered clusters.
4. Public-LB-like layout: each prompt keeps a random 40% of its passages.
"""
import re, sys, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "data" / "train.csv")
LABELS = sorted(train.model.unique()); K = len(LABELS)
y = train.model.map({l: i for i, l in enumerate(LABELS)}).values
pid = train.prompt_id.values
T0 = time.time()

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape_text = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))

def featurize(fit_texts, pred_texts):
    A, B = [], []
    for an, lo, hi, mx, w in SPEC:
        tf = shape_text if an == 'shape' else function_text if an == 'function' else str
        actual = 'char' if an == 'shape' else 'word' if an == 'function' else an
        kw = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mx, dtype=np.float32)
        if actual == 'word': kw['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**kw)
        A.append(v.fit_transform([tf(t) for t in fit_texts]) * w); B.append(v.transform([tf(t) for t in pred_texts]) * w)
    return hstack(A, format='csr'), hstack(B, format='csr')

# ---- grouped 5-fold OOF decision scores
uniq = np.array(sorted(set(pid))); order = uniq[np.random.default_rng(42).permutation(len(uniq))]
fold_of = {g: i % 5 for i, g in enumerate(order)}; fold = np.array([fold_of[g] for g in pid])
D = np.zeros((len(y), K), dtype=np.float32)
texts = train.text.values
for k in range(5):
    tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
    Xa, Xb = featurize(texts[tr], texts[va])
    svc = LinearSVC(C=0.5, max_iter=5000, random_state=42).fit(Xa, y[tr])
    D[va] = svc.decision_function(Xb)
    print(f"fold {k} done ({time.time()-T0:.0f}s)", flush=True)
np.savez(ROOT / "tmp" / "oof_stylegram.npz", D=D, y=y, pid=pid)
pred = D.argmax(1)
print(f"\nplain stylegram SVC: balanced acc {balanced_accuracy_score(y, pred):.4f}")
fam = lambda i: LABELS[i].split('-')[0]
print(f"  family-level acc {np.mean([fam(a)==fam(b) for a,b in zip(pred,y)]):.4f}")

def logp(d, temp=1.0):
    z = d / temp; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))

def assign_groups(L, groups):
    """Hungarian within each group: each label at most once per group (if group size <= K)."""
    out = L.argmax(1).copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if 1 < len(i) <= K:
            r, c = linear_sum_assignment(-L[i]); out[i[r]] = c
        elif len(i) > K:   # bigger than 18: allow each label up to ceil(n/K) times
            reps = int(np.ceil(len(i) / K)); Lrep = np.repeat(L[i], reps, axis=1)
            r, c = linear_sum_assignment(-Lrep); out[i[r]] = c // reps
    return out

def sinkhorn(P, groups, alpha=1.0, iters=50):
    out = P.copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]; Q = P[i].copy()
        target = (1 - alpha) * Q.sum(0) + alpha * np.full(K, len(i) / K)
        for _ in range(iters):
            Q *= (target / (Q.sum(0) + 1e-12))[None, :]; Q /= Q.sum(1, keepdims=True) + 1e-12
        out[i] = Q
    return out

L = logp(D, temp=0.5)
for temp in (0.25, 0.5, 1.0):
    Lt = logp(D, temp)
    print(f"temp {temp}: oracle-group Hungarian BA {balanced_accuracy_score(y, assign_groups(Lt, pid)):.4f} | oracle-group Sinkhorn argmax BA {balanced_accuracy_score(y, sinkhorn(np.exp(Lt), pid).argmax(1)):.4f} | global Sinkhorn {balanced_accuracy_score(y, sinkhorn(np.exp(Lt), np.zeros(len(y)), 0.5).argmax(1)):.4f}")

# ---- public-LB-like layout: random 40% of each prompt's passages (oracle groups)
rng = np.random.default_rng(0); res = []
for rep in range(20):
    keep = np.concatenate([rng.choice(np.where(pid == p)[0], max(1, int(round(0.4 * (pid == p).sum()))), replace=False) for p in uniq])
    res.append((balanced_accuracy_score(y[keep], D[keep].argmax(1)), balanced_accuracy_score(y[keep], assign_groups(L[keep], pid[keep]))))
res = np.array(res); print(f"40%-per-prompt layout (oracle groups), 20 draws: plain {res[:,0].mean():.4f} -> Hungarian {res[:,1].mean():.4f}")

# ---- realistic clustering of held-out passages (per fold, full 18 per prompt)
def clusters(ts, thresh):
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(ts)
    Dm = np.clip(1 - (X @ X.T).toarray(), 0, None); np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), method='average'), thresh, criterion='distance')
def pair_prf(g, p):
    same_c = g[:, None] == g[None, :]; same_p = p[:, None] == p[None, :]; off = ~np.eye(len(g), dtype=bool)
    tp = (same_c & same_p & off).sum(); return tp / max((same_c & off).sum(), 1), tp / max((same_p & off).sum(), 1)
for thresh in (0.75, 0.8, 0.85, 0.9):
    bas, prs, sizes = [], [], []
    for k in range(5):
        va = np.where(fold == k)[0]; g = clusters(texts[va], thresh)
        prs.append(pair_prf(g, pid[va])); sizes.append(np.bincount(g)[1:].mean())
        bas.append(balanced_accuracy_score(y[va], assign_groups(L[va], g)))
    prs = np.array(prs); print(f"tfidf clustering thresh {thresh}: pair precision {prs[:,0].mean():.3f} recall {prs[:,1].mean():.3f} mean cluster size {np.mean(sizes):.1f} -> Hungarian BA {np.mean(bas):.4f}")
# per-class recall for plain vs oracle Hungarian
ph = assign_groups(L, pid)
rec = pd.DataFrame({"plain": [np.mean(pred[y == c] == c) for c in range(K)], "oracle_hungarian": [np.mean(ph[y == c] == c) for c in range(K)]}, index=LABELS)
print(rec.round(3).to_string()); print(f"total {time.time()-T0:.0f}s")
