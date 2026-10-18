# Name That LLM - submission kernel (my own pipeline).
# Members: stylegram TF-IDF LinearSVC + stylometric LightGBM (trained here on the competition data), fine-tuned encoder(s)
# loaded from my training-kernel outputs. Log-linear blend with weights fitted offline on prompt-grouped OOF predictions,
# then label-free test-time steps: content-word TF-IDF clustering of the test passages into recovered prompts,
# per-cluster Sinkhorn balancing and a per-cluster assignment (each model at most once per prompt).
import glob, json, os, re, time
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import hstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.svm import LinearSVC

T0 = time.time()
BLEND = {"members": ["debv3l_L24_6", "debv3l_L24_6_x", "desk_L23ms", "desk_L23ms_x", "electl_L21_24s_x", "q25_7b_L6_20_x", "q8lora_v1", "q8lora_s7", "q8lora_x1", "style_lgb", "stylegram_x", "q25_7b_L6_20", "electl_L21_24s", "q25lora_x1"], "weights": [0.0599, 0.1164, 0.0584, 0.1646, 0.0404, 0.0, 0.1146, 0.1692, 0.1353, 0.0367, 0.6012, 0.0, 0.0413, 0.1284], "temps": {"debv3l_L24_6": 1.0, "debv3l_L24_6_x": 1.0, "desk_L23ms": 1.0, "desk_L23ms_x": 1.0, "electl_L21_24s_x": 1.0, "q25_7b_L6_20_x": 1.0, "q8lora_v1": 1.5, "q8lora_s7": 1.5, "q8lora_x1": 1.5, "style_lgb": 1.0, "stylegram_x": 0.25, "q25_7b_L6_20": 1.0, "electl_L21_24s": 1.0, "q25lora_x1": 1.5}}   # from blend/weights.json (nested CV 0.5520); overwritten by blend.json if present
ENCODERS = {}            # name -> glob of the model dir under /kaggle/input, e.g. {"debv3l_v1": "/kaggle/input/**/job_final/model"}
LORAS = {                # decoder LoRA members from vienas/ntl-train-lora (used only when listed in the blend)
    "q8lora_v1": dict(job="/kaggle/input/**/ntl-train-lora/out/job_final", model_type="qwen3", hidden=4096, n_layers=36, keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447),
    "q8lora_s7": dict(job="/kaggle/input/**/ntl-train-lora-s7/out/job_final", model_type="qwen3", hidden=4096, n_layers=36, keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447),
    "q8lora_x1": dict(job="/kaggle/input/**/ntl-train-lora-x/out/job_final", model_type="qwen3", hidden=4096, n_layers=36, keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447),
    "q34lora_v1": dict(job="/kaggle/input/**/ntl-train-lora-q34/out/job_final", model_type="qwen3", hidden=2560, n_layers=36, keep=36, layers=[4, 8, 12, 16, 20, 24, 30, 36], max_len=447),
    "q25lora_x1": dict(job="/kaggle/input/**/ntl-train-lora-q25x/out/job_final", model_type="qwen2", hidden=3584, n_layers=28, keep=20, layers=[2, 4, 6, 8, 12, 16, 20], max_len=447),
}                        # name -> dict(job="/kaggle/input/**/ntl-train-lora/out/job_final", model_type="qwen3", hidden=4096, n_layers=36, keep=24, layers=[...], max_len=447)
TWIN_W = {"debv3l_L24_6": 0.2, "electl_L21_24s": 0.1, "q25_7b_L6_20": 0.1, "q8lora_v1": 0.2}   # from blend/twins.json (scripts/sim_twins.py)
FROZEN = {               # frozen-embedding members from vienas/ntl-train-frozen (used only when listed in the blend)
    "q3_8b_L7_22": dict(heads="/kaggle/input/**/heads_q3_8b_L7_22.npz", find=dict(model_type="qwen3", hidden=4096, n_layers=36), layers=[7, 22], pool="mean", decoder=True, max_len=512),
    "q34_L7_22": dict(heads="/kaggle/input/**/heads_q34_L7_22.npz", find=dict(model_type="qwen3", hidden=2560, n_layers=36), layers=[7, 22], pool="mean", decoder=True, max_len=512),
    "q25_7b_L6_20": dict(heads="/kaggle/input/**/heads_q25_7b_L6_20.npz", find=dict(model_type="qwen2", hidden=3584, n_layers=28), layers=[6, 20], pool="mean", decoder=True, max_len=512),
    "debv3l_L24_6": dict(heads="/kaggle/input/**/heads_debv3l_L24_6.npz", dir="/kaggle/input/**/models/deberta-v3-large", layers=[24, 6], pool="mean", decoder=False, max_len=512),
    "desk_L23ms": dict(heads="/kaggle/input/**/heads_desk_L23ms.npz", dir="/kaggle/input/**/models/desklib-detector", layers=[23, 23], pools=["mean", "style"], decoder=False, max_len=512),
    "electl_L21_24s": dict(heads="/kaggle/input/**/heads_electl_L21_24s.npz", dir="/kaggle/input/**/models/electra-large", layers=[21, 24], pools=["mean", "style"], decoder=False, max_len=512),
    # plain heads refitted on train + public extra passages (frozen kernel v3); same features as above (cached)
    "q3_8b_L7_22_x": dict(heads="/kaggle/input/**/heads_q3_8b_L7_22_x.npz", find=dict(model_type="qwen3", hidden=4096, n_layers=36), layers=[7, 22], pool="mean", decoder=True, max_len=512),
    "q34_L7_22_x": dict(heads="/kaggle/input/**/heads_q34_L7_22_x.npz", find=dict(model_type="qwen3", hidden=2560, n_layers=36), layers=[7, 22], pool="mean", decoder=True, max_len=512),
    "q25_7b_L6_20_x": dict(heads="/kaggle/input/**/heads_q25_7b_L6_20_x.npz", find=dict(model_type="qwen2", hidden=3584, n_layers=28), layers=[6, 20], pool="mean", decoder=True, max_len=512),
    "debv3l_L24_6_x": dict(heads="/kaggle/input/**/heads_debv3l_L24_6_x.npz", dir="/kaggle/input/**/models/deberta-v3-large", layers=[24, 6], pool="mean", decoder=False, max_len=512),
    "desk_L23ms_x": dict(heads="/kaggle/input/**/heads_desk_L23ms_x.npz", dir="/kaggle/input/**/models/desklib-detector", layers=[23, 23], pools=["mean", "style"], decoder=False, max_len=512),
    "electl_L21_24s_x": dict(heads="/kaggle/input/**/heads_electl_L21_24s_x.npz", dir="/kaggle/input/**/models/electra-large", layers=[21, 24], pools=["mean", "style"], decoder=False, max_len=512),
}                        # name -> dict(heads="/kaggle/input/**/heads_<key>.npz", find=dict(model_type=..., hidden=..., n_layers=...) or dir="/kaggle/input/**/models/<name>", layers=[...], pool="mean", decoder=True, max_len=512)
CLUSTER_THRESH = 0.9     # average-linkage cut on content-word TF-IDF cosine distance (pair precision 0.96 / recall 0.95 in simulation)
GLOBAL_SINKHORN = 0.5    # test-set-wide class-mass balancing strength (0 = off); simulation: never negative, +0.005..+0.011
SINKHORN_ALPHA = 0.5     # per-cluster class-mass balancing strength (0 = off); with assignment: best or near-best in every simulated layout
ASSIGN = True            # Hungarian assignment inside clusters
POSTPROCESS = True       # False = plain blend only (probe submission: measures what clustering/twins/vetoes/Sinkhorn are worth on the leaderboard)
EXTRA = dict(w=0.5, exclude=["kimi-k3"], globs=["/kaggle/input/**/prvsiyans-extended-name-that-llm-dataset/passages.csv", "/kaggle/input/**/ntl_extra_passages_claude.csv", "/kaggle/input/**/ntl-own-prompt-generations-v1/generated_v1.csv", "/kaggle/input/**/ntl_proxy_generations.csv"])   # public extra passages for stylegram_x

LABELS = sorted(["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
                 "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"])
K = len(LABELS); L2I = {l: i for i, l in enumerate(LABELS)}
INP = os.environ.get("NTL_INPUT", "/kaggle/input")   # local dry runs can point this at a folder with kernel outputs
if os.environ.get("NTL_FROZEN_JSON"): FROZEN = json.load(open(os.environ["NTL_FROZEN_JSON"]))   # local dry runs: frozen specs for smoke outputs
def inp(pattern): return pattern.replace("/kaggle/input", INP)
DATA = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm", os.environ.get("NTL_DATA", "data")] if os.path.exists(os.path.join(p, "train.csv")))
train, test = pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")
train["text"] = train.text.astype(str); test["text"] = test.text.fillna("").astype(str)
y = train.model.map(L2I).values; TR, TE = train.text.tolist(), test.text.tolist()
hits = glob.glob("/kaggle/input/**/blend.json", recursive=True) + glob.glob(os.environ.get("NTL_BLEND", "blend.json"))
if hits:
    BLEND = json.load(open(hits[0])); print("blend config", hits[0], BLEND, flush=True)
print("train", train.shape, "test", test.shape, flush=True)
LOGP = {}


def log_softmax(z, temp=1.0):
    z = np.asarray(z, dtype=np.float64) / temp; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


# ---- stylegram: four TF-IDF views + LinearSVC
FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250_000, 1.0), ("word", 1, 3, 180_000, 0.7), ("shape", 2, 6, 60_000, 0.5), ("function", 1, 3, 50_000, 0.5)]
shape_text = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
def stylegram_decision(fit_texts, fit_y, sample_weight=None):   # four TF-IDF views fitted on fit_texts, LinearSVC margins on the test texts
    A, B = [], []
    for an, lo, hi, mx, w in VIEWS:
        tf = shape_text if an == "shape" else function_text if an == "function" else str
        actual = "char" if an == "shape" else "word" if an == "function" else an
        kw = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mx, dtype=np.float32)
        if actual == "word": kw["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**kw); A.append(v.fit_transform([tf(t) for t in fit_texts]) * w); B.append(v.transform([tf(t) for t in TE]) * w)
    svc = LinearSVC(C=0.5, max_iter=5000, random_state=42).fit(hstack(A, format="csr"), fit_y, sample_weight=sample_weight)
    d = np.full((len(TE), K), -10.0); d[:, svc.classes_] = svc.decision_function(hstack(B, format="csr")); return d
if "stylegram" in BLEND["members"]:
    LOGP["stylegram"] = log_softmax(stylegram_decision(TR, y), BLEND["temps"].get("stylegram", 0.5)); print(f"stylegram done ({time.time()-T0:.0f}s)", flush=True)
if "stylegram_x" in BLEND["members"]:   # + public extra passages (class-balanced weights: share EXTRA["w"] per class, Kimi excluded)
    parts = []
    for g in (os.environ.get("NTL_EXTRA_GLOBS").split(",") if os.environ.get("NTL_EXTRA_GLOBS") else EXTRA["globs"]):
        for f in sorted(glob.glob(g, recursive=True)):
            d = pd.read_csv(f).rename(columns={"label": "model"}); parts.append(d[["text", "model"]].dropna())
    assert parts, "stylegram_x: no extra passages attached"
    ex = pd.concat(parts, ignore_index=True); ex = ex[ex.model.isin(LABELS) & ~ex.model.isin(EXTRA["exclude"])].reset_index(drop=True)
    ex_y = ex.model.map(L2I).values; sw_tr = np.ones(len(TR)); sw_ex = np.zeros(len(ex))
    for c in np.unique(ex_y):
        if (y == c).sum(): sw_tr[y == c] = 1 - EXTRA["w"]; sw_ex[ex_y == c] = EXTRA["w"] * (y == c).sum() / (ex_y == c).sum()
    keep = sw_ex > 0; print(f"stylegram_x: {int(keep.sum())} extra rows from {len(parts)} files", flush=True)
    LOGP["stylegram_x"] = log_softmax(stylegram_decision(TR + [t for t, k in zip(ex.text.astype(str), keep) if k], np.concatenate([y, ex_y[keep]]), np.concatenate([sw_tr, sw_ex[keep]])),
                                      BLEND["temps"].get("stylegram_x", 0.5)); print(f"stylegram_x done ({time.time()-T0:.0f}s)", flush=True)

# ---- stylometric features + LightGBM
PUNCT = list(",.;:!?()[]\"'-") + ["—", "–", "‘", "’", "“", "”", "…", "*", "/", "%"]; FUNC = sorted(ENGLISH_STOP_WORDS)
SENT = re.compile(r"(?<=[.!?])\s+"); WORD = re.compile(r"[A-Za-z’']+")
def stylometric_features(t):
    words = WORD.findall(t); nw, nc = max(len(words), 1), max(len(t), 1)
    sents = [x for x in SENT.split(t.strip()) if x]
    sl = np.array([len(WORD.findall(x)) for x in sents]) if sents else np.array([0]); wl = np.array([len(x) for x in words]) if words else np.array([0])
    lw = [x.lower() for x in words]
    f = [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(), (wl >= 9).mean(), len(set(lw)) / nw, sum(c.isupper() for c in t) / nc,
         sum(c.isdigit() for c in t) / nc, sum(ord(c) > 127 for c in t) / nc, t.count(", ") / nw, t.count(" and ") / nw, t.count(" but ") / nw,
         sum(1 for x in sents if x[:1].isupper() and x.split()[0].lower() in ("this", "these", "it", "they")) / max(len(sents), 1),
         sum(1 for x in words if x.endswith("ly")) / nw, sum(1 for x in words if x.endswith("ing")) / nw, sum(1 for x in words if x.endswith("tion")) / nw]
    f += [t.count(q) / nw for q in PUNCT]
    cnt = {}
    for x in lw: cnt[x] = cnt.get(x, 0) + 1
    f += [cnt.get(x, 0) / nw for x in FUNC]
    return f
if "style_lgb" in BLEND["members"]:
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4)
    clf.fit(np.array([stylometric_features(t) for t in TR], dtype=np.float32), y)
    p = np.full((len(TE), K), 1e-6); p[:, clf.classes_] = clf.predict_proba(np.array([stylometric_features(t) for t in TE], dtype=np.float32))
    LOGP["style_lgb"] = np.log(p / p.sum(1, keepdims=True)); print(f"style_lgb done ({time.time()-T0:.0f}s)", flush=True)

# ---- fine-tuned encoders from my training kernels
for name, pattern in ENCODERS.items():
    if name not in BLEND["members"]: continue
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    hits = sorted(glob.glob(pattern, recursive=True)); assert hits, f"{name}: no model dir for {pattern}"
    path = hits[0]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float16 if dev == "cuda" else torch.float32, attn_implementation="sdpa")
    except ValueError:
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float16 if dev == "cuda" else torch.float32, attn_implementation="eager")
    model = model.to(dev).eval()
    cols = [L2I[model.config.id2label[i]] for i in range(K)]
    ids = tok(TE, truncation=True, max_length=512)["input_ids"]; order = np.argsort([len(x) for x in ids]); P = np.zeros((len(TE), K))
    with torch.no_grad():
        for s in range(0, len(order), 32):
            pos = order[s:s + 32]; b = tok.pad({"input_ids": [ids[i] for i in pos]}, return_tensors="pt").to(dev)
            P[np.ix_(pos, cols)] = model(**b).logits.float().softmax(-1).cpu().numpy()
    LOGP[name] = np.log(np.clip(P, 1e-9, 1)); del model; torch.cuda.empty_cache() if dev == "cuda" else None
    print(f"{name} done from {path} ({time.time()-T0:.0f}s)", flush=True)

# ---- frozen-embedding heads from my training kernel (StandardScaler + LR on pooled hidden states)
CENTER_FEATS = globals().setdefault("LORA_FEATS", {}); FEAT_CACHE = {}
for name, cfg in FROZEN.items():
    if name not in BLEND["members"]: continue
    import torch, gc
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    hp = sorted(glob.glob(inp(cfg["heads"]), recursive=True)); assert hp, f"{name}: heads file {cfg['heads']} not attached"; hd = np.load(hp[0], allow_pickle=True)
    assert list(hd["classes"]) == LABELS
    if "dir" in cfg:
        path = sorted(glob.glob(inp(cfg["dir"]) + "/config.json", recursive=True)); assert path, f"{name}: backbone dir {cfg['dir']} not attached"; path = os.path.dirname(path[0])
    else:
        path = None
        for c in sorted(glob.glob(inp("/kaggle/input/**/config.json"), recursive=True)):
            try: j = json.load(open(c))
            except Exception: continue
            f = cfg["find"]
            if j.get("model_type") == f["model_type"] and j.get("hidden_size") == f["hidden"] and j.get("num_hidden_layers") == f["n_layers"]: path = os.path.dirname(c); break
        assert path, f"{name}: backbone not attached"
    dev = "cuda" if torch.cuda.is_available() else "cpu"; decoder = cfg.get("decoder", True)
    ckey = (path, tuple(cfg["layers"]), tuple(cfg.get("pools") or [cfg.get("pool", "mean")] * len(cfg["layers"])), cfg.get("max_len", 512))
    if ckey in FEAT_CACHE:   # same backbone already embedded for another head
        Z = FEAT_CACHE[ckey]; Xs = (Z - hd["mean"]) / hd["scale"]
        if len(TE) >= 50: Xs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        lg = Xs @ hd["coef"].T + hd["intercept"]; LOGP[name] = log_softmax(lg, BLEND["temps"].get(name, 1.0))
        if "c_coef" in hd.files: CENTER_FEATS[name] = (Z, hd["c_coef"] / hd["c_scale"], hd["c_intercept"] - (hd["c_mean"] / hd["c_scale"]) @ hd["c_coef"].T)
        print(f"{name} done from cached features of {path} ({time.time()-T0:.0f}s)", flush=True); continue
    mc = AutoConfig.from_pretrained(path); tc = getattr(mc, "text_config", None) or mc; keep = max(cfg["layers"]) + 1
    tc.num_hidden_layers = min(keep, tc.num_hidden_layers)
    if getattr(tc, "layer_types", None): tc.layer_types = tc.layer_types[:tc.num_hidden_layers]
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.float16 if (dev == "cuda" and decoder) else torch.float32
    kw = dict(config=mc, dtype=dt)
    if dev == "cuda" and decoder: kw["device_map"] = "auto"
    model = AutoModel.from_pretrained(path, **kw).eval()
    if dev == "cuda" and not decoder: model = model.to(dev)
    pools = cfg.get("pools") or [cfg.get("pool", "mean")] * len(cfg["layers"]); smask = None
    if "style" in pools:
        func = frozenset(ENGLISH_STOP_WORDS); smask = torch.tensor([(tok.decode([i]).strip().lower() in func) or bool(re.fullmatch(r"\s*[^\w\s]+\s*", tok.decode([i]))) for i in range(len(tok))])
    H = model.config.hidden_size if not hasattr(model.config, "text_config") else model.config.text_config.hidden_size
    def frozen_embed(texts, bs=8):
        out = np.zeros((len(texts), len(cfg["layers"]) * H), np.float32); order = np.argsort([len(t) for t in texts]); mdev = next(model.parameters()).device
        with torch.no_grad():
            for b0 in range(0, len(texts), bs):
                idx = order[b0:b0 + bs]
                if decoder:
                    batch = [(tok.bos_token + texts[i]) if tok.bos_token else texts[i] for i in idx]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=cfg.get("max_len", 512), add_special_tokens=False).to(mdev)
                else:
                    enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=cfg.get("max_len", 512)).to(mdev)
                hs = model(**enc, output_hidden_states=True).hidden_states
                m0 = enc["attention_mask"].float()
                if decoder: m0[:, 0] = 0
                ms = (m0 * smask.to(m0.device)[enc["input_ids"]].float()) if smask is not None else m0
                for li, Lr in enumerate(cfg["layers"]):
                    h = hs[Lr].float(); mm = (ms if pools[li] == "style" else m0).unsqueeze(-1).to(h.device); out[idx, li * H:(li + 1) * H] = ((h * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu().numpy()
                del hs, enc
        return np.nan_to_num(out)
    arows = train.index[train.id.astype(str).isin([str(i) for i in hd["anchor_ids"]])].tolist() if "anchor_ids" in hd.files else []
    if len(arows) == len(hd["anchors"]):   # parity check: the same train rows must give the same features as in the training kernel
        atexts = [TR[i] for i in arows]; aref = np.stack([hd["anchors"][list(map(str, hd["anchor_ids"])).index(str(train.id.iloc[i]))] for i in arows])
        Za = frozen_embed(atexts, 3); rel = float(np.abs(Za - aref).mean() / (np.abs(aref).mean() + 1e-9))
        print(f"{name} parity vs training-kernel anchors: mean rel diff {rel:.4f}", flush=True); assert rel < 0.05, f"{name}: feature mismatch"
    else:
        print(f"{name}: anchor rows not in this train file, parity check skipped", flush=True)
    Z = frozen_embed(TE); FEAT_CACHE[ckey] = Z
    Xs = (Z - hd["mean"]) / hd["scale"]
    if len(TE) >= 50:   # transductive standardisation with the test set's own statistics (label-free)
        Xs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    lg = Xs @ hd["coef"].T + hd["intercept"]; LOGP[name] = log_softmax(lg, BLEND["temps"].get(name, 1.0))
    if "c_coef" in hd.files: CENTER_FEATS[name] = (Z, hd["c_coef"] / hd["c_scale"], hd["c_intercept"] - (hd["c_mean"] / hd["c_scale"]) @ hd["c_coef"].T)   # centered head folded with its scaler
    del model; gc.collect(); torch.cuda.empty_cache() if dev == "cuda" else None
    print(f"{name} done from {path} ({time.time()-T0:.0f}s)", flush=True)

# ---- decoder LoRA members from my training kernels (plain head; centered head as extra evidence inside clusters)
CENTER_LOGP = {}
def find_base(model_type, hidden, n_layers):
    for c in sorted(glob.glob(inp("/kaggle/input/**/config.json"), recursive=True)):
        try: j = json.load(open(c))
        except Exception: continue
        if j.get("model_type") == model_type and j.get("hidden_size") == hidden and j.get("num_hidden_layers") == n_layers: return os.path.dirname(c)
    return None
for name, cfg in LORAS.items():
    if name not in BLEND["members"]: continue
    import torch, importlib.util as _iu
    _fs = _iu.find_spec; _iu.find_spec = lambda n, *a, **k: None if n.split(".")[0] == "torchao" else _fs(n, *a, **k)
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from peft import PeftModel
    import peft.import_utils as _pi; _pi.is_torchao_available = lambda *a, **k: False
    from safetensors.torch import load_file
    job = sorted(glob.glob(inp(cfg["job"]), recursive=True)); assert job, f"{name}: job dir {cfg['job']} not attached"; job = job[0]
    base_dir = find_base(cfg["model_type"], cfg["hidden"], cfg["n_layers"]); assert base_dir, f"{name}: base model not attached"
    mc = AutoConfig.from_pretrained(base_dir); mc.num_hidden_layers, mc.use_cache = cfg["keep"], False
    if getattr(mc, "layer_types", None) is not None: mc.layer_types = mc.layer_types[:cfg["keep"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"; dt = torch.float16 if dev == "cuda" else torch.float32
    trunk = AutoModel.from_pretrained(base_dir, config=mc, dtype=dt, device_map="auto" if dev == "cuda" else None).eval()
    lm = PeftModel.from_pretrained(trunk, os.path.join(job, "adapter")).eval()
    ltok = AutoTokenizer.from_pretrained(os.path.join(job, "tokenizer")); ltok.padding_side = "right"
    if ltok.pad_token is None: ltok.pad_token = ltok.eos_token
    Lyr = cfg["layers"]; H = cfg["hidden"]
    nst = load_file(os.path.join(job, "norms.safetensors"))
    norms = torch.nn.ModuleList([torch.nn.LayerNorm(H) for _ in Lyr]); norms.load_state_dict({k[len("norms."):]: v for k, v in nst.items()})
    ldev = lm.get_input_embeddings().weight.device; norms = norms.to(ldev).float()
    ro = np.load(os.path.join(job, "readouts.npz")); assert list(ro["classes"]) == LABELS and list(ro["pool_layers"]) == Lyr
    def lora_embed(texts, bs=8):
        toks = [ids + [ltok.eos_token_id] for ids in ltok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]]
        Z = np.zeros((len(toks), H * len(Lyr)), np.float32); order = np.argsort([len(t) for t in toks], kind="stable")
        with torch.inference_mode():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]; batch = ltok.pad({"input_ids": [toks[i] for i in idx]}, return_tensors="pt").to(ldev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    hs = lm(**batch, output_hidden_states=True, use_cache=False).hidden_states
                m = batch["attention_mask"].unsqueeze(-1).float(); m[:, 0] = 0
                Z[idx] = torch.cat([nm((hs[l].float() * m.to(hs[l].device)).sum(1).to(ldev) / m.sum(1).to(ldev).clamp(min=1)) for l, nm in zip(Lyr, norms)], -1).cpu().numpy()
        return Z
    anc = os.path.join(job, "anchor_features.npz")
    if os.path.exists(anc):   # parity check against features saved by the training kernel (by row id)
        az = np.load(anc, allow_pickle=True); aids = [str(i) for i in az["ids"]]; arows = [int(np.where(train.id.astype(str).values == i)[0][0]) for i in aids if i in set(train.id.astype(str))]
        if len(arows) == len(aids):
            a = az["features"].astype(np.float32); Za = lora_embed([TR[i] for i in arows], 3)
            rel = float(np.abs(Za - a).mean() / (np.abs(a).mean() + 1e-9)); print(f"{name} parity vs training-kernel anchors: mean rel diff {rel:.4f}", flush=True); assert rel < 0.05, f"{name}: feature mismatch"
        else:
            print(f"{name}: anchor rows not in this train file, parity check skipped", flush=True)
    Z = lora_embed(TE)
    lg = Z @ ro["plain_coef"].T + ro["plain_intercept"]; LOGP[name] = log_softmax(lg, BLEND["temps"].get(name, 1.0))
    LORA_FEATS = globals().setdefault("LORA_FEATS", {}); LORA_FEATS[name] = (Z, ro["center_coef"], ro["center_intercept"])
    trunk = lm.unload(); del lm, trunk; torch.cuda.empty_cache() if dev == "cuda" else None
    print(f"{name} done from {job} ({time.time()-T0:.0f}s)", flush=True)

# ---- blend
missing = [m for m in BLEND["members"] if m not in LOGP]; assert not missing, f"members without predictions: {missing}"
Z = sum(w * LOGP[m] for m, w in zip(BLEND["members"], BLEND["weights"])); L = log_softmax(Z)
pred = L.argmax(1)

# ---- test-time prompt recovery + assignment (no labels used)
if POSTPROCESS and len(TE) >= 2:
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(TE)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None); np.fill_diagonal(D, 0)
    g = fcluster(linkage(squareform(D, checks=False), method="average"), CLUSTER_THRESH, criterion="distance")
    sz = np.bincount(g)[g]
    print("clusters:", len(np.unique(g)), "| passages in clusters of size 1:", int((sz == 1).sum()), "2-5:", int(((sz >= 2) & (sz <= 5)).sum()), "6-18:", int(((sz >= 6) & (sz <= 18)).sum()), ">18:", int((sz > 18).sum()), flush=True)
    for name, (Zf, cc, ci) in globals().get("LORA_FEATS", {}).items():   # cluster-centered twin evidence
        if TWIN_W.get(name, 0) <= 0: continue
        Zc = Zf.copy(); rows = np.where((sz >= 2) & (sz <= 18))[0]   # clusters above 18 are certainly mixed prompts
        for c in np.unique(g[rows]):
            i = np.where(g == c)[0]; Zc[i] -= Zc[i].mean(0)
        lc = log_softmax(Zc[rows] @ cc.T + ci); L[rows] = L[rows] + TWIN_W[name] * lc; L = log_softmax(L)
        print(f"{name}: centered-twin evidence added on {len(rows)} clustered rows", flush=True)
    # orthographic vetoes seen in train (0 of ~150 per model): curly quotes never in Claude/GLM, em-dashes never in
    # gemini-3.1/3.6/3.8 (1 % in 3.7-flash), straight apostrophes never in gpt-5.6-sol -> log penalties
    pen = np.zeros_like(L); curly = np.array([bool(re.search("[\u2019\u201c\u201d\u2018]", t)) for t in TE]); emd = np.array(["\u2014" in t for t in TE]); apos = np.array(["'" in t for t in TE])
    for mask, models, v in [(curly, [m for m in LABELS if m.startswith(("claude", "glm"))], 4.0), (emd, ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.8-flash"], 4.0),
                            (emd, ["gemini-3.7-flash"], 2.0), (apos, ["gpt-5.6-sol"], 3.0)]:
        for m in models: pen[mask, L2I[m]] -= v
    L2 = log_softmax(L + pen); print("orthographic vetoes changed", int((L2.argmax(1) != L.argmax(1)).sum()), "of", len(L), flush=True); L = L2
    P = np.exp(L)
    if GLOBAL_SINKHORN > 0:
        target = (1 - GLOBAL_SINKHORN) * P.sum(0) + GLOBAL_SINKHORN * np.full(K, len(P) / K)
        for _ in range(50):
            P = P * (target / (P.sum(0) + 1e-12))[None, :]; P = P / (P.sum(1, keepdims=True) + 1e-12)
        print("global balancing changed", int((P.argmax(1) != L.argmax(1)).sum()), "of", len(P), flush=True); L = np.log(np.clip(P, 1e-12, 1))
    if SINKHORN_ALPHA > 0:
        for c in np.unique(g):
            i = np.where(g == c)[0]
            if len(i) < 2: continue
            Q = P[i].copy(); target = (1 - SINKHORN_ALPHA) * Q.sum(0) + SINKHORN_ALPHA * np.full(K, len(i) / K)
            for _ in range(50):
                Q = Q * (target / (Q.sum(0) + 1e-12))[None, :]; Q = Q / (Q.sum(1, keepdims=True) + 1e-12)
            P[i] = Q
        L = np.log(np.clip(P, 1e-12, 1))
    if ASSIGN:
        new = L.argmax(1).copy()
        for c in np.unique(g):
            i = np.where(g == c)[0]
            if len(i) < 2: continue
            reps = int(np.ceil(len(i) / K)); r, cc = linear_sum_assignment(-np.repeat(L[i], reps, axis=1)); new[i[r]] = cc // reps
        print("assignment changed", int((new != pred).sum()), "of", len(pred), flush=True); pred = new

sub = pd.DataFrame({"id": test["id"], "model": [LABELS[i] for i in pred]})
assert sub["id"].is_unique and sub["model"].isin(LABELS).all() and len(sub) == len(test) and sub.model.notna().all()
sub.to_parquet("submission.parquet", index=False)
print(sub.model.value_counts().to_string()); print(f"total {time.time()-T0:.0f}s")
