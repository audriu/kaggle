# Name That LLM - frozen-embedding heads (training only, no submission).
# For each spec: load the backbone truncated to the deepest needed layer, pool hidden states of the chosen layers over
# tokens (masked mean, first token excluded for decoders; optional "style" pooling over stop-word/punctuation tokens),
# concatenate, then fit StandardScaler + LogisticRegression heads: plain (prompt-grouped 5-fold OOF + final) and a
# prompt-centered twin (features minus their prompt mean). Saves oof_<key>.npz, heads_<key>.npz and anchor features
# (train rows 0..2) for a parity check in the submission kernel. Backbones that are not Kaggle Models are downloaded
# from Hugging Face (internet on) and saved under models/<name> for the submission kernel.
import gc, glob, json, os, pathlib, re, time
import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

SPECS = [   # key, how to find the backbone (Kaggle Models dir by config, or an HF id to download), layers, pooling, C
    dict(key="q3_8b_L7_22", find=dict(model_type="qwen3", hidden=4096, n_layers=36), layers=[7, 22], pool="mean", C=0.0025, decoder=True),
    dict(key="q25_7b_L6_20", find=dict(model_type="qwen2", hidden=3584, n_layers=28), layers=[6, 20], pool="mean", C=0.0025, decoder=True),
    dict(key="debv3l_L24_6", hf="microsoft/deberta-v3-large", save="deberta-v3-large", layers=[24, 6], pool="mean", C=0.005, decoder=False),
    dict(key="desk_L23ms", hf="desklib/ai-text-detector-v1.01", save="desklib-detector", layers=[23, 23], pools=["mean", "style"], C=0.005, decoder=False, prefix="model."),
    dict(key="electl_L21_24s", hf="google/electra-large-discriminator", save="electra-large", layers=[21, 24], pools=["mean", "style"], C=0.005, decoder=False),
    dict(key="q34_L7_22", find=dict(model_type="qwen3", hidden=2560, n_layers=36), layers=[7, 22], pool="mean", C=0.0025, decoder=True),
]
ONLY = [k for k in os.environ.get("NTL_ONLY", "").split(",") if k]   # optional subset of spec keys
SMOKE = os.environ.get("NTL_SMOKE", "0") == "1"
MAX_LEN, BS = 512, 8
LABELS = sorted(["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
                 "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"])
K = len(LABELS); L2I = {l: i for i, l in enumerate(LABELS)}
T0 = time.time()
DATA = next(p for p in [pathlib.Path("/kaggle/input/competitions/name-that-llm"), pathlib.Path("/kaggle/input/name-that-llm"), pathlib.Path(os.environ.get("NTL_DATA", "data"))] if (p / "train.csv").exists())
train = pd.read_csv(DATA / "train.csv")
if SMOKE:
    train = train[train.prompt_id.isin(sorted(train.prompt_id.unique())[:12])].reset_index(drop=True)
    SPECS = [dict(key="smoke_q", hf="Qwen/Qwen2.5-0.5B", save="smoke_q", layers=[2, 3], pool="mean", C=0.0025, decoder=True),
             dict(key="smoke_d", hf="microsoft/deberta-v3-xsmall", save="smoke_d", layers=[3, 2], pools=["mean", "style"], C=0.005, decoder=False)]
    MAX_LEN = 48
y = train.model.map(L2I).values; pid = train.prompt_id.astype(str).values; TR = train.text.astype(str).tolist()
prompts = np.array(sorted(train.prompt_id.unique())); rng0 = np.random.RandomState(0); rng0.shuffle(prompts)
fold = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).values
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("train", train.shape, "device", DEV, "gpus", torch.cuda.device_count(), flush=True)
# public extra passages for the plain "_x" heads (train + extra rows, class-balanced weights; twins stay train-only)
EXTRA = dict(w=0.5, exclude=["kimi-k3"], cap=300, globs=["/kaggle/input/**/extra_v3.csv"])
parts = []
for g in (os.environ.get("NTL_EXTRA_GLOBS").split(",") if os.environ.get("NTL_EXTRA_GLOBS") else EXTRA["globs"]):
    for f in sorted(glob.glob(g, recursive=True)):
        d = pd.read_csv(f).rename(columns={"label": "model"}); parts.append(d[["text", "model"]].dropna())
EX = None
if parts:
    EX = pd.concat(parts, ignore_index=True); EX = EX[EX.model.isin(LABELS) & ~EX.model.isin(EXTRA["exclude"])]
    EX = EX.sample(frac=1, random_state=0).groupby("model").head(EXTRA["cap"]).reset_index(drop=True)
    if SMOKE: EX = EX.head(60)
    EX_TEXTS = EX.text.astype(str).tolist(); EX_Y = EX.model.map(L2I).values
    print("extra rows", len(EX), EX.model.value_counts().to_dict(), flush=True)
else:
    print("no extra passages found: _x heads skipped", flush=True)


def find_dir(f):
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try: j = json.load(open(c))
        except Exception: continue
        if j.get("model_type") == f["model_type"] and j.get("hidden_size") == f["hidden"] and j.get("num_hidden_layers") == f["n_layers"]: return os.path.dirname(c)
    raise FileNotFoundError(str(f))


def style_mask(tok):
    func = frozenset(ENGLISH_STOP_WORDS); pieces = [tok.decode([i]) for i in range(len(tok))]
    return torch.tensor([(p.strip().lower() in func) or bool(re.fullmatch(r"\s*[^\w\s]+\s*", p)) for p in pieces])


@torch.no_grad()
def embed(model, tok, texts, layers, pool, decoder, smask=None):
    H = model.config.hidden_size; out = np.zeros((len(texts), len(layers) * H), np.float32)
    order = np.argsort([len(t) for t in texts]); dev = next(model.parameters()).device
    for b0 in range(0, len(texts), BS):
        idx = order[b0:b0 + BS]
        if decoder:
            batch = [(tok.bos_token + texts[i]) if tok.bos_token else texts[i] for i in idx]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN, add_special_tokens=False).to(dev)
        else:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m0 = enc["attention_mask"].float()
        if decoder: m0[:, 0] = 0
        ms = (m0 * smask.to(m0.device)[enc["input_ids"]].float()) if smask is not None else m0
        pools = pool if isinstance(pool, list) else [pool] * len(layers)
        for li, L in enumerate(layers):
            h = hs[L].float(); mm = (ms if pools[li] == "style" else m0).unsqueeze(-1).to(h.device)
            out[idx, li * H:(li + 1) * H] = ((h * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu().numpy()
        del hs, enc
    return np.nan_to_num(out)


def fit_heads(key, X, C, Xe=None):
    Xc = X.copy()
    for p in np.unique(pid): Xc[pid == p] -= Xc[pid == p].mean(0)
    oof = np.zeros((len(y), K), np.float32); oofc = np.zeros((len(y), K), np.float32); fc = {}
    for k in range(5):
        a, b = fold != k, fold == k
        sc = StandardScaler().fit(X[a]); clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[a]), y[a])
        oof[np.ix_(b, clf.classes_)] = clf.predict_proba(sc.transform(X[b]))
        scc = StandardScaler().fit(Xc[a]); cl2 = LogisticRegression(C=C, max_iter=3000).fit(scc.transform(Xc[a]), y[a])
        oofc[np.ix_(b, cl2.classes_)] = cl2.predict_proba(scc.transform(Xc[b]))
        fc[f"c_coef_{k}"] = (cl2.coef_ / scc.scale_).astype(np.float32); fc[f"c_intercept_{k}"] = (cl2.intercept_ - (scc.mean_ / scc.scale_) @ cl2.coef_.T).astype(np.float32)
    ba = lambda P: float(np.mean([(P.argmax(1) == c)[y == c].mean() for c in np.unique(y)]))
    print(f"{key}: OOF balanced acc plain {ba(oof):.4f} | prompt-centered (oracle prompts) {ba(oofc):.4f} ({time.time()-T0:.0f}s)", flush=True)
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y)
    scc = StandardScaler().fit(Xc); cl2 = LogisticRegression(C=C, max_iter=3000).fit(scc.transform(Xc), y)
    np.savez_compressed(f"oof_{key}.npz", ids=train.id.values, scores=oof, kind="prob", classes=np.array(LABELS), scores_centered=oofc)
    np.savez_compressed(f"feats_{key}.npz", ids=train.id.values, X=X.astype(np.float16), fold=fold, classes=np.array(LABELS), **fc)   # for local twin-weight simulation
    if Xe is not None:   # plain head on train + extra rows: share EXTRA["w"] of each class's weight goes to the extra rows
        def weights(yy_tr):
            sw_tr = np.ones(len(yy_tr)); sw_ex = np.zeros(len(EX_Y))
            for c in np.unique(EX_Y):
                if (yy_tr == c).sum(): sw_tr[yy_tr == c] = 1 - EXTRA["w"]; sw_ex[EX_Y == c] = EXTRA["w"] * (yy_tr == c).sum() / (EX_Y == c).sum()
            return sw_tr, sw_ex
        oofx = np.zeros((len(y), K), np.float32)
        for k in range(5):
            a, b = fold != k, fold == k; sw_tr, sw_ex = weights(y[a]); keep = sw_ex > 0
            Xp = np.concatenate([X[a], Xe[keep]]); yp = np.concatenate([y[a], EX_Y[keep]]); sw = np.concatenate([sw_tr, sw_ex[keep]])
            scx = StandardScaler().fit(Xp); clx = LogisticRegression(C=C, max_iter=3000).fit(scx.transform(Xp), yp, sample_weight=sw)
            oofx[np.ix_(b, clx.classes_)] = clx.predict_proba(scx.transform(X[b]))
        print(f"{key}_x: OOF balanced acc plain+extra {ba(oofx):.4f} ({time.time()-T0:.0f}s)", flush=True)
        sw_tr, sw_ex = weights(y); keep = sw_ex > 0
        Xp = np.concatenate([X, Xe[keep]]); yp = np.concatenate([y, EX_Y[keep]]); sw = np.concatenate([sw_tr, sw_ex[keep]])
        scx = StandardScaler().fit(Xp); clx = LogisticRegression(C=C, max_iter=3000).fit(scx.transform(Xp), yp, sample_weight=sw)
        np.savez_compressed(f"oof_{key}_x.npz", ids=train.id.values, scores=oofx, kind="prob", classes=np.array(LABELS))
        np.savez_compressed(f"heads_{key}_x.npz", classes=np.array(LABELS), mean=scx.mean_, scale=scx.scale_, coef=clx.coef_, intercept=clx.intercept_,
                            anchors=X[:3], anchor_ids=train.id.astype(str).values[:3].astype(str))
    np.savez_compressed(f"heads_{key}.npz", classes=np.array(LABELS), mean=sc.mean_, scale=sc.scale_, coef=clf.coef_, intercept=clf.intercept_,
                        c_mean=scc.mean_, c_scale=scc.scale_, c_coef=cl2.coef_, c_intercept=cl2.intercept_, anchors=X[:3], anchor_ids=train.id.astype(str).values[:3].astype(str))


for spec in SPECS:
    key = spec["key"]; t0 = time.time()
    if ONLY and key not in ONLY: continue
    if "hf" in spec:
        path = spec["hf"]; save = os.path.join("models", spec["save"])
    else:
        path = find_dir(spec["find"]); save = None
    cfg = AutoConfig.from_pretrained(path); keep = max(spec["layers"]) + 1
    tc = getattr(cfg, "text_config", None) or cfg; tc.num_hidden_layers = min(keep, tc.num_hidden_layers)
    if getattr(tc, "layer_types", None): tc.layer_types = tc.layer_types[:tc.num_hidden_layers]
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.float16 if (DEV == "cuda" and spec["decoder"]) else torch.float32
    kw = dict(config=cfg, dtype=dt)
    if DEV == "cuda" and spec["decoder"]: kw["device_map"] = "auto"
    model = AutoModel.from_pretrained(path, **kw).eval()
    if spec.get("prefix"):   # trunk stored under a wrapper prefix (desklib: "model.<deberta keys>" + classifier) -> remap and load explicitly
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        pre = spec["prefix"]; sd = {k[len(pre):]: v for k, v in load_file(hf_hub_download(path, "model.safetensors")).items() if k.startswith(pre)}
        r = model.load_state_dict(sd, strict=False); print(f"{key}: remapped {len(sd)} tensors, missing {len(r.missing_keys)}, unexpected {len(r.unexpected_keys)}", flush=True)
        assert not r.missing_keys, r.missing_keys[:5]
    if DEV == "cuda" and not spec["decoder"]: model = model.to(DEV)
    if save:   # keep a fp16 copy (full depth not needed: the truncated config is saved too) for the submission kernel
        os.makedirs(save, exist_ok=True); model.save_pretrained(save, safe_serialization=True); tok.save_pretrained(save)
    pool = spec.get("pools", spec.get("pool", "mean")); smask = style_mask(tok) if "style" in (pool if isinstance(pool, list) else [pool]) else None
    X = embed(model, tok, TR, spec["layers"], pool, spec["decoder"], smask)
    Xe = embed(model, tok, EX_TEXTS, spec["layers"], pool, spec["decoder"], smask) if EX is not None else None
    print(f"{key}: features {X.shape}" + (f" + extra {Xe.shape}" if Xe is not None else "") + f" from {path} ({time.time()-t0:.0f}s)", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    fit_heads(key, X, spec["C"], Xe)
print(f"total {time.time()-T0:.0f}s")
