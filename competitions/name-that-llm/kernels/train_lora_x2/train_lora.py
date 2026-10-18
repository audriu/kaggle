# Name That LLM - decoder LoRA training kernel (training only, no submission). My implementation of the public
# "LoRA24" idea: a base decoder truncated to its first KEEP layers, LoRA on all linear projections, masked mean pooling of
# several hidden layers (first token excluded) each through its own LayerNorm, concatenated into one feature vector z,
# and two linear heads: plain(z) and centered(z - mean of z over the same-prompt micro-batch). Trained on micro-batches
# of 3 answers to the same prompt so the centered head learns prompt-invariant style. Optional extra passages are added
# to the training side (they form their own prompt groups). Jobs: 5 prompt folds (OOF features + readouts) and 'final'
# (adapter + heads saved for inference). Two workers, one per GPU.
import glob, json, os, pathlib, subprocess, sys, time
import numpy as np, pandas as pd

CFG = dict(
    name="q8lora_x2", model_type="qwen3", hidden=4096, n_layers=36,   # how the base model dir is found under /kaggle/input
    keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447,
    r=16, alpha=32, ldrop=0.05, lr_adapter=1e-4, lr_head=1e-3, wd=0.01, warmup=0.10, epochs=3, accum=6, triplet=3,
    seed=7, fold_seed=0, extra_w=0.5, extra_globs=["/kaggle/input/**/prvsiyans-extended-name-that-llm-dataset/passages.csv", "/kaggle/input/**/ntl_extra_passages_claude.csv", "/kaggle/input/**/ntl-own-prompt-generations-v1/generated_v1.csv", "/kaggle/input/**/ntl_proxy_generations.csv"], extra_exclude=["kimi-k3"], extra_cap=150, smoke=False,   # extra_cap: per-class cap on extra rows (pool grows ~2.3x at 150/class)
)
for k, v in list(CFG.items()):
    e = os.environ.get("NTL_" + k.upper())
    if e is not None:
        CFG[k] = (e == "1") if isinstance(v, bool) else [x for x in e.split(",") if x] if isinstance(v, list) else type(v)(e)
JOBS = {0: ["0", "2", "4"], 1: ["1", "3", "final"]}
LABELS = sorted(["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
                 "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"])

WORKER = r'''
import json, math, os, random, sys, time
import importlib.util as _iu
_fs = _iu.find_spec
_iu.find_spec = lambda name, *a, **k: None if name.split(".")[0] == "torchao" else _fs(name, *a, **k)   # peft probes torchao; hide it
import numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import peft.import_utils as _pi
_pi.is_torchao_available = lambda *a, **k: False
from safetensors.torch import save_file

cfg = json.loads(sys.argv[1]); jobs = sys.argv[2].split(",")
LABELS = cfg["labels"]; L2I = {l: i for i, l in enumerate(LABELS)}; K = len(LABELS)
train = pd.read_csv(cfg["train_csv"])
if cfg["smoke"]:
    train = train[train.prompt_id.isin(sorted(train.prompt_id.unique())[:6])].reset_index(drop=True)
y = train.model.map(L2I).values; pid = train.prompt_id.astype(str).values
prompts = np.array(sorted(train.prompt_id.unique())); rng0 = np.random.RandomState(cfg["fold_seed"]); rng0.shuffle(prompts)
fold = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).values
extra = pd.read_csv(cfg["extra_csv"]) if cfg.get("extra_csv") else None
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(cfg["model_dir"]); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def tokens_of(texts):
    return [ids + [tok.eos_token_id] for ids in tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]]
tokens = tokens_of(train.text.astype(str))
ex_tokens = tokens_of(extra.text.astype(str)) if extra is not None else []
ex_y = extra.model.map(L2I).values if extra is not None else np.zeros(0, int)
ex_pid = ("x_" + extra.prompt_id.astype(str)).values if extra is not None else np.zeros(0, object)
L = cfg["layers"]; H = cfg["hidden"]; T = cfg["triplet"]

def triplets(rows_pid, rows, rng):
    """Micro-batches of T distinct answers to the same prompt (short last chunk padded with other answers of the prompt)."""
    out = []
    for p in np.unique(rows_pid):
        m = rng.permutation(rows[rows_pid == p])
        if len(m) < 2: continue
        for s in range(0, len(m), T):
            b = list(m[s:s + T])
            while len(b) < T: b.append(int(rng.choice([x for x in m if x not in b] or list(m))))
            out.append(b)
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % cfg["accum"]: out.append(out[rng.randint(len(out))])
    return out

def run_job(job):
    t0 = time.time(); out = os.path.join("out", f"job_{job}"); os.makedirs(out, exist_ok=True)
    seed = cfg["seed"] + (99 if job == "final" else int(job))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.RandomState(seed)
    fit = np.arange(len(train)) if job == "final" else np.where(fold != int(job))[0]
    valid = np.arange(min(3, len(train))) if job == "final" else np.where(fold == int(job))[0]
    # pool = competition rows (+ extra rows appended with offset indices); sample weights per row
    pool_tok = list(tokens); pool_y = list(y); pool_pid = list(pid); pool_w = np.ones(len(train), dtype=np.float32)
    rows = fit.copy()
    if len(ex_tokens) and cfg["extra_w"] > 0:
        w = cfg["extra_w"]; base = len(pool_tok); yfit = y[fit]; wts = []
        for i in range(len(ex_tokens)):
            c = ex_y[i]; n_tr, n_ex = (yfit == c).sum(), (ex_y == c).sum()
            wts.append(w * n_tr / n_ex if n_tr and n_ex else 0.0)
        pool_tok += ex_tokens; pool_y += list(ex_y); pool_pid += list(ex_pid); pool_w = np.concatenate([pool_w, np.array(wts, dtype=np.float32)])
        for c in np.unique(ex_y): pool_w[fit[yfit == c]] = 1 - w
        rows = np.concatenate([fit, base + np.arange(len(ex_tokens))])
    pool_y = np.array(pool_y); pool_pid = np.array(pool_pid, dtype=object)
    mc = AutoConfig.from_pretrained(cfg["model_dir"]); mc.num_hidden_layers, mc.use_cache = cfg["keep"], False
    if getattr(mc, "layer_types", None) is not None: mc.layer_types = mc.layer_types[:cfg["keep"]]
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    base = AutoModel.from_pretrained(cfg["model_dir"], config=mc, dtype=dt).to(dev)
    if dev.type == "cuda":
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["ldrop"], bias="none",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in L]).to(dev).float()
    plain = nn.Linear(H * len(L), K).to(dev).float(); centered = nn.Linear(H * len(L), K).to(dev).float(); drop = nn.Dropout(0.1)
    ad = [p for p in model.parameters() if p.requires_grad]; hd = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    opt = torch.optim.AdamW([{"params": ad, "lr": cfg["lr_adapter"]}, {"params": hd, "lr": cfg["lr_head"]}], weight_decay=cfg["wd"])
    plan = [triplets(pool_pid[rows], rows, rng) for _ in range(cfg["epochs"])]
    total = max(1, sum(len(e) for e in plan) // cfg["accum"]); warm = max(1, int(cfg["warmup"] * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(init_scale=256.0, enabled=dev.type == "cuda")
    use_amp = dev.type == "cuda"

    def rep(idx):
        batch = tok.pad({"input_ids": [pool_tok[i] for i in idx]}, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        m = batch["attention_mask"].unsqueeze(-1).float(); m[:, 0] = 0
        with torch.autocast("cuda", enabled=False):
            return torch.cat([nm((hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)) for l, nm in zip(L, norms)], -1)

    model.train(); norms.train(); step = 0; skipped = 0; losses = []
    for ep, batches in enumerate(plan):
        for i, b in enumerate(batches):
            z = rep(b); yy = torch.as_tensor(pool_y[b], device=dev); wt = torch.as_tensor(pool_w[b], device=dev)
            l1 = nn.functional.cross_entropy(plain(drop(z)), yy, reduction="none"); l2 = nn.functional.cross_entropy(centered(drop(z - z.mean(0, keepdim=True))), yy, reduction="none")
            loss = ((l1 + l2) * wt).sum() / wt.sum()
            scaler.scale(loss / cfg["accum"]).backward(); losses.append(float(loss.detach()))
            if (i + 1) % cfg["accum"] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(ad + hd, 1.0)
                before = scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale() < before: skipped += 1
                opt.zero_grad(set_to_none=True); sch.step(); step += 1
                if step % 30 == 0: print(f"job {job} ep {ep} step {step}/{total} loss {np.mean(losses[-180:]):.4f} skipped {skipped} ({time.time()-t0:.0f}s)", flush=True)
    model.eval(); norms.eval()
    Z = np.zeros((len(valid), H * len(L)), np.float32); order = np.argsort([len(tokens[i]) for i in valid], kind="stable")
    with torch.inference_mode():
        for s in range(0, len(order), 6):
            pos = order[s:s + 6]; Z[pos] = rep(valid[pos]).cpu().numpy()
    assert np.isfinite(Z).all()
    W = {"plain": plain.weight.detach().cpu().numpy(), "center": centered.weight.detach().cpu().numpy()}
    B = {"plain": plain.bias.detach().cpu().numpy(), "center": centered.bias.detach().cpu().numpy()}
    np.savez_compressed(os.path.join(out, "readouts.npz"), classes=np.array(LABELS), pool_layers=np.array(L), plain_coef=W["plain"], plain_intercept=B["plain"], center_coef=W["center"], center_intercept=B["center"])
    np.savez_compressed(os.path.join(out, "valid_features.npz"), ids=train.id.values[valid], features=Z.astype(np.float16), fold_rows=valid)
    lg = Z @ W["plain"].T + B["plain"]; pr = lg.argmax(1)
    bacc = float(np.mean([(pr == c)[y[valid] == c].mean() for c in np.unique(y[valid])])) if job != "final" else float("nan")
    if job == "final":
        model.save_pretrained(os.path.join(out, "adapter"), safe_serialization=True)
        save_file({f"norms.{k}": v.detach().cpu().contiguous() for k, v in norms.state_dict().items()}, os.path.join(out, "norms.safetensors"))
        tok.save_pretrained(os.path.join(out, "tokenizer"))
        np.savez_compressed(os.path.join(out, "anchor_features.npz"), ids=train.id.values[:3], features=Z[:3].astype(np.float32))
    json.dump(dict(job=job, valid_bacc=bacc, steps=step, skipped=skipped, secs=time.time() - t0, n_rows=int(len(rows)), cfg=cfg), open(os.path.join(out, "COMPLETE.json"), "w"))
    print(f"job {job} done: valid bacc {bacc:.4f} steps {step} skipped {skipped} ({time.time()-t0:.0f}s)", flush=True)
    del model, base, opt, norms, plain, centered
    if dev.type == "cuda": torch.cuda.empty_cache()

for job in jobs:
    run_job(job)
'''

T0 = time.time()
DATA_DIR = next(p for p in [pathlib.Path("/kaggle/input/competitions/name-that-llm"), pathlib.Path("/kaggle/input/name-that-llm"),
                            pathlib.Path(os.environ.get("NTL_DATA", "data"))] if (p / "train.csv").exists())
model_dir = os.environ.get("NTL_MODEL_DIR")
if not model_dir:
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try: j = json.load(open(c))
        except Exception: continue
        if j.get("model_type") == CFG["model_type"] and j.get("hidden_size") == CFG["hidden"] and j.get("num_hidden_layers") == CFG["n_layers"]:
            model_dir = os.path.dirname(c); break
assert model_dir, "base model not found"
print("base", model_dir, flush=True)
cfg = dict(CFG, train_csv=str(DATA_DIR / "train.csv"), model_dir=model_dir, labels=LABELS)
if CFG["extra_globs"] and CFG["extra_w"] > 0:
    parts = []
    for g in CFG["extra_globs"]:
        for f in sorted(glob.glob(g, recursive=True)):
            d = (pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f)).rename(columns={"label": "model"})
            parts.append(d[["text", "model", "prompt_id"]].dropna(subset=["text", "model"]).assign(prompt_id=lambda x: os.path.basename(f) + "_" + x.prompt_id.astype(str)))
    ex = pd.concat(parts, ignore_index=True); ex = ex[ex.model.isin(LABELS) & ~ex.model.isin(CFG["extra_exclude"])]
    if CFG.get("extra_cap"): ex = ex.sample(frac=1, random_state=0).groupby("model").head(CFG["extra_cap"]).reset_index(drop=True)   # per-class cap
    if CFG["smoke"]: ex = ex.sample(min(len(ex), 150), random_state=0)   # keep local smoke tests short
    ex.to_csv("extra_used.csv", index=False); cfg["extra_csv"] = "extra_used.csv"; print("extra rows", len(ex), ex.model.value_counts().to_dict(), flush=True)
if CFG["smoke"]:
    cfg.update(keep=4, layers=[1, 2, 3, 4], epochs=1, max_len=48, accum=2); JOBS = {0: ["0", "final"]}
open("worker.py", "w").write(WORKER)
import torch
ngpu = torch.cuda.device_count()
if ngpu < 2: JOBS = {0: [j for v in JOBS.values() for j in v]}
procs = []
for g, jl in JOBS.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if ngpu else "", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    procs.append((g, subprocess.Popen([sys.executable, "worker.py", json.dumps(cfg), ",".join(jl)], env=env, stdout=open(f"worker{g}.log", "w"), stderr=subprocess.STDOUT)))
for g, p in procs:
    p.wait(); print(f"---- worker {g} (rc {p.returncode})"); print(open(f"worker{g}.log").read()[-6000:])
done = sorted(glob.glob("out/job_*/COMPLETE.json")); summ = [json.load(open(d)) for d in done]
print("completed jobs:", [(s["job"], round(s["valid_bacc"], 4) if s["valid_bacc"] == s["valid_bacc"] else None, int(s["secs"])) for s in summ])
# OOF probabilities of the plain head over the 5 folds
oof = {}
for d in done:
    j = os.path.dirname(d)
    if "final" in j: continue
    ro = np.load(os.path.join(j, "readouts.npz")); vf = np.load(os.path.join(j, "valid_features.npz"), allow_pickle=True)
    lg = vf["features"].astype(np.float32) @ ro["plain_coef"].T + ro["plain_intercept"]
    for i, l in zip(vf["ids"], lg): oof[str(i)] = l
if oof:
    tr = pd.read_csv(cfg["train_csv"]); keep = tr.id.astype(str).isin(oof.keys()).values
    S = np.stack([oof[i] for i in tr.id.astype(str)[keep]]); yy = tr.model.map({l: i for i, l in enumerate(LABELS)}).values[keep]
    pr = S.argmax(1); print(f"OOF rows {keep.sum()} plain-head balanced acc {np.mean([(pr == c)[yy == c].mean() for c in np.unique(yy)]):.4f}", flush=True)
    np.savez_compressed(f"oof_{CFG['name']}.npz", ids=tr.id.values[keep], scores=S, kind="logit", classes=np.array(LABELS))
print(f"total {time.time()-T0:.0f}s")
