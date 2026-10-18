# Name That LLM - encoder fine-tuning kernel (training only, no submission).
# Fine-tunes a bidirectional encoder (default microsoft/deberta-v3-large) as an 18-way classifier on prompt-grouped
# folds, writes out-of-fold probabilities for every fold job and fp16 weights for the 'final' job (all rows).
# Optional public extra passages (new prompts) are added to the TRAINING side only, with class-balanced weights.
# Two workers (one per GPU) each run a list of jobs; every finished job writes its files immediately.
# v2: explicit precision policy (bf16 only on Ampere+, else fp16 + grad scaler), separate head learning rate,
#     worker logs streamed to the kernel log, and a collapse guard: a job whose epoch-0 loss never leaves the
#     uniform prior (ln 18 = 2.89) is restarted at half the learning rate in fp32 with a new seed.
import glob, json, os, pathlib, subprocess, sys, time
import numpy as np, pandas as pd

# ---- run configuration (edit per push; env overrides are for local smoke tests only)
CFG = dict(
    name="debv3l_x2",
    backbone="microsoft/deberta-v3-large",
    max_len=512, lr=1.5e-5, head_lr=1e-4, epochs=6, bs=4, accum=4, warmup=0.10, wd=0.01, label_smoothing=0.05,
    seed=42, fold_seed=0,
    precision="auto",       # auto = bf16 on Ampere+ else fp16 with a grad scaler; or one of fp16 | bf16 | fp32
    collapse_loss=2.85,     # mean train loss over the last quarter of epoch 0 above this => restart (see retries)
    retries=2,              # restarts per job: lr halves each time, precision fp32, new seed
    grad_ckpt=False,        # gradient checkpointing (turned on automatically after a CUDA out-of-memory)
    extra_w=0.5,            # share of each class's training weight given to extra passages (0 = competition data only)
    extra_globs=["/kaggle/input/**/prvsiyans-extended-name-that-llm-dataset/passages.csv", "/kaggle/input/**/ntl_extra_passages_claude.csv", "/kaggle/input/**/ntl-own-prompt-generations-v1/generated_v1.csv", "/kaggle/input/**/ntl_proxy_generations.csv"],  # files under /kaggle/input with text/model columns, e.g. "/kaggle/input/**/passages_clean.csv"
    extra_exclude=["kimi-k3"], extra_cap=150,   # per-class cap on extra rows (GPU time: pool grows by ~2.3x at 150/class)  # labels never taken from extra data
    smoke=False,
)
for k, v in list(CFG.items()):   # local overrides: NTL_NAME, NTL_BACKBONE, NTL_LR, NTL_EPOCHS, NTL_SEED, NTL_EXTRA_W, NTL_SMOKE, ...
    e = os.environ.get("NTL_" + k.upper())
    if e is not None:
        CFG[k] = type(v)(e) if not isinstance(v, (bool, list)) else (e == "1" if isinstance(v, bool) else [x for x in e.split(",") if x])
JOBS = {0: ["0", "2", "4"], 1: ["1", "3", "final"]}
LABELS = sorted(["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview",
                 "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6",
                 "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"])

WORKER = r'''
import json, math, os, random, sys, time
import numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

cfg = json.loads(sys.argv[1]); jobs = sys.argv[2].split(",")
LABELS = cfg["labels"]; L2I = {l: i for i, l in enumerate(LABELS)}
train = pd.read_csv(cfg["train_csv"])
if cfg["smoke"]:
    train = train[train.prompt_id.isin(sorted(train.prompt_id.unique())[:12])].reset_index(drop=True)
y = train.model.map(L2I).values; pid = train.prompt_id.astype(str).values
prompts = np.array(sorted(train.prompt_id.unique())); rng0 = np.random.RandomState(cfg["fold_seed"]); rng0.shuffle(prompts)
fold = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).values   # same folds as the public recipes
extra = pd.read_csv(cfg["extra_csv"]) if cfg.get("extra_csv") else None
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cap = torch.cuda.get_device_capability(0) if dev.type == "cuda" else (0, 0)
tok = AutoTokenizer.from_pretrained(cfg["backbone"])
texts = train.text.astype(str).tolist(); ex_texts = extra.text.astype(str).tolist() if extra is not None else []
ex_y = extra.model.map(L2I).values if extra is not None else np.zeros(0, int)

def encode(ts):
    return tok(ts, truncation=True, max_length=cfg["max_len"], add_special_tokens=True)["input_ids"]
ids_tr = encode(texts); ids_ex = encode(ex_texts) if ex_texts else []

def precision_of(p):
    if p == "auto": p = "bf16" if cap[0] >= 8 else "fp16"
    return p if dev.type == "cuda" else "fp32"

def run_job(job, attempt=0, mem=0):
    t0 = time.time(); out = os.path.join("out", f"job_{job}"); os.makedirs(out, exist_ok=True)
    bs, accum, grad_ckpt = max(1, cfg["bs"] >> mem), cfg["accum"] << mem, (cfg["grad_ckpt"] or mem > 0)
    prec = precision_of(cfg["precision"]) if attempt == 0 else "fp32"
    lr = cfg["lr"] * 0.5 ** attempt; head_lr = cfg["head_lr"] * 0.5 ** attempt
    amp_dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(prec); use_amp = amp_dt is not None
    seed = cfg["seed"] + (99 if job == "final" else int(job)) + 1000 * attempt
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"job {job} attempt {attempt}: precision {prec} lr {lr:.1e} head_lr {head_lr:.1e} seed {seed} bs {bs} x accum {accum} grad_ckpt {grad_ckpt} (gpu cap {cap})", flush=True)
    fit = np.arange(len(train)) if job == "final" else np.where(fold != int(job))[0]
    val = np.arange(min(3, len(train))) if job == "final" else np.where(fold == int(job))[0]
    # training pool: competition rows (+ extra rows with class-balanced weights)
    pool_ids = [ids_tr[i] for i in fit]; pool_y = list(y[fit]); pool_w = [1.0] * len(fit)
    if len(ex_texts) and cfg["extra_w"] > 0:
        w = cfg["extra_w"]; yfit = y[fit]
        for c in np.unique(ex_y):
            n_tr, n_ex = (yfit == c).sum(), (ex_y == c).sum()
            if n_tr == 0 or n_ex == 0: continue
            for i in np.where(yfit == c)[0]: pool_w[i] = 1 - w
            for i in np.where(ex_y == c)[0]:
                pool_ids.append(ids_ex[i]); pool_y.append(int(c)); pool_w.append(w * n_tr / n_ex)
    pool_y = np.array(pool_y); pool_w = np.array(pool_w, dtype=np.float32)
    kw = dict(num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I, dtype=torch.float32)   # fp32 master weights: transformers 5 otherwise keeps the checkpoint dtype (deberta-v3-large is stored in fp16 -> no learning)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(cfg["backbone"], attn_implementation="sdpa", **kw)
    except ValueError:   # e.g. DeBERTa-v2 has no sdpa path in this transformers version
        model = AutoModelForSequenceClassification.from_pretrained(cfg["backbone"], attn_implementation="eager", **kw)
    if grad_ckpt: model.gradient_checkpointing_enable()
    model = model.to(dev)
    no_decay = ("bias", "norm", "layernorm")
    head = {n for n, _ in model.named_parameters() if n.startswith(("pooler", "classifier", "score"))}
    groups = []
    for is_head in (False, True):
        for decay in (True, False):
            ps = [p for n, p in model.named_parameters() if (n in head) == is_head and (not any(k in n.lower() for k in no_decay)) == decay]
            if ps: groups.append({"params": ps, "weight_decay": cfg["wd"] if decay else 0.0, "lr": head_lr if is_head else lr})
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = math.ceil(len(pool_ids) / bs) // accum; total = max(1, steps_per_epoch * cfg["epochs"]); warm = max(1, int(cfg["warmup"] * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dt == torch.float16))
    lossf = nn.CrossEntropyLoss(reduction="none", label_smoothing=cfg["label_smoothing"])
    rng = np.random.RandomState(seed)

    def batches(idx):   # length-bucketed shuffled batches
        idx = rng.permutation(idx); chunks = [idx[i:i + bs * 20] for i in range(0, len(idx), bs * 20)]
        out = []
        for ch in chunks:
            ch = ch[np.argsort([len(pool_ids[i]) for i in ch])]
            out += [ch[i:i + bs] for i in range(0, len(ch), bs)]
        return [out[i] for i in rng.permutation(len(out))]

    def collate(ids_list):
        b = tok.pad({"input_ids": ids_list}, return_tensors="pt"); return {k: v.to(dev) for k, v in b.items()}

    @torch.no_grad()
    def predict(ids_list):
        model.eval(); P = np.zeros((len(ids_list), len(LABELS)), np.float32); order = np.argsort([len(x) for x in ids_list])
        for s in range(0, len(order), 32):
            pos = order[s:s + 32]
            with torch.autocast("cuda", dtype=amp_dt if use_amp else torch.float16, enabled=use_amp):
                P[pos] = model(**collate([ids_list[i] for i in pos])).logits.float().softmax(-1).cpu().numpy()
        model.train(); return P

    model.train(); step = 0; hist = []; skipped = 0
    try:
      for ep in range(cfg["epochs"]):
          losses = []
          for bi, b in enumerate(batches(np.arange(len(pool_ids)))):
              with torch.autocast("cuda", dtype=amp_dt if use_amp else torch.float16, enabled=use_amp):
                  logits = model(**collate([pool_ids[i] for i in b])).logits.float()
              wt = torch.as_tensor(pool_w[b], device=dev)
              loss = (lossf(logits, torch.as_tensor(pool_y[b], device=dev)) * wt).sum() / wt.sum()
              scaler.scale(loss / accum).backward(); losses.append(float(loss.detach()))
              if (bi + 1) % accum == 0:
                  scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                  before = scaler.get_scale() if scaler.is_enabled() else 1.0
                  scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
                  if scaler.is_enabled() and scaler.get_scale() < before: skipped += 1
                  if step % 25 == 0:
                      print(f"job {job} ep {ep} step {step}/{total} loss {np.mean(losses[-50:]):.4f} skipped {skipped} ({time.time() - t0:.0f}s)", flush=True)
          tail = float(np.mean(losses[-max(1, len(losses) // 4):]))
          if ep == 0 and attempt < cfg["retries"] and (not np.isfinite(tail) or tail > cfg["collapse_loss"]):
              print(f"job {job} attempt {attempt}: no learning after epoch 0 (tail loss {tail:.4f}) -> restart", flush=True)
              del model, opt; torch.cuda.empty_cache() if dev.type == "cuda" else None
              return run_job(job, attempt + 1, mem)
          P = predict([ids_tr[i] for i in val]); pr = P.argmax(1)
          bacc = float(np.mean([(pr == c)[y[val] == c].mean() for c in np.unique(y[val])])) if job != "final" else float("nan")
          hist.append(dict(epoch=ep, loss=float(np.mean(losses)), tail_loss=tail, val_bacc=bacc))
          print(f"job {job} epoch {ep} loss {np.mean(losses):.4f} (tail {tail:.4f}) val bacc {bacc:.4f} skipped {skipped} ({time.time() - t0:.0f}s)", flush=True)
    except torch.OutOfMemoryError:
        if mem >= 2: raise
        print(f"job {job}: CUDA out of memory at bs {bs} -> retrying with bs {max(1, bs // 2)} x accum {accum * 2} and gradient checkpointing", flush=True)
        del model, opt; torch.cuda.empty_cache(); return run_job(job, attempt, mem + 1)
    np.savez_compressed(os.path.join(out, "valid_probs.npz"), ids=train.id.values[val], probs=P, classes=np.array(LABELS), fold_rows=val)
    if job == "final":
        model.half().save_pretrained(os.path.join(out, "model"), safe_serialization=True); tok.save_pretrained(os.path.join(out, "model"))
    json.dump(dict(job=job, history=hist, secs=time.time() - t0, n_pool=len(pool_ids), attempt=attempt, precision=prec, lr=lr, bs=bs, accum=accum, grad_ckpt=grad_ckpt, cfg=cfg),
              open(os.path.join(out, "COMPLETE.json"), "w"))
    print(f"job {job} done ({time.time() - t0:.0f}s)", flush=True)
    del model, opt; torch.cuda.empty_cache() if dev.type == "cuda" else None

for job in jobs:
    run_job(job)
'''

T0 = time.time()
DATA_DIR = next(p for p in [pathlib.Path("/kaggle/input/competitions/name-that-llm"), pathlib.Path("/kaggle/input/name-that-llm"),
                            pathlib.Path(os.environ.get("NTL_DATA", "data"))] if (p / "train.csv").exists())
cfg = dict(CFG, train_csv=str(DATA_DIR / "train.csv"), labels=LABELS)
# optional extra passages: any attached dataset with text/model columns listed in extra_globs
extra_globs = CFG["extra_globs"]
if extra_globs and CFG["extra_w"] > 0:
    parts = []
    for g in extra_globs:
        for f in sorted(glob.glob(g, recursive=True)):
            d = pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f)
            d = d.rename(columns={"label": "model"})
            parts.append(d[["text", "model"]].dropna())
    ex = pd.concat(parts, ignore_index=True); ex = ex[ex.model.isin(LABELS) & ~ex.model.isin(CFG["extra_exclude"])]
    if CFG.get("extra_cap"): ex = ex.sample(frac=1, random_state=0).groupby("model").head(CFG["extra_cap"]).reset_index(drop=True)   # per-class cap
    if CFG["smoke"]: ex = ex.sample(min(len(ex), 150), random_state=0)   # keep local smoke tests short
    ex.to_csv("extra_used.csv", index=False); cfg["extra_csv"] = "extra_used.csv"
    print("extra rows", len(ex), ex.model.value_counts().to_dict(), flush=True)
if CFG["smoke"]:
    cfg.update(backbone=os.environ.get("NTL_SMOKE_BACKBONE", "microsoft/deberta-v3-xsmall"), epochs=1, max_len=64, bs=4, accum=1, retries=0)
    JOBS = {0: ["0", "final"]}
open("worker.py", "w").write(WORKER)
import torch
ngpu = torch.cuda.device_count()
if ngpu < 2:   # single GPU or CPU: run all jobs sequentially
    JOBS = {0: [j for v in JOBS.values() for j in v]}
procs = []
for g, jl in JOBS.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if ngpu else "", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    procs.append((g, subprocess.Popen([sys.executable, "worker.py", json.dumps(cfg), ",".join(jl)], env=env, stdout=open(f"worker{g}.log", "w"), stderr=subprocess.STDOUT)))
offs = {g: 0 for g, _ in procs}

def stream():   # forward new worker-log lines to the kernel log so progress is visible while running
    for g, _ in procs:
        try:
            with open(f"worker{g}.log", errors="replace") as f:
                f.seek(offs[g]); new = f.read(); offs[g] = f.tell()
        except FileNotFoundError:
            continue
        lines = [l for l in new.splitlines() if l.strip() and not l.startswith(("Loading weights", "Key ", "----")) and "| UNEXPECTED |" not in l and "| MISSING |" not in l]
        if lines: print("\n".join(f"[w{g}] {l}" for l in lines), flush=True)
while any(p.poll() is None for _, p in procs):
    time.sleep(30); stream()
stream()
for g, p in procs: print(f"---- worker {g} finished (rc {p.returncode})", flush=True)
done = sorted(glob.glob("out/job_*/COMPLETE.json"))
summ = [json.load(open(d)) for d in done]
print("completed jobs:", [(s["job"], round(s["history"][-1]["val_bacc"], 4), s.get("attempt", 0), s.get("precision"), int(s["secs"])) for s in summ])
# assemble OOF over the 5 folds
oof = {}
for d in done:
    j = json.load(open(d))["job"]
    if j != "final":
        z = np.load(os.path.join(os.path.dirname(d), "valid_probs.npz"), allow_pickle=True)
        for i, p in zip(z["ids"], z["probs"]): oof[str(i)] = p
if oof:
    tr = pd.read_csv(cfg["train_csv"]); keep = tr.id.astype(str).isin(oof.keys()).values
    P = np.stack([oof[i] for i in tr.id.astype(str)[keep]]); yy = tr.model.map({l: i for i, l in enumerate(LABELS)}).values[keep]
    pr = P.argmax(1); bacc = np.mean([(pr == c)[yy == c].mean() for c in np.unique(yy)])
    np.savez_compressed(f"oof_{CFG['name']}.npz", ids=tr.id.values[keep], scores=P, kind="prob", classes=np.array(LABELS))
    print(f"OOF rows {keep.sum()} balanced acc {bacc:.4f}", flush=True)
print(f"total {time.time() - T0:.0f}s")
