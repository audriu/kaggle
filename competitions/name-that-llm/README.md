# Name That LLM

Local workspace for [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) (Kaggle, **$50,000**: ten
prizes of $5,000, Featured **code competition**, deadline **13 Jan 2027 23:59 UTC**, entry/merger deadline 6 Jan).

**Task.** 18-way authorship attribution: which of 18 frontier LLMs (5 OpenAI, 4 Google, 5 Anthropic, Qwen, 2 ZAI,
Moonshot) wrote a 40–250-word passage. Train: 2,699 passages = 150 prompts × 18 models (one answer per model per
prompt, `p002` lacks `claude-fable-5`). Hidden test: ~4,000 passages on unseen prompts (public LB = 40%). Metric:
**balanced accuracy**. Submissions run as notebooks (≤ 9 h, internet off, public external data and models allowed),
5 per day, 2 final picks.

## What we submit

`kernels/submit/submit.py` (a Kaggle *script* kernel, pushed with the API) reads the hidden `test.csv`, computes
member predictions, blends them, recovers the test prompts by clustering and assigns labels per cluster, and writes
`submission.parquet`. It is submitted with `kaggle competitions submit -k vienas/ntl-submit -v <version>`.

## Layout

```text
data/                 competition files (gitignored; kaggle competitions download -c name-that-llm)
src/ntl/              shared code: folds (public-recipe prompt folds), style members, grouping/assignment, blend, simulation
scripts/
  cpu_members.py      OOF predictions of the CPU members -> oof/*.npz
  fit_blend.py        fit log-linear weights on OOF files, print nested CV + test-layout simulations -> blend/weights.json
  exp_grouping*.py    how much per-prompt assignment is worth (oracle vs TF-IDF clustering, several layouts)
  exp_extra*.py       fidelity/usefulness of the public extra datasets (exp_extra4: Oct-18 batch; build_extra_v3.py merges them)
kernels/
  submit/             submission kernel (self-contained script + kernel-metadata.json; blend weights embedded by scripts/install_blend.py)
  train_encoder/      Kaggle training kernel: encoder fine-tune (DeBERTa-v3-large) on prompt folds -> OOF + final weights
  train_lora/         Kaggle training kernel: decoder LoRA (Qwen3-8B-Base, 24 layers, layer-pooled plain+centered heads)
  train_frozen/       Kaggle training kernel: frozen-embedding heads (Qwen3-8B, Qwen2.5-7B, DeBERTa-v3-large, desklib detector, ELECTRA-large) + prompt-centred twins + saved OOF features
  train_encoder_x/, train_lora_x/   same recipes with capped public extra passages (extra_w 0.5, 150/class, no Kimi)
  train_lora_s7/, train_lora_x2/    LoRA seed variants (s7: train only, seed 7 — adopted; x2: extra data, seed 7 — no blend gain)
  train_lora_q25x/    Qwen2.5-7B-Base 20-layer extra-data LoRA (adopted: best single member, cross-family diversity)
  train_lora_q34/     Qwen3-4B-Base 36-layer LoRA (no blend gain next to the Qwen3-8B LoRAs)
  train_lora_x3/      extra-data LoRA on the merged extra set v3 (prepared, not run: v3 gave no blend gain for stylegram/frozen heads)
  scripts/cv_compare.py  10-seed nested-CV comparison of member sets — the gate for adding a member
oof/                  out-of-fold member predictions (gitignored)
blend/weights.json    current blend
notes/                progress log, research notes
```

## Day-to-day

```bash
cd /home/a/code/kaggle && source .venv/bin/activate && cd competitions/name-that-llm
python scripts/cpu_members.py                      # OOF for stylegram + style_lgb (CPU, ~2 min)
python scripts/fit_blend.py                        # weights + simulations -> blend/weights.json
python scripts/sim_twins.py --write                # simulate layouts, pick cluster-centred twin weights -> blend/twins.json (needs tmp/out_frozen2 + tmp/out_lora)
python scripts/install_blend.py                    # embed blend/weights.json (+ blend/twins.json) into kernels/submit/submit.py (script kernels upload only the code file)
kaggle kernels push -p kernels/train_encoder       # GPU training on Kaggle (edit CFG in the script first)
kaggle kernels status vienas/ntl-train-encoder; kaggle kernels output vienas/ntl-train-encoder -p tmp/out_encoder
kaggle kernels push -p kernels/submit              # run the submission kernel on the 3-row test
kaggle competitions submit -c name-that-llm -k vienas/ntl-submit -v <version> -f submission.parquet -m "..."
```

Kaggle kernel outputs (`oof_*.npz`, `out/job_final/...`) are attached to the submission kernel through
`kernel_sources` in its `kernel-metadata.json`; copy the `oof_*.npz` files into `oof/` to refit the blend.

## Key facts learned so far

- Prompt-grouped CV is mandatory (random splits read 0.22 for a model that scores 0.37 on the LB).
- Test-time prompt recovery works: content-word TF-IDF, average linkage cut 0.9 gives pair precision 0.96 / recall 0.95
  on 18-per-prompt layouts; Hungarian assignment inside clusters lifts the style blend 0.389 → 0.482 (18/prompt),
  → 0.421 (7/prompt, public-LB-like), → 0.437 (11/prompt, private-like).
- Public extra data: prvsiyan's all-18 passages help most classes (+0.03 for the style model at weight 0.3) but hurt
  GLM and Kimi, whose extra passages do not look like the training ones (straight quotes, no curly quotes); exclude them.
- The public state of the art (0.552) is a log-prob blend of style n-grams, frozen LLM embeddings, a Qwen3-8B LoRA,
  token-attention heads, cluster-centered twins and Sinkhorn balancing; no public notebook uses per-cluster assignment.
