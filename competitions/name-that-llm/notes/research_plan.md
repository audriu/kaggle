# Research plan

Goal: top 10 (prize). Public LB top is 0.600; strongest public notebook 0.552; our simulation says per-cluster
assignment is worth +0.03 (public layout) to +0.09 (18 per prompt) on top of any blend.

## Pipeline

1. Members with prompt-grouped OOF (same folds everywhere): stylegram SVC, stylometric LGBM (CPU);
   DeBERTa-v3-large full fine-tune; Qwen3-8B-Base LoRA with plain + centered heads (GPU, Kaggle kernels);
   later: frozen-LLM embedding heads, token-attention heads, more seeds/backbones, extra-data variants.
2. Log-linear blend, weights by L-BFGS on OOF (class-balanced NLL), temperatures per member.
3. Test time: TF-IDF (+ embedding) clustering of test passages, cluster-centered twins, per-cluster Sinkhorn,
   per-cluster Hungarian assignment; everything validated on simulated layouts (18 / 11 / 7 / 1–8 per prompt).

## Open questions

- Hidden test layout (18 per prompt? public/private split by prompt or passage?). The host has not answered.
  Hedge: assignment with capacity ceil(n/18), Sinkhorn strength tuned on mixed layouts.
- Extra data: selected classes only (no GLM/Kimi); weight 0.3–0.5; verify on the neural members.
- Self-generated data via APIs would need paid keys (not available here); Kaggle Benchmarks credits are an option
  for the user to decide.
