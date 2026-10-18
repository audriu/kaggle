# Progress log

- 2026-10-17 — Joined (rules accepted earlier). Data downloaded; EDA: 150 prompts × 18 models, 36–274 words, strong
  punctuation fingerprints per family (Claude/GLM: em-dashes + straight quotes; GPT/Gemini/Kimi/Qwen: curly quotes;
  Gemini: no em-dashes). Read the top public notebooks and the forum (extra-data debate, test-layout question unanswered).
- 2026-10-17 — Built `src/ntl` + experiments. Style blend OOF 0.390 (nested). Oracle per-prompt Hungarian 0.486;
  TF-IDF clustering (cut 0.9) + Hungarian 0.482 on 18/prompt, 0.421 on 7/prompt. Extra-data study: prvsiyan +0.03,
  GLM/Kimi extra harmful.
- 2026-10-17 — Submission v1 (CPU: stylegram + LGBM, clustering + Hungarian) submitted. Training kernels pushed:
  `ntl-train-encoder` v1 (DeBERTa-v3-large, train only) and `ntl-train-lora` v1 (Qwen3-8B-Base 24-layer LoRA,
  plain + centered heads, train only); both queued for GPUs.
- 2026-10-17 (later) — Submission v1 public LB **0.373** (CPU style blend + clustering/assignment); the assignment gain
  is small on the LB, so the hidden test is probably sparse per prompt. Simulations: global Sinkhorn 0.5 + per-cluster
  Sinkhorn 0.5 + Hungarian is never worse than plain in any layout (1/prompt +0.006 … 18/prompt +0.093); adopted.
  Extra data for the style model: all public sources minus Kimi at weight 0.5 → CV 0.4123 (+0.040). Extra-data
  variants of the training kernels prepared (`train_encoder_x`, `train_lora_x`), waiting for Kaggle GPU slots (max 2).
  `hoody` (12 cores, 6 GB, no GPU) set up for CPU experiments via `scripts/remote.sh`; stylegram sweep running there.
- 2026-10-18 00:30 — `ntl-train-encoder` v2 (P100) finished but **collapsed**: all 6 jobs sit at loss ln 18 = 2.89,
  val balanced acc 0.056, predictions uniform (0.052-0.060). Identical curves across 5 seeds => systematic, not seed luck.
  Suspects: (a) `torch.cuda.is_bf16_supported()` is True on P100/T4 (emulated bf16) so the run used bf16 autocast on a
  pre-Ampere GPU; (b) lr 1.5e-5 / warmup 6 % is on the fragile side for deberta-v3-large with 2.2k rows x 18 classes.
  Local CPU diagnosis is impractical (deberta-v3-xsmall: 44 s per step in transformers 5.19 on this CPU).
  Rewrote `kernels/train_encoder/train_encoder.py` (v2): explicit precision policy (bf16 only on cap>=8, else fp16 +
  GradScaler), head lr 10x body, lr 8e-6 / warmup 10 % / 5 epochs, worker logs streamed to the kernel log, and a
  collapse guard (epoch-0 tail loss > 2.85 => restart the job at half lr in fp32 with a new seed, up to 2 retries).
  `train_encoder_x` regenerated from v2 (name debv3l_x2). Pushed `ntl-train-frozen` v1 (T4 x2, RUNNING); its
  frozen deberta-v3-large head doubles as a forward-pass sanity check. `ntl-train-lora` v1 RUNNING since ~22:20.
  Kernel titles must slugify to the kernel id (409 otherwise) — fixed for frozen / encoder_x / lora_x.
- 2026-10-18 01:12 — `ntl-train-frozen` v1 done in 27 min (T4 x2): OOF balanced acc q3_8b_L7_22 0.4438, q25_7b_L6_20 0.4267,
  debv3l_L24_6 0.4490 (prompt-centered oracle heads 0.56-0.58). Blend of 5 members: nested CV **0.4945** (was 0.3897);
  simulations plain 0.49 -> clustered Hungarian 0.535 (7/prompt), 0.578 (11/prompt), 0.682 (18/prompt).
  Submit kernel v2 pushed (GPU T4 x2, frozen members via kernel_sources, Qwen models via model_sources, TWIN_W off).
  Encoder v2 kernel pushed (version 3, P100 queue). LoRA v1 still running.
- 2026-10-18 01:14 — GPU cap (2 sessions) hit: stopped the just-started encoder v2 run (version 3) to let the submit
  kernel through; `ntl-submit` version 2 queued (T4 x2). Encoder v2 to be re-pushed when the LoRA run finishes.
- 2026-10-18 01:05 — Encoder collapse diagnosis (local, CPU, transformers 5.19): DeBERTa-v2 forward is correct (a
  fine-tuned NLI checkpoint predicts all probes right), every parameter receives finite non-zero gradients, and the
  v2 loop learns a trivial label with deberta-v3-xsmall in fp32 (loss 2.2 -> 1.3 in 35 steps; RoBERTa 2.8 -> 0.9,
  DeBERTa shows a transient spike with a 10x head lr -> head lr lowered to 5e-5). Frozen deberta-v3-large features
  under explicit fp16 on T4 give OOF 0.449, so the P100 run's emulated-bf16 autocast (is_bf16_supported() is True
  there) is the prime suspect for the constant features. v2 forces fp16 + GradScaler below compute capability 8.
  Note: DeBERTa-v2 is ~10x slower per step than RoBERTa on CPU in this transformers version (44 s/step for xsmall).
- 2026-10-18 01:22 — Submission **v2** sent (kernel `ntl-submit` version 3; ref 57327661, pending): frozen Qwen3-8B /
  Qwen2.5-7B / DeBERTa-large heads + stylegram + LGBM, nested CV 0.4945. Lesson: Kaggle script kernels upload only
  the code file, so `blend.json` never reached the kernel (version 2 silently ran the style-only default) ->
  `scripts/install_blend.py` now embeds the weights into `submit.py`.
  `ntl-train-lora` v1 done (2.9 h, T4 x2, no skipped fp16 steps): fold bacc 0.454 / 0.465 / 0.469 / 0.493 / 0.502,
  OOF plain head **0.4764** (best single member). Added to the blend (refit running); submit kernel wired with
  LORAS q8lora_v1 (TWIN_W 0.15) + kernel_sources ntl-train-lora. Encoder v2 re-pushed (version 4, queued).
- 2026-10-18 01:24 — Blend with the LoRA: nested CV **0.5053** (LoRA 0.60, frozen DeBERTa 0.31, stylegram 0.44, LGBM 0.11;
  frozen Qwen3 0.0 and Qwen2.5 0.005 -> dropped from the submission). Simulations: 7/prompt 0.542, 11/prompt 0.583,
  18/prompt 0.696. Submit kernel v4 pushed with LoRA + frozen DeBERTa + style members.
- 2026-10-18 01:27 — **Root cause of the encoder collapse**: transformers 5 `from_pretrained` keeps the checkpoint dtype
  and `microsoft/deberta-v3-large` is stored in fp16, so v1 trained pure-fp16 weights (updates of 1e-5 round away ->
  loss pinned at ln 18). v2 (fp16 GradScaler) exposed it: "ValueError: Attempting to unscale FP16 gradients".
  Fix: load with `dtype=torch.float32` (fp32 master weights + fp16 autocast). Same cause behind the 44 s/step CPU
  tests (fp16 matmuls on CPU). Encoder v2 re-pushed as version 5.
- 2026-10-18 01:29 — fp32 fix confirmed locally: v2 worker with deberta-v3-xsmall on 12 prompts now runs 1.2 s/step on
  CPU (was 44 s) and learns (train loss 2.90 -> 2.30 over 6 epochs, val bacc 0.056 -> 0.148 on 43 rows).
- 2026-10-18 01:31 — Submission **v3** sent (kernel version 4; ref 57327812, pending): LoRA + frozen DeBERTa + style,
  nested CV 0.5053; parity of both neural members vs their training kernels 0.0000. 2 submissions left today.
  Encoder v5 hit CUDA OOM at bs 8 x 512 with fp32 weights -> bs 4 x accum 4 + OOM fallback (halve bs, enable
  gradient checkpointing, restart); version 6 queued. Smoke mode now caps extra rows at 150 so the `_x` variants can
  be smoke-tested locally; both smokes running.
- 2026-10-18 01:37 — `_x` variants: extra rows capped per class (`extra_cap=150`, shuffled `groupby().head()`) so the
  GPU pool grows ~2.3x instead of ~6x (all 12k extra rows would blow the 9 h limit for the LoRA). Both variants
  smoke-tested locally with the real public files (extra-data weighting path exercised). `ntl-train-lora-x` v1
  pushed (T4 x2, queued; expected ~6 h). `ntl-train-encoder-x` waits for the encoder v6 result first.
- 2026-10-18 01:45 — **Submission v3 public LB 0.503** (v1 0.373). Nested CV was 0.505 and the clustered-Hungarian
  simulation 0.527-0.542, so the test-time assignment adds little on the public split -> sparse per-prompt layout
  confirmed; model strength is the lever. v2 (frozen heads only) still pending.
- 2026-10-18 01:51 — LB analysis: v3 0.503 ranks ~507/1374; 10th place 0.562, 232 teams >= 0.55 (the public 0.552
  lineage). Its biggest cheap lever that I lacked: "adaptive centering" twins for every frozen head (cluster-centred
  embeddings scored by prompt-centred heads on clusters of size 2..18; their simulation +0.024 even at 1-2/prompt).
  Submit kernel v5 pushed: TWIN_W debv3l 0.25 / q3_8b 0.33 / q25_7b 0.21 (public recipe values), LoRA 0.15; the two
  zero-weight Qwen heads re-added for their twins; twin rows capped at cluster size 18. (Stopped the 15-min-old
  LoRA-x run to free the GPU slot; re-push after.) Frozen kernel v2: saves OOF features + per-fold centred heads for
  a local twin-weight simulation, per-layer pooling, + desklib AI-detector (L23 mean+style) and ELECTRA-large
  (L21 mean + L24 style) encoders from the public recipe; smoke-tested.
- 2026-10-18 01:55 — Submission **v4** sent (kernel version 5): v3 + frozen-head twins. 1 submission left today (reset 03:00).
  Frozen kernel v2 pushed (queued): + desklib detector (trunk keys live under a `model.` prefix -> explicit remap,
  0 missing keys) + ELECTRA-large, per-layer mean/style pooling, saved OOF features + per-fold centred heads.
  Submit kernel: per-layer pooling ported, specs for the two new encoders registered (inactive until blended).
- 2026-10-18 01:56 — `scripts/sim_twins.py`: simulates layouts with cluster-centred twin evidence and picks TWIN_W by
  coordinate ascent (global + cluster Sinkhorn + Hungarian decode, eligible clusters 2..18). LoRA-only check: twin
  0.6 lifts the fit-layout mean 0.548 -> 0.569 (1-5/prompt 0.528 -> 0.548, 7/prompt 0.568 -> 0.589). `install_blend.py`
  now also writes TWIN_W from blend/twins.json. Full run once frozen v2 (feats_*.npz) lands.
- 2026-10-18 01:57 — Kaggle GPU quota: 5 h 31 min of 30 h used this week (T4 x2 sessions count as wall time).
  Planned: encoder v6 (~2 h, running), frozen v2 (~0.6 h, queued), LoRA-x (~6 h), encoder-x (~4 h).
- 2026-10-18 02:00 — Submission v2 (frozen Qwen3/Qwen2.5/DeBERTa heads + style, no LoRA) public LB **0.480**;
  v3 (+LoRA) 0.503. v4 (+twins) pending.
- 2026-10-18 02:13 — Encoder v6 (fp32 weights + fp16 autocast, lr 8e-6, 5 epochs, bs 4x4) **trains** (loss 2.9 -> 1.65,
  1-3 skipped scaler steps per job, no collapse) but is underfit: fold bacc 0.313/0.314/0.324/0.328/0.346, OOF
  **0.3249** (< its frozen head 0.449), val still rising at epoch 4. 39 min on T4 x2. Next config (v7, name debv3l_v3):
  lr 1.5e-5, head 1e-4, 6 epochs. Blend refit with debv3l_v2 running to measure its marginal value.
- 2026-10-18 02:16 — Blend + fine-tuned DeBERTa (debv3l_v2, OOF 0.325): weight 0.31, nested CV 0.5053 -> **0.5071**
  (+0.002, diversity). Deferred from the submission until the stronger rerun (v7) lands; 6-member weights restored.
  `ntl-train-lora-x` re-pushed (version 2, queued; ~6 h).
- 2026-10-18 02:21 — Submission v4 (frozen-head twins, public-recipe weights) public LB **0.505** (+0.002 over v3).
  Cluster-based evidence is ~worthless on the public split for this pipeline (Hungarian/Sinkhorn earlier, twins now)
  => keep them (harmless) but invest GPU time in members: frozen v2 encoders, LoRA-x, DeBERTa v7, more LoRA seeds.
  Possible later probe: a submission without clustering to confirm its zero effect on LB.
- 2026-10-18 02:36 — Frozen v2 done (41 min): desk_L23ms OOF 0.4260 (centred oracle 0.5135), electl_L21_24s 0.4383
  (0.5435); older heads reproduced exactly. feats_*.npz saved for the twin simulation. Blend refit (8 members) +
  sim_twins running. Encoder v7 (debv3l_v3: lr 1.5e-5, head 1e-4, 6 epochs) pushed into the free slot.
- 2026-10-18 02:43 — 8-member blend nested CV **0.5093** (LoRA 0.52, stylegram 0.41, desk 0.23, debv3l 0.15, electl 0.11,
  LGBM 0.08; Qwen frozen heads 0). sim_twins (3 draws, layouts 1-2/1-5/3-8/7): TWIN_W debv3l 0.6, q25 0.45, LoRA 0.2,
  others 0 -> simulated fit-layout mean 0.554 -> 0.577. Installed with debv3l capped at 0.45; q3 head dropped from the
  submission (no plain or twin weight). Submit kernel v6 queued behind the two running GPU jobs.
- 2026-10-18 02:50 — Orthographic regularities in train: curly quotes 0 % in all Claude + GLM (6-58 % elsewhere), em-dash
  0 % in gemini-3.1/3.6/3.8 (63-83 % in Claude/GLM), straight apostrophe 0 % in gpt-5.6-sol. As log penalties
  (-4/-4/-2/-3) they change 4 of 2699 OOF predictions (+0.0004: the stylegram already encodes them); added to the
  submit kernel anyway (cheap, safe), before the global Sinkhorn.
- 2026-10-18 03:30 — Encoder v7 (lr 1.5e-5, head 1e-4, 6 epochs): fold bacc 0.343/0.362/0.348/0.380/0.350, OOF **0.3564**
  (v6 0.325; still < frozen head 0.449; train loss 1.2 and falling) — 46 min. Minor member; blend refit running.
  Submit kernel v6 (8 members + tuned twins + vetoes) pushed at 03:28, running. Prepared `train_lora_s7` (same
  LoRA recipe, seed 7) for the next free slot — the LoRA is the strongest member, a second seed adds diversity.
- 2026-10-18 03:36 — Blend + debv3l_v3: weight 0.18 but nested CV 0.5093 -> 0.5097 (+0.0004): not worth the extra
  inference member for now; 8-member blend kept for the running submit v6. `train_lora_s7` push loop armed (waits for
  a GPU slot behind LoRA-x and submit v6).
- 2026-10-18 03:34 — Submission **v5** sent (kernel version 6): 8-member blend (nested CV 0.5093) + simulated twins +
  orthographic vetoes. 4 submissions left today.
- 2026-10-18 04:16 — Submission v5 public LB **0.515** (+0.010; nested CV 0.5093). LoRA-x and LoRA-s7 both running.
  Next (CPU, while GPUs are busy): deploy the extra-data stylegram (exp_extra: 0.3715 -> 0.4123 OOF with all public
  sources minus Kimi at weight 0.5) as a blend member and in the submit kernel.
- 2026-10-18 04:19 — Submit kernel: new member `stylegram_x` (same four TF-IDF views + LinearSVC, fitted on train +
  11,949 public extra passages with class-balanced weights, share 0.5, Kimi excluded; datasets attached via
  dataset_sources; local dry run 52 s). `scripts/cpu_members_x.py` computes its OOF (running).
- 2026-10-18 04:21 — `stylegram_x` OOF **0.4123** (plain 0.3726). Blend refit: stylegram_x weight 0.79 replaces the plain
  stylegram (0.0); nested CV 0.5093 -> **0.5279**. Re-tuning twins, then submit kernel v7 (queued behind the LoRA runs).
- 2026-10-18 04:23 — Frozen kernel v3 (written, smoke running): embeds capped public extra passages (300/class) and
  fits plain "_x" heads (train + extra, class-balanced weights) for all 5 backbones; twins stay train-only. Zero extra
  test-time cost: the submit kernel now caches features per backbone and applies both heads; local dry runs can
  point NTL_INPUT / NTL_FROZEN_JSON at smoke outputs.
- 2026-10-18 04:25 — Frozen v3 smoke OK (extra rows embedded, `_x` heads + OOF written); submit kernel dry run against
  the smoke outputs passes end to end (parity, cached features for `_x` heads, vetoes, decode). Frozen v3 to be pushed
  after the submit v7 run takes its slot. LORAS pre-wired for q8lora_s7 / q8lora_x1.
- 2026-10-18 06:49 — Seed-7 LoRA (`ntl-train-lora-s7` v1, 3.1 h) OOF **0.4808** (seed-42 LoRA 0.4764; the two agree on 74 % of rows, mean-logit 0.4882). Blend refit with both LoRAs: nested CV 0.5279 -> **0.5316** (weights q8lora_v1 0.26, q8lora_s7 0.28, stylegram_x 0.78, desk 0.20, deb 0.14, electra 0.09, lgb 0.03, q25 0). Twins re-tuned (sim_twins v3), blend installed, `vienas/ntl-train-lora-s7` added to submit kernel_sources. Submit kernel v7 (stylegram_x blend) started 06:41 after the slot freed; v8 (two LoRAs) goes next.
- 2026-10-18 06:54 — Submit kernel v7 ran clean (stylegram_x: 12,714 extra rows from the 4 public sets on Kaggle vs 11,949 locally; all parity checks <= 0.0008). **Submitted as v6** (id 57337340; 3 submissions left today). Submit kernel v8 (two LoRAs, nested CV 0.5316) pushed into the freed slot; frozen v3 (extra-data heads) queued behind it. New kernel `kernels/train_lora_q34`: Qwen3-4B-Base full depth (36 layers, pooled layers 4-36, 20,480 features) — diversity vs the truncated 8B LoRAs; not yet pushed.
- 2026-10-18 07:05 — Submit kernel v8 ran clean (both LoRAs parity 0.0000, 419 s on the 3-row sample). **Submitted as v7** (nested CV 0.5316). Next kernel version v9 is a probe: identical blend with POSTPROCESS=False (no clusters / twins / vetoes / Sinkhorn / Hungarian) to measure what the post-processing is worth on the public leaderboard. Then frozen v3 (+ Qwen3-4B frozen head q34_L7_22) and the Qwen3-4B LoRA kernel take the GPU slots.
- 2026-10-18 07:07 — Frozen v3 (extra-data `_x` heads for all backbones + new Qwen3-4B head `q34_L7_22`) started 07:02 on T4 x2. Kaggle's active-events panel shows competition scoring reruns as events but they do not count toward the 2-session cap (frozen v3 was accepted while v6 was scoring and lora-x running). Probe v9 (POSTPROCESS=False) waits for a slot (~08:30 when lora-x ends), then the Qwen3-4B LoRA kernel.
- 2026-10-18 08:28 — **Public LB: v6 (stylegram_x) 0.509, v7 (two LoRAs) 0.516** (v5 0.515). The extra-data stylegram's +0.019 nested-CV gain did not transfer (-0.006), the second LoRA's +0.004 roughly did. Extra-data LoRA (`ntl-train-lora-x` v2, 6.1 h) OOF 0.4805, agreement with the plain LoRAs only 0.58, 9-member nested CV 0.5460 — suspect for the same reason. Probe v9 (plain blend) running. Pseudo-labelling simulation (cross-fit): hurts in 4 of 5 folds — dropped. Extra-data leave-one-class-out ablation (refreshed datasets, 13.2k rows): every class's extra rows help or are neutral (max +0.003 from dropping qwen3.8-max) — within noise. Hypothesis under test: public extra passages answer the train prompts, leaking held-out prompts into the training side of the CV.
- 2026-10-18 08:30 — Probe submitted as **v8** (kernel v9: the v7 blend with POSTPROCESS=False; 1 submission left today). Its score vs v7's 0.516 measures the whole cluster/twin/veto/Sinkhorn/Hungarian stack on the public split.
- 2026-10-18 08:31 — Leakage check (`scripts/check_extra_overlap.py`): train passages' max TF-IDF cosine to any extra passage (mean 0.081, none > 0.3) equals that to other train prompts (0.078); stylegram_x's per-prompt gain is flat across similarity terciles (Spearman 0.00), q8lora_x1 likewise. No topic leakage. Public LB noise for paired submissions is ~±0.01 (1,600 rows), so v6's -0.006 vs v5 is not evidence against the extra data; decisions stay CV-driven. `oof/stylegram_x.npz` replaced by the refreshed-data run (13.2k rows minus Kimi = what the kernel sees; OOF 0.4134).
- 2026-10-18 09:00 — Frozen v3 done (5,985 s): extra-data heads lift every backbone (q3 0.444->0.469, q25 0.427->0.451, deb 0.449->0.465, desklib 0.426->0.434, electra 0.438->0.447, new Qwen3-4B head 0.430->0.444). Blend refits: all 17 members nested 0.5442 (overfits), 8 `_x`-only 0.5438, chosen 11-member set (deb/desk plain+x, electra_x, q25_x, 3 LoRAs, lgb, stylegram_x) **nested 0.5468**; decoder `_x` heads get ~0 weight. Twins re-tuned on frozen v3 features (deb 0.2, electra 0.1, q25 0.1, lora 0.2). Submit kernel v10 pushed (lora-x in kernel_sources).
- 2026-10-18 09:12 — **Probe result: plain v7 blend 0.513 vs 0.516 with post-processing** -> clusters/twins/vetoes/Sinkhorn/Hungarian are worth ~+0.003 on the public split (noise level). Kept in the kernel but no further tuning. Submit kernel v10 ran clean (`_x` heads reuse cached backbone features: 494 s on the sample); **submitted as v9** (nested CV 0.5468; 0 submissions left today, 5 more from 03:00). Qwen2.5-7B extra-data LoRA (`kernels/train_lora_q25x`) pushes itself into the freed slot; Qwen3-4B LoRA still training.
2026-10-18 11:34 — v9 (submit kernel v10: 3 Qwen3-8B LoRAs + extra-data frozen heads, nested CV 0.5468) scored **0.518** on public LB (new best; v7 0.516). Qwen3-4B LoRA (ntl-train-lora-q34 v1) finished 11:33: OOF plain-head BA 0.4772, 10696 s.
2026-10-18 11:44 — q34lora_v1 (Qwen3-4B 36-layer LoRA, OOF 0.4772, agreement 0.72 with q8lora_v1) REJECTED: new scripts/cv_compare.py (10 fold seeds, nested weight fit) gives −0.0026 ± 0.0006 vs the 13-member v10 blend; q34_L7_22_x frozen head gets weight 0. Same-family members add nothing over the three Qwen3-8B LoRAs. blend/weights.json restored (13 members, nested 0.5468). Kernel log of submit v10 confirms only 3 sample test rows are visible at push time (494 s), so runtime is unmeasured locally; scoring rerun took ~40 min.
2026-10-18 12:40 — GPU quota (editor panel) 24:08 / 30 h remaining. Pushed ntl-train-lora-x2 (Qwen3-8B extra-data LoRA, seed 7, name q8lora_x2) into the free slot. New Oct-18 external datasets (scratchpad ntl_external_v3; scripts/exp_extra4.py): scraped corpora (oleg/speechmap/bullshit/HF traces) have family fidelity ≤0.12 on gemini/glm/kimi/qwen and HURT stylegram CV (−0.004 to −0.024); fresh proxy generations ≈ 0 to +0.004 (afify +0.004). semenoff own_or (glm/kimi/qwen via OpenRouter) fidelity family 0.06–0.33 → distribution mismatch.
2026-10-18 13:59 — Merged extra set v3 (scripts/build_extra_v3.py → data/extra_v3.csv, 19,535 rows from 11 public sources; private dataset vienas/ntl-extra-v3). stylegram_x3 on it: OOF 0.4245 (vs 0.4134) but in the 13-member blend −0.0008 ± 0.0007 (cv_compare, 10 seeds) → no blend gain; the other members already cover gemini-3.7/qwen. Frozen v4 (extra_v3, cap 300) queued for next GPU slot; train_lora_x3 (extra_v3, seed 42) prepared; train_lora_x2 (seed 7, old extras) running since 12:40.
2026-10-18 15:42 — q25lora_x1 (Qwen2.5-7B 20-layer extra-data LoRA, ntl-train-lora-q25x v1, 22,799 s): OOF 0.4831 (best single member), agreement with Qwen3 LoRAs 0.54–0.64. cv_compare 10 seeds: +0.0054 ± 0.0009 → ADOPTED. 14-member blend nested CV 0.5520 (v10 0.5468), installed into submit.py; kernel_sources + ntl-train-lora-q25x. Frozen v4 (extra_v3) pushed 15:34. Submit v11 push queued for next free slot; competition submission after 03:00 reset.
2026-10-18 17:28 — CORRECTION: editor panel 'Quota: 24:08 / 30 hrs' was hours USED; weekly GPU quota hit 30 h at 17:14 ('Maximum weekly GPU quota of 30.00 hours reached'). Submit v11 push blocked until the weekly reset; ntl-train-lora-x2 (started 10:54 UTC) keeps running. Frozen v4 heads on extra_v3 (cap 300): individually better (desk_x 0.4527 vs 0.4338, debv3l_x 0.4727 vs 0.4649, q25_x 0.4594 vs 0.4512) but swapped into the 14-member blend +0.0002 ± 0.0008 (x3heads) / −0.0009 with stylegram_x3 → NOT adopted; submit kernel stays on the old extra set. train_lora_x3 (extra_v3) dropped from the queue. Stopped push chain balyeh983; replaced by an hourly retry loop (push v11 → wait → submit after 03:00 → monitor).
2026-10-18 20:52 — q8lora_x2 (Qwen3-8B extra-data LoRA, seed 7, ntl-train-lora-x2 v1, 24,640 s): OOF 0.4712; cv_compare +0.0005 ± 0.0007 on the 14-member blend → NOT adopted. Submit v11 (14 members, nested 0.5520) still waiting for the weekly GPU quota reset (hourly retry loop, tmp/push_submit_v11.log).
2026-10-18 22:54 — STOPPED on user request: all local loops/watchers killed (v11 push loop, score watcher). No Kaggle kernels running. Submit v11 (14 members, nested 0.5520) is installed in kernels/submit/submit.py but NOT pushed/submitted.
